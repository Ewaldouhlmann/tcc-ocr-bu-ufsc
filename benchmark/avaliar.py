"""Etapa 2 do benchmark: compara as saídas salvas com a regra ouro (CER/WER).

Não roda nenhum modelo: lê os textos gerados por extrair.py (e os coletados
por sessao_claude.py) em data/results/extracao/{modelo}/{obra}/{pagina}.txt e
compara cada um com data/corpus/regra_ouro/{obra}/{pagina}.txt. Toda pasta de
modelo encontrada em --saidas entra na avaliação.

Normalização (aplicada igualmente à referência e às saídas):
- Unicode NFC;
- s longo (ſ) -> s;
- marcação Markdown/HTML removida (#, *, _, `, >, |, <tags>);
- palavra hifenizada no fim da linha é reunida (afe-\\ntividade -> afetividade);
- aspas, travessões e reticências unificados (“ ” -> ", – -> —, … -> ...);
- espaço antes de , ; : . ! ? removido;
- espaços e quebras de linha colapsados em um espaço.
Maiúsculas, acentos e pontuação são mantidos: a fidelidade ortográfica é
justamente o que se quer medir.

Saídas em data/results/avaliacao/:
- por_pagina.csv          CER/WER por modelo x página
- resumo_modelos.csv      médias por modelo, geral e por dificuldade
- benchmark.xlsx          planilha (abas Resultados e Resumo por Modelo)
e a tabela escrita tcc/tabelas/avaliacao_resumo.tex.

Uso:
    python benchmark/avaliar.py
"""

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from benchmark.metrics import compute_cer, compute_wer, save_report  # noqa: E402

REGRA_OURO = RAIZ / "data" / "corpus" / "regra_ouro"
MANIFESTO = REGRA_OURO / "manifest.csv"
SAIDAS = RAIZ / "data" / "results" / "extracao"
DESTINO = RAIZ / "data" / "results" / "avaliacao"
TABELAS = RAIZ / "escrita tcc" / "tabelas"

#: Nome de exibição e categoria; modelos fora daqui aparecem pelo nome da pasta.
MODELOS = {
    "abbyy": ("ABBYY FineReader", "Comercial"),
    "tesseract": ("Tesseract", "Clássico"),
    "paddleocr": ("PaddleOCR", "Clássico"),
    "doctr": ("DocTR", "Clássico"),
    "easyocr": ("EasyOCR", "Clássico"),
    "glm-ocr": ("GLM-OCR", "VLM local"),
    "qwen": ("Qwen2.5-VL 3B", "VLM local"),
    "qwen3-vl": ("Qwen3-VL 2B", "VLM local"),
    "qwen-7b": ("Qwen2.5-VL 7B", "VLM local (GPU)"),
    "gemma4-e4b": ("Gemma 4 E4B", "VLM local (GPU)"),
    "deepseek-ocr": ("DeepSeek-OCR", "VLM local (GPU)"),
    "paddleocr-vl": ("PaddleOCR-VL", "VLM local (GPU)"),
    "llamaparse-free": ("LlamaParse (básico)", "VLM via API"),
    "gemini": ("Gemini", "VLM via API"),
    "mistral": ("Mistral OCR", "VLM via API"),
    "claude": ("Claude Opus 5.5", "VLM via API"),
}
#: Modelos cujo resultado não é comparável aos demais (ver nota da tabela).
RESSALVA = {"claude"}
DIFICULDADES = ["Fácil", "Médio", "Difícil"]


def normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFC", texto).replace("ſ", "s")
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"^\s*(#+|>+|[-*+]\s)\s*", "", t, flags=re.M)
    t = re.sub(r"[*_`|]", " ", t)
    t = re.sub(r"(\w)-[ \t]*\n\s*(\w)", r"\1\2", t)
    t = t.translate(str.maketrans({"“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
                                   "‘": "'", "’": "'", "–": "—"}))
    t = t.replace("…", "...")
    t = re.sub(r"\s+([,;:.!?])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def ler_manifesto() -> dict[tuple[str, str], dict]:
    with open(MANIFESTO, encoding="utf-8") as f:
        return {(l["obra_slug"], l["pagina"]): l for l in csv.DictReader(f)}


def avaliar(saidas: Path) -> pd.DataFrame:
    manifesto = ler_manifesto()
    referencias = {k: normalizar((REGRA_OURO / k[0] / f"{k[1]}.txt").read_text(encoding="utf-8"))
                   for k in manifesto}
    linhas = []
    for pasta in sorted(p for p in saidas.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for (obra, pagina), info in manifesto.items():
            arquivo = pasta / obra / f"{pagina}.txt"
            if not arquivo.exists():
                continue
            hipotese = normalizar(arquivo.read_text(encoding="utf-8"))
            referencia = referencias[(obra, pagina)]
            linhas.append({
                "modelo": pasta.name,
                "obra": obra,
                "pagina": pagina,
                "dificuldade": info["dificuldade"],
                # jiwer não aceita hipótese vazia; saída vazia = 100% de erro.
                "cer": compute_cer(hipotese, referencia) if hipotese else 1.0,
                "wer": compute_wer(hipotese, referencia) if hipotese else 1.0,
            })
    return pd.DataFrame(linhas)


def resumir(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for modelo, d in df.groupby("modelo"):
        nome, categoria = MODELOS.get(modelo, (modelo, ""))
        linha = {"modelo": modelo, "nome": nome, "categoria": categoria, "paginas": len(d),
                 "cer": d["cer"].mean(), "wer": d["wer"].mean()}
        for dif in DIFICULDADES:
            dd = d[d["dificuldade"] == dif]
            linha[f"cer_{dif}"] = dd["cer"].mean() if len(dd) else None
            linha[f"wer_{dif}"] = dd["wer"].mean() if len(dd) else None
        linhas.append(linha)
    return pd.DataFrame(linhas).sort_values("cer").reset_index(drop=True)


def _pct(v) -> str:
    return "--" if pd.isna(v) else f"{100 * v:.1f}".replace(".", ",")


def tabela_tex(resumo: pd.DataFrame, total: int):
    linhas, notas = [], []
    for _, r in resumo.iterrows():
        nome = r["nome"]
        if r["modelo"] in RESSALVA:
            nome += r"$^{\dagger}$"
        if r["paginas"] < total:
            nome += r"$^{*}$"
        celulas = [f"{_pct(r[f'cer_{d}'])} / {_pct(r[f'wer_{d}'])}" for d in DIFICULDADES]
        linhas.append(f"{nome} & {r['paginas']} & {_pct(r['cer'])} & {_pct(r['wer'])} & "
                      + " & ".join(celulas) + r" \\")
    if (resumo["paginas"] < total).any():
        notas.append(r"$^{*}$ Avaliado só nas páginas concluídas; as médias não são "
                     r"diretamente comparáveis às dos demais modelos.")
    if resumo["modelo"].isin(RESSALVA).any():
        notas.append(r"$^{\dagger}$ A transcrição inicial da regra ouro foi gerada por este "
                     r"modelo e revisada pelo autor; o resultado tende a ser otimista.")
    nota = "\n".join(r"\par\footnotesize{" + n + "}" for n in notas)
    tex = r"""% Gerado automaticamente por benchmark/avaliar.py -- não editar à mão.
\begin{table}[H]
\centering
\caption{CER e WER médios (\%) de cada modelo sobre as páginas da amostra, em
geral e por classe de dificuldade (CER / WER).}
\label{tab:avaliacao_resumo}
\renewcommand{\arraystretch}{1.25}
{\small
\begin{tabular}{l r r r c c c}
\hline
& & \multicolumn{2}{c}{\textbf{Geral}} & \textbf{Fácil} & \textbf{Médio} & \textbf{Difícil} \\
\textbf{Modelo} & \textbf{Págs.} & \textbf{CER} & \textbf{WER} & \textbf{CER / WER} & \textbf{CER / WER} & \textbf{CER / WER} \\
\hline
""" + "\n".join(linhas) + r"""
\hline
\end{tabular}
}
\fonte{Elaborado pelo autor.}
""" + nota + "\n" + r"\end{table}" + "\n"
    (TABELAS / "avaliacao_resumo.tex").write_text(tex, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--saidas", default=str(SAIDAS), help="pasta com uma subpasta por modelo")
    args = parser.parse_args()

    df = avaliar(Path(args.saidas))
    resumo = resumir(df)
    DESTINO.mkdir(parents=True, exist_ok=True)
    df.to_csv(DESTINO / "por_pagina.csv", index=False)
    resumo.to_csv(DESTINO / "resumo_modelos.csv", index=False)

    # Planilha com tempo e tokens da extração, quando houver (o Claude via
    # sessão e o ABBYY não têm essas medições). metricas_gpu.csv vem da rodada
    # em GPU na nuvem (notebooks/benchmark_gpu.ipynb), em outro hardware.
    arquivos = [a for a in (SAIDAS / "metricas.csv", SAIDAS / "metricas_gpu.csv") if a.exists()]
    if arquivos:
        m = pd.concat(pd.read_csv(a, dtype={"pagina": str}) for a in arquivos)
        m = m[["modelo", "obra", "pagina", "tempo_s", "tokens"]]
        df = df.merge(m.drop_duplicates(["modelo", "obra", "pagina"], keep="last"),
                      on=["modelo", "obra", "pagina"], how="left")
    save_report(df.astype(object).where(df.notna(), None).to_dict("records"), DESTINO / "benchmark.xlsx")

    tabela_tex(resumo, total=len(ler_manifesto()))

    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(resumo[["nome", "paginas", "cer", "wer"] + [f"cer_{d}" for d in DIFICULDADES]]
              .round(3).to_string(index=False))
    print(f"\nArquivos em {DESTINO} e tabela em {TABELAS / 'avaliacao_resumo.tex'}")


if __name__ == "__main__":
    main()
