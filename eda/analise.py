"""Consolida os dados do perfil de recursos e dos textos extraídos.

Lê eda/dados/perfil_linha_tempo.csv, perfil_eventos.csv e os textos em
eda/dados/saidas/, e produz:

- eda/dados/resumo_modelos.csv : uma linha por modelo (tempo, CPU, RAM, CER...)
- eda/dados/ortografia.csv     : preservação de cada forma arcaica por modelo
- escrita tcc/tabelas/eda_resumo.tex : tabela pronta para \\input no TCC

Normalização antes do CER/WER: espaços e quebras de linha são colapsados e
marcações Markdown (#, *, _) removidas, porque alguns serviços devolvem o texto
formatado. Maiúsculas, acentos e pontuação são mantidos: são justamente o que
interessa avaliar em textos históricos.

Uso:
    python eda/analise.py
"""

import csv
import re
import sys
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(AQUI))

from benchmark.metrics import compute_cer, compute_wer  # noqa: E402
from pagina_sintetica import FORMAS_ARCAICAS  # noqa: E402

DADOS = AQUI / "dados"
PAGINAS = DADOS / "paginas" / "sintetica"
TABELAS = RAIZ / "escrita tcc" / "tabelas"

#: Rótulos e categorias usados em tabelas e gráficos.
MODELOS = {
    "tesseract": ("Tesseract", "Clássico"),
    "easyocr": ("EasyOCR", "Clássico"),
    "paddleocr": ("PaddleOCR", "Clássico"),
    "doctr": ("DocTR", "Clássico"),
    "qwen": ("Qwen2.5-VL 3B", "VLM local"),
    "qwen3-vl": ("Qwen3-VL 2B", "VLM local"),
    "glm-ocr": ("GLM-OCR", "VLM local"),
    "gemma4": ("Gemma 4", "VLM local"),
    "deepseek": ("DeepSeek-VL2", "VLM local"),
    "llamaparse-free": ("LlamaParse", "VLM via API"),
    "mistral": ("Mistral OCR", "VLM via API"),
    "gemini": ("Gemini 3.8 Flash", "VLM via API"),
}
MODELOS_OLLAMA = {"qwen", "qwen3-vl", "glm-ocr", "gemma4", "deepseek"}
VERSOES = {"001": "limpa", "002": "degradada"}


def normalizar(texto: str) -> str:
    texto = re.sub(r"[#*_`]", "", texto)
    return re.sub(r"\s+", " ", texto).strip()


def classificar_forma(texto: str, arcaica: str, moderna: str) -> str:
    """preservada | modernizada | ausente, para uma forma arcaica no texto."""
    palavras = re.findall(r"\w+", texto.lower())
    if arcaica in palavras:
        return "preservada"
    if moderna in palavras:
        return "modernizada"
    return "ausente"


def main():
    lt = pd.read_csv(DADOS / "perfil_linha_tempo.csv")
    ev = pd.read_csv(DADOS / "perfil_eventos.csv", dtype={"pagina": str})
    ev["pagina"] = ev["pagina"].str.zfill(3)

    # RAM atribuída ao modelo: processo Python + Ollama (só para VLMs locais).
    lt["ram_modelo_mb"] = lt["ram_python_mb"] + lt["ram_ollama_mb"].where(
        lt["modelo"].isin(MODELOS_OLLAMA), 0)
    lt["cpu_modelo_pct"] = lt["cpu_python_pct"] + lt["cpu_ollama_pct"].where(
        lt["modelo"].isin(MODELOS_OLLAMA), 0)

    linhas, orto = [], []
    for modelo, (rotulo, categoria) in MODELOS.items():
        amostras = lt[lt["modelo"] == modelo]
        eventos = ev[ev["modelo"] == modelo]
        if amostras.empty:
            continue
        def t(evento, pagina=None):
            sel = eventos[eventos["evento"] == evento]
            if pagina is not None:
                sel = sel[sel["pagina"] == pagina]
            return sel["t_s"].iloc[0] if not sel.empty else None

        carga = (t("fim_carga") - t("inicio_carga")) if t("fim_carga") is not None else None
        erros = eventos[eventos["evento"] == "erro"]
        inferencia = amostras[amostras["fase"] == "inferencia"]
        linha = {
            "modelo": modelo, "rotulo": rotulo, "categoria": categoria,
            "tempo_carga_s": round(carga, 1) if carga is not None else None,
            "ram_pico_mb": round(amostras["ram_modelo_mb"].max()),
            "cpu_media_inferencia_pct": round(inferencia["cpu_modelo_pct"].mean(), 1) if not inferencia.empty else None,
            "ram_sistema_inicio_mb": round(amostras["ram_sistema_usada_mb"].iloc[0]),
            "status": "ok" if erros.empty else erros["erro"].iloc[0].split(":")[0].strip(),
            "erro": "" if erros.empty else erros["erro"].iloc[0][:200],
        }
        referencia = normalizar((PAGINAS / "001.txt").read_text(encoding="utf-8"))
        for pagina, versao in VERSOES.items():
            ini, fim = t("inicio_pagina", pagina), t("fim_pagina", pagina)
            linha[f"tempo_{versao}_s"] = round(fim - ini, 1) if ini is not None and fim is not None else None
            saida = DADOS / "saidas" / modelo / f"{pagina}.txt"
            if fim is None or not saida.exists():
                linha[f"cer_{versao}"] = linha[f"wer_{versao}"] = None
                continue
            texto = saida.read_text(encoding="utf-8")
            hip = normalizar(texto)
            linha[f"cer_{versao}"] = round(compute_cer(hip, referencia), 4)
            linha[f"wer_{versao}"] = round(compute_wer(hip, referencia), 4)
            for arcaica, moderna in FORMAS_ARCAICAS.items():
                orto.append({"modelo": modelo, "rotulo": rotulo, "categoria": categoria,
                             "versao": versao, "forma": arcaica,
                             "resultado": classificar_forma(texto, arcaica, moderna)})
        if _limite_api(linha["erro"]):
            # O cliente da API repete a requisição com espera após o erro 429;
            # o tempo medido mistura inferência e espera, então não é reportado.
            linha["tempo_limpa_s"] = linha["tempo_degradada_s"] = None
        linhas.append(linha)

    resumo = pd.DataFrame(linhas)
    resumo.to_csv(DADOS / "resumo_modelos.csv", index=False)
    pd.DataFrame(orto).to_csv(DADOS / "ortografia.csv", index=False)
    escrever_tabela_latex(resumo)
    print(resumo.drop(columns=["erro"]).to_string(index=False))


def _fmt(v, casas=1, pct=False):
    if v is None or pd.isna(v):
        return "--"
    if pct:
        return f"{100 * v:.1f}".replace(".", ",")
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def escrever_tabela_latex(resumo: pd.DataFrame):
    """Gera a tabela de resumo em LaTeX (números no formato brasileiro)."""
    TABELAS.mkdir(exist_ok=True)
    linhas = []
    categoria_atual = None
    for _, r in resumo.iterrows():
        if categoria_atual is not None and r["categoria"] != categoria_atual:
            linhas.append("\\hline")
        categoria_atual = r["categoria"]
        status = "" if r["status"] == "ok" else "$^{*}$"
        linhas.append(
            f"{r['rotulo']}{status} & {_fmt(r['tempo_carga_s'])} & {_fmt(r['tempo_limpa_s'])} & "
            f"{_fmt(r['tempo_degradada_s'])} & {_fmt(r['ram_pico_mb'], 0)} & "
            f"{_fmt(r['cer_limpa'], pct=True)} & {_fmt(r['cer_degradada'], pct=True)} \\\\"
        )
    falhas = resumo[resumo["status"] != "ok"]
    notas = "; ".join(f"{r['rotulo']}: {descrever_falha(r)}" for _, r in falhas.iterrows())
    tex = r"""% Gerado automaticamente por eda/analise.py -- não editar à mão.
\begin{table}[H]
\centering
\caption{Perfil de execução dos modelos nas páginas sintéticas (A5, 300~DPI), no
ambiente da Tabela~\ref{tab:ambiente}.}
\label{tab:eda_resumo}
\renewcommand{\arraystretch}{1.25}
{\small
\begin{tabular}{l r r r r r r}
\hline
& \multicolumn{3}{c}{\textbf{Tempo (s)}} & \textbf{RAM} & \multicolumn{2}{c}{\textbf{CER (\%)}} \\
\textbf{Modelo} & \textbf{Carga} & \textbf{Limpa} & \textbf{Degr.} & \textbf{pico (MB)} & \textbf{Limpa} & \textbf{Degr.} \\
\hline
""" + "\n".join(linhas) + r"""
\hline
\end{tabular}
}
\fonte{Elaborado pelo autor.}
""" + (f"\\par\\footnotesize{{$^{{*}}$ {notas}.}}\n" if notas else "") + "\\end{table}\n"
    (TABELAS / "eda_resumo.tex").write_text(tex, encoding="utf-8")


def _limite_api(erro: str) -> bool:
    return "429" in erro or "quota" in erro.lower() or "Rate limit" in erro


def descrever_falha(r) -> str:
    erro = r["erro"]
    if "-9" in erro:
        return "processo encerrado pelo sistema por falta de memória"
    if _limite_api(erro):
        return ("limite de requisições da API atingido (tempos omitidos, pois incluem "
                "as novas tentativas automáticas do cliente)")
    if "unexpected EOF" in erro or "Server disconnected" in erro:
        return "servidor do Ollama interrompido durante a inferência"
    if "not found" in erro:
        return "modelo não disponível no Ollama"
    if "timeout" in erro:
        return "tempo limite excedido"
    return erro.split(":")[0]


if __name__ == "__main__":
    main()
