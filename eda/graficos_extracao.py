"""Figuras e tabela da extração sobre as páginas reais do corpus (Cap. 6).

Lê data/results/extracao/metricas.csv (gerado por benchmark/extrair.py) e o
manifesto da amostra, e produz, sem depender da regra ouro:

- escrita tcc/imagens/eda_real_tempo.pdf       tempo por página, cada página um ponto
- escrita tcc/imagens/eda_real_resolucao.pdf   tempo médio em páginas de ~150 e 300 ppi
- escrita tcc/imagens/eda_real_ram.pdf         pico de memória por modelo
- escrita tcc/tabelas/extracao_resumo.tex      tabela-resumo com projeção para o corpus

A CPU não é reportada: o Tesseract cria um processo novo por página e a
amostragem por intervalo não mede esse caso de forma confiável.

Uso:
    python eda/graficos_extracao.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from graficos import CORES, EIXO, GRADE, MUDO, TINTA, TINTA_2, _num  # estilo comum
import matplotlib.ticker

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
METRICAS = RAIZ / "data" / "results" / "extracao" / "metricas.csv"
MANIFESTO = RAIZ / "data" / "corpus" / "regra_ouro" / "manifest.csv"
FIGURAS = RAIZ / "escrita tcc" / "imagens"
TABELAS = RAIZ / "escrita tcc" / "tabelas"

MODELOS = {
    "tesseract": ("Tesseract", "Clássico"),
    "paddleocr": ("PaddleOCR", "Clássico"),
    "doctr": ("DocTR", "Clássico"),
    "easyocr": ("EasyOCR", "Clássico"),
    "glm-ocr": ("GLM-OCR", "VLM local"),
    "qwen": ("Qwen2.5-VL 3B", "VLM local"),
    "qwen3-vl": ("Qwen3-VL 2B", "VLM local"),
    "llamaparse-free": ("LlamaParse (básico)", "VLM via API"),
    "llamaparse-premium": ("LlamaParse (Agentic)", "VLM via API"),
    "llamaparse-lvm": ("LlamaParse (LVM)", "VLM via API"),
    "llamaparse-layout": ("LlamaParse (Layout)", "VLM via API"),
    "gemini": ("Gemini", "VLM via API"),
    "mistral": ("Mistral OCR", "VLM via API"),
    "claude": ("Claude Opus 5.5", "VLM via API"),
}

# Resolução nativa das digitalizações (ppi das imagens embutidas nos PDFs).
PPI = {"heloiza": 150, "culto_mulher_v1": 137, "culto_mulher_v2": 150,
       "amante_militar": 300, "feliz_independente": 300}
# Páginas de cada obra no corpus completo (data/catalog.yaml).
PAGINAS_OBRA = {"heloiza": 144, "culto_mulher_v1": 254, "culto_mulher_v2": 219,
                "amante_militar": 110, "feliz_independente": 1043}


def carregar() -> pd.DataFrame:
    m = pd.read_csv(METRICAS, dtype={"pagina": str})
    man = pd.read_csv(MANIFESTO, dtype={"pagina": str})
    m = m.merge(man[["obra_slug", "pagina", "dificuldade"]],
                left_on=["obra", "pagina"], right_on=["obra_slug", "pagina"], how="left")
    m["resolucao"] = m["obra"].map(lambda o: "300 ppi" if PPI[o] == 300 else "~150 ppi")
    m = m[m["modelo"].isin(MODELOS)]
    m["rotulo"] = m["modelo"].map(lambda k: MODELOS[k][0])
    m["categoria"] = m["modelo"].map(lambda k: MODELOS[k][1])
    return m


def figura_tempo(m: pd.DataFrame):
    ordem = m.groupby("modelo")["tempo_s"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(6.2, 0.45 * len(ordem) + 1.2))
    rng = np.random.default_rng(0)
    for y, modelo in enumerate(ordem):
        d = m[m["modelo"] == modelo]
        cor = CORES[d["categoria"].iloc[0]]
        jitter = rng.uniform(-0.16, 0.16, len(d))
        ax.scatter(d["tempo_s"], y + jitter, s=16, color=cor, alpha=0.75, edgecolor="white", linewidth=0.4, zorder=2)
        med = d["tempo_s"].median()
        ax.plot([med, med], [y - 0.3, y + 0.3], color=TINTA, linewidth=1.6, zorder=3)
        texto = f"{_num(med)} s" if med < 60 else f"{_num(med / 60)} min"
        ax.text(d["tempo_s"].max() * 1.25, y, texto, va="center", fontsize=8, color=TINTA)
    ax.set_yticks(range(len(ordem)), [MODELOS[k][0] for k in ordem])
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _num(v, 0) if v >= 1 else _num(v)))
    ax.set_xlim(0.4, m["tempo_s"].max() * 4)
    ax.set_xlabel("segundos por página (escala logarítmica); traço = mediana")
    ax.set_title("Tempo de processamento por página")
    ax.xaxis.grid(True, color=GRADE, linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", color="none", markerfacecolor=CORES[c], markersize=6, label=c)
               for c in CORES if c in set(m["categoria"])]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURAS / "eda_real_tempo.pdf")
    plt.close(fig)


def figura_resolucao(m: pd.DataFrame):
    t = m.groupby(["modelo", "resolucao"])["tempo_s"].mean().unstack()
    t = t.dropna(subset=["~150 ppi"]).sort_values("~150 ppi")
    fig, ax = plt.subplots(figsize=(6.2, 0.45 * len(t) + 1.2))
    for y, (modelo, r) in enumerate(t.iterrows()):
        cor = CORES[MODELOS[modelo][1]]
        a, b = r["~150 ppi"], r.get("300 ppi")
        if pd.notna(b):
            ax.plot([a, b], [y, y], color=EIXO, linewidth=2, zorder=1)
            ax.scatter([b], [y], s=46, color=cor, edgecolor="white", linewidth=1, zorder=3)
            ax.text(max(a, b) * 1.25, y, f"×{_num(b / a)}", va="center", fontsize=8, color=TINTA)
        ax.scatter([a], [y], s=46, facecolor="white", edgecolor=cor, linewidth=1.6, zorder=2)
    ax.set_yticks(range(len(t)), [MODELOS[k][0] for k in t.index])
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _num(v, 0) if v >= 1 else _num(v)))
    ax.set_xlim(0.8, t.max().max() * 4)
    ax.set_xlabel("segundos por página, média (escala logarítmica)")
    ax.set_title("Efeito da resolução da digitalização no tempo")
    ax.xaxis.grid(True, color=GRADE, linewidth=0.6)
    ax.set_axisbelow(True)
    vazio = plt.Line2D([], [], marker="o", color="none", markerfacecolor="white",
                       markeredgecolor=TINTA_2, markersize=6, label="~150 ppi (Heloíza, Culto)")
    cheio = plt.Line2D([], [], marker="o", color="none", markerfacecolor=TINTA_2,
                       markersize=6, label="300 ppi (Amante, Feliz)")
    ax.legend(handles=[vazio, cheio], loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURAS / "eda_real_resolucao.pdf")
    plt.close(fig)


def figura_ram(m: pd.DataFrame):
    r = m.groupby("modelo")["ram_pico_mb"].max().sort_values()
    total = psutil.virtual_memory().total / 1024**2
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(r) + 1.2))
    y = np.arange(len(r))
    ax.barh(y, r.values, color=[CORES[MODELOS[k][1]] for k in r.index], height=0.62,
            edgecolor="white", linewidth=1.5)
    for yi, v in zip(y, r.values):
        ax.text(v + total * 0.012, yi, _num(v, 0), va="center", fontsize=8, color=TINTA)
    ax.axvline(total, color=MUDO, linestyle="--", linewidth=0.8)
    ax.text(total, -0.45, f" RAM total\n ({_num(total / 1024)} GB)", color=TINTA_2, fontsize=7.5, va="bottom")
    ax.set_yticks(y, [MODELOS[k][0] for k in r.index])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, total * 1.18)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _num(v, 0)))
    ax.set_xlabel("MB (maior valor entre as páginas)")
    ax.set_title("Pico de memória RAM por modelo")
    ax.xaxis.grid(True, color=GRADE, linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CORES[c]) for c in CORES if c in set(m["categoria"])]
    ax.legend(handles, [c for c in CORES if c in set(m["categoria"])], loc="center right", fontsize=8, handlelength=1)
    fig.tight_layout()
    fig.savefig(FIGURAS / "eda_real_ram.pdf")
    plt.close(fig)


def tabela(m: pd.DataFrame):
    """Resumo por modelo, com projeção do tempo para o corpus completo.

    A projeção usa o tempo médio de cada resolução, ponderado pelo número de
    páginas do corpus em cada uma (617 a ~150 ppi e 1.153 a 300 ppi).
    """
    pag150 = sum(v for o, v in PAGINAS_OBRA.items() if PPI[o] != 300)
    pag300 = sum(v for o, v in PAGINAS_OBRA.items() if PPI[o] == 300)
    linhas = []
    ordem = m.groupby("modelo")["tempo_s"].median().sort_values().index
    for modelo in ordem:
        d = m[m["modelo"] == modelo]
        t150 = d.loc[d["resolucao"] == "~150 ppi", "tempo_s"].mean()
        t300 = d.loc[d["resolucao"] == "300 ppi", "tempo_s"].mean()
        horas = (pag150 * t150 + pag300 * t300) / 3600 if pd.notna(t150) and pd.notna(t300) else None
        proj = "--" if horas is None else (f"{_num(horas * 60, 0)} min" if horas < 1 else f"{_num(horas)} h")
        if horas is not None and horas >= 48:
            proj = f"{_num(horas / 24)} dias"
        nota = "$^{*}$" if len(d) < 20 else ""
        linhas.append(
            f"{MODELOS[modelo][0]}{nota} & {len(d)} & {_num(d['tempo_s'].median())} & "
            f"{_num(d['tempo_s'].min())}--{_num(d['tempo_s'].max())} & "
            f"{_num(d['ram_pico_mb'].max(), 0)} & {proj} \\\\")
    incompletos = [MODELOS[k][0] for k in ordem if (m["modelo"] == k).sum() < 20]
    nota = ("\\par\\footnotesize{$^{*}$ " + ", ".join(incompletos) +
            ": páginas pendentes nesta rodada (ver texto); projeção feita com as páginas concluídas.}\n"
            if incompletos else "")
    tex = r"""% Gerado automaticamente por eda/graficos_extracao.py -- não editar à mão.
\begin{table}[H]
\centering
\caption{Execução dos modelos sobre as 20 páginas da amostra, no ambiente da
Tabela~\ref{tab:ambiente}.}
\label{tab:extracao_resumo}
\renewcommand{\arraystretch}{1.25}
{\small
\begin{tabular}{l r r r r r}
\hline
& & \multicolumn{2}{c}{\textbf{Tempo por página (s)}} & \textbf{RAM} & \textbf{Projeção} \\
\textbf{Modelo} & \textbf{Págs.} & \textbf{Mediana} & \textbf{Mín.--máx.} & \textbf{pico (MB)} & \textbf{corpus} \\
\hline
""" + "\n".join(linhas) + r"""
\hline
\end{tabular}
}
\fonte{Elaborado pelo autor.}
""" + nota + "\\end{table}\n"
    TABELAS.mkdir(exist_ok=True)
    (TABELAS / "extracao_resumo.tex").write_text(tex, encoding="utf-8")


def main():
    m = carregar()
    figura_tempo(m)
    figura_resolucao(m)
    figura_ram(m)
    tabela(m)
    print("Figuras eda_real_*.pdf e tabela extracao_resumo.tex geradas.")


if __name__ == "__main__":
    main()
