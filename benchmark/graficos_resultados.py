"""Figuras dos resultados de qualidade do benchmark (Cap. 6 do TCC).

Lê data/results/avaliacao/por_pagina.csv (gerado por avaliar.py) e as
métricas de execução (metricas.csv e metricas_gpu.csv) e salva PNGs (300 dpi) em
escrita tcc/imagens/:

- resultados_cer_custo.png    CER médio x tempo por página e x pico de RAM
- resultados_cer_paginas.png  CER de cada página, por modelo
- resultados_dificuldade.png  CER médio por classe de dificuldade
- resultados_heatmap.png      CER modelo x página, agrupado por obra

Segue as convenções de eda/graficos.py: a cor identifica a categoria do
modelo (Clássico, VLM local, VLM via API). Modelos executados em GPU na nuvem
aparecem com marcador vazado, porque o tempo deles foi medido em outro
hardware.

Uso:
    python benchmark/avaliar.py              # antes, para atualizar o CSV
    python benchmark/graficos_resultados.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "eda"))
from graficos import CORES, GRADE, MUDO, TINTA, TINTA_2, _num  # noqa: E402  (estilo comum)

AVALIACAO = RAIZ / "data" / "results" / "avaliacao" / "por_pagina.csv"
EXTRACAO = RAIZ / "data" / "results" / "extracao"
MANIFESTO = RAIZ / "data" / "corpus" / "regra_ouro" / "manifest.csv"
FIGURAS = RAIZ / "escrita tcc" / "imagens"

#: Nome de exibição e categoria (cor). GPU = executado na nuvem (marcador vazado).
MODELOS = {
    "abbyy": ("ABBYY FineReader", "Clássico"),
    "tesseract": ("Tesseract", "Clássico"),
    "paddleocr": ("PaddleOCR", "Clássico"),
    "doctr": ("DocTR", "Clássico"),
    "easyocr": ("EasyOCR", "Clássico"),
    "glm-ocr": ("GLM-OCR", "VLM local"),
    "qwen-7b": ("Qwen2.5-VL 7B", "VLM local"),
    "gemma4-e4b": ("Gemma 4 E4B", "VLM local"),
    "deepseek-ocr": ("DeepSeek-OCR", "VLM local"),
    "llamaparse-free": ("LlamaParse (básico)", "VLM via API"),
    "claude": ("Claude Opus 5.5", "VLM via API"),
}
GPU = {"qwen-7b", "gemma4-e4b", "deepseek-ocr"}
#: Regra ouro derivada da transcrição deste modelo (ver Cap. 4).
RESSALVA = {"claude"}
DIFICULDADES = ["Fácil", "Médio", "Difícil"]
#: Deslocamento (pontos) e alinhamento dos rótulos que colidiriam no padrão (5, 3).
ROTULO_TEMPO = {"gemma4-e4b": (6, 4, "left"), "qwen-7b": (-7, -3, "right")}
ROTULO_RAM = {"glm-ocr": (6, -9, "left"), "llamaparse-free": (6, 4, "left")}
MARCADORES = {"Fácil": "o", "Médio": "s", "Difícil": "^"}
TONS = {"Fácil": "#b5b4ad", "Médio": "#6f6e69", "Difícil": TINTA}
OBRAS = {"heloiza": "Heloíza", "culto_mulher_v1": "Culto v1", "culto_mulher_v2": "Culto v2",
         "amante_militar": "Amante", "feliz_independente": "Feliz Indep."}


def rotulo(modelo: str, n_paginas: int, total: int) -> str:
    nome = MODELOS.get(modelo, (modelo, ""))[0]
    if modelo in GPU:
        nome += " (GPU)"
    if modelo in RESSALVA:
        nome += " †"
    if n_paginas < total:
        nome += " *"
    return nome


def carregar():
    df = pd.read_csv(AVALIACAO, dtype={"pagina": str})
    df = df[df["modelo"].isin(MODELOS)]
    total = df.groupby("modelo").size().max()
    contagem = df.groupby("modelo").size()
    df["rotulo"] = [rotulo(m, contagem[m], total) for m in df["modelo"]]
    df["categoria"] = df["modelo"].map(lambda m: MODELOS[m][1])
    arquivos = [a for a in (EXTRACAO / "metricas.csv", EXTRACAO / "metricas_gpu.csv") if a.exists()]
    exec_ = pd.concat(pd.read_csv(a, dtype={"pagina": str}) for a in arquivos)
    exec_ = exec_.drop_duplicates(["modelo", "obra", "pagina"], keep="last")
    return df, exec_, total


def _salvar(fig, nome):
    fig.savefig(FIGURAS / nome, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figura_cer_custo(df, exec_):
    """CER médio contra custo de execução: tempo mediano e pico de RAM."""
    cer = df.groupby("modelo").agg(cer=("cer", "mean"), rotulo=("rotulo", "first"),
                                   categoria=("categoria", "first"))
    custo = exec_.groupby("modelo").agg(tempo=("tempo_s", "median"), ram=("ram_pico_mb", "max"))
    d = cer.join(custo, how="inner")

    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for modelo, r in d.iterrows():
        cor = CORES[r["categoria"]]
        vazado = modelo in GPU
        ax_t.scatter(r["tempo"], 100 * r["cer"], s=42, zorder=3, linewidths=1.6,
                     facecolors="white" if vazado else cor, edgecolors=cor)
        dx, dy, ha = ROTULO_TEMPO.get(modelo, (5, 3, "left"))
        ax_t.annotate(r["rotulo"], (r["tempo"], 100 * r["cer"]), xytext=(dx, dy), ha=ha,
                      textcoords="offset points", fontsize=7, color=TINTA_2)
        if not vazado:  # RAM da nuvem não é comparável à do equipamento do trabalho
            ax_r.scatter(r["ram"] / 1024, 100 * r["cer"], s=42, zorder=3, color=cor,
                         edgecolors="white", linewidths=1)
            dx, dy, ha = ROTULO_RAM.get(modelo, (5, 3, "left"))
            ax_r.annotate(r["rotulo"], (r["ram"] / 1024, 100 * r["cer"]), xytext=(dx, dy), ha=ha,
                          textcoords="offset points", fontsize=7, color=TINTA_2)

    ax_t.set_xscale("log")
    ax_t.set_xlabel("Tempo mediano por página (s, escala log)")
    ax_t.set_ylabel("CER médio (%)")
    ax_t.set_title("Qualidade x tempo")
    ax_r.axvline(7.5, color=MUDO, linewidth=0.8, linestyle="--")
    ax_r.text(7.4, ax_r.get_ylim()[1] * 0.97, "RAM total\ndo equipamento", ha="right", va="top",
              fontsize=7, color=MUDO)
    ax_r.set_xlabel("Pico de RAM (GB), equipamento sem GPU")
    ax_r.set_title("Qualidade x memória")
    for ax in (ax_t, ax_r):
        ax.yaxis.grid(True, color=GRADE, linewidth=0.6)
        ax.set_axisbelow(True)

    alcas = [plt.Line2D([], [], marker="o", linestyle="", color=CORES[c], label=c)
             for c in ["Clássico", "VLM local", "VLM via API"]]
    alcas.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
                            markeredgecolor=CORES["VLM local"], label="GPU (nuvem)"))
    fig.legend(handles=alcas, loc="upper center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    _salvar(fig, "resultados_cer_custo.png")


def _ordem(df):
    """Modelos do menor para o maior CER médio (de cima para baixo)."""
    return df.groupby("rotulo")["cer"].mean().sort_values().index.tolist()


def figura_cer_paginas(df):
    """Cada página é um ponto; o traço vertical marca a mediana do modelo."""
    ordem = _ordem(df)
    fig, ax = plt.subplots(figsize=(6.4, 0.36 * len(ordem) + 1.2))
    rng = np.random.default_rng(0)
    for y, nome in enumerate(ordem):
        d = df[df["rotulo"] == nome]
        cor = CORES[d["categoria"].iloc[0]]
        for dif in DIFICULDADES:
            dd = d[d["dificuldade"] == dif]
            ax.scatter(100 * dd["cer"], y + rng.uniform(-0.18, 0.18, len(dd)), s=18,
                       marker=MARCADORES[dif], color=cor, alpha=0.85, edgecolors="white",
                       linewidths=0.5, zorder=3)
        ax.plot([100 * d["cer"].median()] * 2, [y - 0.32, y + 0.32], color=TINTA, linewidth=1.4, zorder=4)
    ax.set_yticks(range(len(ordem)), ordem)
    ax.invert_yaxis()
    ax.set_xlim(-2, 105)
    ax.set_xlabel("CER da página (%)")
    ax.xaxis.grid(True, color=GRADE, linewidth=0.6)
    ax.set_axisbelow(True)
    alcas = [plt.Line2D([], [], marker=MARCADORES[d], linestyle="", color=MUDO, label=d)
             for d in DIFICULDADES]
    alcas.append(plt.Line2D([], [], color=TINTA, linewidth=1.4, label="Mediana"))
    ax.legend(handles=alcas, loc="lower right", fontsize=7.5)
    _salvar(fig, "resultados_cer_paginas.png")


def figura_dificuldade(df):
    """CER médio por classe; a linha liga as três classes de cada modelo."""
    ordem = _ordem(df)
    medias = df.groupby(["rotulo", "dificuldade"])["cer"].mean().unstack()
    fig, ax = plt.subplots(figsize=(6.4, 0.36 * len(ordem) + 1.2))
    for y, nome in enumerate(ordem):
        valores = medias.loc[nome].reindex(DIFICULDADES).dropna()
        ax.plot(100 * valores.values, [y] * len(valores), color=GRADE, linewidth=2, zorder=2)
        for dif, v in valores.items():
            ax.scatter(100 * v, y, marker=MARCADORES[dif], s=36, color=TONS[dif],
                       edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_yticks(range(len(ordem)), ordem)
    ax.invert_yaxis()
    ax.set_xlabel("CER médio (%)")
    ax.xaxis.grid(True, color=GRADE, linewidth=0.6)
    ax.set_axisbelow(True)
    alcas = [plt.Line2D([], [], marker=MARCADORES[d], linestyle="", color=TONS[d], label=d)
             for d in DIFICULDADES]
    ax.legend(handles=alcas, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5,
              title="Dificuldade", title_fontsize=7.5)
    _salvar(fig, "resultados_dificuldade.png")


def figura_heatmap(df):
    """CER de cada página (colunas, agrupadas por obra) para cada modelo (linhas)."""
    ordem = _ordem(df)
    manifesto = pd.read_csv(MANIFESTO, dtype={"pagina": str})
    colunas = list(zip(manifesto["obra_slug"], manifesto["pagina"]))
    matriz = (df.assign(chave=list(zip(df["obra"], df["pagina"])))
                .pivot(index="rotulo", columns="chave", values="cer")
                .reindex(index=ordem, columns=colunas))

    # Rampa sequencial de um só tom (azul da paleta), clara -> escura.
    cmap = LinearSegmentedColormap.from_list("cer", ["#f4f7fc", CORES["Clássico"], "#0d2f5c"])
    cmap.set_bad("#ffffff")
    fig, ax = plt.subplots(figsize=(7.4, 0.33 * len(ordem) + 1.6))
    im = ax.imshow(100 * matriz.values, cmap=cmap, vmin=0, vmax=50, aspect="auto")
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            v = matriz.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=6, color=MUDO)
            elif v >= 0.10:
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if v >= 0.25 else TINTA)
    ax.set_yticks(range(len(ordem)), ordem, fontsize=7.5)
    ax.set_xticks(range(len(colunas)), [p.lstrip("0") for _, p in colunas], fontsize=6.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # Separadores e nomes das obras acima das colunas.
    inicio = 0
    for obra, grupo in manifesto.groupby("obra_slug", sort=False):
        fim = inicio + len(grupo)
        if inicio:
            ax.axvline(inicio - 0.5, color="white", linewidth=2.5)
        dif = grupo["dificuldade"].iloc[0]
        ax.text((inicio + fim - 1) / 2, -0.9, f"{OBRAS[obra]}\n({dif})", ha="center", va="bottom",
                fontsize=7, color=TINTA_2)
        inicio = fim
    ax.set_xlabel("Página do PDF")
    barra = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, extend="max")
    barra.set_label("CER (%)", fontsize=7.5)
    barra.ax.tick_params(labelsize=7)
    barra.outline.set_visible(False)
    _salvar(fig, "resultados_heatmap.png")


def main():
    df, exec_, _ = carregar()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    figura_cer_custo(df, exec_)
    figura_cer_paginas(df)
    figura_dificuldade(df)
    figura_heatmap(df)
    print(f"Figuras resultados_*.png geradas em {FIGURAS}")


if __name__ == "__main__":
    main()
