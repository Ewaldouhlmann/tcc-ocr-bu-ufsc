"""Gera as figuras da análise exploratória para o TCC.

Lê os CSVs produzidos por perfil_recursos.py e analise.py e salva PDFs
vetoriais em escrita tcc/imagens/ (prefixo eda_), prontos para
\\includegraphics.

Convenções visuais (iguais em todas as figuras):
- a cor identifica a CATEGORIA do modelo (Clássico, VLM local, VLM via API),
  nunca a posição no ranking;
- barras recebem o valor escrito ao lado, para leitura em impressão P&B;
- um único eixo por gráfico; medidas diferentes vão em painéis separados.

Uso:
    python eda/graficos.py
"""

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

from analise import MODELOS, MODELOS_OLLAMA  # noqa: E402

AQUI = Path(__file__).resolve().parent
DADOS = AQUI / "dados"
FIGURAS = AQUI.parent / "escrita tcc" / "imagens"

# Paleta categórica validada (ordem fixa; ver README da pasta).
CORES = {"Clássico": "#2a78d6", "VLM local": "#eb6834", "VLM via API": "#1baf7a"}
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
MUDO = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.edgecolor": EIXO,
    "axes.labelcolor": TINTA_2,
    "axes.titlesize": 10,
    "axes.titlecolor": TINTA,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": MUDO,
    "ytick.color": MUDO,
    "xtick.labelcolor": TINTA_2,
    "ytick.labelcolor": TINTA_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def _num(v: float, casas: int = 1) -> str:
    """Formato brasileiro: 1.234,5"""
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _grade_x(ax):
    ax.xaxis.grid(True, color=GRADE, linewidth=0.6)
    ax.set_axisbelow(True)


def _legenda_categorias(ax, categorias, loc="lower right"):
    handles = [plt.Rectangle((0, 0), 1, 1, color=CORES[c]) for c in categorias]
    ax.legend(handles, categorias, loc=loc, fontsize=8, handlelength=1)


def barras_horizontais(df, coluna, titulo, rotulo_x, arquivo, casas=1, log=False, linhas_ref=()):
    df = df.dropna(subset=[coluna]).sort_values(coluna)
    fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(df) + 1.1))
    y = np.arange(len(df))
    cores = [CORES[c] for c in df["categoria"]]
    ax.barh(y, df[coluna], color=cores, height=0.62, edgecolor="white", linewidth=1.5)
    ax.set_yticks(y, df["rotulo"])
    ax.tick_params(axis="y", length=0)
    if log:
        ax.set_xscale("log")
    for yi, v in zip(y, df[coluna]):
        ax.text(v * (1.08 if log else 1) + (0 if log else df[coluna].max() * 0.015), yi,
                _num(v, casas), va="center", fontsize=8, color=TINTA)
    for valor, texto in linhas_ref:
        ax.axvline(valor, color=MUDO, linestyle="--", linewidth=0.8)
        ax.text(valor, -0.45, f" {texto}", color=TINTA_2, fontsize=7.5, va="bottom", ha="left")
    limite = max(df[coluna].max(), *(v for v, _ in linhas_ref)) if linhas_ref else df[coluna].max()
    ax.set_xlim(right=limite * (3 if log else 1.18))
    ax.set_xlabel(rotulo_x)
    ax.set_title(titulo)
    if not log:
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _num(v, 0)))
    _grade_x(ax)
    _legenda_categorias(ax, [c for c in CORES if c in set(df["categoria"])],
                        loc="center right" if linhas_ref else "lower right")
    fig.tight_layout()
    fig.savefig(FIGURAS / arquivo)
    plt.close(fig)
    print(f"  {arquivo}")


def figura_tempo(resumo):
    """Tempo médio por página (limpa e degradada)."""
    df = resumo.copy()
    df["tempo_pagina_s"] = df[["tempo_limpa_s", "tempo_degradada_s"]].mean(axis=1)
    barras_horizontais(df, "tempo_pagina_s", "Tempo de processamento por página",
                       "segundos por página", "eda_tempo_pagina.pdf")


def figura_ram(resumo):
    total = psutil.virtual_memory().total / 1024**2
    inicio = resumo["ram_sistema_inicio_mb"].median()
    livre = total - inicio
    barras_horizontais(
        resumo, "ram_pico_mb", "Pico de memória RAM por modelo", "MB",
        "eda_ram_pico.pdf", casas=0,
        linhas_ref=[(livre, f"livre no início\n(~{_num(livre / 1024)} GB)"),
                    (total, f"RAM total\n({_num(total / 1024)} GB)")],
    )


def figura_linha_tempo(lt, resumo):
    """Pequenos múltiplos: CPU (linha de cima) e RAM (linha de baixo) ao longo
    da execução de cada modelo local. Modelos via API ficam de fora: o custo
    computacional deles está no provedor, não nesta máquina."""
    locais = [m for m in resumo["modelo"] if MODELOS[m][1] != "VLM via API"]
    n = len(locais)
    fig, eixos = plt.subplots(2, n, figsize=(1.65 * n + 0.6, 3.6), sharey="row", squeeze=False)
    for j, modelo in enumerate(locais):
        rotulo, categoria = MODELOS[modelo]
        a = lt[lt["modelo"] == modelo]
        cor = CORES[categoria]
        for i, (col, lim) in enumerate((("cpu_modelo_pct", 100), ("ram_modelo_mb", None))):
            ax = eixos[i][j]
            ax.plot(a["t_s"], a[col], color=cor, linewidth=1.3)
            # Faixas de inferência (cinza claro) para separar da carga do modelo.
            for pagina, trecho in a[a["fase"] == "inferencia"].groupby("pagina"):
                ax.axvspan(trecho["t_s"].min(), trecho["t_s"].max(), color=GRADE, alpha=0.6, lw=0)
            if lim:
                ax.set_ylim(0, lim)
            else:
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _num(v, 0)))
            ax.yaxis.grid(True, color=GRADE, linewidth=0.6)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(rotulo, fontsize=8.5)
            else:
                ax.set_xlabel("tempo (s)", fontsize=8)
            ax.tick_params(labelsize=7)
        status = resumo.loc[resumo["modelo"] == modelo, "status"].iloc[0]
        if status != "ok":
            pico = a["ram_modelo_mb"].idxmax()
            # Código -9 (OOM killer no processo Python) ou queda do servidor do
            # Ollama: nos dois casos o log do kernel registrou falta de memória.
            eixos[1][j].annotate("encerrado por\nfalta de memória",
                                 xy=(a.loc[pico, "t_s"], a.loc[pico, "ram_modelo_mb"]),
                                 xytext=(-8, -6), textcoords="offset points", ha="right", va="top",
                                 fontsize=7, color=TINTA_2)
    eixos[0][0].set_ylabel("CPU (% da máquina)", fontsize=8)
    eixos[1][0].set_ylabel("RAM (MB)", fontsize=8)
    fig.suptitle("Uso de CPU e memória ao longo da execução (faixas cinza: inferência)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", color=TINTA)
    fig.tight_layout()
    fig.savefig(FIGURAS / "eda_linha_tempo.pdf")
    plt.close(fig)
    print("  eda_linha_tempo.pdf")


def figura_cer(resumo):
    """CER por modelo nas duas versões da página: pontos ligados (limpa -> degradada)."""
    df = resumo.dropna(subset=["cer_limpa"]).copy()
    df["ordem"] = df[["cer_limpa", "cer_degradada"]].max(axis=1)
    df = df.sort_values("ordem")
    fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(df) + 1.2))
    y = np.arange(len(df))
    for yi, (_, r) in zip(y, df.iterrows()):
        cor = CORES[r["categoria"]]
        xs = [100 * r["cer_limpa"]] + ([100 * r["cer_degradada"]] if pd.notna(r["cer_degradada"]) else [])
        ax.plot(xs, [yi] * len(xs), color=EIXO, linewidth=2, zorder=1)
        ax.scatter([xs[0]], [yi], s=46, facecolor="white", edgecolor=cor, linewidth=1.6, zorder=2)
        if len(xs) > 1:
            ax.scatter([xs[1]], [yi], s=46, color=cor, edgecolor="white", linewidth=1, zorder=3)
            ax.text(xs[1] + 0.35, yi, f"{_num(xs[1])}%", va="center", fontsize=8, color=TINTA)
        else:
            ax.text(xs[0] + 0.35, yi, "degradada: n/d", va="center", fontsize=7.5, color=MUDO)
    ax.set_yticks(y, df["rotulo"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(left=-0.5, right=max(100 * df[["cer_limpa", "cer_degradada"]].max().max() * 1.2, 3))
    ax.set_xlabel("CER (%)")
    ax.set_title("Taxa de erro de caracteres: página limpa vs. degradada")
    _grade_x(ax)
    vazio = plt.Line2D([], [], marker="o", color="none", markerfacecolor="white",
                       markeredgecolor=TINTA_2, markersize=6, label="página limpa")
    cheio = plt.Line2D([], [], marker="o", color="none", markerfacecolor=TINTA_2,
                       markersize=6, label="página degradada")
    ax.legend(handles=[vazio, cheio], loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURAS / "eda_cer.pdf")
    plt.close(fig)
    print("  eda_cer.pdf")


def figura_paginas():
    """As duas versões da página sintética, lado a lado (PNG reduzido)."""
    from PIL import Image

    pasta = DADOS / "paginas" / "sintetica"
    paginas = [Image.open(pasta / f"{n}.png").convert("RGB") for n in ("001", "002")]
    largura = 620
    reduzidas = [im.resize((largura, round(im.height * largura / im.width)), Image.LANCZOS) for im in paginas]
    folga = 40
    mosaico = Image.new("RGB", (2 * largura + folga, reduzidas[0].height), "white")
    for i, im in enumerate(reduzidas):
        mosaico.paste(im, (i * (largura + folga), 0))
    # Borda fina para a página limpa não se confundir com o fundo do documento.
    from PIL import ImageDraw
    ImageDraw.Draw(mosaico).rectangle([0, 0, largura - 1, reduzidas[0].height - 1], outline=EIXO, width=2)
    mosaico.save(FIGURAS / "eda_paginas_sinteticas.png", dpi=(150, 150))
    print("  eda_paginas_sinteticas.png")


def main():
    FIGURAS.mkdir(exist_ok=True)
    resumo = pd.read_csv(DADOS / "resumo_modelos.csv")
    lt = pd.read_csv(DADOS / "perfil_linha_tempo.csv", dtype={"pagina": str})
    lt["ram_modelo_mb"] = lt["ram_python_mb"] + lt["ram_ollama_mb"].where(lt["modelo"].isin(MODELOS_OLLAMA), 0)
    lt["cpu_modelo_pct"] = lt["cpu_python_pct"] + lt["cpu_ollama_pct"].where(lt["modelo"].isin(MODELOS_OLLAMA), 0)
    print("Figuras geradas:")
    figura_tempo(resumo)
    figura_ram(resumo)
    figura_linha_tempo(lt, resumo)
    figura_cer(resumo)
    figura_paginas()


if __name__ == "__main__":
    main()
