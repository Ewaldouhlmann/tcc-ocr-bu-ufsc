"""Estilo comum das figuras do TCC (usado por analise_resultados.ipynb).

Convenções visuais:
- a cor identifica a CATEGORIA do modelo (Clássico, VLM local, Serviço via API),
  nunca a posição no ranking; paleta validada para daltonismo;
- modelos executados em GPU na nuvem aparecem com marcador vazado;
- um único eixo por gráfico; medidas diferentes vão em painéis separados;
- números no formato brasileiro (1.234,5).
"""

import matplotlib
import matplotlib.pyplot as plt

CORES = {"Clássico": "#2a78d6", "VLM local": "#eb6834", "Serviço via API": "#1baf7a"}
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
    "savefig.dpi": 300,
})


def num(v: float, casas: int = 1) -> str:
    """Formato brasileiro: 1.234,5"""
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def formatador_br(casas_acima_de_1: int = 0):
    """Rótulos de eixo em formato brasileiro (útil em escala log)."""
    return matplotlib.ticker.FuncFormatter(
        lambda v, _: num(v, casas_acima_de_1) if v >= 1 else num(v))
