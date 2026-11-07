"""
Caminhos compartilhados pelos scripts deste exercicio.

Os scripts moram em code/, mas escrevem em figures/ (o que o relatorio exibe)
e em data/ (numeros gerados). Assim rodar `python3 ex2_dados.py` de dentro de
code/ ja deixa cada arquivo no lugar certo.
"""

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent   # docs/exercises/mlp
FIGURAS = BASE / "figures"
DADOS = BASE / "data"

SEED = 42

CORES = ["#2a78d6", "#eb6834", "#2e9e6a"]
SURFACE, TINTA, TINTA_2 = "#fcfcfb", "#1a1a19", "#5c5b55"


def figura(nome):
    """Caminho de uma figura, criando figures/ se preciso."""
    FIGURAS.mkdir(exist_ok=True)
    return FIGURAS / nome


def dado(nome):
    """Caminho de um arquivo de dados gerado."""
    DADOS.mkdir(exist_ok=True)
    return DADOS / nome


def estilo(ax):
    """Estilo comum a todas as figuras do exercicio."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#d5d4ce")
    ax.tick_params(colors=TINTA_2, labelsize=9)
