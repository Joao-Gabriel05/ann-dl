"""
Caminhos compartilhados pelos scripts deste exercicio.

Os scripts moram em code/, mas escrevem em figures/ (o que o relatorio exibe)
e em data/ (entradas do Kaggle e saidas geradas). Assim rodar
`python3 ex1_nuvens.py` de dentro de code/ ja deixa cada arquivo no lugar certo.
"""

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent   # docs/exercises/data
FIGURAS = BASE / "figures"
DADOS = BASE / "data"


def figura(nome):
    """Caminho de uma figura, criando figures/ se preciso."""
    FIGURAS.mkdir(exist_ok=True)
    return FIGURAS / nome


def dado(nome):
    """Caminho de um arquivo de dados (entrada ou saida)."""
    DADOS.mkdir(exist_ok=True)
    return DADOS / nome
