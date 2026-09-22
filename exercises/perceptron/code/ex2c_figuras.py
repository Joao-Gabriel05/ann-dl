"""
Exercicio 2 - Parte C: figuras.

Figura 5 - as duas fronteiras (final e pocket) sobre os pontos, com os mal
           classificados de cada uma marcados;
Figura 6 - acuracia dos pesos atuais e do pocket, epoca a epoca.
"""

import numpy as np
import matplotlib.pyplot as plt

from _saida import figura
from ex1_dados import CORES, SURFACE, TINTA, TINTA_2, estilo
from ex2_dados import gerar_dados, melhor_reta
from pocket import PerceptronPocket

COR_FINAL, COR_POCKET = "#1a1a19", "#4a3aa7"


def figura_5(p, X, y, caminho=None):
    caminho = caminho or figura("fig05_fronteiras.png")
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    painéis = [
        ("final", p.predizer(X), p.reta_de_decisao(x_min, x_max),
         p.acuracia(X, y), COR_FINAL),
        ("pocket", p.predizer_pocket(X), p.reta_de_decisao_pocket(x_min, x_max),
         p.acuracia_pocket(X, y), COR_POCKET),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), dpi=150,
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, (nome, pred, reta, acc, cor) in zip(axes, painéis):
        estilo(ax)
        errados = pred != y
        for c in (0, 1):
            pts = X[y == c]
            ax.scatter(pts[:, 0], pts[:, 1], s=14, c=CORES[c], alpha=0.45,
                       linewidths=0, label=f"Classe {c}" if nome == "final" else None,
                       zorder=2)
        # mal classificados: 'x' escuro por cima do ponto
        ax.scatter(X[errados, 0], X[errados, 1], s=13, marker="x",
                   c="#1a1a19", alpha=0.55, linewidths=0.7, zorder=3,
                   label=f"mal classificados ({int(errados.sum())})")

        # a outra fronteira, de referencia, apagada
        outra = painéis[1][2] if nome == "final" else painéis[0][2]
        ax.plot(*outra, color=TINTA_2, linewidth=1.2, linestyle=":", zorder=4,
                label="a outra fronteira")
        ax.plot(*reta, color=cor, linewidth=2.6, zorder=5,
                label=f"fronteira {nome}")

        ax.set_title(f"fronteira {nome} — acurácia {acc:.2%}\n"
                     f"{int(errados.sum())} pontos errados de {len(X)}",
                     fontsize=12, color=TINTA, loc="left", pad=10)
        ax.set_xlabel("$x_1$", color=TINTA_2)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.legend(frameon=False, fontsize=9, loc="upper left")

    axes[0].set_ylabel("$x_2$", color=TINTA_2)
    fig.suptitle("Figura 5 — as duas fronteiras sobre os dados sobrepostos",
                 fontsize=13, color=TINTA, x=0.02, ha="left", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


def figura_6(p, teto, caminho=None):
    caminho = caminho or figura("fig06_acuracia_pocket.png")
    ep = [h["epoca"] for h in p.historico_]
    atual = [h["acuracia"] for h in p.historico_]
    bolso = [h["acuracia_pocket"] for h in p.historico_]

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    estilo(ax)

    ax.axhline(teto, color=TINTA_2, linewidth=1.2, linestyle="--", zorder=2)
    ax.annotate(f"melhor reta possível: {teto:.2%}", (ep[-1], teto),
                textcoords="offset points", xytext=(-6, 8), ha="right",
                fontsize=9.5, color=TINTA_2)
    ax.axhline(0.5, color="#c0392b", linewidth=1, linestyle=":", zorder=2)
    ax.annotate("acaso: 50%", (ep[-1], 0.5), textcoords="offset points",
                xytext=(-6, 6), ha="right", fontsize=9.5, color="#c0392b")

    ax.plot(ep, bolso, color=COR_POCKET, linewidth=2.2, zorder=4,
            label="melhor até agora (pocket)")
    ax.plot(ep, atual, color=CORES[1], linewidth=1.6, alpha=0.9, zorder=3,
            label="pesos atuais")

    ax.set_title("Figura 6 — acurácia por época: pesos atuais × pocket",
                 fontsize=12.5, color=TINTA, loc="left", pad=12)
    ax.set_xlabel("época", color=TINTA_2)
    ax.set_ylabel("acurácia no dataset completo", color=TINTA_2)
    ax.set_ylim(0.44, 0.80)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, fontsize=10, loc="center right")
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


if __name__ == "__main__":
    X, y = gerar_dados()
    p = PerceptronPocket(eta=0.01, max_epocas=100, seed=42).treinar(X, y)
    teto, _, _ = melhor_reta(X, y)

    print(f"final  : w = [{p.w_[0]:.5f}, {p.w_[1]:.5f}] | b = {p.b_:.5f} | "
          f"acurácia {p.acuracia(X, y):.4f}")
    print(f"pocket : w = [{p.w_pocket_[0]:.5f}, {p.w_pocket_[1]:.5f}] | "
          f"b = {p.b_pocket_:.5f} | acurácia {p.acuracia_pocket(X, y):.4f}")
    print(f"teto de qualquer reta: {teto:.4f}")

    figura_5(p, X, y)
    figura_6(p, teto)
