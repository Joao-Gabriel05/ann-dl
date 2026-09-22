"""
Exercicio 1 - Parte C: treine e meca.

Treina o perceptron no dataset separavel e produz:
  Figura 2 - fronteira de decisao w.x + b = 0 sobre os pontos, com os pontos
             mal classificados marcados de forma distinta;
  Figura 3 - acuracia x epoca.
"""

import numpy as np
import matplotlib.pyplot as plt

from _saida import figura
from ex1_dados import gerar_dados, CORES, SURFACE, TINTA, TINTA_2, estilo, MU
from perceptron import Perceptron


def figura_2(p, X, y, caminho=None):
    """Fronteira de decisao sobre a nuvem de pontos."""
    caminho = caminho or figura("fig02_fronteira.png")
    errados = p.predizer(X) != y

    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    estilo(ax)

    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6

    # regioes de decisao, para deixar visivel de que lado o modelo diz 0 e 1
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    Z = p.predizer(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5],
                colors=[CORES[0], CORES[1]], alpha=0.08, zorder=1)

    for c in (0, 1):
        pts = X[y == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=15, c=CORES[c], alpha=0.5,
                   linewidths=0.3, edgecolors=SURFACE, label=f"Classe {c}",
                   zorder=2)

    xs, ys = p.reta_de_decisao(x_min, x_max)
    ax.plot(xs, ys, color=TINTA, linewidth=2, zorder=4,
            label="fronteira $w \\cdot x + b = 0$")

    # marcacao distinta dos erros (fica vazia quando a acuracia e 100%)
    ax.scatter(X[errados, 0], X[errados, 1], s=110, facecolors="none",
               edgecolors="#c0392b", linewidths=1.8, zorder=5,
               label=f"mal classificados ({int(errados.sum())})")

    ax.set_title("Figura 2 — fronteira de decisão aprendida\n"
                 f"acurácia {p.acuracia(X, y):.2%} · "
                 f"{int(errados.sum())} pontos mal classificados de {len(X)}",
                 fontsize=12.5, color=TINTA, loc="left", pad=12)
    ax.set_xlabel("$x_1$", color=TINTA_2)
    ax.set_ylabel("$x_2$", color=TINTA_2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


def figura_3(p, caminho=None):
    """Acuracia no dataset completo ao fim de cada epoca."""
    caminho = caminho or figura("fig03_acuracia.png")
    ep = [h["epoca"] for h in p.historico_]
    acc = [h["acuracia"] for h in p.historico_]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    estilo(ax)

    ax.plot(ep, acc, color=CORES[0], linewidth=2, zorder=3)
    ax.scatter(ep, acc, s=34, color=CORES[0], edgecolors=SURFACE,
               linewidths=1.5, zorder=4)
    ax.axhline(1.0, color=TINTA_2, linewidth=1, linestyle=":", zorder=2)
    ax.annotate(f"convergiu na época {p.epocas_}\n(acurácia {acc[-1]:.2%})",
                (ep[-1], acc[-1]), textcoords="offset points", xytext=(-30, -190),
                ha="right", fontsize=10, color=TINTA,
                arrowprops=dict(arrowstyle="->", color=TINTA_2, lw=1.2))

    ax.set_title("Figura 3 — acurácia no dataset completo × época",
                 fontsize=12.5, color=TINTA, loc="left", pad=12)
    ax.set_xlabel("época", color=TINTA_2)
    ax.set_ylabel("acurácia", color=TINTA_2)
    ax.set_ylim(0.45, 1.06)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


def figura_a1(p, caminho=None):
    """Suplementar: atualizacoes por epoca - o que de fato governa a parada."""
    caminho = caminho or figura("figA1_atualizacoes.png")
    ep = [h["epoca"] for h in p.historico_]
    upd = [h["atualizacoes"] for h in p.historico_]

    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    estilo(ax)
    ax.bar(ep, upd, color=CORES[1], alpha=0.85, zorder=2, width=0.7)
    ax.annotate("época sem nenhuma\natualização → parada",
                (ep[-1], 0), textcoords="offset points", xytext=(-8, 40),
                ha="right", fontsize=9.5, color=TINTA,
                arrowprops=dict(arrowstyle="->", color=TINTA_2, lw=1.2))
    ax.set_title("Figura A1 — atualizações por época "
                 f"({sum(upd)} no total, em {p.epocas_ * 2000} amostras vistas)",
                 fontsize=12, color=TINTA, loc="left", pad=12)
    ax.set_xlabel("época", color=TINTA_2)
    ax.set_ylabel("atualizações", color=TINTA_2)
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


if __name__ == "__main__":
    X, y = gerar_dados()
    p = Perceptron(eta=0.01, max_epocas=100, seed=42).treinar(X, y)

    print("=== C.1 — resultado do treino ===")
    print(f"  w final   : [{p.w_[0]:.5f}, {p.w_[1]:.5f}]")
    print(f"  b final   : {p.b_:.5f}")
    print(f"  épocas    : {p.epocas_} "
          f"({'convergiu' if p.convergiu_ else 'parou no teto'})")
    print(f"  acurácia  : {p.acuracia(X, y):.4f}")
    print(f"  erros     : {int((p.predizer(X) != y).sum())} de {len(X)}")
    print(f"  atualizações totais: {sum(h['atualizacoes'] for h in p.historico_)}")

    figura_2(p, X, y)
    figura_3(p)
    figura_a1(p)
