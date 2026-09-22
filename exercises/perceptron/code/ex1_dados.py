"""
Exercicio 1 - Parte A: dados linearmente separaveis.

Duas classes 2D, 1000 amostras cada, de normais multivariadas com a mesma
covariancia isotropica e medias bem afastadas - o caso para o qual o perceptron
foi projetado.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _saida import figura, dado

SEED = 42
N_POR_CLASSE = 1000

MU = {
    0: np.array([1.5, 1.5]),
    1: np.array([5.0, 5.0]),
}
COV = {
    0: np.array([[0.5, 0.0], [0.0, 0.5]]),
    1: np.array([[0.5, 0.0], [0.0, 0.5]]),
}

CORES = {0: "#2a78d6", 1: "#eb6834"}
SURFACE, TINTA, TINTA_2 = "#fcfcfb", "#1a1a19", "#5c5b55"


def gerar_dados(seed=SEED, n=N_POR_CLASSE):
    """Retorna X (2000, 2) e y (2000,) com rotulos 0/1."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for c in (0, 1):
        Xs.append(rng.multivariate_normal(MU[c], COV[c], size=n))
        ys.append(np.full(n, c))
    return np.vstack(Xs), np.concatenate(ys)


def estilo(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#d5d4ce")
    ax.tick_params(colors=TINTA_2, labelsize=9)


def figura_1(X, y, caminho=None):
    caminho = caminho or figura("fig01_dados.png")
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    estilo(ax)

    for c in (0, 1):
        pts = X[y == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=16, c=CORES[c], alpha=0.55,
                   linewidths=0.3, edgecolors=SURFACE, label=f"Classe {c}",
                   zorder=2)
    for c in (0, 1):
        # media teorica de cada nuvem
        ax.scatter(*MU[c], marker="X", s=190, c=CORES[c], edgecolors=SURFACE,
                   linewidths=2.0, zorder=4)
        ax.annotate(f"$\\mu_{c}$ = ({MU[c][0]:g}, {MU[c][1]:g})", MU[c],
                    textcoords="offset points", xytext=(12, 10), fontsize=9.5,
                    color=TINTA, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE,
                              ec=CORES[c], lw=1.0, alpha=0.9))

    ax.set_title("Figura 1 — duas classes linearmente separáveis "
                 "(1000 amostras cada)",
                 fontsize=12.5, color=TINTA, loc="left", pad=12)
    ax.set_xlabel("$x_1$", color=TINTA_2)
    ax.set_ylabel("$x_2$", color=TINTA_2)
    ax.set_aspect("equal")
    ax.legend(title="X = média da classe", frameon=False, fontsize=10,
              title_fontsize=9.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


def margem_na_direcao_das_medias(X, y):
    """Projeta os pontos na direcao que liga as medias e mede a folga entre as
    duas classes. Folga > 0 => existe reta perpendicular a essa direcao que
    separa tudo (condicao suficiente para separabilidade linear)."""
    w = MU[1] - MU[0]
    p = X @ w / np.linalg.norm(w)
    return p[y == 1].min() - p[y == 0].max(), p


if __name__ == "__main__":
    X, y = gerar_dados()
    pd.DataFrame(X, columns=["x1", "x2"]).assign(classe=y) \
      .to_csv(dado("dados_separaveis.csv"), index=False)

    print(f"X: {X.shape} | y: {y.shape} | por classe: {np.bincount(y)}")
    for c in (0, 1):
        Xc = X[y == c]
        print(f"\n--- Classe {c} ---")
        print("  media amostral :", np.array2string(Xc.mean(axis=0), precision=3))
        print("  media teorica  :", np.array2string(MU[c], precision=3))
        print("  cov amostral   :",
              np.array2string(np.cov(Xc, rowvar=False), precision=3).replace("\n", "\n" + " " * 19))

    d = np.linalg.norm(MU[1] - MU[0])
    sigma = np.sqrt(0.5)
    print(f"\ndistancia entre as medias: {d:.3f}")
    print(f"desvio por eixo: {sigma:.3f}  ->  razao de separacao "
          f"d / (2*sigma) = {d / (2 * sigma):.3f}")

    folga, p = margem_na_direcao_das_medias(X, y)
    print(f"\nprojecao na direcao mu1 - mu0:")
    print(f"  classe 0 vai ate {p[y == 0].max():.3f} | "
          f"classe 1 comeca em {p[y == 1].min():.3f}")
    print(f"  folga entre as duas nuvens: {folga:+.3f}  -> "
          f"{'SEPARAVEL por essa reta' if folga > 0 else 'ha sobreposicao nessa direcao'}")

    figura_1(X, y)
