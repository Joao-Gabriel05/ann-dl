"""
Exercicio 1 - Nuvens de Pontos: Geometria e Espalhamento em 2D
Parte A: geracao das nuvens gaussianas + Figura 1 (scatter 2D com os centros).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _saida import figura, dado

SEED = 42
N_POR_CLASSE = 100

# (media, desvio padrao) por classe, em (x1, x2)
PARAMS = {
    0: {"mu": np.array([2.0, 3.0]), "sigma": np.array([0.8, 2.5])},
    1: {"mu": np.array([5.0, 6.0]), "sigma": np.array([1.2, 1.9])},
    2: {"mu": np.array([8.0, 1.0]), "sigma": np.array([0.9, 0.9])},
    3: {"mu": np.array([15.0, 4.0]), "sigma": np.array([0.5, 2.0])},
}

# paleta categorica validada para scatter (todas as duplas, visao normal e CVD)
CORES = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a", 3: "#4a3aa7"}
SURFACE = "#fcfcfb"
TINTA = "#1a1a19"
TINTA_2 = "#5c5b55"


def gerar_dados(seed=SEED, n=N_POR_CLASSE):
    """Gera n amostras gaussianas 2D por classe. Retorna X (400,2) e y (400,)."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for classe, p in PARAMS.items():
        # covariancia diagonal: cada eixo sorteado com seu proprio desvio
        pontos = rng.normal(loc=p["mu"], scale=p["sigma"], size=(n, 2))
        Xs.append(pontos)
        ys.append(np.full(n, classe))
    return np.vstack(Xs), np.concatenate(ys)


def figura_1(X, y, caminho=None):
    caminho = caminho or figura("fig01_nuvens.png")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for classe in sorted(PARAMS):
        pts = X[y == classe]
        ax.scatter(pts[:, 0], pts[:, 1], s=26, c=CORES[classe], alpha=0.70,
                   linewidths=0.5, edgecolors=SURFACE, label=f"Classe {classe}",
                   zorder=2)

    for classe, p in PARAMS.items():
        mx, my = p["mu"]
        # centro teorico (media usada na geracao)
        ax.scatter(mx, my, marker="X", s=190, c=CORES[classe],
                   edgecolors=SURFACE, linewidths=2.0, zorder=4)
        ax.annotate(f"$\\mu_{classe}$ = ({mx:g}, {my:g})", (mx, my),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=9, color=TINTA, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE,
                              ec=CORES[classe], lw=1.0, alpha=0.9))

    ax.set_title("Figura 1 - Nuvens gaussianas 2D (400 amostras, 4 classes)",
                 fontsize=12, color=TINTA, pad=12, loc="left")
    ax.set_xlabel("$x_1$", color=TINTA_2)
    ax.set_ylabel("$x_2$", color=TINTA_2)
    ax.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#d5d4ce")
    ax.tick_params(colors=TINTA_2, labelsize=9)
    ax.legend(title="X = centro da nuvem", frameon=False, fontsize=9,
              title_fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")
    return fig


if __name__ == "__main__":
    X, y = gerar_dados()
    df = pd.DataFrame(X, columns=["x1", "x2"]).assign(classe=y)
    df.to_csv(dado("dados_nuvens.csv"), index=False)

    print(f"X: {X.shape} | y: {y.shape} | por classe: {np.bincount(y)}")
    print("\nMedia e desvio AMOSTRAIS por classe (vs. os teoricos):")
    for classe, p in PARAMS.items():
        pts = X[y == classe]
        m, s = pts.mean(axis=0), pts.std(axis=0, ddof=1)
        print(f"  Classe {classe}: mu_amostral=({m[0]:6.3f}, {m[1]:6.3f})"
              f"  teorico=({p['mu'][0]:g}, {p['mu'][1]:g})"
              f" | sigma_amostral=({s[0]:5.3f}, {s[1]:5.3f})"
              f"  teorico=({p['sigma'][0]:g}, {p['sigma'][1]:g})")

    figura_1(X, y)
