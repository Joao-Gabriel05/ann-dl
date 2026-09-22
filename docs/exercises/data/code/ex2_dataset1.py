"""
Exercicio 2 - Parte A: Dataset I, gaussianas multivariadas deslocadas em 5D.

Classe A: 500 amostras, mu_A = 0, Sigma_A (correlacoes positivas)
Classe B: 500 amostras, mu_B = 1.5, Sigma_B (variancia maior, x1-x2 anticorrelacionados)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _saida import figura, dado

SEED = 42
N_POR_CLASSE = 500
D = 5
FEATURES = [f"x{i+1}" for i in range(D)]

MU = {
    "A": np.zeros(D),
    "B": np.full(D, 1.5),
}

SIGMA = {
    "A": np.array([
        [1.0, 0.8, 0.1, 0.0, 0.0],
        [0.8, 1.0, 0.3, 0.0, 0.0],
        [0.1, 0.3, 1.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 1.0, 0.2],
        [0.0, 0.0, 0.0, 0.2, 1.0],
    ]),
    "B": np.array([
        [ 1.5, -0.7, 0.2, 0.0, 0.0],
        [-0.7,  1.5, 0.4, 0.0, 0.0],
        [ 0.2,  0.4, 1.5, 0.6, 0.0],
        [ 0.0,  0.0, 0.6, 1.5, 0.3],
        [ 0.0,  0.0, 0.0, 0.3, 1.5],
    ]),
}

CORES = {"A": "#2a78d6", "B": "#eb6834"}
SURFACE = "#fcfcfb"
TINTA = "#1a1a19"
TINTA_2 = "#5c5b55"


def checar_covariancias():
    """Uma matriz de covariancia precisa ser simetrica e positiva definida."""
    for c, S in SIGMA.items():
        simetrica = np.allclose(S, S.T)
        autoval = np.linalg.eigvalsh(S)
        print(f"  Sigma_{c}: simetrica={simetrica} | "
              f"autovalores {np.array2string(autoval, precision=3)} | "
              f"pos. definida={bool(autoval.min() > 0)}")


def gerar_dados(seed=SEED, n=N_POR_CLASSE):
    """500 amostras por classe de uma normal multivariada 5D."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for c in ("A", "B"):
        Xs.append(rng.multivariate_normal(MU[c], SIGMA[c], size=n))
        ys.append(np.full(n, c))
    return np.vstack(Xs), np.concatenate(ys)


def _estilo(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#ecebe6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#d5d4ce")
    ax.tick_params(colors=TINTA_2, labelsize=7)


def figura_1(X, y, caminho=None):
    caminho = caminho or figura("figA3_dataset1_matriz.png")
    """Matriz 5x5: diagonal = distribuicao marginal; fora = dispersao 2D."""
    fig, axes = plt.subplots(D, D, figsize=(13, 12), dpi=130)
    fig.patch.set_facecolor(SURFACE)

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            _estilo(ax)
            if i == j:
                for c in ("A", "B"):
                    ax.hist(X[y == c, i], bins=28, color=CORES[c], alpha=0.55,
                            density=True, zorder=2)
                ax.set_yticks([])
            else:
                for c in ("A", "B"):
                    ax.scatter(X[y == c, j], X[y == c, i], s=5, c=CORES[c],
                               alpha=0.45, linewidths=0, zorder=2)
            if i == D - 1:
                ax.set_xlabel(FEATURES[j], color=TINTA, fontsize=10)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(FEATURES[i], color=TINTA, fontsize=10)
            elif i != j:
                ax.set_yticklabels([])

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                          color=CORES[c], label=f"Classe {c}") for c in ("A", "B")]
    fig.legend(handles=handles, frameon=False, fontsize=11, ncol=2,
               loc="upper right", bbox_to_anchor=(0.99, 0.985))
    fig.suptitle("Figura 1 — Dataset I: 1000 amostras em 5D "
                 "(diagonal: marginais · fora: pares de variáveis)",
                 fontsize=14, color=TINTA, x=0.02, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


if __name__ == "__main__":
    print("=== validacao das matrizes de covariancia ===")
    checar_covariancias()

    X, y = gerar_dados()
    df = pd.DataFrame(X, columns=FEATURES).assign(classe=y)
    df.to_csv(dado("dataset1.csv"), index=False)
    print(f"\nX: {X.shape} | classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    for c in ("A", "B"):
        Xc = X[y == c]
        print(f"\n--- Classe {c} ---")
        print("  media amostral :", np.array2string(Xc.mean(axis=0), precision=3,
                                                    suppress_small=True))
        print("  media teorica  :", np.array2string(MU[c], precision=3))
        print("  erro max |cov_amostral - Sigma| =",
              f"{np.abs(np.cov(Xc, rowvar=False) - SIGMA[c]).max():.3f}")

    figura_1(X, y)
