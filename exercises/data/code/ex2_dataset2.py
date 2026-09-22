"""
Exercicio 2 - Parte B: Dataset II, cascas concentricas em 5D.

Direcao u uniforme na esfera unitaria de R^5 (v ~ N(0, I5), u = v/||v||);
raio rho ~ N(2.0, 0.4) para a classe C (nucleo) e N(5.0, 0.4) para a D (casca);
cada ponto e x = rho * u.

Nota: o enunciado escreve rho ~ N(2.0, 0.4). Adotamos 0.4 como DESVIO PADRAO
(SIGMA_RHO abaixo); se o 0.4 for variancia, basta usar sqrt(0.4) = 0.632.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _saida import figura, dado

from ex2_dataset1 import D, FEATURES, SEED, N_POR_CLASSE, SURFACE, TINTA, TINTA_2, _estilo

RAIOS = {"C": 2.0, "D": 5.0}   # raio medio de cada casca
SIGMA_RHO = 0.4                # desvio padrao do raio
CORES = {"C": "#1baf7a", "D": "#4a3aa7"}


def gerar_dados(seed=SEED, n=N_POR_CLASSE):
    """500 amostras por classe: direcao isotropica x raio gaussiano."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for c, raio in RAIOS.items():
        v = rng.standard_normal((n, D))                  # N(0, I5)
        u = v / np.linalg.norm(v, axis=1, keepdims=True)  # uniforme na esfera
        rho = rng.normal(raio, SIGMA_RHO, size=(n, 1))
        Xs.append(rho * u)
        ys.append(np.full(n, c))
    return np.vstack(Xs), np.concatenate(ys)


def figura_2(X, y, caminho=None):
    caminho = caminho or figura("figA4_cascas.png")
    """(a) o que se ve numa projecao 2D  vs  (b) o que a norma revela."""
    r = np.linalg.norm(X, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=140)
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    _estilo(ax)
    for c in RAIOS:
        pts = X[y == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=11, c=CORES[c], alpha=0.55,
                   linewidths=0, label=f"Classe {c}", zorder=2)
    ax.set_title("(a) projeção nos eixos $x_1 \\times x_2$ — as cascas colapsam",
                 fontsize=11.5, color=TINTA, loc="left", pad=10)
    ax.set_xlabel("$x_1$", color=TINTA_2)
    ax.set_ylabel("$x_2$", color=TINTA_2)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=9)
    ax.legend(frameon=False, fontsize=10, loc="upper right", markerscale=1.8)

    ax = axes[1]
    _estilo(ax)
    for c in RAIOS:
        ax.hist(r[y == c], bins=34, color=CORES[c], alpha=0.75, zorder=2,
                label=f"Classe {c}  (raio ≈ {RAIOS[c]:g})")
    ax.set_title("(b) norma $\\|x\\|$ — separação perfeita em 1 dimensão",
                 fontsize=11.5, color=TINTA, loc="left", pad=10)
    ax.set_xlabel("$\\|x\\|$", color=TINTA_2)
    ax.set_ylabel("contagem", color=TINTA_2)
    ax.tick_params(labelsize=9)
    ax.legend(frameon=False, fontsize=10, loc="upper center")

    fig.suptitle("Figura 2 — Dataset II: cascas concêntricas em 5D "
                 "(500 amostras por classe)",
                 fontsize=13.5, color=TINTA, x=0.02, ha="left", y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


def figura_3(X, y, caminho=None):
    caminho = caminho or figura("figA5_dataset2_matriz.png")
    """Matriz 5x5 - para comparar com a Figura 1 do Dataset I."""
    fig, axes = plt.subplots(D, D, figsize=(13, 12), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            _estilo(ax)
            if i == j:
                for c in RAIOS:
                    ax.hist(X[y == c, i], bins=28, color=CORES[c], alpha=0.55,
                            density=True, zorder=2)
                ax.set_yticks([])
            else:
                for c in RAIOS:
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
                          color=CORES[c], label=f"Classe {c}") for c in RAIOS]
    fig.legend(handles=handles, frameon=False, fontsize=11, ncol=2,
               loc="upper right", bbox_to_anchor=(0.99, 0.985))
    fig.suptitle("Figura 3 — Dataset II nas coordenadas originais: "
                 "nenhum par de eixos separa as classes",
                 fontsize=14, color=TINTA, x=0.02, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


if __name__ == "__main__":
    X, y = gerar_dados()
    r = np.linalg.norm(X, axis=1)
    pd.DataFrame(X, columns=FEATURES).assign(classe=y, raio=r) \
      .to_csv(dado("dataset2.csv"), index=False)

    print(f"X: {X.shape} | classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    for c, raio in RAIOS.items():
        Xc, rc = X[y == c], r[y == c]
        print(f"\n--- Classe {c} (raio alvo {raio:g}, sigma {SIGMA_RHO:g}) ---")
        print(f"  raio amostral : media {rc.mean():.3f} | desvio {rc.std(ddof=1):.3f}"
              f" | min {rc.min():.3f} | max {rc.max():.3f}")
        print("  media por eixo:", np.array2string(Xc.mean(axis=0), precision=3,
                                                   suppress_small=True),
              " (esperado ~0: a direcao e isotropica)")
        print(f"  desvio por eixo: "
              f"{np.array2string(Xc.std(axis=0, ddof=1), precision=3)}"
              f"  (esperado ~ raio/sqrt(5) = {raio / np.sqrt(D):.3f})")

    fronteira = (r[y == "C"].max() + r[y == "D"].min()) / 2
    print(f"\nMaior raio de C = {r[y == 'C'].max():.3f} | "
          f"menor raio de D = {r[y == 'D'].min():.3f}")
    print(f"-> as cascas nao se tocam: ||x|| = {fronteira:.3f} separa as duas "
          f"classes com 0% de erro")

    figura_2(X, y)
    figura_3(X, y)
