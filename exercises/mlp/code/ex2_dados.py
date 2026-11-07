"""
Exercicio 2 - Partes A e B: dados que uma reta nao separa, e o baseline linear.

Classe 1 = dois aglomerados em lados opostos da classe 0, centrada na origem.
Qualquer reta que tente isolar a classe 1 so consegue ficar com UM dos dois
aglomerados externos - dai o baseline linear ficar abaixo do acaso.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from _saida import figura, estilo, CORES, SURFACE, TINTA, TINTA_2, SEED

CENTROS = [[0, 0], [3, 3], [-3, -3]]
N_AMOSTRAS = [500, 250, 250]


def gerar_dados():
    """Retorna X (1000, 2), y (1000,) e o indice do aglomerado de origem."""
    X, aglomerado = make_blobs(n_samples=N_AMOSTRAS, centers=CENTROS,
                               cluster_std=1.2, random_state=SEED)
    y = (aglomerado != 0).astype(int)        # os dois aglomerados externos = classe 1
    return X, y, aglomerado


def dividir(X, y):
    """Divisao 80/20 estratificada, com a semente do enunciado."""
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def figura_1(X, y, caminho=None):
    caminho = caminho or figura("fig01_dados.png")
    fig, ax = plt.subplots(figsize=(7.5, 7), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    estilo(ax)
    for c in (0, 1):
        pts = X[y == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=14, c=CORES[c], alpha=0.6,
                   linewidths=0.3, edgecolors=SURFACE, zorder=2,
                   label=f"Classe {c} ({len(pts)} pontos)")
    for cx, cy in CENTROS:
        ax.scatter(cx, cy, marker="X", s=160, c=TINTA, edgecolors=SURFACE,
                   linewidths=1.8, zorder=4)
    ax.set_title("Figura 1 — classe 1 em dois aglomerados, um de cada lado da classe 0",
                 fontsize=12, color=TINTA, loc="left", pad=12)
    ax.set_xlabel("$x_1$", color=TINTA_2)
    ax.set_ylabel("$x_2$", color=TINTA_2)
    ax.set_aspect("equal")
    ax.legend(title="X = centro do aglomerado", frameon=False, fontsize=10,
              title_fontsize=9.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


def baseline_linear(X_tr, y_tr, X_te, y_te):
    """Uma unica fronteira linear (regressao logistica), como permite o item B."""
    lr = LogisticRegression().fit(X_tr, y_tr)
    return lr, lr.score(X_te, y_te)


if __name__ == "__main__":
    X, y, ag = gerar_dados()
    X_tr, X_te, y_tr, y_te = dividir(X, y)
    print(f"X: {X.shape} | por classe: {np.bincount(y)} | por aglomerado: {np.bincount(ag)}")
    print(f"treino: {len(X_tr)} ({np.bincount(y_tr)}) | teste: {len(X_te)} ({np.bincount(y_te)})")
    figura_1(X, y)

    lr, acc = baseline_linear(X_tr, y_tr, X_te, y_te)
    print("\n=== B - baseline linear (regressao logistica) ===")
    print(f"  w = {np.array2string(lr.coef_[0], precision=4)} | b = {lr.intercept_[0]:.4f}")
    print(f"  acuracia de treino: {lr.score(X_tr, y_tr):.4f}")
    print(f"  acuracia de teste : {acc:.4f}")
    pred = lr.predict(X_te)
    print(f"  no teste, preve classe 1 para {int(pred.sum())} de {len(pred)} pontos")
