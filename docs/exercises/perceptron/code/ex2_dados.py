"""
Exercicio 2 - Parte A: dados sobrepostos.

Mesmas 1000 amostras por classe, mas agora as medias estao proximas (distancia
sqrt(2)) e o espalhamento e tres vezes maior (variancia 1.5 contra 0.5).
Nenhuma reta separa as duas nuvens.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _saida import figura, dado
from ex1_dados import CORES, SURFACE, TINTA, TINTA_2, estilo

SEED = 42
N_POR_CLASSE = 1000

MU = {
    0: np.array([3.0, 3.0]),
    1: np.array([4.0, 4.0]),
}
COV = {
    0: np.array([[1.5, 0.0], [0.0, 1.5]]),
    1: np.array([[1.5, 0.0], [0.0, 1.5]]),
}


def gerar_dados(seed=SEED, n=N_POR_CLASSE):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for c in (0, 1):
        Xs.append(rng.multivariate_normal(MU[c], COV[c], size=n))
        ys.append(np.full(n, c))
    return np.vstack(Xs), np.concatenate(ys)


def acuracia_otima_teorica():
    """Duas gaussianas de mesma covariancia isotropica: a fronteira otima e a
    mediatriz entre as medias, e o acerto e Phi(d / 2*sigma)."""
    from math import erf, sqrt
    d = np.linalg.norm(MU[1] - MU[0])
    sigma = np.sqrt(COV[0][0, 0])
    z = d / (2 * sigma)
    return 0.5 * (1 + erf(z / sqrt(2))), d, sigma


def melhor_reta(X, y, n_direcoes=720):
    """Teto empirico de qualquer reta: varre direcoes e, para cada uma, o melhor
    corte possivel (busca exaustiva sobre os cortes candidatos)."""
    melhor = (0.0, None, None)
    for ang in np.linspace(0, np.pi, n_direcoes, endpoint=False):
        w = np.array([np.cos(ang), np.sin(ang)])
        v = X @ w
        ordem = np.argsort(v)
        vs, ts = v[ordem], y[ordem]
        cortes = (vs[:-1] + vs[1:]) / 2
        acima = ts.sum() - np.cumsum(ts)[:-1]      # positivos acima do corte
        abaixo = np.cumsum(1 - ts)[:-1]            # negativos abaixo do corte
        acc = (acima + abaixo) / len(y)
        acc = np.maximum(acc, 1 - acc)             # permite inverter o sentido
        i = int(np.argmax(acc))
        if acc[i] > melhor[0]:
            melhor = (float(acc[i]), w, float(cortes[i]))
    return melhor


def figura_4(X, y, caminho=None):
    caminho = caminho or figura("fig04_dados_sobrepostos.png")
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    estilo(ax)

    for c in (0, 1):
        pts = X[y == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=16, c=CORES[c], alpha=0.5,
                   linewidths=0.3, edgecolors=SURFACE, label=f"Classe {c}",
                   zorder=2)
    for c in (0, 1):
        ax.scatter(*MU[c], marker="X", s=190, c=CORES[c], edgecolors=SURFACE,
                   linewidths=2.0, zorder=4)
        ax.annotate(f"$\\mu_{c}$ = ({MU[c][0]:g}, {MU[c][1]:g})", MU[c],
                    textcoords="offset points",
                    xytext=(14, 12) if c else (-96, -26), fontsize=9.5,
                    color=TINTA, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE,
                              ec=CORES[c], lw=1.0, alpha=0.9))

    ax.set_title("Figura 4 — duas classes sobrepostas (1000 amostras cada)",
                 fontsize=12.5, color=TINTA, loc="left", pad=12)
    ax.set_xlabel("$x_1$", color=TINTA_2)
    ax.set_ylabel("$x_2$", color=TINTA_2)
    ax.set_aspect("equal")
    ax.legend(title="X = média da classe", frameon=False, fontsize=10,
              title_fontsize=9.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


if __name__ == "__main__":
    X, y = gerar_dados()
    pd.DataFrame(X, columns=["x1", "x2"]).assign(classe=y) \
      .to_csv(dado("dados_sobrepostos.csv"), index=False)

    print(f"X: {X.shape} | por classe: {np.bincount(y)}")
    for c in (0, 1):
        Xc = X[y == c]
        print(f"  classe {c}: média amostral "
              f"{np.array2string(Xc.mean(axis=0), precision=3)} | "
              f"desvio {np.array2string(Xc.std(axis=0, ddof=1), precision=3)}")

    otimo, d, sigma = acuracia_otima_teorica()
    print(f"\ndistância entre as médias: {d:.4f}")
    print(f"desvio por eixo: {sigma:.4f}")
    print(f"razão de separação d / (2*sigma) = {d / (2 * sigma):.4f}  "
          f"(no Exercício 1 era 3.5)")
    print(f"acurácia da fronteira ótima (teórica): {otimo:.4f}")

    acc, w, corte = melhor_reta(X, y)
    print(f"melhor reta possível nesta amostra: {acc:.4f}  "
          f"(direção [{w[0]:.3f}, {w[1]:.3f}])")

    # projecao na direcao das medias: aqui as nuvens se invadem
    u = (MU[1] - MU[0]) / np.linalg.norm(MU[1] - MU[0])
    p = X @ u
    print(f"\nprojeção na direção mu1 - mu0:")
    print(f"  classe 0 vai até {p[y == 0].max():.3f} | "
          f"classe 1 começa em {p[y == 1].min():.3f}")
    print(f"  folga: {p[y == 1].min() - p[y == 0].max():+.3f}  -> "
          f"{'separável' if p[y == 1].min() > p[y == 0].max() else 'SOBREPOSTAS'}")

    figura_4(X, y)
