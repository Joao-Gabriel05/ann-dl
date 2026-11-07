"""
Exercicio 2 - Parte C: MLP do zero na classificacao binaria.

Arquitetura [2, 8, 1]: 2 entradas, 1 camada oculta de 8 neuronios tanh,
1 saida sigmoide; perda = entropia cruzada binaria. Gradiente descendente em
lote completo.

Produz:
  Figura 2 - perda de treino x epoca;
  Figura 3 - fronteira de decisao sobre os pontos de TESTE.
"""

import numpy as np
import matplotlib.pyplot as plt

from _saida import figura, estilo, CORES, SURFACE, TINTA, TINTA_2, SEED
from ex2_dados import gerar_dados, dividir, baseline_linear
from mlp import MLP, SigmoideBCE

CAMADAS = [2, 8, 1]
ETA = 0.1
EPOCAS = 2000


def treinar_rede(X_tr, y_tr, rng):
    return MLP(CAMADAS, SigmoideBCE(), rng).treinar(X_tr, y_tr, ETA, EPOCAS)


def figura_2(rede, caminho=None):
    caminho = caminho or figura("fig02_perda.png")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    estilo(ax)
    ep = np.arange(1, len(rede.historico_) + 1)
    ax.plot(ep, rede.historico_, color=CORES[0], linewidth=2, zorder=3,
            label=f"MLP {CAMADAS}, η = {ETA}")
    ax.annotate(f"perda final {rede.perda_final_:.4f}", (ep[-1], rede.historico_[-1]),
                textcoords="offset points", xytext=(-10, 40), ha="right",
                fontsize=10, color=TINTA,
                arrowprops=dict(arrowstyle="->", color=TINTA_2, lw=1.2))
    ax.set_title("Figura 2 — perda de treino (entropia cruzada binária) × época",
                 fontsize=12.5, color=TINTA, loc="left", pad=12)
    ax.set_xlabel("época", color=TINTA_2)
    ax.set_ylabel("perda de treino (BCE)", color=TINTA_2)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


def figura_3(rede, lr, X_te, y_te, caminho=None):
    caminho = caminho or figura("fig03_fronteira.png")
    fig, ax = plt.subplots(figsize=(7.8, 7.3), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    estilo(ax)

    lim = 7.5
    gx, gy = np.meshgrid(np.linspace(-lim, lim, 400), np.linspace(-lim, lim, 400))
    grade = np.column_stack([gx.ravel(), gy.ravel()])
    Z = rede.predizer(grade).reshape(gx.shape)
    ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5], colors=[CORES[0], CORES[1]],
                alpha=0.13, zorder=1)
    ax.contour(gx, gy, Z, levels=[0.5], colors=[TINTA], linewidths=1.8, zorder=3)

    # a reta do baseline, para comparacao
    w, b = lr.coef_[0], lr.intercept_[0]
    xs = np.array([-lim, lim])
    ax.plot(xs, -(w[0] * xs + b) / w[1], color=TINTA_2, linestyle="--",
            linewidth=1.3, zorder=3, label="fronteira do baseline linear")

    errados = rede.predizer(X_te) != y_te
    for c in (0, 1):
        pts = X_te[y_te == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=22, c=CORES[c], alpha=0.85,
                   linewidths=0.4, edgecolors=SURFACE, zorder=4, label=f"Classe {c} (teste)")
    ax.scatter(X_te[errados, 0], X_te[errados, 1], s=95, facecolors="none",
               edgecolors="#c0392b", linewidths=1.6, zorder=5,
               label=f"erros da MLP ({int(errados.sum())})")
    ax.plot([], [], color=TINTA, linewidth=1.8, label="fronteira da MLP")

    ax.set_title(f"Figura 3 — fronteira da MLP {CAMADAS} nos 200 pontos de teste\n"
                 f"acurácia de teste {rede.acuracia(X_te, y_te):.1%} · "
                 "região sombreada = classe prevista",
                 fontsize=12, color=TINTA, loc="left", pad=12)
    ax.set_xlabel("$x_1$", color=TINTA_2)
    ax.set_ylabel("$x_2$", color=TINTA_2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor="#d5d4ce", fontsize=9,
              loc="upper left")
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    X, y, _ = gerar_dados()
    X_tr, X_te, y_tr, y_te = dividir(X, y)

    rede = treinar_rede(X_tr, y_tr, rng)
    lr, acc_lr = baseline_linear(X_tr, y_tr, X_te, y_te)

    print("=== C - MLP do zero ===")
    print(f"  arquitetura      : {CAMADAS} (tanh na oculta, sigmoide na saida)")
    print(f"  parametros       : {rede.n_parametros()}")
    print(f"  eta, epocas      : {ETA}, {EPOCAS} (lote completo)")
    print(f"  perda inicial    : {rede.historico_[0]:.4f}")
    print(f"  perda final      : {rede.perda_final_:.4f}")
    print(f"  acuracia treino  : {rede.acuracia(X_tr, y_tr):.4f}")
    print(f"  acuracia teste   : {rede.acuracia(X_te, y_te):.4f}")
    print(f"  baseline linear  : {acc_lr:.4f}")
    for e in (1, 10, 50, 100, 250, 500, 1000, 2000):
        print(f"    epoca {e:>5}: perda {rede.historico_[e - 1]:.4f}")

    figura_2(rede)
    figura_3(rede, lr, X_te, y_te)
