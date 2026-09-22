"""
Exercicio 1 - Parte C: Analise da sobreposicao e das fronteiras de decisao.

Figura 4: sobre os pontos do item A (s=1), duas leituras da fronteira -
  (a) linear por partes: particao de Voronoi das 4 medias (o que uma camada
      linear/softmax consegue traçar);
  (b) otima de Bayes: argmax da densidade gaussiana verdadeira - fronteiras
      curvas, o alvo que uma MLP treinada persegue.
Figura 5: erro irredutivel (Bayes) x fator de escala s, comparado com a taxa
  de mistura do item B.
"""

import numpy as np
import matplotlib.pyplot as plt

from _saida import figura, dado
from matplotlib.colors import ListedColormap

from ex1_nuvens import PARAMS, CORES, SURFACE, TINTA, TINTA_2, gerar_dados
from ex1b_espalhamento import MU, SIGMA, ESCALAS, taxa_de_mistura

CMAP = ListedColormap([CORES[k] for k in sorted(PARAMS)])


def log_posterior(pontos, s=1.0):
    """log da densidade gaussiana diagonal de cada classe (priors iguais).
    Retorna (N, 4) - o argmax e o classificador de Bayes."""
    sig = SIGMA * s                                     # (4, 2)
    d = pontos[:, None, :] - MU[None, :, :]             # (N, 4, 2)
    return (-0.5 * (d / sig) ** 2 - np.log(sig)).sum(axis=2)


def regiao_voronoi(pontos):
    """Classe do centro mais proximo -> particao linear por partes."""
    return np.linalg.norm(pontos[:, None, :] - MU[None, :, :], axis=2).argmin(axis=1)


def erro_de_bayes(s, n=200_000, seed=7):
    """Erro irredutivel: fracao de amostras da propria distribuicao que o
    classificador otimo (que conhece mu e sigma) ainda erra."""
    rng = np.random.default_rng(seed)
    m = n // len(MU)
    X = np.vstack([MU[k] + rng.standard_normal((m, 2)) * (SIGMA[k] * s)
                   for k in range(len(MU))])
    y = np.repeat(np.arange(len(MU)), m)
    return float((log_posterior(X, s).argmax(axis=1) != y).mean())


def _malha(X, passo=600):
    pad = 1.5
    gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, passo)
    gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, passo)
    GX, GY = np.meshgrid(gx, gy)
    return GX, GY, np.column_stack([GX.ravel(), GY.ravel()])


def _estilo(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e5e0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#d5d4ce")
    ax.tick_params(colors=TINTA_2, labelsize=9)
    ax.set_xlabel("$x_1$", color=TINTA_2)


def figura_4(X, y, caminho=None):
    caminho = caminho or figura("figA1_fronteiras.png")
    GX, GY, malha = _malha(X)
    mapas = {
        "(a) fronteiras lineares por partes — centro mais próximo":
            regiao_voronoi(malha).reshape(GX.shape),
        "(b) fronteiras ótimas (Bayes) — o alvo de uma MLP":
            log_posterior(malha).argmax(axis=1).reshape(GX.shape),
    }
    erros = {"(a)": taxa_de_mistura(X, y)[0],
             "(b)": float((log_posterior(X).argmax(axis=1) != y).mean())}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, (titulo, Z), (tag, err) in zip(axes, mapas.items(), erros.items()):
        ax.contourf(GX, GY, Z, levels=[-.5, .5, 1.5, 2.5, 3.5], cmap=CMAP,
                    alpha=0.13, zorder=1)
        ax.contour(GX, GY, Z, levels=[.5, 1.5, 2.5], colors=TINTA_2,
                   linewidths=1.4, linestyles="--", zorder=3)
        errados = log_posterior(X).argmax(axis=1) != y if tag == "(b)" \
            else regiao_voronoi(X) != y
        for k in sorted(PARAMS):
            pts = X[y == k]
            ax.scatter(pts[:, 0], pts[:, 1], s=22, c=CORES[k], alpha=0.75,
                       linewidths=0.4, edgecolors=SURFACE,
                       label=f"Classe {k}" if tag == "(a)" else None, zorder=4)
        ax.scatter(X[errados, 0], X[errados, 1], s=95, facecolors="none",
                   edgecolors=TINTA, linewidths=1.3, zorder=5,
                   label="ponto do lado errado" if tag == "(a)" else None)
        ax.scatter(MU[:, 0], MU[:, 1], marker="X", s=150,
                   c=[CORES[k] for k in sorted(PARAMS)], edgecolors=SURFACE,
                   linewidths=1.8, zorder=6)
        ax.set_title(f"{titulo}\nerro sobre as 400 amostras: {err:.1%}",
                     fontsize=11, color=TINTA, loc="left", pad=10)
        _estilo(ax)

    axes[0].set_ylabel("$x_2$", color=TINTA_2)
    axes[0].legend(frameon=False, fontsize=9, loc="upper center", ncol=2,
                   markerscale=1.2)
    fig.suptitle("Figura 4 — fronteiras de decisão sobre as nuvens do item A (s = 1)",
                 fontsize=13, color=TINTA, x=0.02, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


def figura_5(tabela, caminho=None):
    caminho = caminho or figura("figA2_erro_bayes.png")
    s, bayes, mistura = (np.array([t[i] for t in tabela]) for i in range(3))
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(s, bayes, color=CORES[0], linewidth=2, zorder=4,
            label="erro irredutível (Bayes) — nenhuma rede supera")
    ax.scatter(s, bayes, s=70, color=CORES[0], edgecolors=SURFACE, linewidths=2, zorder=5)
    ax.plot(s, mistura, color=CORES[1], linewidth=2, linestyle="--", zorder=3,
            label="taxa de mistura (centro mais próximo) — item B")
    ax.scatter(s, mistura, s=70, color=CORES[1], edgecolors=SURFACE, linewidths=2, zorder=3)
    for xs, yb in zip(s, bayes):
        ax.annotate(f"{yb:.1%}", (xs, yb), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=9.5, color=TINTA)

    ax.set_title("Figura 5 — a região onde a rede necessariamente erra cresce com s",
                 fontsize=12, color=TINTA, loc="left", pad=12)
    ax.set_xlabel("fator de escala $s$", color=TINTA_2)
    ax.set_ylabel("fração de pontos", color=TINTA_2)
    ax.set_xticks(ESCALAS)
    ax.set_xticklabels([f"{v:g}" for v in ESCALAS])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(-0.04, 0.55)
    ax.grid(True, axis="y", color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#d5d4ce")
    ax.tick_params(colors=TINTA_2, labelsize=9)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


if __name__ == "__main__":
    X, y = gerar_dados()

    print("=== C — erro das duas fronteiras sobre as 400 amostras (s = 1) ===")
    print(f"  linear por partes (Voronoi): {taxa_de_mistura(X, y)[0]:.2%}")
    print(f"  ótima de Bayes (curva)     : "
          f"{(log_posterior(X).argmax(axis=1) != y).mean():.2%}")

    print("\n=== C — região de erro necessário x espalhamento ===")
    print("   s | erro de Bayes | taxa de mistura (B)")
    tabela = []
    for s in ESCALAS:
        from ex1b_espalhamento import gerar_escalado
        Xs, ys = gerar_escalado(s)
        b, m = erro_de_bayes(s), taxa_de_mistura(Xs, ys)[0]
        tabela.append((s, b, m))
        print(f" {s:>3g} | {b:12.2%}  | {m:.2%}")

    figura_4(X, y)
    figura_5(tabela)
