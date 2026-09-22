"""
Exercicio 1 - Parte B: Mais ou menos espalhado.
4 datasets (mesmas 4 classes, mesmas medias), com todos os desvios multiplicados
por s in {0.5, 1.0, 2.0, 4.0}.

Produz: Figura 2 (4 subplots, eixos compartilhados), tabela de razoes de
separacao r_ij (s=1), taxa de mistura por s e Figura 3 (mistura x s).
"""

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _saida import figura, dado

from ex1_nuvens import PARAMS, CORES, SURFACE, TINTA, TINTA_2, N_POR_CLASSE, SEED

ESCALAS = [0.5, 1.0, 2.0, 4.0]
MU = np.array([PARAMS[k]["mu"] for k in sorted(PARAMS)])        # (4, 2)
SIGMA = np.array([PARAMS[k]["sigma"] for k in sorted(PARAMS)])  # (4, 2)


def gerar_escalado(s, seed=SEED, n=N_POR_CLASSE):
    """Mesmas medias, desvios multiplicados por s. Seed fixa por escala:
    o mesmo sorteio normal padrao e reescalado, entao as 4 nuvens sao
    literalmente a mesma nuvem 'inflada' - a comparacao isola o efeito de s."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for k in sorted(PARAMS):
        z = rng.standard_normal((n, 2))              # ruido comum as escalas
        Xs.append(MU[k] + z * (SIGMA[k] * s))        # mu + z * (sigma*s)
        ys.append(np.full(n, k))
    return np.vstack(Xs), np.concatenate(ys)


def razoes_separacao(sigma=SIGMA, mu=MU):
    """r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j), sigma_bar = media dos eixos."""
    sigma_bar = sigma.mean(axis=1)
    linhas = []
    for i, j in itertools.combinations(range(len(mu)), 2):
        d = float(np.linalg.norm(mu[i] - mu[j]))
        linhas.append({
            "par": f"({i}, {j})",
            "||mu_i - mu_j||": d,
            "sigma_bar_i": sigma_bar[i],
            "sigma_bar_j": sigma_bar[j],
            "r_ij": d / (sigma_bar[i] + sigma_bar[j]),
        })
    return pd.DataFrame(linhas)


def taxa_de_mistura(X, y, mu=MU):
    """Fracao de pontos cujo centro de classe mais proximo nao e o da propria classe.
    Distancia de cada ponto as 4 medias por broadcasting: (N,1,2) - (1,4,2)."""
    d = np.linalg.norm(X[:, None, :] - mu[None, :, :], axis=2)  # (N, 4)
    mais_proximo = d.argmin(axis=1)
    return float((mais_proximo != y).mean()), mais_proximo


def figura_2(datasets, caminho=None):
    caminho = caminho or figura("fig02_escalas.png")
    todos = np.vstack([X for X, _ in datasets.values()])
    pad = 0.04 * (todos.max(axis=0) - todos.min(axis=0))
    xlim = (todos[:, 0].min() - pad[0], todos[:, 0].max() + pad[0])
    ylim = (todos[:, 1].min() - pad[1], todos[:, 1].max() + pad[1])

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150,
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, (s, (X, y)) in zip(axes.ravel(), datasets.items()):
        ax.set_facecolor(SURFACE)
        for k in sorted(PARAMS):
            pts = X[y == k]
            ax.scatter(pts[:, 0], pts[:, 1], s=14, c=CORES[k], alpha=0.65,
                       linewidths=0.4, edgecolors=SURFACE,
                       label=f"Classe {k}" if s == ESCALAS[0] else None, zorder=2)
        ax.scatter(MU[:, 0], MU[:, 1], marker="X", s=110,
                   c=[CORES[k] for k in sorted(PARAMS)],
                   edgecolors=SURFACE, linewidths=1.6, zorder=4)
        mistura, _ = taxa_de_mistura(X, y)
        ax.set_title(f"s = {s:g}   ·   taxa de mistura = {mistura:.1%}",
                     fontsize=11, color=TINTA, loc="left", pad=8)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, color="#e6e5e0", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color("#d5d4ce")
        ax.tick_params(colors=TINTA_2, labelsize=9)

    for ax in axes[1, :]:
        ax.set_xlabel("$x_1$", color=TINTA_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("$x_2$", color=TINTA_2)

    fig.suptitle("Figura 2 — mesmas 4 classes, desvios multiplicados por s "
                 "(eixos compartilhados)", fontsize=13, color=TINTA, x=0.02,
                 ha="left", y=0.985)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left",
                      markerscale=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


def figura_3(escalas, misturas, caminho=None):
    caminho = caminho or figura("fig03_mistura.png")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(escalas, misturas, color=CORES[0], linewidth=2, zorder=3)
    ax.scatter(escalas, misturas, s=70, color=CORES[0], edgecolors=SURFACE,
               linewidths=2, zorder=4)
    for s, m in zip(escalas, misturas):
        ax.annotate(f"{m:.1%}", (s, m), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9.5, color=TINTA)

    ax.set_title("Figura 3 — taxa de mistura em função do fator de escala s",
                 fontsize=12, color=TINTA, loc="left", pad=12)
    ax.set_xlabel("fator de escala $s$ (multiplica todos os $\\sigma$)", color=TINTA_2)
    ax.set_ylabel("fração de pontos com centro mais próximo errado", color=TINTA_2)
    ax.set_xticks(escalas)
    ax.set_xticklabels([f"{s:g}" for s in escalas])
    ax.set_ylim(-0.03, max(misturas) * 1.30 + 0.03)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(True, axis="y", color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#d5d4ce")
    ax.tick_params(colors=TINTA_2, labelsize=9)
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


if __name__ == "__main__":
    datasets = {s: gerar_escalado(s) for s in ESCALAS}
    for s, (X, y) in datasets.items():
        pd.DataFrame(X, columns=["x1", "x2"]).assign(classe=y) \
          .to_csv(dado(f"dados_s{s:g}.csv"), index=False)

    print("=== B.2 — razoes de separacao r_ij (s = 1) ===")
    df_r = razoes_separacao()
    print(df_r.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    menor = df_r.loc[df_r["r_ij"].idxmin()]
    print(f"\nMenor razao: par {menor['par']} -> r = {menor['r_ij']:.3f}")
    print(f"Como r_ij escala com 1/s, em s = 2 esse par vale "
          f"{menor['r_ij'] / 2:.3f}  (e em s=0.5: {menor['r_ij'] / 0.5:.3f}; "
          f"s=4: {menor['r_ij'] / 4:.3f})")

    print("\n=== B.3 — taxa de mistura por escala ===")
    misturas = []
    for s, (X, y) in datasets.items():
        m, pred = taxa_de_mistura(X, y)
        misturas.append(m)
        por_classe = [f"C{k}: {(pred[y == k] != k).mean():.0%}" for k in sorted(PARAMS)]
        print(f"  s = {s:<4g} mistura = {m:6.2%}   ({', '.join(por_classe)})"
              f"   | menor r_ij = {menor['r_ij'] / s:.3f}")

    figura_2(datasets)
    figura_3(ESCALAS, misturas)
