"""
Exercicio 2 - Parte C: Visualize e compare.

Figura 4: PCA de cada dataset em 2D, lado a lado, colorido por classe.
Reporta a variancia explicada por PC1 e PC2 e a distancia entre os centros em 5D.
Figura 5: histograma do raio ||x||, as duas classes sobrepostas, por dataset.
"""

import numpy as np
import matplotlib.pyplot as plt

from _saida import figura, dado

import ex2_dataset1 as d1
import ex2_dataset2 as d2
from ex2_dataset1 import SURFACE, TINTA, TINTA_2, _estilo

DATASETS = {
    "Dataset I — gaussianas deslocadas": {
        "classes": ("A", "B"), "cores": d1.CORES, "gerar": d1.gerar_dados},
    "Dataset II — cascas concêntricas": {
        "classes": ("C", "D"), "cores": d2.CORES, "gerar": d2.gerar_dados},
}


def pca(X, k=2):
    """PCA por SVD sobre os dados centrados (nao padronizados: as escalas ja
    sao comparaveis). Retorna scores (N,k), razao de variancia explicada (d,)."""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (len(X) - 1)
    return Xc @ Vt[:k].T, var / var.sum()


def figura_4(dados, caminho=None):
    caminho = caminho or figura("fig04_pca.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=140)
    fig.patch.set_facecolor(SURFACE)

    for ax, (nome, d) in zip(axes, dados.items()):
        _estilo(ax)
        Z, ratio = d["pca"]
        for c in d["classes"]:
            pts = Z[d["y"] == c]
            ax.scatter(pts[:, 0], pts[:, 1], s=12, c=d["cores"][c], alpha=0.55,
                       linewidths=0, label=f"Classe {c}", zorder=2)
        ax.set_title(f"{nome}\nPC1 + PC2 explicam "
                     f"{ratio[:2].sum():.1%} da variância "
                     f"(PC1 {ratio[0]:.1%} · PC2 {ratio[1]:.1%})",
                     fontsize=11.5, color=TINTA, loc="left", pad=10)
        ax.set_xlabel("PC1", color=TINTA_2)
        ax.set_ylabel("PC2", color=TINTA_2)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=9)
        ax.legend(frameon=False, fontsize=10, loc="upper right", markerscale=1.8)

    fig.suptitle("Figura 4 — projeção PCA em 2D dos dois datasets 5D",
                 fontsize=13.5, color=TINTA, x=0.02, ha="left", y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


def figura_5(dados, caminho=None):
    caminho = caminho or figura("fig05_raios.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=140)
    fig.patch.set_facecolor(SURFACE)

    for ax, (nome, d) in zip(axes, dados.items()):
        _estilo(ax)
        r = d["raio"]
        bins = np.histogram_bin_edges(r, bins=40)
        for c in d["classes"]:
            rc = r[d["y"] == c]
            ax.hist(rc, bins=bins, color=d["cores"][c], alpha=0.70, zorder=2,
                    label=f"Classe {c}  (mediana {np.median(rc):.2f})")
        ax.set_title(f"{nome}\nsobreposição dos raios: {d['sobrep']:.1%} das amostras",
                     fontsize=11.5, color=TINTA, loc="left", pad=10)
        ax.set_xlabel("$\\|x\\|$", color=TINTA_2)
        ax.set_ylabel("contagem", color=TINTA_2)
        ax.tick_params(labelsize=9)
        ax.legend(frameon=False, fontsize=10, loc="upper right")

    fig.suptitle("Figura 5 — histograma do raio $\\|x\\|$ (as duas classes no mesmo eixo)",
                 fontsize=13.5, color=TINTA, x=0.02, ha="left", y=0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


def sobreposicao_1d(r, y, classes):
    """Fracao de pontos na faixa em que as duas classes coexistem
    [max(min), min(max)] - medida simples de quanto o raio sozinho confunde."""
    r1, r2 = r[y == classes[0]], r[y == classes[1]]
    lo, hi = max(r1.min(), r2.min()), min(r1.max(), r2.max())
    return 0.0 if lo >= hi else float(((r >= lo) & (r <= hi)).mean())


if __name__ == "__main__":
    dados = {}
    for nome, d in DATASETS.items():
        X, y = d["gerar"]()
        d = dict(d)
        d["X"], d["y"] = X, y
        d["pca"] = pca(X)
        d["raio"] = np.linalg.norm(X, axis=1)
        d["sobrep"] = sobreposicao_1d(d["raio"], y, d["classes"])
        dados[nome] = d

    print("=== C.2 — variancia explicada pela PCA ===")
    for nome, d in dados.items():
        r = d["pca"][1]
        print(f"\n{nome}")
        print("  por componente:", " ".join(f"PC{i+1} {v:6.1%}" for i, v in enumerate(r)))
        print(f"  PC1 + PC2 = {r[:2].sum():.1%}")

    print("\n=== C.3 — geometria em 5D ===")
    for nome, d in dados.items():
        c1, c2 = d["classes"]
        m1, m2 = d["X"][d["y"] == c1].mean(axis=0), d["X"][d["y"] == c2].mean(axis=0)
        print(f"\n{nome}")
        print(f"  centro {c1}: {np.array2string(m1, precision=3, suppress_small=True)}")
        print(f"  centro {c2}: {np.array2string(m2, precision=3, suppress_small=True)}")
        print(f"  ||mu_{c1} - mu_{c2}|| = {np.linalg.norm(m1 - m2):.3f}")
        for c in (c1, c2):
            rc = d["raio"][d["y"] == c]
            print(f"  raio da classe {c}: media {rc.mean():.3f} | "
                  f"desvio {rc.std(ddof=1):.3f} | faixa [{rc.min():.2f}, {rc.max():.2f}]")
        print(f"  sobreposicao dos raios: {d['sobrep']:.1%} das amostras")

    figura_4(dados)
    figura_5(dados)
