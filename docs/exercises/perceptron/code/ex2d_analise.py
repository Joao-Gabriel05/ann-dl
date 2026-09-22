"""
Exercicio 2 - Parte D: analise.

D.1 - por que os pesos finais ficam em ~50% e onde a fronteira final para;
D.2 - o que o teorema da convergencia garante e qual hipotese este dataset viola;
D.3 - mais epocas resolvem? um eta menor resolve?
"""

import numpy as np
import matplotlib.pyplot as plt

from _saida import figura
from ex1_dados import CORES, SURFACE, TINTA, TINTA_2, estilo
from ex2_dados import gerar_dados, melhor_reta, MU
from pocket import PerceptronPocket

COR_POCKET = "#4a3aa7"


def distancia_da_reta_a_origem(w, b):
    """|b| / ||w|| - o quanto a fronteira esta deslocada da origem."""
    return abs(b) / np.linalg.norm(w)


def figura_a2(p, centro, caminho=None):
    """Suplementar: o deslocamento da fronteira encolhe epoca a epoca."""
    caminho = caminho or figura("figA2_deslocamento.png")
    ep = [h["epoca"] for h in p.historico_]
    desloc = [distancia_da_reta_a_origem(h["w"], h["b"]) for h in p.historico_]

    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    estilo(ax)

    ax.axhline(centro, color=TINTA_2, linewidth=1.3, linestyle="--", zorder=2)
    ax.annotate(f"centro da nuvem: {centro:.2f}", (ep[-1], centro),
                textcoords="offset points", xytext=(-6, 8), ha="right",
                fontsize=9.5, color=TINTA_2)
    ax.plot(ep, desloc, color=CORES[1], linewidth=2, zorder=3,
            label="pesos atuais")
    ax.axhline(distancia_da_reta_a_origem(p.w_pocket_, p.b_pocket_),
               color=COR_POCKET, linewidth=2, zorder=4, label="pocket")

    ax.set_title("Figura A2 — deslocamento da fronteira em relação à origem, "
                 "$|b| / \\|w\\|$", fontsize=12.5, color=TINTA, loc="left", pad=12)
    ax.set_xlabel("época", color=TINTA_2)
    ax.set_ylabel("$|b| / \\|w\\|$", color=TINTA_2)
    ax.set_ylim(-0.25, 5.8)
    ax.annotate("a fronteira final nunca chega perto da nuvem:\nfica abaixo de 1.6 nas 100 épocas",
                (52, 1.55), textcoords="offset points", xytext=(18, 46),
                fontsize=9.5, color=TINTA,
                arrowprops=dict(arrowstyle="->", color=TINTA_2, lw=1.2))
    ax.legend(frameon=False, fontsize=10, loc="center left")
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


if __name__ == "__main__":
    X, y = gerar_dados()
    p = PerceptronPocket(eta=0.01, max_epocas=100, seed=42).treinar(X, y)
    eta = 0.01

    print("=== D.1 — por que a fronteira final acerta ~50% ===")
    norma_x = np.linalg.norm(X, axis=1)
    print(f"  ||x|| médio nos dados: {norma_x.mean():.3f} "
          f"(faixa {norma_x.min():.2f} a {norma_x.max():.2f})")
    print(f"  cada engano move w em eta*||x|| ≈ {eta * norma_x.mean():.4f}")
    print(f"  cada engano move b em eta        = {eta:.4f}")
    print(f"  razão: w anda {norma_x.mean():.1f}x mais rápido que b\n")

    centro = np.linalg.norm((MU[0] + MU[1]) / 2)
    for nome, w, b in (("final ", p.w_, p.b_), ("pocket", p.w_pocket_, p.b_pocket_)):
        print(f"  {nome}: ||w|| = {np.linalg.norm(w):.5f} | |b| = {abs(b):.5f} | "
              f"|b|/||w|| = {distancia_da_reta_a_origem(w, b):6.3f}")
    print(f"  distância do centro da nuvem à origem: {centro:.3f}")
    print(f"  -> a fronteira final passa a {distancia_da_reta_a_origem(p.w_, p.b_):.2f} "
          f"da origem, com a nuvem a {centro:.2f}: ela corta FORA dos dados")

    print("\n=== D.3 — mais épocas resolvem? ===")
    for max_ep in (100, 500):
        q = PerceptronPocket(eta=0.01, max_epocas=max_ep, seed=42).treinar(X, y)
        print(f"  {max_ep:>3} épocas -> final {q.acuracia(X, y):.4f} | "
              f"pocket {q.acuracia_pocket(X, y):.4f} | "
              f"atualizações {sum(h['atualizacoes'] for h in q.historico_):>5} | "
              f"|b|/||w|| final {distancia_da_reta_a_origem(q.w_, q.b_):.3f}")

    print("\n=== D.3 — um eta menor resolve? (100 épocas) ===")
    for e in (0.01, 0.001, 0.0001):
        q = PerceptronPocket(eta=e, max_epocas=100, seed=42).treinar(X, y)
        print(f"  eta = {e:<7} -> final {q.acuracia(X, y):.4f} | "
              f"pocket {q.acuracia_pocket(X, y):.4f} | "
              f"atualizações {sum(h['atualizacoes'] for h in q.historico_):>5} | "
              f"|b|/||w|| final {distancia_da_reta_a_origem(q.w_, q.b_):.3f}")

    teto, _, _ = melhor_reta(X, y)
    print(f"\n  teto de qualquer reta: {teto:.4f}")

    figura_a2(p, centro)
