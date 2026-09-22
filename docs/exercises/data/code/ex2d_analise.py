"""
Exercicio 2 - Parte D: Analise.

Verifica empiricamente as tres afirmacoes discutidas no README:
  1. o melhor classificador LINEAR no Dataset II fica no nivel do acaso;
  2. isso nao melhora com mais dados (varre n = 500 ... 50000);
  3. a feature ||x||^2 (soma dos quadrados) separa perfeitamente.

Regressao logistica implementada com numpy (gradiente em batch completo).
"""

import numpy as np
import matplotlib.pyplot as plt

from _saida import figura, dado

import ex2_dataset1 as d1
import ex2_dataset2 as d2
from ex2_dataset1 import SURFACE, TINTA, TINTA_2, _estilo


def logistica(X, t, epocas=4000, lr=0.5):
    """Regressao logistica binaria (t in {0,1}) por gradiente descendente.
    Padroniza as entradas so para o gradiente se comportar bem."""
    m, s = X.mean(axis=0), X.std(axis=0)
    Z = np.column_stack([(X - m) / s, np.ones(len(X))])
    w = np.zeros(Z.shape[1])
    for _ in range(epocas):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - t)) / len(Z)
    return w, (Z @ w)


def acuracia(score, t):
    return float(((score > 0).astype(int) == t).mean())


def melhor_limiar(v, t):
    """Melhor acuracia possivel com UM corte sobre a variavel v.
    Varre todos os cortes candidatos (pontos medios entre valores consecutivos)."""
    ordem = np.argsort(v)
    vs, ts = v[ordem], t[ordem]
    cortes = (vs[:-1] + vs[1:]) / 2
    # acertos de "v > corte -> classe 1", para cada corte, via soma acumulada
    acima = ts.sum() - np.cumsum(ts)[:-1]          # positivos acima do corte
    abaixo = np.cumsum(1 - ts)[:-1]                # negativos abaixo do corte
    acc = (acima + abaixo) / len(v)
    acc = np.maximum(acc, 1 - acc)                 # permite inverter o sentido
    i = int(np.argmax(acc))
    return float(acc[i]), float(cortes[i])


def melhor_hiperplano(X, t, n_dir=20, seed=0):
    """Teto do que QUALQUER hiperplano consegue: varre direcoes aleatorias e,
    para cada uma, o melhor corte. A regressao logistica fica abaixo disso
    porque otimiza log-loss, e no Dataset II o otimo de log-loss e w = 0."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n_dir, X.shape[1]))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    return max(melhor_limiar(X @ w, t)[0] for w in W)


def figura_6(X, t, caminho=None):
    caminho = caminho or figura("figA6_linear_quadratica.png")
    w, score = logistica(X, t)
    q = (X ** 2).sum(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    cores = [d2.CORES["C"], d2.CORES["D"]]
    nomes = ["Classe C", "Classe D"]

    for ax, v, titulo, xlabel in [
        (axes[0], score,
         f"(a) melhor combinação LINEAR $w\\cdot x + b$\nacurácia "
         f"{acuracia(score, t):.1%} — nível do acaso", "$w \\cdot x + b$"),
        (axes[1], q,
         f"(b) feature QUADRÁTICA $\\|x\\|^2 = \\sum_i x_i^2$\nacurácia "
         f"{melhor_limiar(q, t)[0]:.1%} com um único corte", "$\\|x\\|^2$"),
    ]:
        _estilo(ax)
        bins = np.histogram_bin_edges(v, bins=45)
        for k in (0, 1):
            ax.hist(v[t == k], bins=bins, color=cores[k], alpha=0.72,
                    label=nomes[k], zorder=2)
        ax.set_title(titulo, fontsize=11.5, color=TINTA, loc="left", pad=10)
        ax.set_xlabel(xlabel, color=TINTA_2)
        ax.set_ylabel("contagem", color=TINTA_2)
        ax.tick_params(labelsize=9)
        ax.legend(frameon=False, fontsize=10, loc="upper right")

    axes[1].axvline(melhor_limiar(q, t)[1], color=TINTA, linewidth=1.6,
                    linestyle="--", zorder=5)
    axes[1].annotate(f"corte em {melhor_limiar(q, t)[1]:.2f}",
                     (melhor_limiar(q, t)[1], 0), xytext=(8, 60),
                     textcoords="offset points", fontsize=9.5, color=TINTA)

    fig.suptitle("Figura 6 — Dataset II: por que a fronteira linear falha e a "
                 "quadrática resolve", fontsize=13.5, color=TINTA, x=0.02,
                 ha="left", y=0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


if __name__ == "__main__":
    X1, y1 = d1.gerar_dados()
    t1 = (y1 == "B").astype(int)
    X2, y2 = d2.gerar_dados()
    t2 = (y2 == "D").astype(int)

    print("=== D.1/D.2 — melhor fronteira LINEAR em cada dataset ===")
    for nome, X, t in [("Dataset I ", X1, t1), ("Dataset II", X2, t2)]:
        _, score = logistica(X, t)
        print(f"  {nome}: acuracia da regressao logistica = {acuracia(score, t):.1%}")

    print("\n=== D.2 — mais dados resolvem? (Dataset II, so linear) ===")
    print("   n/classe | regressao logistica | melhor hiperplano possivel")
    for n in (500, 2_000, 10_000, 50_000):
        Xn, yn = d2.gerar_dados(seed=1, n=n)
        tn = (yn == "D").astype(int)
        _, score = logistica(Xn, tn)
        print(f"   {n:>8} | {acuracia(score, tn):18.1%} | "
              f"{melhor_hiperplano(Xn, tn):.1%}")

    print("\n=== D.3 — a feature quadratica ===")
    q2 = (X2 ** 2).sum(axis=1)
    acc, corte = melhor_limiar(q2, t2)
    print(f"  ||x||^2 no Dataset II: acuracia {acc:.1%} com corte em {corte:.3f} "
          f"(raio {np.sqrt(corte):.3f})")
    q1 = (X1 ** 2).sum(axis=1)
    acc1, corte1 = melhor_limiar(q1, t1)
    print(f"  ||x||^2 no Dataset I : acuracia {acc1:.1%} (abaixo dos 88.7% da fronteira linear)")

    print("\n  fronteira sugerida:  f(x) = x1^2+x2^2+x3^2+x4^2+x5^2 - "
          f"{corte:.2f}   ->  f(x) > 0 : classe D | f(x) < 0 : classe C")

    figura_6(X2, t2)
