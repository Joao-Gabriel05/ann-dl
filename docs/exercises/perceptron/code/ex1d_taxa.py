"""
Exercicio 1 - Parte D: analise da taxa de aprendizado.

D.2 - re-executa o treino com eta = 1.0 e compara com eta = 0.01: epocas,
      acuracia e a DIRECAO de w (w/||w||).
D.3 - demonstracao empirica do argumento algebrico: partindo de w = 0 e b = 0,
      treinar com eta_1 e eta_2 produz pesos que diferem apenas pelo fator
      eta_2/eta_1 - mesma fronteira, mesmo numero de epocas.
"""

import numpy as np

from ex1_dados import gerar_dados
from perceptron import Perceptron


def direcao(w):
    return w / np.linalg.norm(w)


def angulo(w1, w2):
    c = np.clip(direcao(w1) @ direcao(w2), -1, 1)
    return float(np.degrees(np.arccos(c)))


class PerceptronDoZero(Perceptron):
    """So para o item D.3: identica, mas comecando de w = 0 e b = 0."""

    def treinar(self, X, y):
        guardado = np.random.default_rng
        try:
            # substitui o sorteio da inicializacao por zeros, sem tocar no resto
            np.random.default_rng = lambda seed=None: _RngZero(guardado(seed))
            return super().treinar(X, y)
        finally:
            np.random.default_rng = guardado


class _RngZero:
    """Embrulho do gerador: normal() devolve zeros; o resto passa direto."""

    def __init__(self, rng):
        self._rng = rng

    def normal(self, *a, size=None, **k):
        return np.zeros(size)

    def __getattr__(self, nome):
        return getattr(self._rng, nome)


if __name__ == "__main__":
    X, y = gerar_dados()

    print("=== D.2 — mesma inicialização, taxas diferentes ===")
    modelos = {}
    for eta in (0.01, 1.0):
        p = Perceptron(eta=eta, max_epocas=100, seed=42).treinar(X, y)
        modelos[eta] = p
        print(f"\n  eta = {eta}")
        print(f"    épocas   : {p.epocas_}  ({'convergiu' if p.convergiu_ else 'teto'})")
        print(f"    acurácia : {p.acuracia(X, y):.4f}")
        print(f"    w final  : [{p.w_[0]:.5f}, {p.w_[1]:.5f}]   b: {p.b_:.5f}")
        print(f"    ||w||    : {np.linalg.norm(p.w_):.5f}")
        print(f"    direção  : [{direcao(p.w_)[0]:.5f}, {direcao(p.w_)[1]:.5f}]")
        print(f"    atualizações: {sum(h['atualizacoes'] for h in p.historico_)}")

    a, b = modelos[0.01], modelos[1.0]
    print(f"\n  ângulo entre as duas direções de w: {angulo(a.w_, b.w_):.2f}°")
    print(f"  razão entre as normas ||w(1.0)|| / ||w(0.01)||: "
          f"{np.linalg.norm(b.w_) / np.linalg.norm(a.w_):.1f}")
    print(f"  predições idênticas nos 2000 pontos? "
          f"{np.array_equal(a.predizer(X), b.predizer(X))}")

    print("\n=== D.3 — partindo de w = 0, b = 0 (o que o item B proíbe) ===")
    zeros = {}
    for eta in (0.01, 1.0):
        p = PerceptronDoZero(eta=eta, max_epocas=100, seed=42).treinar(X, y)
        zeros[eta] = p
        print(f"  eta = {eta:<5} -> épocas {p.epocas_:>3} | "
              f"acurácia {p.acuracia(X, y):.4f} | "
              f"w = [{p.w_[0]:.5f}, {p.w_[1]:.5f}] | b = {p.b_:.5f}")

    z1, z2 = zeros[0.01], zeros[1.0]
    fator = 1.0 / 0.01
    print(f"\n  w(eta=1.0) == (1.0/0.01) * w(eta=0.01)? "
          f"{np.allclose(z2.w_, fator * z1.w_)}")
    print(f"  b(eta=1.0) == (1.0/0.01) * b(eta=0.01)? "
          f"{np.isclose(z2.b_, fator * z1.b_)}")
    print(f"  mesmo número de épocas? {z1.epocas_ == z2.epocas_}")
    print(f"  ângulo entre as direções: {angulo(z1.w_, z2.w_):.4f}°")
    print(f"  predições idênticas? {np.array_equal(z1.predizer(X), z2.predizer(X))}")
