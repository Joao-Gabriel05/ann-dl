"""
Perceptron multicamadas (MLP) escrito do zero, so com numpy.

Este modulo e o unico lugar onde vivem forward, backward e atualizacao. Os
Exercicios 2, 3 e 4 importam a classe MLP SEM alterar nenhuma linha: o que muda
entre eles e so o que o enunciado permite mudar -

  - os tamanhos das camadas (argumento `camadas`),
  - a saida + perda (argumento `saida`: SigmoideBCE ou SoftmaxCCE),
  - os hiperparametros (eta, epocas, semente).

Convencoes (as da aula, com W(l) de forma (saida, entrada)):

  A(0) = X                                   X: (N, d), uma amostra por linha
  Z(l) = A(l-1) W(l)^T + b(l)
  A(l) = tanh(Z(l))                          camadas ocultas
  P    = saida.ativar(Z(L))                  ultima camada: sigmoide ou softmax

Por que a mesma retropropagacao serve aos dois problemas: tanto sigmoide + BCE
quanto softmax + entropia cruzada categorica tem o MESMO gradiente na
pre-ativacao da saida,

  dL/dZ(L) = (P - Y) / N,

(derivacao no relatorio, Exercicio 3 B). Entao cada perda so precisa fornecer
esse delta inicial; dali para tras o codigo e identico.
"""

import numpy as np


# ============================================================ saidas + perdas
class SigmoideBCE:
    """Classificacao binaria: 1 unidade de saida, sigmoide, entropia cruzada binaria."""

    def ativar(self, Z):
        return 1.0 / (1.0 + np.exp(-Z))

    def perda(self, P, Y):
        P = np.clip(P, 1e-12, 1 - 1e-12)          # evita log(0)
        return float(-np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P)))

    def delta(self, P, Y):
        """dL/dZ(L) = (P - Y) / N."""
        return (P - Y) / len(Y)

    def alvo(self, y):
        """Rotulos 0/1 -> coluna (N, 1)."""
        return y.reshape(-1, 1).astype(float)

    def classificar(self, P):
        return (P[:, 0] >= 0.5).astype(int)


class SoftmaxCCE:
    """Classificacao multiclasse: K unidades, softmax, entropia cruzada categorica."""

    def ativar(self, Z):
        Z = Z - Z.max(axis=1, keepdims=True)      # estabilidade numerica: nao muda o softmax
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def perda(self, P, Y):
        return float(-np.mean(np.sum(Y * np.log(np.clip(P, 1e-12, 1.0)), axis=1)))

    def delta(self, P, Y):
        """dL/dZ(L) = (P - Y) / N, com Y one-hot."""
        return (P - Y) / len(Y)

    def alvo(self, y, K=None):
        """Rotulos 0..K-1 -> one-hot (N, K)."""
        K = K or int(y.max()) + 1
        return np.eye(K)[y]

    def classificar(self, P):
        return P.argmax(axis=1)


# ============================================================ a rede
class MLP:
    """MLP com ativacao tanh nas camadas ocultas, treinada por gradiente descendente.

    Parametros
    ----------
    camadas : tamanhos de TODAS as camadas, da entrada a saida.
              Ex.: [2, 8, 1] = 2 entradas, 1 oculta de 8, 1 saida.
    saida   : objeto de saida+perda (SigmoideBCE() ou SoftmaxCCE()).
    rng     : np.random.Generator usado na inicializacao dos pesos.
    """

    def __init__(self, camadas, saida, rng):
        self.camadas = list(camadas)
        self.saida = saida
        # Xavier/Glorot: variancia 2 / (fan_in + fan_out), adequada a tanh.
        # Pesos aleatorios quebram a simetria - com tudo zero, todos os
        # neuronios de uma camada receberiam o mesmo gradiente para sempre.
        self.W, self.b = [], []
        for n_in, n_out in zip(self.camadas[:-1], self.camadas[1:]):
            desvio = np.sqrt(2.0 / (n_in + n_out))
            self.W.append(rng.normal(0.0, desvio, size=(n_out, n_in)))
            self.b.append(np.zeros(n_out))

    def n_parametros(self):
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))

    # -------------------------------------------------------------- forward
    def forward(self, X):
        """Propaga X e guarda as ativacoes de cada camada (o backward precisa delas)."""
        A = [X]
        n = len(self.W)
        for l in range(n):
            Z = A[-1] @ self.W[l].T + self.b[l]
            A.append(np.tanh(Z) if l < n - 1 else self.saida.ativar(Z))
        self.A_ = A
        return A[-1]

    # -------------------------------------------------------------- backward
    def backward(self, Y):
        """Gradientes de todos os pesos e vieses, da saida para a entrada.

        delta = dL/dZ(l). Para cada camada l:
            dL/dW(l) = delta^T A(l-1)
            dL/db(l) = soma de delta nas amostras
            dL/dA(l-1) = delta W(l)
            dL/dZ(l-1) = dL/dA(l-1) * (1 - A(l-1)^2)      derivada da tanh
        """
        A = self.A_
        delta = self.saida.delta(A[-1], Y)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = delta.T @ A[l]
            gb[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ self.W[l]) * (1.0 - A[l] ** 2)
        return gW, gb

    # -------------------------------------------------------------- update
    def atualizar(self, gW, gb, eta):
        """Passo de gradiente descendente: theta <- theta - eta * dL/dtheta."""
        for l in range(len(self.W)):
            self.W[l] -= eta * gW[l]
            self.b[l] -= eta * gb[l]

    # -------------------------------------------------------------- treino
    def treinar(self, X, y, eta, epocas):
        """Gradiente descendente em lote completo; registra a perda de treino por epoca.

        A perda guardada em historico_[e] e a da rede ANTES do passo da epoca e,
        ou seja, a perda que gerou aquele gradiente.
        """
        Y = self.saida.alvo(y)
        self.historico_ = []
        for _ in range(epocas):
            P = self.forward(X)
            self.historico_.append(self.saida.perda(P, Y))
            gW, gb = self.backward(Y)
            self.atualizar(gW, gb, eta)
        self.perda_final_ = self.saida.perda(self.forward(X), Y)
        return self

    # -------------------------------------------------------------- uso
    def predizer(self, X):
        return self.saida.classificar(self.forward(X))

    def acuracia(self, X, y):
        return float((self.predizer(X) == y).mean())


# ============================================================ verificacao
def checar_gradientes(rede, X, y, eps=1e-6):
    """Compara o backward com diferencas finitas centrais, parametro a parametro.

    Retorna o maior erro relativo encontrado. Valores ~1e-7 ou menores indicam
    que as derivadas escritas a mao estao corretas.
    """
    Y = rede.saida.alvo(y)
    rede.forward(X)
    gW, gb = rede.backward(Y)

    pior = 0.0
    for params, grads in ((rede.W, gW), (rede.b, gb)):
        for theta, g in zip(params, grads):
            for idx in np.ndindex(theta.shape):
                orig = theta[idx]
                theta[idx] = orig + eps
                lp = rede.saida.perda(rede.forward(X), Y)
                theta[idx] = orig - eps
                lm = rede.saida.perda(rede.forward(X), Y)
                theta[idx] = orig
                num = (lp - lm) / (2 * eps)
                pior = max(pior, abs(num - g[idx]) / max(1e-12, abs(num) + abs(g[idx])))
    return pior


if __name__ == "__main__":
    # Checagem numerica das derivadas, nos dois tipos de saida e com 1 e 2
    # camadas ocultas (as quatro configuracoes usadas no exercicio).
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 4))
    y_bin = rng.integers(0, 2, size=20)
    y_mc = rng.integers(0, 3, size=20)

    casos = [
        ("binaria, 1 oculta", [4, 5, 1], SigmoideBCE(), y_bin),
        ("binaria, 2 ocultas", [4, 5, 3, 1], SigmoideBCE(), y_bin),
        ("3 classes, 1 oculta", [4, 5, 3], SoftmaxCCE(), y_mc),
        ("3 classes, 2 ocultas", [4, 5, 4, 3], SoftmaxCCE(), y_mc),
    ]
    print("checagem de gradientes (maior erro relativo, backward x diferencas finitas)")
    for nome, camadas, saida, y in casos:
        rede = MLP(camadas, saida, rng)
        print(f"  {nome:<22} {str(camadas):<14} {checar_gradientes(rede, X, y):.2e}")
