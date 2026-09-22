"""
Perceptron de camada unica, escrito do zero.

Esta implementacao e reutilizada no Exercicio 2 - por isso mora num modulo
proprio e nao depende de nada especifico do dataset separavel.

Convencao de rotulos: y in {0, 1}.
  predicao:    y_hat = degrau(w . x + b),  degrau(z) = 1 se z >= 0, senao 0
  atualizacao: w <- w + eta (y - y_hat) x
               b <- b + eta (y - y_hat)

O erro (y - y_hat) vale 0 quando acerta (nao atualiza), +1 quando deixou de
prever a classe 1 e -1 quando previu 1 indevidamente. Escrever a regra como
w <- w + eta*y*x so vale na convencao de rotulos -1/+1: com rotulos 0/1 ela
nunca atualizaria na classe 0 e o falso positivo jamais seria corrigido.
"""

import numpy as np


def degrau(z):
    """Funcao de ativacao: 1 se z >= 0, 0 caso contrario."""
    return (z >= 0).astype(int)


class Perceptron:
    """Perceptron de camada unica treinado pela regra classica, amostra a amostra.

    Parametros
    ----------
    eta : taxa de aprendizado.
    max_epocas : teto de passagens pelo dataset.
    seed : semente do sorteio de w.
    embaralhar : se True, sorteia a ordem das amostras a cada epoca.
    """

    def __init__(self, eta=0.01, max_epocas=100, seed=42, embaralhar=False):
        self.eta = eta
        self.max_epocas = max_epocas
        self.seed = seed
        self.embaralhar = embaralhar

    # ---------------------------------------------------------------- predicao
    def ativacao(self, X):
        """Pre-ativacao w . x + b (o score bruto, antes do degrau)."""
        return X @ self.w_ + self.b_

    def predizer(self, X):
        return degrau(self.ativacao(X))

    def acuracia(self, X, y):
        return float((self.predizer(X) == y).mean())

    # ---------------------------------------------------------------- treino
    def treinar(self, X, y):
        """Treina ate uma epoca sem nenhuma atualizacao, ou ate max_epocas.

        Guarda em self.historico_, por epoca: numero de atualizacoes, acuracia
        no dataset COMPLETO ao fim da epoca, e os pesos naquele momento.
        """
        rng = np.random.default_rng(self.seed)

        # inicializacao pedida: w pequeno e aleatorio, b = 0.
        # comecar de w = 0 tornaria eta irrelevante (ela so reescalaria w,
        # deixando fronteira e numero de epocas identicos).
        self.w_ = rng.normal(0, 0.01, size=X.shape[1])
        self.b_ = 0.0

        self.w_inicial_ = self.w_.copy()
        self.historico_ = []
        self.convergiu_ = False

        for epoca in range(1, self.max_epocas + 1):
            ordem = rng.permutation(len(X)) if self.embaralhar else np.arange(len(X))

            atualizacoes = 0
            for i in ordem:
                xi, yi = X[i], y[i]
                y_hat = 1 if (xi @ self.w_ + self.b_) >= 0 else 0
                erro = yi - y_hat
                if erro:                       # acerto => erro 0 => nao mexe nos pesos
                    self.w_ += self.eta * erro * xi
                    self.b_ += self.eta * erro
                    atualizacoes += 1

            self.historico_.append({
                "epoca": epoca,
                "atualizacoes": atualizacoes,
                "acuracia": self.acuracia(X, y),
                "w": self.w_.copy(),
                "b": self.b_,
            })

            if atualizacoes == 0:              # passagem inteira sem erro: parou
                self.convergiu_ = True
                break

        self.epocas_ = len(self.historico_)
        return self

    # ---------------------------------------------------------------- utilidades
    def reta_de_decisao(self, x_min, x_max):
        """Pontos (x1, x2) da fronteira w1*x1 + w2*x2 + b = 0, para plotar."""
        w1, w2 = self.w_
        xs = np.array([x_min, x_max])
        if abs(w2) < 1e-12:                    # fronteira vertical
            x = -self.b_ / w1
            return np.array([x, x]), np.array([x_min, x_max])
        return xs, -(w1 * xs + self.b_) / w2


if __name__ == "__main__":
    from ex1_dados import gerar_dados

    X, y = gerar_dados()
    p = Perceptron(eta=0.01, max_epocas=100, seed=42).treinar(X, y)

    print(f"w inicial : {np.array2string(p.w_inicial_, precision=5)}  | b inicial: 0.0")
    print(f"convergiu : {p.convergiu_} em {p.epocas_} épocas "
          f"(parada: {'época sem atualizações' if p.convergiu_ else 'teto de épocas'})")
    print(f"w final   : {np.array2string(p.w_, precision=5)}")
    print(f"b final   : {p.b_:.5f}")
    print(f"acurácia final no dataset completo: {p.acuracia(X, y):.4f}")

    print("\n época | atualizações | acurácia")
    for h in p.historico_:
        print(f"  {h['epoca']:>4} | {h['atualizacoes']:>12} | {h['acuracia']:.4f}")

    total = sum(h["atualizacoes"] for h in p.historico_)
    print(f"\ntotal de atualizações: {total} "
          f"(de {p.epocas_ * len(X)} amostras vistas)")
