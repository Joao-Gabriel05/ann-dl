"""
Algoritmo pocket.

O enunciado do Exercicio 2 pede para reutilizar a implementacao do Exercicio 1
"sem alteracoes", acrescentando ao laco apenas a copia dos melhores pesos. Por
isso esta classe HERDA de Perceptron: perceptron.py fica intacto (e o Exercicio
1 continua reproduzindo exatamente os mesmos numeros), e aqui so o metodo
treinar e reescrito, com uma unica linha de ideia nova - guardar (w, b) sempre
que uma atualizacao melhorar a acuracia no dataset completo.

Tudo o mais - degrau, predizer, acuracia, reta_de_decisao - vem da classe mae.
"""

import numpy as np

from perceptron import Perceptron


class PerceptronPocket(Perceptron):
    """Perceptron com bolso: mantem os pesos finais E os melhores ja vistos."""

    def treinar(self, X, y):
        rng = np.random.default_rng(self.seed)

        self.w_ = rng.normal(0, 0.01, size=X.shape[1])
        self.b_ = 0.0
        self.w_inicial_ = self.w_.copy()

        # o bolso comeca com os pesos iniciais como "melhor ate agora"
        self.w_pocket_ = self.w_.copy()
        self.b_pocket_ = self.b_
        self.melhor_acuracia_ = self.acuracia(X, y)
        self.trocas_de_bolso_ = 0

        self.historico_ = []
        self.convergiu_ = False

        for epoca in range(1, self.max_epocas + 1):
            ordem = rng.permutation(len(X)) if self.embaralhar else np.arange(len(X))

            atualizacoes = 0
            for i in ordem:
                xi, yi = X[i], y[i]
                y_hat = 1 if (xi @ self.w_ + self.b_) >= 0 else 0
                erro = yi - y_hat
                if erro:
                    self.w_ += self.eta * erro * xi
                    self.b_ += self.eta * erro
                    atualizacoes += 1

                    # unica adicao ao laco: avaliar e, se melhorou, guardar
                    acc = self.acuracia(X, y)
                    if acc > self.melhor_acuracia_:
                        self.melhor_acuracia_ = acc
                        self.w_pocket_ = self.w_.copy()
                        self.b_pocket_ = self.b_
                        self.trocas_de_bolso_ += 1

            self.historico_.append({
                "epoca": epoca,
                "atualizacoes": atualizacoes,
                "acuracia": self.acuracia(X, y),        # pesos atuais
                "acuracia_pocket": self.melhor_acuracia_,  # melhor ate agora
                "w": self.w_.copy(),
                "b": self.b_,
            })

            if atualizacoes == 0:
                self.convergiu_ = True
                break

        self.epocas_ = len(self.historico_)
        return self

    # --------------------------------------------------- versao "de bolso"
    def predizer_pocket(self, X):
        return ((X @ self.w_pocket_ + self.b_pocket_) >= 0).astype(int)

    def acuracia_pocket(self, X, y):
        return float((self.predizer_pocket(X) == y).mean())

    def reta_de_decisao_pocket(self, x_min, x_max):
        w1, w2 = self.w_pocket_
        xs = np.array([x_min, x_max])
        if abs(w2) < 1e-12:
            x = -self.b_pocket_ / w1
            return np.array([x, x]), np.array([x_min, x_max])
        return xs, -(w1 * xs + self.b_pocket_) / w2


if __name__ == "__main__":
    from ex2_dados import gerar_dados

    X, y = gerar_dados()
    p = PerceptronPocket(eta=0.01, max_epocas=100, seed=42).treinar(X, y)

    print(f"épocas executadas: {p.epocas_} "
          f"(convergiu? {p.convergiu_} — nunca houve época sem atualização)")
    print(f"atualizações totais: {sum(h['atualizacoes'] for h in p.historico_)}")
    print(f"trocas de bolso: {p.trocas_de_bolso_}")

    print("\n=== pesos FINAIS (os que o laço tinha na mão na última época) ===")
    print(f"  w = [{p.w_[0]:.5f}, {p.w_[1]:.5f}]")
    print(f"  b = {p.b_:.5f}")
    print(f"  acurácia = {p.acuracia(X, y):.4f}")

    print("\n=== pesos do POCKET (melhor acurácia vista durante o treino) ===")
    print(f"  w = [{p.w_pocket_[0]:.5f}, {p.w_pocket_[1]:.5f}]")
    print(f"  b = {p.b_pocket_:.5f}")
    print(f"  acurácia = {p.acuracia_pocket(X, y):.4f}")

    accs = [h["acuracia"] for h in p.historico_]
    print(f"\nacurácia dos pesos atuais ao longo das 100 épocas: "
          f"min {min(accs):.4f} | máx {max(accs):.4f} | "
          f"média {np.mean(accs):.4f} | última {accs[-1]:.4f}")
