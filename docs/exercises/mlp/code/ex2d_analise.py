"""
Exercicio 2 - Parte D: por que o baseline linear fica abaixo de 50%.

Duas medidas que separam "o que uma reta CONSEGUE" de "o que a regressao
logistica ESCOLHE":
  1. a melhor reta possivel no teste, por busca exaustiva (direcao x corte);
  2. as probabilidades que a regressao logistica de fato produz.
E, do lado da MLP, de que forma e a regiao que ela atribui a classe 1.
"""

import numpy as np

from _saida import SEED
from ex2_dados import gerar_dados, dividir, baseline_linear
from ex2c_treino import treinar_rede


def melhor_reta(X, y, n_direcoes=720):
    """Acuracia maxima de qualquer reta: para cada direcao, testa todos os cortes
    entre projecoes consecutivas, nos dois sentidos."""
    melhor = (0.0, None)
    for ang in np.linspace(0, np.pi, n_direcoes, endpoint=False):
        d = np.array([np.cos(ang), np.sin(ang)])
        p = X @ d
        ordem = np.argsort(p)
        ys = y[ordem]
        # acerto com "classe 1 acima do corte" para cada posicao de corte k
        uns_acima = np.concatenate([[ys.sum()], ys.sum() - np.cumsum(ys)])
        zeros_abaixo = np.concatenate([[0], np.cumsum(1 - ys)])
        acc = (uns_acima + zeros_abaixo) / len(y)
        acc = np.maximum(acc, 1 - acc)          # sentido oposto
        k = acc.argmax()
        if acc[k] > melhor[0]:
            melhor = (float(acc[k]), np.degrees(ang))
    return melhor


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    X, y, ag = gerar_dados()
    X_tr, X_te, y_tr, y_te = dividir(X, y)
    lr, acc_lr = baseline_linear(X_tr, y_tr, X_te, y_te)

    print("=== o que a regressao logistica escolheu ===")
    p = lr.predict_proba(X_te)[:, 1]
    print(f"  acuracia de teste: {acc_lr:.4f}")
    print(f"  probabilidades previstas no teste: min {p.min():.3f} | max {p.max():.3f}")
    print(f"  media de cada classe no treino: classe 0 {np.array2string(X_tr[y_tr == 0].mean(0), precision=3)}"
          f" | classe 1 {np.array2string(X_tr[y_tr == 1].mean(0), precision=3)}")
    pred = lr.predict(X_te)
    # mesmo split (estratificado em y) aplicado aos indices, para recuperar o aglomerado
    _, idx_te, _, _ = dividir(np.arange(len(y)), y)
    ag_te = ag[idx_te]
    for a, nome in ((0, "centro (0,0)  classe 0"), (1, "(3,3)         classe 1"), (2, "(-3,-3)       classe 1")):
        m = ag_te == a
        print(f"  aglomerado {nome}: acerto {np.mean(pred[m] == y_te[m]):.3f} ({m.sum()} pontos)")

    print("\n=== o que uma reta consegue ===")
    for nome, (Xs, ys) in (("treino", (X_tr, y_tr)), ("teste", (X_te, y_te))):
        acc, ang = melhor_reta(Xs, ys)
        print(f"  melhor reta no {nome}: {acc:.4f} (normal a {ang:.1f} graus)")

    print("\n=== a regiao da MLP ===")
    rede = treinar_rede(X_tr, y_tr, rng)
    # percorre a diagonal x1 = x2 e a antidiagonal, de -7 a 7
    t = np.linspace(-7, 7, 2801)
    for nome, d in (("diagonal (1,1)", np.array([1, 1]) / np.sqrt(2)),
                    ("antidiagonal (1,-1)", np.array([1, -1]) / np.sqrt(2))):
        c = rede.predizer(np.outer(t, d))
        trocas = t[1:][np.diff(c) != 0]
        print(f"  ao longo da {nome}: classe muda em t = {np.array2string(trocas, precision=2)}"
              f"  (classe nas pontas: {c[0]}, centro: {c[len(t) // 2]}, {c[-1]})")
