"""
Exercicio 1 - conferencia numerica da conta feita a mao.

A derivacao do relatorio e feita em Markdown, passo a passo. Este script NAO a
substitui: ele refaz as mesmas contas em numpy, na mesma ordem, so para
conferir cada valor intermediario (e checar a direcao da perda depois do passo).
"""

import numpy as np

x = np.array([0.5, -0.2])
y = 1.0
W1 = np.array([[0.3, -0.1],
               [0.2, 0.4]])
b1 = np.array([0.1, -0.2])
W2 = np.array([0.5, -0.3])
b2 = 0.2
eta = 0.3


def forward(W1, b1, W2, b2):
    z1 = W1 @ x + b1
    h1 = np.tanh(z1)
    u2 = W2 @ h1 + b2
    y_hat = np.tanh(u2)
    L = (y - y_hat) ** 2
    return z1, h1, u2, y_hat, L


def mostra(nome, v):
    print(f"  {nome:<12} = {np.array2string(np.asarray(v), precision=6, floatmode='fixed')}")


if __name__ == "__main__":
    z1, h1, u2, y_hat, L = forward(W1, b1, W2, b2)
    print("A - forward")
    for n, v in [("z1", z1), ("h1", h1), ("u2", u2), ("y_hat", y_hat), ("L", L)]:
        mostra(n, v)

    dL_dyhat = -2 * (y - y_hat)
    dL_du2 = dL_dyhat * (1 - y_hat ** 2)
    dL_dW2 = dL_du2 * h1
    dL_db2 = dL_du2
    dL_dh1 = dL_du2 * W2
    dL_dz1 = dL_dh1 * (1 - h1 ** 2)
    dL_dW1 = np.outer(dL_dz1, x)
    dL_db1 = dL_dz1
    print("\nB - backward")
    for n, v in [("dL/dyhat", dL_dyhat), ("1-yhat^2", 1 - y_hat ** 2),
                 ("dL/du2", dL_du2), ("dL/dW2", dL_dW2), ("dL/db2", dL_db2),
                 ("dL/dh1", dL_dh1), ("1-h1^2", 1 - h1 ** 2), ("dL/dz1", dL_dz1),
                 ("dL/dW1", dL_dW1), ("dL/db1", dL_db1)]:
        mostra(n, v)

    W2n, b2n = W2 - eta * dL_dW2, b2 - eta * dL_db2
    W1n, b1n = W1 - eta * dL_dW1, b1 - eta * dL_db1
    print("\nC - parametros atualizados (eta = 0.3)")
    for n, v in [("W2", W2n), ("b2", b2n), ("W1", W1n), ("b1", b1n)]:
        mostra(n, v)

    *_, y_hat_n, L_n = forward(W1n, b1n, W2n, b2n)
    print("\nsegundo forward com os parametros novos")
    mostra("y_hat", y_hat_n)
    mostra("L", L_n)
    print(f"  variacao da perda: {L_n - L:+.6f}")
