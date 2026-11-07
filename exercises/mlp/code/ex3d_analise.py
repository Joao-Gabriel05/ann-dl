"""
Exercicio 3 - Parte D: explicar a matriz de confusao pela estrutura dos aglomerados.

make_classification nao devolve o aglomerado de cada ponto, mas ele e
recuperavel: gerando de novo com shuffle=False (mesma semente), os pontos saem
em blocos de 250, um bloco por aglomerado, com o aglomerado k pertencendo a
classe k % 3. O shuffle=True do dataset original permuta as linhas E as
colunas, entao o casamento e feito pela linha com coordenadas ordenadas.

Mede:
  - centros dos 6 aglomerados e em quais vertices do hipercubo eles estao;
  - distancias entre aglomerados de classes diferentes;
  - rotulos invertidos pelo flip_y (ruido que nenhum modelo acerta);
  - para onde vao os erros da MLP, aglomerado a aglomerado.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix

from _saida import SEED
from ex3_multiclasse import gerar_dados, dividir, treinar_rede

N_AGLOMERADOS = 6


def recuperar_aglomerados(X):
    Xn, _ = make_classification(n_samples=1500, n_features=4, n_informative=4,
                                n_redundant=0, n_repeated=0, n_classes=3,
                                n_clusters_per_class=2, class_sep=1.0,
                                random_state=SEED, shuffle=False)
    chave = lambda r: tuple(np.sort(np.round(r, 9)))
    por_linha = {chave(r): i // 250 for i, r in enumerate(Xn)}
    return np.array([por_linha[chave(r)] for r in X])


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    X, y = gerar_dados()
    ag = recuperar_aglomerados(X)
    classe_do_ag = np.arange(N_AGLOMERADOS) % 3

    print("=== aglomerados ===")
    centros = np.array([X[ag == k].mean(axis=0) for k in range(N_AGLOMERADOS)])
    for k in range(N_AGLOMERADOS):
        sinais = "".join("+" if v > 0 else "-" for v in centros[k])
        trocados = int((y[ag == k] != classe_do_ag[k]).sum())
        print(f"  A{k} (classe {classe_do_ag[k]}): centro {np.array2string(centros[k], precision=2)}"
              f"  vertice {sinais}  | rotulos invertidos: {trocados}")
    flip = int((y != classe_do_ag[ag]).sum())
    print(f"  total de rotulos invertidos (flip_y): {flip} de {len(y)}")

    print("\n=== aglomerados de classes diferentes, do mais sobreposto ao mais separado ===")
    # distancia entre centros dividida pela soma dos desvios NA DIRECAO que liga
    # os dois centros: os aglomerados sao esticados por uma transformacao linear
    # aleatoria, entao a distancia entre centros sozinha engana.
    pares = []
    for i in range(N_AGLOMERADOS):
        for j in range(i + 1, N_AGLOMERADOS):
            if classe_do_ag[i] == classe_do_ag[j]:
                continue
            d = centros[j] - centros[i]
            u = d / np.linalg.norm(d)
            si, sj = (X[ag == i] @ u).std(), (X[ag == j] @ u).std()
            difs = int((np.sign(centros[i]) != np.sign(centros[j])).sum())
            pares.append((np.linalg.norm(d) / (si + sj), i, j, np.linalg.norm(d), si, sj, difs))
    for sep, i, j, d, si, sj, difs in sorted(pares):
        print(f"  A{i}(c{classe_do_ag[i]}) - A{j}(c{classe_do_ag[j]}): separacao {sep:.2f}"
              f" | distancia {d:.2f}, desvios {si:.2f} e {sj:.2f}"
              f" | {difs} coordenada(s) de sinal diferente")

    print("\n=== erros da MLP no teste, por aglomerado de origem ===")
    _, idx_te, _, _ = dividir(np.arange(len(y)), y)
    X_tr, X_te, y_tr, y_te = dividir(X, y)
    rede = treinar_rede(X_tr, y_tr, rng)
    pred = rede.predizer(X_te)
    ag_te = ag[idx_te]
    for k in range(N_AGLOMERADOS):
        m = ag_te == k
        errados = m & (pred != y_te)
        destino = np.bincount(pred[errados], minlength=3)
        inv = int((y_te[m] != classe_do_ag[k]).sum())
        print(f"  A{k} (classe {classe_do_ag[k]}): {m.sum():>3} pontos, {errados.sum():>2} erros "
              f"-> previstos como {destino.tolist()}  | rotulos invertidos no teste: {inv}")
    inv_te = int((y_te != classe_do_ag[ag_te]).sum())
    print(f"  rotulos invertidos no teste: {inv_te} de {len(y_te)}")

    print("\n=== confusoes por par de classes, em 5 sementes de inicializacao ===")
    # a Figura 5 e uma unica semente; aqui se ve qual par e confundido de forma estavel
    total = np.zeros((3, 3), dtype=int)
    for s in (42, 43, 44, 45, 46):
        cm = confusion_matrix(y_te, treinar_rede(X_tr, y_tr, np.random.default_rng(s)).predizer(X_te))
        total += cm
        print(f"  semente {s}: 0-1 {cm[0, 1] + cm[1, 0]:>2} | 0-2 {cm[0, 2] + cm[2, 0]:>2} | "
              f"1-2 {cm[1, 2] + cm[2, 1]:>2} | recall por classe "
              f"{np.array2string(np.diag(cm) / cm.sum(axis=1), precision=2)}")
    print(f"  soma      : 0-1 {total[0, 1] + total[1, 0]:>2} | 0-2 {total[0, 2] + total[2, 0]:>2} | "
          f"1-2 {total[1, 2] + total[2, 1]:>2}")
