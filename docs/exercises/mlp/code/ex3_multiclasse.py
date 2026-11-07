"""
Exercicio 3: tres classes, a MESMA rede.

A classe MLP de mlp.py e usada exatamente como no Exercicio 2 - forward,
backward e atualizar nao foram tocados. O que muda e so o que o enunciado
permite mudar:
  - tamanho da saida: 3 unidades (camadas = [4, 16, 3]);
  - saida + perda: SoftmaxCCE no lugar de SigmoideBCE;
  - hiperparametros.

Produz:
  Figura 4 - perda de treino x epoca;
  Figura 5 - matriz de confusao no teste.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from _saida import figura, estilo, CORES, SURFACE, TINTA, TINTA_2, SEED
from mlp import MLP, SoftmaxCCE

CAMADAS = [4, 16, 3]
ETA = 0.1
EPOCAS = 2000


def gerar_dados():
    """1500 amostras, 4 features informativas, 3 classes x 2 aglomerados cada."""
    X, y = make_classification(n_samples=1500, n_features=4, n_informative=4,
                               n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=2, class_sep=1.0,
                               random_state=SEED)
    return X, y


def dividir(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def treinar_rede(X_tr, y_tr, rng, camadas=CAMADAS):
    return MLP(camadas, SoftmaxCCE(), rng).treinar(X_tr, y_tr, ETA, EPOCAS)


def figura_4(rede, caminho=None):
    caminho = caminho or figura("fig04_perda_multiclasse.png")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    estilo(ax)
    ep = np.arange(1, len(rede.historico_) + 1)
    ax.plot(ep, rede.historico_, color=CORES[0], linewidth=2, zorder=3,
            label=f"MLP {CAMADAS}, η = {ETA}")
    ax.axhline(np.log(3), color=TINTA_2, linestyle=":", linewidth=1,
               label="ln 3 ≈ 1.099 (chute uniforme)")
    ax.annotate(f"perda final {rede.perda_final_:.4f}", (ep[-1], rede.historico_[-1]),
                textcoords="offset points", xytext=(-10, 45), ha="right",
                fontsize=10, color=TINTA,
                arrowprops=dict(arrowstyle="->", color=TINTA_2, lw=1.2))
    ax.set_title("Figura 4 — perda de treino (entropia cruzada categórica) × época",
                 fontsize=12.5, color=TINTA, loc="left", pad=12)
    ax.set_xlabel("época", color=TINTA_2)
    ax.set_ylabel("perda de treino (CCE)", color=TINTA_2)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


def figura_5(cm, acc, caminho=None):
    caminho = caminho or figura("fig05_confusao.png")
    fig, ax = plt.subplots(figsize=(6.6, 5.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(cm, cmap="Blues", zorder=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.0%})", ha="center",
                    va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() / 2 else TINTA)
    ax.set_xticks(range(3), [f"classe {k}" for k in range(3)])
    ax.set_yticks(range(3), [f"classe {k}" for k in range(3)])
    ax.tick_params(colors=TINTA_2, labelsize=9.5)
    ax.set_xlabel("classe prevista", color=TINTA_2)
    ax.set_ylabel("classe real", color=TINTA_2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("número de amostras de teste", color=TINTA_2)
    ax.set_title(f"Figura 5 — matriz de confusão no teste\n"
                 f"MLP {CAMADAS} · acurácia {acc:.1%} · % por linha",
                 fontsize=12, color=TINTA, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    X, y = gerar_dados()
    X_tr, X_te, y_tr, y_te = dividir(X, y)
    print(f"X: {X.shape} | por classe: {np.bincount(y)}")
    print(f"treino: {len(X_tr)} {np.bincount(y_tr)} | teste: {len(X_te)} {np.bincount(y_te)}")

    rede = treinar_rede(X_tr, y_tr, rng)
    acc = rede.acuracia(X_te, y_te)
    print("\n=== C - treino ===")
    print(f"  arquitetura     : {CAMADAS} (tanh na oculta, softmax na saida)")
    print(f"  parametros      : {rede.n_parametros()}")
    print(f"  eta, epocas     : {ETA}, {EPOCAS} (lote completo)")
    print(f"  perda inicial   : {rede.historico_[0]:.4f}")
    print(f"  perda final     : {rede.perda_final_:.4f}")
    print(f"  acuracia treino : {rede.acuracia(X_tr, y_tr):.4f}")
    print(f"  acuracia teste  : {acc:.4f}")

    cm = confusion_matrix(y_te, rede.predizer(X_te))
    print("\n=== D - matriz de confusao (linhas = real, colunas = prevista) ===")
    print(cm)
    pares = {(i, j): cm[i, j] + cm[j, i] for i in range(3) for j in range(i + 1, 3)}
    for (i, j), n in sorted(pares.items(), key=lambda kv: -kv[1]):
        print(f"  par {i}-{j}: {n} confusoes ({cm[i, j]} de {i} -> {j}, {cm[j, i]} de {j} -> {i})")

    lr = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    print(f"\nbaseline linear (regressao logistica multinomial): {lr.score(X_te, y_te):.4f}")
    print("matriz de confusao do baseline:")
    print(confusion_matrix(y_te, lr.predict(X_te)))

    figura_4(rede)
    figura_5(cm, acc)
