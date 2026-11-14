"""A rede e o laço de treino da entrega de classificação.

Separado de `run.py` para que o relatório possa incluir só a arquitetura.
O alvo é desbalanceado (19,4 : 1), então a saída é **um neurônio linear** (logit)
e a perda é `BinaryCrossentropy(from_logits=True)`: o sigmoide vive dentro da
perda, o que mantém o log-sum-exp estável em vez de deixar um sigmoide saturado
arredondar para 0 ou 1. O peso da classe positiva entra via `class_weight`.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

tf.get_logger().setLevel("ERROR")


def build_mlp(n_features, widths=(32, 16), activation="relu", dropout=0.0,
              weight_decay=0.0, batchnorm=False, lr=1e-3, seed=42):
    """MLP com uma única saída linear.

    `widths` curto por decisão: o treino tem 199 positivos, então a capacidade
    precisa ser contada contra esse número, não contra as 4.069 linhas.
    """
    keras.utils.set_random_seed(seed)
    init = (keras.initializers.HeUniform(seed=seed) if activation == "relu"
            else keras.initializers.GlorotUniform(seed=seed))

    model = keras.Sequential([layers.Input(shape=(n_features,))])
    for w in widths:
        model.add(layers.Dense(w, activation=None, kernel_initializer=init))
        if batchnorm:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))
        if dropout > 0:
            model.add(layers.Dropout(dropout, seed=seed))
    model.add(layers.Dense(1, activation=None, kernel_initializer=init))

    opt = (keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
           if weight_decay > 0 else keras.optimizers.Adam(learning_rate=lr))
    model.compile(
        optimizer=opt,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.AUC(name="pr_auc", curve="PR", from_logits=True)],
    )
    return model


def fit(model, Xtr, ytr, Xva, yva, batch_size=64, max_epochs=200, patience=20,
        pos_weight=1.0, verbose=0):
    """Treina com early stopping na validação, restaurando os melhores pesos.

    O monitor é `val_pr_auc` (modo max), não a perda: com 40 positivos na
    validação a perda ponderada oscila muito, e o que a triagem precisa é da
    ordenação dos pacientes, que é exatamente o que a PR-AUC mede.
    """
    es = keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max",
                                       patience=patience,
                                       restore_best_weights=True)
    hist = model.fit(
        Xtr, ytr, validation_data=(Xva, yva), epochs=max_epochs,
        batch_size=batch_size, callbacks=[es], verbose=verbose, shuffle=True,
        class_weight={0: 1.0, 1: float(pos_weight)},
    )
    h = hist.history
    best = int(np.argmax(h["val_pr_auc"]))
    return {
        "history": {k: [float(v) for v in vals] for k, vals in h.items()},
        "epochs_run": len(h["loss"]),
        "best_epoch": best + 1,
        "best_val_pr_auc": float(h["val_pr_auc"][best]),
        "stopped_early": len(h["loss"]) < max_epochs,
    }


def logits(model, X):
    return model.predict(X, verbose=0).ravel()


def n_params(model):
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))
