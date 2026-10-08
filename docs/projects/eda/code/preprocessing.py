"""Pipeline de pré-processamento do projeto (Stroke Prediction Dataset).

Importável pelas entregas seguintes:

    from preprocessing import load_raw, clean, split, build_preprocessor

    X_train, X_test, y_train, y_test = split(clean(load_raw()))
    pre = build_preprocessor().fit(X_train)   # estatísticas só do treino
    Xt_train, Xt_test = pre.transform(X_train), pre.transform(X_test)

Executado como script, ajusta o pipeline no treino e reporta shape, nomes e NaN.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "healthcare-dataset-stroke-data.csv"

TARGET = "stroke"
DROP = ["id"]  # identificador: não descreve o paciente
NUM_SYMMETRIC = ["age"]  # sem assimetria nem outliers pelo IQR
NUM_SKEWED = ["avg_glucose_level", "bmi"]  # assimétricas à direita, com cauda longa
BINARY = ["hypertension", "heart_disease"]  # já 0/1
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
FEATURES = NUM_SYMMETRIC + NUM_SKEWED + BINARY + CATEGORICAL
# Categorias raras demais para a rede aprender algo delas (1 e 22 linhas, nenhum AVC).
RARE = {"gender": ["Other"], "work_type": ["Never_worked"]}


def load_raw(path=DATA_PATH):
    """Lê o CSV bruto. O arquivo grava ausência de BMI como a string 'N/A'."""
    return pd.read_csv(path, na_values=["N/A"])


def clean(df):
    """Remove as linhas das categorias raras (RARE). Filtro por valor, sem estatística
    ajustada: pode rodar antes do split sem vazar informação do teste."""
    rare = np.zeros(len(df), dtype=bool)
    for col, values in RARE.items():
        rare |= df[col].isin(values).to_numpy()
    return df.loc[~rare].reset_index(drop=True)


def split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split estratificado pelo alvo; descarta o id e devolve X_train, X_test, y_train, y_test."""
    X = df.drop(columns=DROP + [TARGET])[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Winsoriza cada coluna nos quantis aprendidos no treino (não remove linhas)."""

    def __init__(self, lower=0.005, upper=0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)


def build_preprocessor(bmi_indicator=True):
    """Pipeline não ajustado; chame .fit(X_train) antes de usar.

    Duas etapas: "columns" transforma cada tipo de coluna (imputação, winsorização,
    Yeo-Johnson, one-hot) e "scale" aplica um único StandardScaler a TODAS as features
    de saída, numéricas e 0/1 — toda entrada da rede chega com média 0 e desvio 1.

    bmi_indicator=False remove a coluna "BMI ausente" — para a ablação de vazamento
    prevista na entrega seguinte (ver 1B no relatório).
    """
    skewed = Pipeline([
        ("impute", SimpleImputer(strategy="median")),  # mediana: robusta à cauda
        ("clip", QuantileClipper(lower=0.005, upper=0.995)),
        # Yeo-Johnson reduz a assimetria; a escala fica para o StandardScaler final
        ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
    ])
    # one-hot completo: categoria nova (ex. gender=Other em produção) vira o vetor de
    # zeros antes da escala, distinto de toda categoria conhecida
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    blocks = [
        ("num", "passthrough", NUM_SYMMETRIC),
        ("skew", skewed, NUM_SKEWED),
        # 0/1 fora do pipeline numérico: preserva o sinal de "BMI ausente"
        ("miss", MissingIndicator(features="all"), ["bmi"]),
        ("bin", "passthrough", BINARY),
        ("cat", categorical, CATEGORICAL),
    ]
    if not bmi_indicator:
        blocks = [b for b in blocks if b[0] != "miss"]
    return Pipeline([
        ("columns", ColumnTransformer(blocks, verbose_feature_names_out=False)),
        ("scale", StandardScaler()),
    ])


def main():
    X_train, X_test, y_train, y_test = split(clean(load_raw()))
    pre = build_preprocessor().fit(X_train)
    Xt_train = pre.transform(X_train)
    Xt_test = pre.transform(X_test)
    names = pre.get_feature_names_out()

    print(f"train: {X_train.shape} -> {Xt_train.shape}")
    print(f"test:  {X_test.shape} -> {Xt_test.shape}")
    print(f"NaN restantes: treino={np.isnan(Xt_train).sum()}, teste={np.isnan(Xt_test).sum()}")
    print(f"{len(names)} features:", ", ".join(names))


if __name__ == "__main__":
    main()
