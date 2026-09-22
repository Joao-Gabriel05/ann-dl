"""
Exercicio 3 - Parte A: Conheca os dados (Spaceship Titanic, train.csv).

Objetivo do dataset, balanceamento do alvo, tipos de feature, valores faltantes
e estatisticas das colunas de gasto.

Coloque train.csv nesta pasta (ver README) e rode:  python3 ex3a_conhecer.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _saida import figura, dado

CAMINHO = dado("train.csv")

ALVO = "Transported"
ID = ["PassengerId", "Name"]           # identificadores, nao sao features
GASTOS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

CORES = {True: "#2a78d6", False: "#eb6834"}
SURFACE = "#fcfcfb"
TINTA = "#1a1a19"
TINTA_2 = "#5c5b55"


def carregar():
    if not CAMINHO.exists():
        sys.exit(f"train.csv nao encontrado em {CAMINHO}.\n"
                 "Baixe em https://www.kaggle.com/competitions/spaceship-titanic/data "
                 "e coloque o arquivo nesta pasta.")
    return pd.read_csv(CAMINHO)


def tipos_de_feature(df):
    """Separa as colunas em numericas e categoricas (ignorando id e alvo)."""
    features = [c for c in df.columns if c not in ID + [ALVO]]
    num = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in features if c not in num]
    return num, cat


def tabela_faltantes(df):
    n = len(df)
    falt = df.isna().sum()
    return (pd.DataFrame({"faltantes": falt, "% do total": 100 * falt / n})
            .sort_values("faltantes", ascending=False))


def _estilo(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color="#e6e5e0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#d5d4ce")
    ax.tick_params(colors=TINTA_2, labelsize=9)


def figura_1(df, caminho=None):
    caminho = caminho or figura("figA7_gastos.png")
    """Distribuicao das 5 colunas de gasto - a assimetria em imagem."""
    fig, axes = plt.subplots(1, len(GASTOS), figsize=(17, 4.2), dpi=140,
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, col in zip(axes, GASTOS):
        _estilo(ax)
        v = df[col].dropna()
        # log1p so para o eixo: sem isso, tudo colapsa numa barra em zero
        ax.hist(np.log1p(v), bins=40, color="#2a78d6", alpha=0.8, zorder=2)
        ax.axvline(np.log1p(v.mean()), color="#eb6834", linewidth=2, zorder=4)
        ax.axvline(np.log1p(v.median()), color=TINTA, linewidth=2,
                   linestyle="--", zorder=4)
        zeros = (v == 0).mean()
        ax.set_title(f"{col}\n{zeros:.0%} de zeros", fontsize=10.5, color=TINTA,
                     loc="left", pad=8)
        ax.set_xlabel("log(1 + gasto)", color=TINTA_2)
    axes[0].set_ylabel("passageiros", color=TINTA_2)
    handles = [plt.Line2D([], [], color="#eb6834", linewidth=2, label="média"),
               plt.Line2D([], [], color=TINTA, linewidth=2, linestyle="--",
                          label="mediana")]
    fig.legend(handles=handles, frameon=False, fontsize=10, ncol=2,
               loc="upper right", bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("Figura 1 — distribuição dos gastos (eixo em log): "
                 "cauda longa e pilha de zeros",
                 fontsize=13, color=TINTA, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"\nFigura salva em: {caminho}")


if __name__ == "__main__":
    df = carregar()
    pd.set_option("display.width", 120)

    print("=== formato ===")
    print(f"  {df.shape[0]} linhas x {df.shape[1]} colunas")
    print("  colunas:", ", ".join(df.columns))

    print("\n=== A.2 — balanceamento do alvo (Transported) ===")
    cont = df[ALVO].value_counts()
    prop = df[ALVO].value_counts(normalize=True)
    for v in cont.index:
        print(f"  {str(v):>6}: {cont[v]:>5} passageiros  ({prop[v]:.2%})")
    print(f"  razao entre classes: {cont.max() / cont.min():.3f} : 1")

    num, cat = tipos_de_feature(df)
    print("\n=== A.3 — tipos de feature ===")
    print(f"  numericas ({len(num)}):", ", ".join(num))
    print(f"  categoricas ({len(cat)}):", ", ".join(cat))
    print(f"  identificadores (fora do modelo): {', '.join(ID)}")
    for c in cat:
        vals = df[c].dropna().unique()
        amostra = ", ".join(map(str, vals[:4])) + (" ..." if len(vals) > 4 else "")
        print(f"    {c:<12} {len(vals):>4} valores distintos  [{amostra}]")

    print("\n=== A.4 — valores faltantes ===")
    print(tabela_faltantes(df).to_string(
        float_format=lambda v: f"{v:5.2f}"))
    linhas_ok = df.notna().all(axis=1).mean()
    print(f"\n  linhas completas (sem nenhum NaN): {linhas_ok:.2%}")
    print(f"  linhas com pelo menos 1 NaN      : {1 - linhas_ok:.2%}")

    print("\n=== A.5 — colunas de gasto (todos os passageiros) ===")
    est = df[GASTOS].agg(["mean", "median", "max", "std"]).T
    est["% zeros"] = [(df[c] == 0).mean() * 100 for c in GASTOS]
    est["p90"] = [df[c].quantile(0.90) for c in GASTOS]
    est["p99"] = [df[c].quantile(0.99) for c in GASTOS]
    est["assimetria"] = [df[c].skew() for c in GASTOS]
    print(est.to_string(float_format=lambda v: f"{v:9.2f}"))

    print("\n=== A.5 — os mesmos gastos, so entre quem gastou (valor > 0) ===")
    est2 = pd.DataFrame({
        "n": [(df[c] > 0).sum() for c in GASTOS],
        "mean": [df.loc[df[c] > 0, c].mean() for c in GASTOS],
        "median": [df.loc[df[c] > 0, c].median() for c in GASTOS],
        "max": [df[c].max() for c in GASTOS],
    }, index=GASTOS)
    est2["média/mediana"] = est2["mean"] / est2["median"]
    est2["% do total gasto"] = (100 * df[GASTOS].sum() / df[GASTOS].sum().sum()).values
    print(est2.to_string(float_format=lambda v: f"{v:9.2f}"))

    top = df[GASTOS].sum(axis=1).sort_values(ascending=False)
    total = top.sum()
    for frac in (0.01, 0.05, 0.10):
        k = int(len(top) * frac)
        print(f"\n  os {frac:.0%} maiores gastadores ({k} passageiros) concentram "
              f"{100 * top.head(k).sum() / total:.1f}% de todo o gasto")

    figura_1(df)
