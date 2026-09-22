"""
Exercicio 3 - Partes B, C e D.

B: separacao treino/teste 80/20 estratificada pelo alvo, ANTES de qualquer
   transformacao (evita vazamento de dados).
C: imputacao, one-hot, engenharia de features, log1p nos gastos e escalonamento.
D: verificacao (Figura 2, ausencia de NaN, shape, faixa compativel com tanh).

Todas as estatisticas (medianas, categorias, media/desvio) sao aprendidas SO no
treino e depois aplicadas ao teste.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _saida import figura, dado

SEED = 42
ALVO = "Transported"
DESCARTAR = ["PassengerId", "Name", "Cabin"]
GASTOS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
NUMERICAS = ["Age"] + GASTOS
LOG = GASTOS + ["TotalSpend"]          # colunas de cauda pesada
FALTANTE = "Desconhecido"              # nivel explicito para categorica ausente

SURFACE, TINTA, TINTA_2 = "#fcfcfb", "#1a1a19", "#5c5b55"
COR_ANTES, COR_DEPOIS = "#eb6834", "#2a78d6"


# ----------------------------------------------------------------- B
def separar_estratificado(df, alvo=ALVO, teste=0.2, seed=SEED):
    """80/20 estratificado: a proporcao do alvo e preservada nos dois lados.
    Embaralha dentro de cada classe e corta - sem sklearn."""
    rng = np.random.default_rng(seed)
    idx_teste = []
    for _, grupo in df.groupby(alvo).groups.items():
        idx = np.array(grupo)
        rng.shuffle(idx)
        idx_teste.extend(idx[:int(round(len(idx) * teste))])
    mascara = df.index.isin(idx_teste)
    return df[~mascara].copy(), df[mascara].copy()


# ----------------------------------------------------------------- C
class PreProcessador:
    """Aprende tudo no fit (so com o treino) e reaplica no transform."""

    def fit(self, df):
        self.medianas_ = df[NUMERICAS].median()          # C.1 numericas
        self.categorias_ = {                             # C.2 categorias vistas
            c: sorted(df[c].dropna().astype(str).unique().tolist()) + [FALTANTE]
            for c in CATEGORICAS
        }
        Z = self._ate_log(df)                            # C.3 e C.4
        self.mu_ = Z[self.num_cols_].mean()              # C.5 escalonamento
        self.sigma_ = Z[self.num_cols_].std(ddof=0).replace(0, 1.0)
        return self

    def _ate_log(self, df):
        """Imputacao -> TotalSpend -> log1p. Sem escalonar (o fit usa isto)."""
        X = df.drop(columns=[c for c in DESCARTAR if c in df], errors="ignore")
        X[NUMERICAS] = X[NUMERICAS].fillna(self.medianas_)     # mediana do TREINO
        X["TotalSpend"] = X[GASTOS].sum(axis=1)                # C.3
        X[LOG] = np.log1p(X[LOG])                              # C.4
        self.num_cols_ = NUMERICAS + ["TotalSpend"]
        return X

    def transform(self, df):
        X = self._ate_log(df)
        X[self.num_cols_] = (X[self.num_cols_] - self.mu_) / self.sigma_  # z-score
        partes = [X[self.num_cols_]]
        for c in CATEGORICAS:                                   # C.2 one-hot
            v = X[c].astype(str).where(X[c].notna(), FALTANTE)
            # categoria nunca vista no treino -> vira linha toda de zeros
            dummies = pd.DataFrame(
                {f"{c}={k}": (v == k).astype(float) for k in self.categorias_[c]},
                index=X.index)
            partes.append(dummies)
        return pd.concat(partes, axis=1)


# ----------------------------------------------------------------- D
def figura_2(bruto, processado, col="FoodCourt", caminho=None):
    caminho = caminho or figura("fig06_foodcourt.png")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    for ax, v, cor, titulo, xlabel in [
        (axes[0], bruto, COR_ANTES,
         f"ANTES — {col} bruto\nfaixa [{bruto.min():.0f}, {bruto.max():.0f}] "
         f"· assimetria {bruto.skew():.2f}", f"{col} (unidades monetárias)"),
        (axes[1], processado, COR_DEPOIS,
         f"DEPOIS — log1p + padronização\nfaixa [{processado.min():.2f}, "
         f"{processado.max():.2f}] · assimetria {processado.skew():.2f}",
         f"{col} transformado (z)"),
    ]:
        ax.set_facecolor(SURFACE)
        ax.hist(v, bins=45, color=cor, alpha=0.85, zorder=2)
        ax.grid(True, axis="y", color="#e6e5e0", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color("#d5d4ce")
        ax.tick_params(colors=TINTA_2, labelsize=9)
        ax.set_title(titulo, fontsize=11.5, color=TINTA, loc="left", pad=10)
        ax.set_xlabel(xlabel, color=TINTA_2)
        ax.set_ylabel("passageiros", color=TINTA_2)

    for lim, rotulo in ((-2, "−2"), (2, "+2")):
        axes[1].axvline(lim, color=TINTA, linestyle="--", linewidth=1.4, zorder=4)
    axes[1].annotate("faixa útil da tanh: [−2, +2]\n(fora daqui ela satura)",
                     (2, 0), xytext=(12, 120), textcoords="offset points",
                     fontsize=9.5, color=TINTA)
    fig.suptitle("Figura 2 — FoodCourt antes e depois do pré-processamento "
                 "(“Figura 6” do enunciado)",
                 fontsize=13, color=TINTA, x=0.02, ha="left", y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"\nFigura salva em: {caminho}")


if __name__ == "__main__":
    df = pd.read_csv(dado("train.csv"))

    # ---------------- B ----------------
    treino, teste = separar_estratificado(df)
    print("=== B — separação 80/20 estratificada (seed 42) ===")
    for nome, parte in (("treino", treino), ("teste ", teste)):
        p = parte[ALVO].mean()
        print(f"  {nome}: {len(parte):>5} linhas ({len(parte)/len(df):.1%})"
              f" | Transported=True: {p:.2%}")
    print(f"  dataset completo: {df[ALVO].mean():.2%} de True "
          f"(a estratificação preservou a proporção)")

    # ---------------- C ----------------
    pp = PreProcessador().fit(treino)          # <- APENAS o treino
    Xtr, Xte = pp.transform(treino), pp.transform(teste)
    ytr = treino[ALVO].astype(int).to_numpy()
    yte = teste[ALVO].astype(int).to_numpy()

    print("\n=== C.1 — imputação (estatísticas do treino) ===")
    for c in NUMERICAS:
        print(f"  {c:<13} mediana do treino = {pp.medianas_[c]:8.1f}"
              f" | {df[c].isna().sum():>3} NaN preenchidos no dataset")
    print(f"  categóricas -> nível explícito '{FALTANTE}' (não é a moda)")

    print("\n=== C.2 — one-hot ===")
    for c, cats in pp.categorias_.items():
        print(f"  {c:<12} -> {len(cats)} colunas: {cats}")

    novo = teste.head(3).copy()
    novo["HomePlanet"] = "Titan"               # categoria inexistente no treino
    cols = [c for c in Xte.columns if c.startswith("HomePlanet=")]
    print("  categoria nunca vista no treino ('Titan'):")
    print(pp.transform(novo)[cols].to_string(index=False))
    print("  -> todas as colunas de HomePlanet ficam em 0 (nenhuma coluna nova,"
          " nenhum erro, shape preservado)")

    print("\n=== C.5 — escalonamento (z-score ajustado no treino) ===")
    faixa = pd.DataFrame({"mín": Xtr.min(), "máx": Xtr.max()})
    print(faixa.to_string(float_format=lambda v: f"{v:7.3f}"))

    # ---------------- D ----------------
    print("\n=== D.2 — checagens finais ===")
    print(f"  X_treino: {Xtr.shape} | X_teste: {Xte.shape} | "
          f"{Xtr.shape[1]} features")
    print(f"  NaN no treino: {int(Xtr.isna().sum().sum())} | "
          f"no teste: {int(Xte.isna().sum().sum())}")
    print(f"  faixa global treino: [{Xtr.values.min():.3f}, {Xtr.values.max():.3f}]"
          f" | teste: [{Xte.values.min():.3f}, {Xte.values.max():.3f}]")
    dentro = float((np.abs(Xtr.values) <= 2).mean())
    print(f"  valores dentro de [-2, 2] (faixa útil da tanh): {dentro:.2%}")
    print(f"  alvo: treino {ytr.mean():.2%} True | teste {yte.mean():.2%} True")

    np.savez(dado("dados_processados.npz"), Xtr=Xtr.to_numpy(), ytr=ytr,
             Xte=Xte.to_numpy(), yte=yte, colunas=np.array(Xtr.columns))
    print("  arrays salvos em dados_processados.npz")

    figura_2(treino["FoodCourt"].dropna(), Xtr["FoodCourt"])
