"""Gera todas as figuras e números do EDA (Stroke Prediction Dataset).

    python docs/projects/eda/code/eda.py

Figuras vão para ../figures/; os números citados no relatório, para results.json.
Seções 1A–1B descrevem o arquivo bruto com contagens (nada ajustado, nada relacionado ao
alvo); 1C usa o arquivo limpo; da 1D em diante, tudo usa só o treino.
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from scipy.stats import chi2_contingency, skew
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from preprocessing import (
    BINARY, CATEGORICAL, NUM_SKEWED, NUM_SYMMETRIC, RANDOM_STATE, TARGET,
    RARE, build_preprocessor, clean, load_raw, split,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")

FIG = Path(__file__).resolve().parents[1] / "figures"
OUT = Path(__file__).resolve().parent / "results.json"
FIG.mkdir(exist_ok=True)

NUM = NUM_SYMMETRIC + NUM_SKEWED
NO, YES = "#2a78d6", "#eb6834"  # sem AVC / com AVC
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {0: "Sem AVC (0)", 1: "Com AVC (1)"}
UNITS = {"age": "idade (anos)", "avg_glucose_level": "glicose média (mg/dL)", "bmi": "IMC (kg/m²)"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})

R = {}  # tudo o que o relatório cita


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)


def r(x, n=4):
    return None if x is None else round(float(x), n)


def wilson(k, n, z=1.96):
    """IC 95% de Wilson para uma proporção (estável com poucos positivos)."""
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


def cramers_v(a, b):
    t = pd.crosstab(a, b)
    chi2, p, dof, _ = chi2_contingency(t)
    n = t.values.sum()
    k = min(t.shape) - 1
    return np.sqrt(chi2 / (n * k)), chi2, p, dof


def scatter_by_target(ax, xy, y, s=6):
    """Positivos por cima, para não sumirem sob a classe majoritária."""
    for cls, color, alpha in [(0, NO, 0.35), (1, YES, 0.9)]:
        m = y == cls
        ax.scatter(xy[m, 0], xy[m, 1], s=s if cls == 0 else s * 2.5, c=color, alpha=alpha,
                   linewidths=0, label=LABEL[cls])


def knn_enrichment(emb, y, k=10):
    """Fração média de vizinhos com AVC entre os k vizinhos de cada ponto com AVC."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    idx = nn.kneighbors(emb[y == 1], return_distance=False)[:, 1:]
    return float(y[idx].mean())


# =============================================================================
# 1A–1C · arquivo bruto
# =============================================================================
df = load_raw()
R["shape_raw"] = list(df.shape)
R["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

# 1B · qualidade
miss = df.isna().sum()
R["missing"] = {c: {"n": int(v), "pct": r(100 * v / len(df), 2)} for c, v in miss.items() if v}
R["duplicates"] = int(df.duplicated().sum())
R["duplicates_without_id"] = int(df.drop(columns="id").duplicated().sum())
R["id_unique"] = bool(df["id"].is_unique)
R["gender_other"] = int((df.gender == "Other").sum())
R["smoking_unknown"] = int((df.smoking_status == "Unknown").sum())
R["smoking_unknown_pct"] = r(100 * (df.smoking_status == "Unknown").mean(), 2)
R["smoking_unknown_under18"] = int(((df.smoking_status == "Unknown") & (df.age < 18)).sum())
R["age_fractional"] = int((df.age % 1 != 0).sum())
R["age_fractional_max"] = r(df.loc[df.age % 1 != 0, "age"].max(), 2)
R["children_worktype_max_age"] = r(df.loc[df.work_type == "children", "age"].max(), 1)
R["never_worked_age_range"] = [r(df.loc[df.work_type == "Never_worked", "age"].min(), 1),
                               r(df.loc[df.work_type == "Never_worked", "age"].max(), 1)]
R["married_under_18"] = int(((df.age < 18) & (df.ever_married == "Yes")).sum())
R["smokers_under_10"] = int(((df.age < 10) & (df.smoking_status.isin(["smokes", "formerly smoked"]))).sum())
R["bmi_over_60"] = int((df.bmi > 60).sum())
R["bmi_min_max"] = [r(df.bmi.min(), 1), r(df.bmi.max(), 1)]
R["glucose_min_max"] = [r(df.avg_glucose_level.min(), 2), r(df.avg_glucose_level.max(), 2)]
R["age_min_max"] = [r(df.age.min(), 2), r(df.age.max(), 1)]
R["youngest_strokes"] = df.loc[df[TARGET] == 1, "age"].sort_values().head(3).tolist()
R["constant_columns"] = [c for c in df if df[c].nunique() <= 1]

# Vazamento: o arquivo vem ordenado pelo alvo, e o id carregaria isso?
first_neg = int((df[TARGET] == 0).idxmax())
R["rows_sorted_by_target"] = {"first_negative_row": first_neg,
                              "all_positives_first": bool(df.loc[: first_neg - 1, TARGET].all())}
R["auc_id"] = r(roc_auc_score(df[TARGET], df["id"]), 3)
R["auc_row_index"] = r(roc_auc_score(df[TARGET], -df.index.to_numpy()), 3)
R["raw_cat_freq"] = {c: {str(k): int(v) for k, v in df[c].value_counts().items()} for c in CATEGORICAL}

# Limpeza: categorias raras demais para a rede (ver RARE em preprocessing.py)
raw = df
df = clean(raw)
removed = raw.loc[~raw["id"].isin(df["id"])]
R["clean"] = {"removed_rows": int(len(removed)), "removed_pos": int(removed[TARGET].sum()),
              "removed_bmi_missing": int(removed.bmi.isna().sum()),
              "per_rule": {f"{c}={v}": int((raw[c] == v).sum()) for c, vs in RARE.items() for v in vs},
              "shape_clean": list(df.shape)}

# 1C · alvo (arquivo limpo)
vc = df[TARGET].value_counts().sort_index()
R["target_counts"] = {int(k): int(v) for k, v in vc.items()}
R["target_pct"] = {int(k): r(100 * v / len(df), 2) for k, v in vc.items()}
R["imbalance_ratio"] = r(vc[0] / vc[1], 2)
R["baseline_accuracy"] = r(100 * vc[0] / len(df), 2)

fig, ax = plt.subplots(figsize=(6, 3.4))
bars = ax.bar([LABEL[0], LABEL[1]], vc.values, color=[NO, YES], width=0.55)
for b, v in zip(bars, vc.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v:,} ({100 * v / len(df):.2f}%)".replace(",", "."),
            ha="center", va="bottom", color=INK)
ax.set(title=f"Distribuição do alvo: AVC é a classe rara ({vc[0] / vc[1]:.1f} : 1)".replace(".", ","),
       xlabel="stroke",
       ylabel="Número de pacientes", ylim=(0, vc.max() * 1.15))
ax.legend(handles=bars, labels=[LABEL[0], LABEL[1]], loc="upper right")
save(fig, "fig01-alvo.png")

# =============================================================================
# 1D · split — daqui em diante, só treino
# =============================================================================
X_train, X_test, y_train, y_test = split(df)
R["split"] = {
    "train": list(X_train.shape), "test": list(X_test.shape),
    "train_pos": int(y_train.sum()), "test_pos": int(y_test.sum()),
    "train_pos_pct": r(100 * y_train.mean(), 2), "test_pos_pct": r(100 * y_test.mean(), 2),
}
tr = X_train.assign(stroke=y_train.to_numpy())

# 1B (no treino) · relações com o alvo que investigam vazamento
R["bmi_missing_by_target"] = {
    int(k): r(100 * v, 2) for k, v in tr.groupby(TARGET)["bmi"].apply(lambda s: s.isna().mean()).items()
}
R["stroke_rate_bmi_missing"] = r(100 * tr.loc[tr.bmi.isna(), TARGET].mean(), 2)
R["stroke_rate_bmi_present"] = r(100 * tr.loc[tr.bmi.notna(), TARGET].mean(), 2)
R["bmi_missing_counts_train"] = {"missing": int(tr.bmi.isna().sum()),
                                 "missing_pos": int(tr.loc[tr.bmi.isna(), TARGET].sum())}
_, _, p_miss, _ = cramers_v(tr.bmi.isna(), tr[TARGET])
R["bmi_missing_chi2_p"] = float(f"{p_miss:.3g}")
auc_single = {}
for c in NUM + BINARY:
    s = tr[c].fillna(tr[c].median())
    auc_single[c] = r(roc_auc_score(tr[TARGET], s), 3)
R["auc_single_feature"] = auc_single

# =============================================================================
# 2A · numéricas
# =============================================================================
desc = {}
for c in NUM:
    s = tr[c].dropna()
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    desc[c] = {"n": int(s.size), "mean": r(s.mean(), 2), "median": r(q2, 2), "std": r(s.std(), 2),
               "min": r(s.min(), 2), "q1": r(q1, 2), "q3": r(q3, 2), "max": r(s.max(), 2),
               "skew": r(skew(s), 2), "iqr_outliers": int(out),
               "iqr_outliers_pct": r(100 * out / s.size, 2),
               "upper_fence": r(q3 + 1.5 * iqr, 2)}
R["num_describe_train"] = desc
g = tr.avg_glucose_level
R["glucose_over_150_pct"] = r(100 * (g > 150).mean(), 2)
R["glucose_valley_150_170_pct"] = r(100 * ((g > 150) & (g < 170)).mean(), 2)
R["age_under18_pct"] = r(100 * (tr.age < 18).mean(), 2)

fig, axes = plt.subplots(2, 3, figsize=(12, 5.6), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
for j, c in enumerate(NUM):
    s = tr[c].dropna()
    ax = axes[0, j]
    ax.hist(s, bins=40, color=NO, alpha=0.85, edgecolor="white", linewidth=0.5, label="contagem")
    ax.axvline(s.mean(), color=INK, lw=1.5, ls="--", label=f"média {s.mean():.1f}")
    ax.axvline(s.median(), color=YES, lw=2, label=f"mediana {s.median():.1f}")
    ax.set(title=f"{c} — assimetria {skew(s):.2f}", ylabel="Pacientes (treino)")
    ax.legend(loc="upper right", fontsize=8)
    bx = axes[1, j]
    bx.boxplot(s, orientation="horizontal", widths=0.6, patch_artist=True,
               boxprops={"facecolor": "#d6e4f6", "edgecolor": NO},
               medianprops={"color": YES, "lw": 2},
               flierprops={"marker": "o", "ms": 3, "mfc": YES, "mec": "none", "alpha": 0.5})
    bx.set(xlabel=UNITS[c], yticks=[])
    bx.set_title(f"{desc[c]['iqr_outliers']} outliers pelo IQR", fontsize=9, fontweight="normal")
n_tr = f"{len(tr):,}".replace(",", ".")
fig.suptitle(f"Distribuições numéricas no treino (n = {n_tr}): idade simétrica, glicose bimodal, IMC com cauda",
             fontweight="bold")
fig.tight_layout()
save(fig, "fig02-numericas.png")

# =============================================================================
# 2B · categóricas (inclui as binárias 0/1)
# =============================================================================
cats = CATEGORICAL + BINARY
freq = {}
for c in cats:
    vc = tr[c].value_counts()
    freq[c] = {"cardinality": int(vc.size),
               "counts": {str(k): int(v) for k, v in vc.items()},
               "pct": {str(k): r(100 * v / len(tr), 2) for k, v in vc.items()},
               "rare_lt_1pct": [str(k) for k, v in vc.items() if v / len(tr) < 0.01]}
R["cat_freq_train"] = freq

fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
for ax, c in zip(axes.flat, cats):
    vc = tr[c].value_counts().sort_values()
    rare = vc / len(tr) < 0.01
    colors = [YES if x else NO for x in rare]
    ax.barh(vc.index.astype(str), vc.values, color=colors, height=0.6)
    for i, v in enumerate(vc.values):
        ax.text(v + len(tr) * 0.01, i, f"{v} ({100 * v / len(tr):.1f}%)", va="center", fontsize=8, color=INK)
    ax.set(title=f"{c} (k = {vc.size})", xlabel="Pacientes (treino)", xlim=(0, vc.max() * 1.45))
    ax.grid(axis="y", visible=False)
axes.flat[-1].axis("off")
from matplotlib.patches import Patch
n_rare = sum(len(v["rare_lt_1pct"]) for v in freq.values())
handles = [Patch(color=NO, label="categoria ≥ 1% do treino")]
if n_rare:
    handles.append(Patch(color=YES, label="categoria rara (< 1%)"))
axes.flat[-1].legend(handles=handles, loc="center", fontsize=10, title="Legenda")
rare_txt = f"{n_rare} categorias raras" if n_rare else "nenhuma rara (< 1%)"
fig.suptitle(f"Frequência das categóricas no treino: nenhuma de alta cardinalidade; {rare_txt}",
             fontweight="bold")
fig.tight_layout()
save(fig, "fig03-categoricas.png")

# =============================================================================
# 3A · numéricas × numéricas (e binárias, como 0/1)
# =============================================================================
cols = NUM + BINARY + [TARGET]
sp = tr[cols].corr(method="spearman")
pe = tr[cols].corr(method="pearson")
R["spearman"] = sp.round(3).to_dict()
R["pearson"] = pe.round(3).to_dict()
pairs = []
for i, a in enumerate(cols[:-1]):
    for b in cols[i + 1:-1]:
        pairs.append((a, b, sp.loc[a, b], pe.loc[a, b]))
pairs.sort(key=lambda t: -abs(t[2]))
R["top_pairs"] = [{"a": a, "b": b, "spearman": r(s, 3), "pearson": r(p, 3)} for a, b, s, p in pairs[:5]]
R["corr_with_target"] = {c: {"spearman": r(sp.loc[c, TARGET], 3), "pearson": r(pe.loc[c, TARGET], 3)}
                         for c in cols[:-1]}
adults = tr[tr.age >= 18]
R["spearman_age_bmi_adults"] = r(adults[["age", "bmi"]].corr(method="spearman").iloc[0, 1], 3)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, m, name in [(axes[0], sp, "Spearman"), (axes[1], pe, "Pearson")]:
    im = ax.imshow(m.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else INK)
    ax.set_title(f"Correlação de {name}")
    ax.set_xlabel("feature")
    ax.set_ylabel("feature")
cb = fig.colorbar(im, ax=axes, shrink=0.85)
cb.set_label("coeficiente (−1 a 1)")
a, b, top = pairs[0][:3]
fig.suptitle(f"Nenhum par redundante: maior |ρ| de Spearman = {top:.2f} ({a} × {b})".replace(".", ","),
             fontweight="bold")
save(fig, "fig04-correlacao.png")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (a, b) in zip(axes, [("age", "bmi"), ("age", "avg_glucose_level"), ("bmi", "avg_glucose_level")]):
    d = tr[[a, b, TARGET]].dropna()
    scatter_by_target(ax, d[[a, b]].to_numpy(), d[TARGET].to_numpy(), s=5)
    ax.set(xlabel=UNITS[a], ylabel=UNITS[b], title=f"{a} × {b}  (ρ = {sp.loc[a, b]:.2f})")
axes[0].legend(loc="upper left", markerscale=2)
fig.suptitle("Dispersão das numéricas no treino, por alvo: AVC concentrado acima dos ~50 anos",
             fontweight="bold")
fig.tight_layout()
save(fig, "fig05-dispersao.png")

# Associação entre categóricas (Cramér's V) — redundância do lado categórico
cv = {}
for i, a in enumerate(cats):
    for b in cats[i + 1:]:
        cv[f"{a} × {b}"] = r(cramers_v(tr[a], tr[b])[0], 3)
R["cramers_v_pairs"] = dict(sorted(cv.items(), key=lambda kv: -kv[1])[:5])
R["eta_age_by_worktype"] = None  # preenchido em 3C

# =============================================================================
# 3B · categóricas × alvo
# =============================================================================
base = y_train.mean()
R["base_rate_train"] = r(100 * base, 2)
rates = {}
for c in cats:
    g = tr.groupby(c)[TARGET].agg(["sum", "count"])
    v, chi2, p, dof = cramers_v(tr[c], tr[TARGET])
    rows = {}
    for k, (pos, n) in g.iterrows():
        lo, hi = wilson(pos, n)
        rows[str(k)] = {"n": int(n), "pos": int(pos), "rate": r(100 * pos / n, 2),
                        "ci95": [r(100 * lo, 2), r(100 * hi, 2)]}
    rates[c] = {"rows": rows, "cramers_v": r(v, 3), "chi2": r(chi2, 2), "p": float(f"{p:.3g}"), "dof": int(dof)}
R["cat_target_train"] = rates

fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
for ax, c in zip(axes.flat, cats):
    rows = rates[c]["rows"]
    keys = sorted(rows, key=lambda k: rows[k]["rate"])
    y = np.arange(len(keys))
    rate = np.array([rows[k]["rate"] for k in keys])
    lo = rate - np.array([rows[k]["ci95"][0] for k in keys])
    hi = np.array([rows[k]["ci95"][1] for k in keys]) - rate
    ax.errorbar(rate, y, xerr=[lo, hi], fmt="o", color=YES, ecolor=MUTED, capsize=3, ms=6,
                label="taxa de AVC (IC 95%)")
    ax.axvline(100 * base, color=NO, ls="--", lw=1.5, label=f"taxa geral {100 * base:.1f}%")
    ax.set_yticks(y, [f"{k} (n={rows[k]['n']})" for k in keys], fontsize=8)
    ax.set(title=f"{c} — V = {rates[c]['cramers_v']:.2f}", xlabel="% com AVC", xlim=(-1, 32))
    ax.grid(axis="y", visible=False)
axes.flat[-1].axis("off")
h, l = axes.flat[0].get_legend_handles_labels()
axes.flat[-1].legend(h, l, loc="center", fontsize=10, title="Legenda")
fig.suptitle("Taxa de AVC por categoria (treino): doença cardíaca, hipertensão e casamento multiplicam o risco por 3–4×",
             fontweight="bold")
fig.tight_layout()
save(fig, "fig06-categoricas-alvo.png")

# =============================================================================
# 3C · numéricas × categóricas (boxplots agrupados)
# =============================================================================
def grouped_box(ax, data, num, cat, order=None, colors=None):
    order = order or sorted(data[cat].dropna().unique(), key=str)
    vals = [data.loc[data[cat] == k, num].dropna() for k in order]
    bp = ax.boxplot(vals, widths=0.6, patch_artist=True,
                    medianprops={"color": INK, "lw": 2},
                    flierprops={"marker": "o", "ms": 2.5, "alpha": 0.4, "mec": "none", "mfc": MUTED})
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i] if colors else "#d6e4f6")
        patch.set_edgecolor(MUTED)
    ax.set_xticks(range(1, len(order) + 1), [str(k) for k in order], rotation=20, ha="right")
    ax.grid(axis="x", visible=False)
    return {str(k): {"median": r(v.median(), 2), "iqr": r(v.quantile(.75) - v.quantile(.25), 2), "n": int(v.size)}
            for k, v in zip(order, vals)}


box = {}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, c in zip(axes, NUM):
    box[f"{c} × stroke"] = grouped_box(ax, tr, c, TARGET, order=[0, 1], colors=[NO, YES])
    ax.set_xticks([1, 2], [LABEL[0], LABEL[1]], rotation=0, ha="center")
    ax.set(title=f"{c} por alvo", xlabel="stroke", ylabel=UNITS[c])
axes[0].legend(handles=[Patch(color=NO, label=LABEL[0]), Patch(color=YES, label=LABEL[1])], loc="lower left")
fig.suptitle("Numéricas por alvo (treino): idade separa as classes; glicose desloca a cauda; IMC quase não muda",
             fontweight="bold")
fig.tight_layout()
save(fig, "fig07-numericas-alvo.png")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
specs = [("age", "smoking_status"), ("bmi", "work_type"), ("avg_glucose_level", "heart_disease")]
for ax, (n, c) in zip(axes, specs):
    box[f"{n} × {c}"] = grouped_box(ax, tr, n, c)
    ax.set(title=f"{n} por {c}", xlabel=c, ylabel=UNITS[n])
fig.legend(handles=[Patch(facecolor="#d6e4f6", edgecolor=MUTED, label="IQR (Q1–Q3)"),
                    plt.Line2D([], [], color=INK, lw=2, label="mediana")],
           loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
fig.suptitle("Boxplots agrupados (treino): 'Unknown' e 'children' são, sobretudo, crianças",
             fontweight="bold")
fig.tight_layout()
save(fig, "fig08-boxplots-agrupados.png")
R["grouped_boxes"] = box
# razão de correlação η: quanto da variância de age é explicada por work_type
gm = tr.groupby("work_type").age.transform("mean")
R["eta_age_by_worktype"] = r(np.sqrt(((gm - tr.age.mean()) ** 2).sum() / ((tr.age - tr.age.mean()) ** 2).sum()), 3)

# =============================================================================
# 4A · efeito das estratégias (medido no treino)
# =============================================================================
pre = build_preprocessor().fit(X_train)
Xt_train = pre.transform(X_train)
Xt_test = pre.transform(X_test)
names = list(pre.get_feature_names_out())
cols_ct = pre.named_steps["columns"]
clip = cols_ct.named_transformers_["skew"].named_steps["clip"]
imp = cols_ct.named_transformers_["skew"].named_steps["impute"]
raw_sk = X_train[NUM_SKEWED].fillna(pd.Series(imp.statistics_, index=NUM_SKEWED))
clipped = (raw_sk < clip.low_) | (raw_sk > clip.high_)
R["clip_bounds"] = {c: [r(lo, 2), r(hi, 2)] for c, lo, hi in zip(NUM_SKEWED, clip.low_, clip.high_)}
R["clip_rows_per_feature"] = {c: int(clipped[c].sum()) for c in NUM_SKEWED}
R["clip_rows_any"] = int(clipped.any(axis=1).sum())
R["clip_rows_any_pct"] = r(100 * clipped.any(axis=1).mean(), 2)
R["bmi_imputed_train"] = int(X_train.bmi.isna().sum())
R["bmi_imputed_test"] = int(X_test.bmi.isna().sum())
R["impute_median"] = {c: r(v, 2) for c, v in zip(NUM_SKEWED, imp.statistics_)}
R["skew_after"] = {c: r(skew(Xt_train[:, names.index(c)]), 3) for c in NUM}
probe = X_test.iloc[:1].copy()
probe["gender"] = "Other"  # categoria removida na limpeza: inexistente no treino
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)  # o aviso é exatamente o comportamento testado
    pv = pre.transform(probe)[0]
R["unknown_category_probe"] = {n: r(pv[names.index(n)], 3) for n in names if n.startswith("gender")}
R["scaled"] = {"max_abs_mean": r(np.abs(Xt_train.mean(0)).max(), 6),
               "std_min_max": [r(Xt_train.std(0).min(), 4), r(Xt_train.std(0).max(), 4)],
               "col_max": {n: r(Xt_train[:, i].max(), 2) for i, n in enumerate(names)},
               "col_min": {n: r(Xt_train[:, i].min(), 2) for i, n in enumerate(names)},
               "heart_disease_values": sorted(float(v) for v in set(np.round(Xt_train[:, names.index("heart_disease")], 3)))}

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for k, c in enumerate(NUM_SKEWED):
    ax = axes[2 * k]
    ax.hist(X_train[c].dropna(), bins=40, color=NO, edgecolor="white", lw=0.5, label="bruto")
    ax.set(title=f"{c} bruto (assim. {desc[c]['skew']:.2f})", xlabel=UNITS[c], ylabel="Pacientes (treino)")
    ax.legend(loc="upper right")
    ax = axes[2 * k + 1]
    ax.hist(Xt_train[:, names.index(c)], bins=40, color=YES, edgecolor="white", lw=0.5,
            label="imputado + clip + Yeo-Johnson + escala")
    ax.set(title=f"{c} transformado (assim. {R['skew_after'][c]:.2f})", xlabel="valor padronizado (z)",
           ylabel="Pacientes (treino)")
    ax.legend(loc="upper right", fontsize=8)
fig.suptitle("Efeito do pré-processamento nas numéricas assimétricas (treino)", fontweight="bold")
fig.tight_layout()
save(fig, "fig09-transformacao.png")

# =============================================================================
# 4B · redução de dimensionalidade (features escaladas do treino)
# =============================================================================
yt = y_train.to_numpy()
pca = PCA(random_state=RANDOM_STATE).fit(Xt_train)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
R["pca_evr"] = [r(x, 4) for x in evr]
R["pca_cum"] = [r(x, 4) for x in cum]
R["pca_pc12"] = r(100 * cum[1], 2)
R["pca_n_for"] = {str(t): int(np.searchsorted(cum, t) + 1) for t in (0.8, 0.9, 0.95)}
load = pd.DataFrame(pca.components_[:2].T, index=names, columns=["PC1", "PC2"])
R["pca_loadings_top"] = {pc: load[pc].reindex(load[pc].abs().sort_values(ascending=False).index)[:6].round(3).to_dict()
                         for pc in ["PC1", "PC2"]}
Z = pca.transform(Xt_train)[:, :2]
R["pca_pc1_by_target"] = {int(k): r(Z[yt == k, 0].mean(), 3) for k in (0, 1)}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axes[0]
ax.bar(range(1, len(evr) + 1), 100 * evr, color=NO, label="variância de cada PC")
ax.plot(range(1, len(evr) + 1), 100 * cum, "-o", color=YES, ms=4, lw=2, label="variância acumulada")
for t in (80, 90):
    ax.axhline(t, color=MUTED, ls=":", lw=1)
    ax.text(len(evr) + 0.3, t, f"{t}%", va="center", fontsize=8, color=MUTED)
ax.set_xticks(range(1, len(evr) + 1, 2))
ax.set(title=f"PCA: PC1 + PC2 = {100 * cum[1]:.1f}% da variância", xlabel="componente principal",
       ylabel="% da variância explicada", ylim=(0, 105))
ax.legend(loc="center right")
ax = axes[1]
scatter_by_target(ax, Z, yt)
ax.set(title="PCA 2D colorido pelo alvo", xlabel=f"PC1 ({100 * evr[0]:.1f}%)", ylabel=f"PC2 ({100 * evr[1]:.1f}%)")
ax.legend(loc="upper right", markerscale=2)
fig.suptitle("PCA nas 20 features escaladas do treino", fontweight="bold")
fig.tight_layout()
save(fig, "fig10-pca.png")

enrich = {"pca": r(knn_enrichment(Z, yt), 4)}
PERPLEXITIES = [5, 30, 100]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for ax, perp in zip(axes, PERPLEXITIES):
    E = TSNE(n_components=2, perplexity=perp, init="pca", random_state=RANDOM_STATE).fit_transform(Xt_train)
    enrich[f"tsne_p{perp}"] = r(knn_enrichment(E, yt), 4)
    scatter_by_target(ax, E, yt, s=4)
    ax.set(title=f"t-SNE, perplexity = {perp}", xlabel="t-SNE 1", ylabel="t-SNE 2")
axes[0].legend(loc="upper right", markerscale=2)
fig.suptitle("t-SNE do treino com três perplexidades", fontweight="bold")
fig.tight_layout()
save(fig, "fig11-tsne.png")

NEIGHBORS = [5, 15, 100]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for ax, nn in zip(axes, NEIGHBORS):
    E = umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=RANDOM_STATE).fit_transform(Xt_train)
    enrich[f"umap_n{nn}"] = r(knn_enrichment(E, yt), 4)
    if nn == 15:  # o que define as "ilhas"? pureza de cada categórica dentro dos clusters
        lab = DBSCAN(eps=0.5, min_samples=10).fit_predict(E)
        ok = lab >= 0
        R["umap_islands"] = {"n_clusters": int(len(set(lab[ok]))), "noise": int((~ok).sum())}
        for c in cats + [TARGET]:
            v = tr[c].astype(str).to_numpy()
            hits = sum(pd.Series(v[lab == k]).value_counts().iloc[0] for k in set(lab[ok]))
            R["umap_islands"][f"purity_{c}"] = r(hits / ok.sum(), 3)
    scatter_by_target(ax, E, yt, s=4)
    ax.set(title=f"UMAP, n_neighbors = {nn}", xlabel="UMAP 1", ylabel="UMAP 2")
axes[0].legend(loc="upper right", markerscale=2)
fig.suptitle("UMAP do treino com três tamanhos de vizinhança", fontweight="bold")
fig.tight_layout()
save(fig, "fig12-umap.png")
enrich["full_20d"] = r(knn_enrichment(Xt_train, yt), 4)
# Variante para comparação: numéricas padronizadas, colunas 0/1 só centralizadas
Xalt = cols_ct.transform(X_train)
k = len(NUM)
Xalt[:, :k] = (Xalt[:, :k] - Xalt[:, :k].mean(0)) / Xalt[:, :k].std(0)
Xalt[:, k:] = Xalt[:, k:] - Xalt[:, k:].mean(0)
enrich["full_20d_binary_centered_only"] = r(knn_enrichment(Xalt, yt), 4)
R["knn10_enrichment"] = enrich

# =============================================================================
# 4C · pipeline
# =============================================================================
R["pipeline"] = {"train_shape": list(Xt_train.shape), "test_shape": list(Xt_test.shape),
                 "nan_train": int(np.isnan(Xt_train).sum()), "nan_test": int(np.isnan(Xt_test).sum()),
                 "feature_names": names}

OUT.write_text(json.dumps(R, indent=2, ensure_ascii=False))
print(f"ok: {len(list(FIG.glob('fig*.png')))} figuras, números em {OUT.name}")
