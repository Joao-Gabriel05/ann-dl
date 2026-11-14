"""Entrega 2 (classificação) -- passos 0 a 6. Todo número do relatório sai daqui.

Reusa o pipeline do EDA sem reescrevê-lo: `preprocessing.py` é importado de
`docs/projects/eda/code/`, e o split externo (80/20 estratificado, seed 42) é o
mesmo das três entregas. A validação é recortada **do treino**, e o
pré-processador é reajustado só nesse treino interno -- assim a validação fica
tão intocada quanto o teste.

    python run.py          # ~3 min em CPU; escreve results.json e figures/
"""

import os
import sys
import json
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "eda" / "code"))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score, roc_curve,
                             precision_recall_curve, confusion_matrix,
                             precision_recall_fscore_support, f1_score,
                             accuracy_score, recall_score, precision_score)

from preprocessing import load_raw, clean, split, build_preprocessor, FEATURES, TARGET
from network import build_mlp, fit, logits, n_params

FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)
SEED = 42
SEEDS = [42, 7, 2024]
res = {}
t0 = time.time()

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 140,
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "600",
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": False,
})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#86857f"]

# ======================================== Passo 0: ponto de partida, sem vazamento
raw = clean(load_raw())
X_tr_full, X_te, y_tr_full, y_te = split(raw)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tr_full, y_tr_full, test_size=0.20, stratify=y_tr_full, random_state=SEED)

pre = build_preprocessor().fit(X_tr)          # estatísticas: treino interno só
names = [str(n) for n in pre.get_feature_names_out()]
Xtr, Xva, Xte = pre.transform(X_tr), pre.transform(X_va), pre.transform(X_te)
ytr, yva, yte = y_tr.to_numpy(), y_va.to_numpy(), y_te.to_numpy()
POS_W = float((ytr == 0).sum() / (ytr == 1).sum())

res["setup"] = {
    "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
    "pos_train": int(ytr.sum()), "pos_val": int(yva.sum()), "pos_test": int(yte.sum()),
    "pos_rate_train": float(ytr.mean()), "pos_rate_val": float(yva.mean()),
    "pos_rate_test": float(yte.mean()),
    "n_features": len(names), "features": names, "pos_weight": POS_W,
    "nan_train": int(np.isnan(Xtr).sum()), "nan_test": int(np.isnan(Xte).sum()),
}
res["step0"] = {
    "main_metric": "PR-AUC (average precision) no teste",
    "why": "19,4 : 1 de desbalanceamento: a acurácia do trivial já é 95,11% sem "
           "detectar um único AVC, e a ROC-AUC infla com os 4.838 negativos. A "
           "PR-AUC só sobe se os positivos subirem no ranking.",
    "secondary": ["recall da classe 1", "macro F1", "ROC-AUC"],
    "criterion": "PR-AUC de teste pelo menos 15% acima da melhor baseline, "
                 "média de 3 seeds",
    "margin_required_rel": 0.15,
}
print("[0] train %d (%d pos) | val %d (%d pos) | test %d (%d pos) | %d features | pos_weight %.2f"
      % (len(ytr), ytr.sum(), len(yva), yva.sum(), len(yte), yte.sum(), len(names), POS_W))


def ev(y, s, th=0.5):
    p = (s >= th).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average=None, zero_division=0)
    return {
        "acc": float(accuracy_score(y, p)),
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "precision": [float(v) for v in pr], "recall": [float(v) for v in rc],
        "f1": [float(v) for v in f1], "cm": confusion_matrix(y, p).tolist(),
        "threshold": float(th),
    }


# ============================================================ Passo 1: baselines
bl = {}
dc = DummyClassifier(strategy="most_frequent", random_state=SEED).fit(Xtr, ytr)
bl["dummy"] = ev(yva, dc.predict_proba(Xva)[:, 1])
lg = LogisticRegression(max_iter=5000, class_weight="balanced",
                        random_state=SEED).fit(Xtr, ytr)
bl["logreg"] = ev(yva, lg.predict_proba(Xva)[:, 1])
bl["logreg"]["n_params"] = int(Xtr.shape[1] + 1)
rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                            class_weight="balanced_subsample",
                            random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
bl["rf"] = ev(yva, rf.predict_proba(Xva)[:, 1])
res["baselines_val"] = bl
print("[1] val PR-AUC  dummy %.4f | logreg %.4f | rf %.4f"
      % (bl["dummy"]["pr_auc"], bl["logreg"]["pr_auc"], bl["rf"]["pr_auc"]))

# ================================================ Passo 2 + 3: a rede, treinada
net = build_mlp(Xtr.shape[1], widths=(32, 16), activation="relu", lr=1e-3, seed=SEED)
res["network"] = {
    "widths": [32, 16], "activation": "relu", "output": "1 neurônio linear (logit)",
    "loss": "BinaryCrossentropy(from_logits=True)", "init": "He uniform",
    "n_params": n_params(net), "params_per_pos": float(n_params(net) / ytr.sum()),
    "params_per_row": float(n_params(net) / len(ytr)),
}
h = fit(net, Xtr, ytr, Xva, yva, batch_size=64, max_epochs=200, patience=20,
        pos_weight=POS_W)
res["training"] = {
    "optimizer": "Adam", "lr": 1e-3, "batch_size": 64, "max_epochs": 200,
    "patience": 20, "monitor": "val_pr_auc (max)",
    "steps_per_epoch": int(np.ceil(len(ytr) / 64)),
    **{k: v for k, v in h.items() if k != "history"}, "history": h["history"],
}
sv = 1 / (1 + np.exp(-logits(net, Xva)))
res["network_val"] = ev(yva, sv)
print("[2-3] rede: %d params | val PR-AUC %.4f | melhor época %d/%d"
      % (res["network"]["n_params"], res["network_val"]["pr_auc"],
         h["best_epoch"], h["epochs_run"]))

# ------------------------------------------------------------------- Figura 1
hh = h["history"]
ep = np.arange(1, len(hh["loss"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
ax[0].plot(ep, hh["loss"], color=C[0], label="treino")
ax[0].plot(ep, hh["val_loss"], color=C[1], label="validação")
ax[0].axvline(h["best_epoch"], color=C[4], ls="--", lw=1)
ax[0].set_title("Perda ponderada (BCE com logits)")
ax[0].set_xlabel("época"); ax[0].set_ylabel("perda"); ax[0].legend()
ax[1].plot(ep, hh["pr_auc"], color=C[0], label="treino")
ax[1].plot(ep, hh["val_pr_auc"], color=C[1], label="validação")
ax[1].axvline(h["best_epoch"], color=C[4], ls="--", lw=1,
              label="época restaurada (%d)" % h["best_epoch"])
ax[1].set_title("PR-AUC")
ax[1].set_xlabel("época"); ax[1].set_ylabel("PR-AUC"); ax[1].legend()
fig.suptitle("Figura 1 — Curvas de treino e validação por época (MLP 32-16, seed 42)",
             x=0.02, ha="left", fontsize=12, weight="600")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(FIG / "fig01-curvas.png"); plt.close(fig)

# ======================================= Passo 4a: regularização, uma a uma
regs = [
    ("nenhuma", {}),
    ("dropout 0,2", {"dropout": 0.2}),
    ("weight decay 1e-3", {"weight_decay": 1e-3}),
    ("dropout + decay", {"dropout": 0.2, "weight_decay": 1e-3}),
    ("batch norm", {"batchnorm": True}),
]
reg_rows = []
for name, kw in regs:
    m = build_mlp(Xtr.shape[1], widths=(32, 16), activation="relu", lr=1e-3,
                  seed=SEED, **kw)
    hi = fit(m, Xtr, ytr, Xva, yva, batch_size=64, max_epochs=200, patience=20,
             pos_weight=POS_W)
    a_tr = float(average_precision_score(ytr, logits(m, Xtr)))
    a_va = float(average_precision_score(yva, logits(m, Xva)))
    reg_rows.append({"name": name, "train_pr_auc": a_tr, "val_pr_auc": a_va,
                     "gap": a_tr - a_va, "best_epoch": hi["best_epoch"],
                     "epochs_run": hi["epochs_run"]})
    print("[4a] %-20s treino %.4f val %.4f gap %+.4f" % (name, a_tr, a_va, a_tr - a_va))
res["regularization"] = reg_rows

fig, ax = plt.subplots(figsize=(9, 4.3))
pos = np.arange(len(reg_rows)); w = 0.38
ax.bar(pos - w / 2, [r["train_pr_auc"] for r in reg_rows], w, color=C[0], label="treino")
ax.bar(pos + w / 2, [r["val_pr_auc"] for r in reg_rows], w, color=C[1], label="validação")
for i, r in enumerate(reg_rows):
    ax.text(i, max(r["train_pr_auc"], r["val_pr_auc"]) + 0.012,
            "gap %+.3f" % r["gap"], ha="center", fontsize=7.5, color="#52514e")
ax.set_xticks(pos); ax.set_xticklabels([r["name"] for r in reg_rows], fontsize=8)
ax.set_ylabel("PR-AUC"); ax.legend(loc="upper right")
ax.set_title("Figura 2 — Efeito de cada regularizador na distância treino–validação\n"
             "mesma arquitetura 32-16, mesma seed, mesmo early stopping", loc="left")
fig.tight_layout(); fig.savefig(FIG / "fig02-regularizacao.png"); plt.close(fig)

# ================================== Passo 4b: busca de hiperparâmetros (validação)
grid = [
    dict(widths=(16,),      act="relu", do=0.0, wd=0.0,  lr=1e-3, bs=64),
    dict(widths=(32, 16),   act="relu", do=0.0, wd=0.0,  lr=1e-3, bs=64),
    dict(widths=(32, 16),   act="relu", do=0.2, wd=1e-3, lr=1e-3, bs=64),
    dict(widths=(32, 16),   act="tanh", do=0.2, wd=1e-3, lr=1e-3, bs=64),
    dict(widths=(64, 32),   act="relu", do=0.3, wd=1e-3, lr=1e-3, bs=64),
    dict(widths=(64, 32),   act="relu", do=0.3, wd=1e-2, lr=5e-4, bs=64),
    dict(widths=(64, 32, 16), act="relu", do=0.3, wd=1e-3, lr=1e-3, bs=128),
    dict(widths=(128, 64),  act="relu", do=0.4, wd=1e-2, lr=5e-4, bs=128),
    dict(widths=(32, 16),   act="relu", do=0.1, wd=1e-4, lr=3e-3, bs=32),
    dict(widths=(16, 8),    act="relu", do=0.2, wd=1e-3, lr=1e-3, bs=32),
]
search = []
for i, g in enumerate(grid):
    m = build_mlp(Xtr.shape[1], widths=g["widths"], activation=g["act"],
                  dropout=g["do"], weight_decay=g["wd"], lr=g["lr"], seed=SEED)
    hi = fit(m, Xtr, ytr, Xva, yva, batch_size=g["bs"], max_epochs=200,
             patience=20, pos_weight=POS_W)
    s = logits(m, Xva)
    search.append({
        "id": i + 1, "widths": list(g["widths"]), "activation": g["act"],
        "dropout": g["do"], "weight_decay": g["wd"], "lr": g["lr"],
        "batch_size": g["bs"], "n_params": n_params(m),
        "best_epoch": hi["best_epoch"],
        "val_pr_auc": float(average_precision_score(yva, s)),
        "val_roc_auc": float(roc_auc_score(yva, s)),
    })
    print("[4b] cfg %2d %-12s %s do=%.1f wd=%g lr=%g bs=%3d -> val PR-AUC %.4f"
          % (i + 1, "x".join(map(str, g["widths"])), g["act"], g["do"], g["wd"],
             g["lr"], g["bs"], search[-1]["val_pr_auc"]))
search.sort(key=lambda r: -r["val_pr_auc"])
res["search"] = search
best = search[0]
res["best_config"] = best
print("[4b] vencedora: cfg %d, val PR-AUC %.4f" % (best["id"], best["val_pr_auc"]))


def build_best(seed):
    return build_mlp(Xtr.shape[1], widths=tuple(best["widths"]),
                     activation=best["activation"], dropout=best["dropout"],
                     weight_decay=best["weight_decay"], lr=best["lr"], seed=seed)


# ============================ limiar escolhido na validação (custo assimétrico)
mb = build_best(SEED)
fit(mb, Xtr, ytr, Xva, yva, batch_size=best["batch_size"], max_epochs=200,
    patience=20, pos_weight=POS_W)
pv = 1 / (1 + np.exp(-logits(mb, Xva)))
ths = np.linspace(0.02, 0.98, 193)
rows = [{"th": float(t),
         "recall": float(recall_score(yva, (pv >= t).astype(int), zero_division=0)),
         "precision": float(precision_score(yva, (pv >= t).astype(int), zero_division=0)),
         "f2": float((5 * precision_score(yva, (pv >= t).astype(int), zero_division=0)
                      * recall_score(yva, (pv >= t).astype(int), zero_division=0))
                     / (4 * precision_score(yva, (pv >= t).astype(int), zero_division=0)
                        + recall_score(yva, (pv >= t).astype(int), zero_division=0))
                     if precision_score(yva, (pv >= t).astype(int), zero_division=0)
                     + recall_score(yva, (pv >= t).astype(int), zero_division=0) > 0 else 0.0)}
        for t in ths]
th = max(rows, key=lambda r: r["f2"])
res["threshold"] = {
    "value": th["th"], "rule": "maximiza F2 na validação",
    "why": "F2 pesa recall 2x a precisão: numa triagem de AVC, um falso negativo "
           "é um paciente mandado para casa, e um falso positivo é um exame a mais.",
    "val_recall": th["recall"], "val_precision": th["precision"], "val_f2": th["f2"],
    "at_half": {"recall": float(recall_score(yva, (pv >= 0.5).astype(int), zero_division=0)),
                "precision": float(precision_score(yva, (pv >= 0.5).astype(int), zero_division=0))},
    "curve": rows[::6],
}
TH = th["th"]
print("[4c] limiar %.3f -> val recall %.3f precisão %.3f (F2 %.3f)"
      % (TH, th["recall"], th["precision"], th["f2"]))

json.dump(res, open(HERE / "results.json", "w"), indent=2, ensure_ascii=False)
print("[ok] passos 0-4 em %.1fs" % (time.time() - t0))

# ============================================== Passo 5: teste, uma vez, 3 seeds
runs = []
scores = []
for s in SEEDS:
    m = build_best(s)
    hi = fit(m, Xtr, ytr, Xva, yva, batch_size=best["batch_size"], max_epochs=200,
             patience=20, pos_weight=POS_W)
    p = 1 / (1 + np.exp(-logits(m, Xte)))
    scores.append(p)
    e = ev(yte, p, TH)
    e["seed"] = s
    e["best_epoch"] = hi["best_epoch"]
    runs.append(e)
    print("[5] seed %4d -> teste PR-AUC %.4f ROC-AUC %.4f recall %.3f precisão %.3f"
          % (s, e["pr_auc"], e["roc_auc"], e["recall"][1], e["precision"][1]))


def ms(key, idx=None):
    v = [r[key] if idx is None else r[key][idx] for r in runs]
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
            "values": [float(x) for x in v]}


res["test_net"] = {
    "seeds": SEEDS, "runs": runs, "threshold": TH,
    "pr_auc": ms("pr_auc"), "roc_auc": ms("roc_auc"), "acc": ms("acc"),
    "macro_f1": ms("macro_f1"), "recall_pos": ms("recall", 1),
    "precision_pos": ms("precision", 1), "f1_pos": ms("f1", 1),
}

bt = {}
dc_t = DummyClassifier(strategy="most_frequent", random_state=SEED).fit(Xtr, ytr)
bt["dummy"] = ev(yte, dc_t.predict_proba(Xte)[:, 1])
bt["logreg"] = ev(yte, lg.predict_proba(Xte)[:, 1], TH)
bt["rf"] = ev(yte, rf.predict_proba(Xte)[:, 1], TH)
res["test_baselines"] = bt
best_base_pr = max(bt["logreg"]["pr_auc"], bt["rf"]["pr_auc"])
res["verdict"] = {
    "net_pr_auc_mean": res["test_net"]["pr_auc"]["mean"],
    "best_baseline_pr_auc": float(best_base_pr),
    "relative_gain": float(res["test_net"]["pr_auc"]["mean"] / best_base_pr - 1),
    "criterion_met": bool(res["test_net"]["pr_auc"]["mean"] >= 1.15 * best_base_pr),
}
print("[5] rede %.4f +- %.4f vs melhor baseline %.4f -> ganho %+.1f%% | critério: %s"
      % (res["test_net"]["pr_auc"]["mean"], res["test_net"]["pr_auc"]["std"],
         best_base_pr, 100 * res["verdict"]["relative_gain"],
         "ATENDIDO" if res["verdict"]["criterion_met"] else "NÃO atendido"))

mean_score = np.mean(scores, axis=0)
res["test_net"]["ensemble_of_seeds"] = ev(yte, mean_score, TH)

# --------------------------------------------------- Figuras 3 e 4
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for nm, s, c in [("rede (média 3 seeds)", mean_score, C[0]),
                 ("regressão logística", lg.predict_proba(Xte)[:, 1], C[1]),
                 ("random forest", rf.predict_proba(Xte)[:, 1], C[2])]:
    fpr, tpr, _ = roc_curve(yte, s)
    ax[0].plot(fpr, tpr, color=c, label="%s (AUC %.3f)" % (nm, roc_auc_score(yte, s)))
    p_, r_, _ = precision_recall_curve(yte, s)
    ax[1].plot(r_, p_, color=c,
               label="%s (PR-AUC %.3f)" % (nm, average_precision_score(yte, s)))
ax[0].plot([0, 1], [0, 1], color=C[4], ls="--", lw=1, label="aleatório")
ax[0].set_title("Curva ROC"); ax[0].set_xlabel("falso positivo"); ax[0].set_ylabel("recall")
ax[0].legend(fontsize=7.5, loc="lower right")
ax[1].axhline(yte.mean(), color=C[4], ls="--", lw=1,
              label="prevalência (%.3f)" % yte.mean())
ax[1].set_title("Curva precisão–recall"); ax[1].set_xlabel("recall")
ax[1].set_ylabel("precisão"); ax[1].legend(fontsize=7.5, loc="upper right")
fig.suptitle("Figura 3 — Rede contra as baselines no conjunto de teste (1.018 pacientes, 50 AVCs)",
             x=0.02, ha="left", fontsize=12, weight="600")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(FIG / "fig03-roc-pr.png"); plt.close(fig)

cm = np.array(res["test_net"]["ensemble_of_seeds"]["cm"])
fig, ax = plt.subplots(figsize=(4.6, 4.1))
ax.imshow(cm, cmap="Blues"); ax.grid(False)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=15,
                color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
ax.set_xticks([0, 1]); ax.set_xticklabels(["prev. 0", "prev. 1"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["real 0", "real 1"])
ax.set_title("Figura 4 — Matriz de confusão no teste\nlimiar %.3f, escolhido na validação" % TH,
             loc="left")
fig.tight_layout(); fig.savefig(FIG / "fig04-confusao.png"); plt.close(fig)

# ======================================================= Passo 6: análise de erro
pred = (mean_score >= TH).astype(int)
te = X_te.reset_index(drop=True).copy()
te["y"] = yte
te["p"] = mean_score
te["pred"] = pred
te["erro"] = (te["pred"] != te["y"]).astype(int)

groups = []
bands = pd.cut(te["age"], [0, 30, 45, 60, 75, 100],
               labels=["0-30", "30-45", "45-60", "60-75", "75+"])
for lbl, g in te.groupby(bands, observed=True):
    groups.append({"dim": "faixa de idade", "level": str(lbl), "n": int(len(g)),
                   "pos": int(g["y"].sum()), "err_rate": float(g["erro"].mean()),
                   "recall": float(g.loc[g["y"] == 1, "pred"].mean()) if g["y"].sum() else None,
                   "fp_rate": float(g.loc[g["y"] == 0, "pred"].mean()) if (g["y"] == 0).any() else None})
for dim in ["gender", "hypertension", "heart_disease", "smoking_status", "work_type"]:
    for lvl, g in te.groupby(dim, observed=True):
        groups.append({"dim": dim, "level": str(lvl), "n": int(len(g)),
                       "pos": int(g["y"].sum()), "err_rate": float(g["erro"].mean()),
                       "recall": float(g.loc[g["y"] == 1, "pred"].mean()) if g["y"].sum() else None,
                       "fp_rate": float(g.loc[g["y"] == 0, "pred"].mean()) if (g["y"] == 0).any() else None})
g_bmi = te["bmi"].isna()
for lvl, g in te.groupby(g_bmi, observed=True):
    groups.append({"dim": "bmi ausente", "level": "sim" if lvl else "não",
                   "n": int(len(g)), "pos": int(g["y"].sum()),
                   "err_rate": float(g["erro"].mean()),
                   "recall": float(g.loc[g["y"] == 1, "pred"].mean()) if g["y"].sum() else None,
                   "fp_rate": float(g.loc[g["y"] == 0, "pred"].mean()) if (g["y"] == 0).any() else None})
res["subgroups"] = groups

fn = te[(te["y"] == 1)].nsmallest(8, "p")
fp = te[(te["y"] == 0)].nlargest(8, "p")
cols = FEATURES + ["y", "p"]
res["worst"] = {
    "false_negatives": fn[cols].round(3).to_dict("records"),
    "false_positives": fp[cols].round(3).to_dict("records"),
}
print("[6] %d subgrupos | %d FN e %d FP no limiar %.3f"
      % (len(groups), int(((te.y == 1) & (te.pred == 0)).sum()),
         int(((te.y == 0) & (te.pred == 1)).sum()), TH))

# permutação sobre as features BRUTAS, re-transformadas pelo pipeline ajustado:
# embaralhar uma coluna one-hot isolada criaria pacientes impossíveis
rng = np.random.default_rng(SEED)
base_pr = float(average_precision_score(yte, mean_score))
models = []
for s in SEEDS:
    m = build_best(s)
    fit(m, Xtr, ytr, Xva, yva, batch_size=best["batch_size"], max_epochs=200,
        patience=20, pos_weight=POS_W)
    models.append(m)


def score_raw(df):
    Z = pre.transform(df)
    return np.mean([1 / (1 + np.exp(-logits(m, Z))) for m in models], axis=0)


perm = []
for f in FEATURES:
    drops = []
    for _ in range(10):
        d = X_te.reset_index(drop=True).copy()
        d[f] = d[f].to_numpy()[rng.permutation(len(d))]
        drops.append(base_pr - float(average_precision_score(yte, score_raw(d))))
    perm.append({"feature": f, "drop_mean": float(np.mean(drops)),
                 "drop_std": float(np.std(drops, ddof=1))})
perm.sort(key=lambda r: -r["drop_mean"])
res["permutation"] = {"base_pr_auc": base_pr, "n_repeats": 10, "rows": perm}
print("[6] importância por permutação (top 3): "
      + ", ".join("%s %.4f" % (r["feature"], r["drop_mean"]) for r in perm[:3]))

fig, ax = plt.subplots(figsize=(8.2, 4.6))
o = perm[::-1]
y_ = np.arange(len(o))
ax.barh(y_, [r["drop_mean"] for r in o], xerr=[r["drop_std"] for r in o],
        color=C[0], error_kw={"ecolor": "#86857f", "lw": 1})
ax.axvline(0, color="#52514e", lw=1)
ax.set_yticks(y_); ax.set_yticklabels([r["feature"] for r in o], fontsize=8.5)
ax.set_xlabel("queda na PR-AUC quando a feature é embaralhada")
ax.set_title("Figura 5 — Importância por permutação no teste\n"
             "média ± desvio de 10 embaralhamentos; PR-AUC de base %.3f" % base_pr,
             loc="left")
fig.tight_layout(); fig.savefig(FIG / "fig05-permutacao.png"); plt.close(fig)

sg = [g for g in groups if g["dim"] == "faixa de idade"]
fig, ax = plt.subplots(figsize=(8.4, 4.3))
x_ = np.arange(len(sg)); w = 0.38
ax.bar(x_ - w / 2, [g["recall"] if g["recall"] is not None else 0 for g in sg], w,
       color=C[0], label="recall (dos AVCs do grupo)")
ax.bar(x_ + w / 2, [g["fp_rate"] if g["fp_rate"] is not None else 0 for g in sg], w,
       color=C[1], label="taxa de falso positivo")
for i, g in enumerate(sg):
    ax.text(i, 1.02, "n=%d\n%d AVC" % (g["n"], g["pos"]), ha="center", fontsize=7.5,
            color="#52514e")
ax.set_xticks(x_); ax.set_xticklabels([g["level"] for g in sg])
ax.set_ylim(0, 1.18); ax.set_xlabel("faixa de idade"); ax.set_ylabel("taxa")
ax.legend(loc="lower right", fontsize=8)
ax.set_title("Figura 6 — Onde o erro se concentra: recall e falso positivo por idade\n"
             "limiar %.3f no teste" % TH, loc="left")
fig.tight_layout(); fig.savefig(FIG / "fig06-subgrupos.png"); plt.close(fig)

json.dump(res, open(HERE / "results.json", "w"), indent=2, ensure_ascii=False)
print("[ok] tudo em %.1fs -> results.json + 6 figuras" % (time.time() - t0))
