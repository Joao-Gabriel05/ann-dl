"""
Exercicio 4: uma segunda camada oculta ajuda?

Mesmos dados, mesma divisao, mesmo eta (0.1) e mesmo numero de epocas (2000) do
Exercicio 3. A unica mudanca e a profundidade:

  rasa     [4, 16, 3]       - a rede do Exercicio 3
  profunda [4, 16, 16, 3]   - a mesma, com uma segunda oculta de 16

As duas sao treinadas com 5 sementes de inicializacao (42 a 46); a semente so
afeta os pesos iniciais, os dados sao sempre os mesmos.

Controle extra: a profunda tem ~3x mais parametros que a rasa, entao "mais
profunda" e "maior" estao misturados. A rede [4, 48, 3] - rasa, mas com quase o
mesmo numero de parametros da profunda - separa os dois efeitos.

Produz a Figura 6: perda de treino das duas redes, no mesmo eixo.
"""

import numpy as np
import matplotlib.pyplot as plt

from _saida import figura, estilo, CORES, SURFACE, TINTA, TINTA_2
from ex3_multiclasse import gerar_dados, dividir, treinar_rede, ETA, EPOCAS

SEMENTES = [42, 43, 44, 45, 46]
REDES = {
    "rasa (Ex. 3)": [4, 16, 3],
    "profunda (Ex. 4)": [4, 16, 16, 3],
    "rasa e larga (controle)": [4, 48, 3],
}


def rodar(X_tr, y_tr, X_te, y_te):
    """Treina cada rede com cada semente; guarda perda, acuracias e historico."""
    res = {}
    for nome, camadas in REDES.items():
        res[nome] = []
        for s in SEMENTES:
            rede = treinar_rede(X_tr, y_tr, np.random.default_rng(s), camadas)
            res[nome].append({
                "semente": s,
                "rede": rede,
                "perda": rede.perda_final_,
                "acc_treino": rede.acuracia(X_tr, y_tr),
                "acc_teste": rede.acuracia(X_te, y_te),
            })
    return res


def figura_6(res, caminho=None):
    caminho = caminho or figura("fig06_perda_rasa_vs_profunda.png")
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    estilo(ax)
    ep = np.arange(1, EPOCAS + 1)
    for cor, nome in zip(CORES, ("rasa (Ex. 3)", "profunda (Ex. 4)")):
        for r in res[nome][1:]:                      # demais sementes, finas
            ax.plot(ep, r["rede"].historico_, color=cor, linewidth=0.8, alpha=0.35, zorder=2)
        r = res[nome][0]                              # semente 42, em destaque
        ax.plot(ep, r["rede"].historico_, color=cor, linewidth=2.2, zorder=3,
                label=f"{nome} {REDES[nome]} — perda final {r['perda']:.3f}")
    ax.plot([], [], color=TINTA_2, linewidth=0.8, alpha=0.5, label="sementes 43–46 (linhas finas)")
    ax.set_title("Figura 6 — perda de treino: 1 camada oculta × 2 camadas ocultas\n"
                 f"mesmos dados, η = {ETA}, {EPOCAS} épocas · linha grossa = semente 42",
                 fontsize=12, color=TINTA, loc="left", pad=12)
    ax.set_xlabel("época", color=TINTA_2)
    ax.set_ylabel("perda de treino (CCE)", color=TINTA_2)
    ax.set_ylim(0.2, 1.4)
    ax.legend(frameon=False, fontsize=9.5)
    # detalhe do trecho final, onde as curvas se separam
    ins = ax.inset_axes([0.47, 0.33, 0.5, 0.33])
    estilo(ins)
    for cor, nome in zip(CORES, ("rasa (Ex. 3)", "profunda (Ex. 4)")):
        for r in res[nome]:
            ins.plot(ep[999:], r["rede"].historico_[999:], color=cor,
                     linewidth=2 if r["semente"] == 42 else 0.8,
                     alpha=1 if r["semente"] == 42 else 0.4)
    ins.set_title("épocas 1000–2000", fontsize=8.5, color=TINTA_2)
    ins.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SURFACE)
    print(f"Figura salva em: {caminho}")


if __name__ == "__main__":
    X, y = gerar_dados()
    X_tr, X_te, y_tr, y_te = dividir(X, y)
    res = rodar(X_tr, y_tr, X_te, y_te)

    print(f"eta = {ETA}, epocas = {EPOCAS}, sementes = {SEMENTES}\n")
    for nome, rs in res.items():
        print(f"=== {nome} {REDES[nome]} - {rs[0]['rede'].n_parametros()} parametros ===")
        for r in rs:
            print(f"  semente {r['semente']}: perda {r['perda']:.4f} | "
                  f"treino {r['acc_treino']:.4f} | teste {r['acc_teste']:.4f}")
        for k in ("perda", "acc_treino", "acc_teste"):
            v = np.array([r[k] for r in rs])
            print(f"  {k:<10} media {v.mean():.4f} | desvio {v.std(ddof=1):.4f}")
        print()

    rasa = np.array([r["acc_teste"] for r in res["rasa (Ex. 3)"]])
    prof = np.array([r["acc_teste"] for r in res["profunda (Ex. 4)"]])
    larga = np.array([r["acc_teste"] for r in res["rasa e larga (controle)"]])
    print("diferenca de acuracia de teste, profunda - rasa, semente a semente:")
    print(f"  {np.array2string(prof - rasa, precision=4)}  -> media {np.mean(prof - rasa):+.4f}")
    print(f"diferenca larga - rasa: media {np.mean(larga - rasa):+.4f}")
    print(f"diferenca profunda - larga: media {np.mean(prof - larga):+.4f}")
    p = rasa.mean()
    print(f"\nreferencia: 1 ponto de teste = {1 / len(y_te):.4f}; "
          f"erro-padrao amostral de uma acuracia de {p:.2f} em {len(y_te)} pontos = "
          f"{np.sqrt(p * (1 - p) / len(y_te)):.4f}")

    figura_6(res)
