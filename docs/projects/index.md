# Projeto

!!! abstract "Enunciados"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

O projeto é **um só**, feito em equipe sobre **o mesmo dataset**, e entregue em três partes
ao longo do semestre, cada uma com data e peso próprios.

## Equipe

!!! danger "Preencha antes de qualquer entrega"

    Toda entrega do projeto é avaliada em equipe. Se os nomes não estiverem aqui, não há
    como atribuir a nota — e o mesmo vale para o `mkdocs.yml`, cujo `site_author` deve
    listar o grupo.

| Nome completo | E-mail | GitHub |
|---------------|--------|--------|
| João Gabriel Faraco | | [Joao-Gabriel05](https://github.com/Joao-Gabriel05) |
| Henrique Turco | | |
| Leonardo Freitas | | |

Times de 2 a 3 pessoas. Repita esses nomes no cabeçalho de cada entrega — quem corrige pode
abrir uma página sozinha, sem passar por aqui.

## As três entregas

| # | Entrega | Página |
|---|---------|--------|
| 1 | EDA | [EDA](eda/index.md) |
| 2 | Classificação | [Classificação](classification/index.md) |
| 3 | Generativo | [Generativo](generative/index.md) |

Datas e pesos são da sua edição — veja o
[overview](https://insper.github.io/ann-dl/){:target='_blank'}.

!!! danger "A nota do projeto costuma ser limitada por uma prova sobre o próprio projeto"

    Deliverables bem escritos não sustentam uma equipe que não consegue explicar o que
    entregou. Escreva os relatórios de modo que você consiga defendê-los meses depois, e
    confira no overview da sua edição como a prova entra na nota.

## Dataset

O mesmo dataset atravessa as três entregas — escolhê-lo bem no EDA é o que torna as outras
duas viáveis.

| | |
|---|---|
| **Nome** | Stroke Prediction Dataset |
| **Fonte (URL)** | <https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset> |
| **Licença / termos de uso** | "Confidential Source — use only for educational purposes" |
| **Amostras** | 5.110 pacientes (5.087 após remover 23 linhas de categorias raras) |
| **Features** | 10 (3 numéricas, 7 categóricas) + `id`, descartado |
| **Variável alvo** | `stroke` (0/1), 4,89% de positivos |
| **Tarefa escolhida** | Classificação |

Prever AVC a partir de dados demográficos e clínicos é um problema real de triagem, e o
dataset traz as dificuldades que tornam a classificação não trivial: desbalanceamento de
19,4 : 1 (o classificador trivial acerta 95,11% sem detectar nenhum caso), uma feature
dominante (idade, AUC 0,833 sozinha) que o modelo precisa superar, e ausência informativa
no IMC. Detalhes no [EDA](eda/index.md).

## Status

- [x] **1. EDA**
- [ ] **2. Classificação**
- [ ] **3. Generativo**

## Registro de decisões

Anote aqui as decisões que atravessam as entregas — troca de dataset, mudança de alvo,
recorte de features — com a data. É o que permite reconstruir o raciocínio na prova de
projeto.

| Data | Decisão | Motivo |
|------|---------|--------|
| 2026-10-08 | Dataset: Stroke Prediction (Kaggle); tarefa: classificação de `stroke` | tabular, misto numérico/categórico, desbalanceado; ver [EDA](eda/index.md) |
| 2026-10-08 | Split 80/20 estratificado, `random_state=42`, fixo para as três entregas | só 249 positivos; o teste (50 AVCs) é usado uma única vez |
| 2026-10-08 | Descartar `id`; manter `smoking_status = Unknown` como categoria | `id` sem informação (AUC 0,509); `Unknown` concentra crianças |
| 2026-10-08 | Manter o indicador de IMC ausente, com ablação na entrega 2 | ausência associada ao alvo (p = 1,9 × 10⁻²¹), possível vazamento |
| 2026-10-08 | Remover as 23 linhas com `gender = Other` ou `work_type = Never_worked`, antes do split | categorias raras demais para a rede (1 e 22 linhas), nenhuma com AVC |
| 2026-10-08 | Categóricas em one-hot completo; `StandardScaler` em todas as 20 features (numéricas e 0/1) | toda entrada da rede com média 0 e desvio 1; binárias raras ficam grandes (`heart_disease = 1` → +4,30), o que concentra os AVCs nas vizinhanças (4B); comparar com só centralizar na entrega 2 |
