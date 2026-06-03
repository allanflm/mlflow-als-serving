# 🎬 mlflow-als-serving

> Pipeline end-to-end de recomendação de filmes com **ALS + MLflow + Databricks Free Edition**

[![Databricks](https://img.shields.io/badge/Databricks-Free%20Edition-FF3621?style=flat&logo=databricks&logoColor=white)](https://www.databricks.com/try-databricks)
[![MLflow](https://img.shields.io/badge/MLflow-2.13+-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org)
[![PySpark](https://img.shields.io/badge/PySpark-3.5+-E25A1C?style=flat&logo=apachespark&logoColor=white)](https://spark.apache.org)
[![Delta Lake](https://img.shields.io/badge/Delta%20Lake-3.0+-00ADD8?style=flat)](https://delta.io)

---

## 📌 Sobre o projeto

Sistema de recomendação de filmes construído com **Collaborative Filtering** usando o algoritmo **ALS (Alternating Least Squares)** distribuído via Apache Spark. O pipeline cobre todas as etapas de um projeto de ML em produção:

- **Ingestão** de dados brutos com Delta Lake
- **Feature engineering** com filtros de cold-start
- **Treinamento distribuído** com grid search e rastreamento via MLflow
- **Versionamento** de modelos no Unity Catalog Model Registry
- **Serving** via API REST com Databricks Model Serving

> 💡 Projeto desenvolvido inteiramente no **Databricks Free Edition** (serverless compute + Unity Catalog).

---

## 🏗️ Arquitetura

```
MovieLens CSV
     │
     ▼
┌─────────────────────────────────────┐
│         Unity Catalog               │
│  🟤 Bronze  ──►  🥈 Silver          │
│  (dado bruto)    (cold-start filter) │
└─────────────────────────────────────┘
          │
          ▼ fit
     🤖 ALS (Spark MLlib)
       grid search
          │
          ▼
   📊 MLflow Experiment
   + Model Registry
   (workspace.default.movielens-als)
          │
          ▼ @Production
   🚀 Model Serving
   REST API endpoint
          │
          ▼
   🖥️ App / Cliente
   POST /invocations
   → top-N recomendações
```

---

## 📂 Estrutura do repositório

```
mlflow-als-serving/
├── notebooks/
│   ├── 00_pipeline_orchestrator.py   ← roda o pipeline completo
│   ├── 01_ingestion.py               ← Bronze Layer
│   ├── 02_feature_engineering.py     ← Silver Layer
│   ├── 03_train_als.py               ← ALS + MLflow tracking
│   ├── 04_model_registry.py          ← Unity Catalog Registry
│   └── 05_model_serving.py           ← REST API endpoint
├── images/
│   └── architecture.png              ← diagrama da arquitetura
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Stack

| Camada | Tecnologia |
|---|---|
| Plataforma | Databricks Free Edition |
| Compute | Serverless (sem cluster manual) |
| Armazenamento | Delta Lake + Unity Catalog Volumes |
| Processamento | Apache Spark 3.5 (PySpark) |
| Algoritmo | ALS — Alternating Least Squares (Spark MLlib) |
| Rastreamento | MLflow Experiments |
| Governança | MLflow Model Registry (Unity Catalog) |
| Serving | Databricks Model Serving (scale-to-zero) |
| Dataset | [MovieLens Small](https://grouplens.org/datasets/movielens/) — 100k ratings |

---

## 🚀 Como reproduzir

### Pré-requisitos

- Conta no [Databricks Free Edition](https://www.databricks.com/try-databricks)
- Volume criado: `Catalog > workspace > default > Create Volume > mlflow-als-serving`

### Passo a passo

**1. Clone o repositório**
```bash
git clone https://github.com/seu-usuario/mlflow-als-serving.git
```

**2. Importe os notebooks no Databricks**

No Databricks: `Workspace > sua pasta > Import > From URL`

Ou conecte via Git:
`Workspace > Repos > Add Repo > cole a URL do repositório`

**3. Execute o pipeline**

Abra o notebook `00_pipeline_orchestrator` e clique em **Run All**.

O orquestrador vai rodar todos os notebooks em sequência e gerar um dashboard de métricas ao final.

> ⏱️ Tempo estimado: ~20 minutos no serverless do Free Edition.

---

## 📊 Resultados

| Métrica | Valor |
|---|---|
| Dataset | MovieLens Small (100k ratings) |
| Ratings após filtro cold-start | ~81k |
| Usuários ativos | 610 |
| Filmes ativos | 2.269 |
| Melhor RMSE | ~0.87 |
| Melhores hiperparâmetros | rank=30, regParam=0.01 |

### Exemplo de recomendações para o usuário 1

| # | Filme | Score |
|---|---|---|
| 1 | Low Down Dirty Shame (1994) | 5.36 |
| 2 | Before Sunrise (1995) | 5.11 |
| 3 | RoboCop 3 (1993) | 5.09 |
| 4 | Total Eclipse (1995) | 5.08 |
| 5 | Screamers (1995) | 5.06 |

> 💡 Scores acima de 5.0 são normais no ALS com `nonnegative=True` — o modelo extrapola a escala original. Na prática usa-se o ranking relativo, não o valor absoluto.

---

## ⚠️ Limitações do Free Edition

Durante o desenvolvimento, foram encontradas algumas restrições do Databricks Free Edition e suas respectivas soluções:

| Limitação | Solução adotada |
|---|---|
| DBFS desabilitado | Unity Catalog Volumes (`/Volumes/...`) |
| `CrossValidator` falha no serverless | Loop manual de grid search |
| `MLFLOW_DFS_TMP` obrigatório | Variável de ambiente apontando para Volume |
| Signature obrigatória no UC | `infer_signature` antes do `log_model` |
| Higher-order functions bloqueadas | Conversão para Pandas + processamento local |
| Kafka externo bloqueado | Não abordado neste projeto |

---

## 📚 Referências

- [Databricks Free Edition — Limitações](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Spark MLlib — ALS](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/latest/)
- [Unity Catalog Volumes](https://docs.databricks.com/aws/en/connect/unity-catalog/volumes.html)

---

## 👤 Autor

Feito com ☕ e muito `%run` no Databricks Free Edition.
