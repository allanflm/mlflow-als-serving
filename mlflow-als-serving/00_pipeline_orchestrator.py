# Databricks notebook source
# MAGIC %md
# MAGIC # 🎬 mlflow-als-serving
# MAGIC ## Pipeline de Recomendação de Filmes com ALS + MLflow + Databricks
# MAGIC
# MAGIC > Pipeline end-to-end de Machine Learning para recomendação de filmes,
# MAGIC > usando Collaborative Filtering com ALS, rastreamento com MLflow
# MAGIC > e serving via API REST no Databricks Free Edition.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC | Componente | Tecnologia |
# MAGIC |---|---|
# MAGIC | 📦 Armazenamento | Delta Lake + Unity Catalog |
# MAGIC | ⚙️ Processamento | Apache Spark (PySpark) |
# MAGIC | 🤖 Algoritmo | ALS — Alternating Least Squares |
# MAGIC | 📊 Rastreamento | MLflow Experiments + Model Registry |
# MAGIC | 🚀 Serving | Databricks Model Serving (REST API) |
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🟤 Etapa 1 — Ingestão de Dados
# MAGIC ### Bronze Layer
# MAGIC
# MAGIC Baixando o dataset **MovieLens Small** (100k ratings, 9.7k filmes, 610 usuários)
# MAGIC diretamente da GroupLens e persistindo como tabelas **Delta Lake** na camada Bronze.
# MAGIC
# MAGIC > 💡 Bronze = dado bruto, sem transformação, preservado exatamente como veio da fonte.

# COMMAND ----------

# MAGIC %run ./01_ingestion

# COMMAND ----------

# DBTITLE 1,Display ingestion results
print(f"✅ Ingestão concluída")
print(f"   bronze_ratings : {spark.table('bronze_ratings').count():,} linhas")
print(f"   bronze_movies  : {spark.table('bronze_movies').count():,} linhas")
print(f"   bronze_tags    : {spark.table('bronze_tags').count():,} linhas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🥈 Etapa 2 — Feature Engineering
# MAGIC ### Silver Layer
# MAGIC
# MAGIC Aplicando filtros de **cold-start** para remover usuários e filmes com poucas avaliações
# MAGIC — um dos principais problemas em sistemas de recomendação.
# MAGIC
# MAGIC | Filtro | Critério |
# MAGIC |---|---|
# MAGIC | 👤 Usuários | mínimo 5 avaliações |
# MAGIC | 🎬 Filmes | mínimo 10 avaliações |
# MAGIC
# MAGIC > 💡 Silver = dado limpo, confiável, pronto para análise e modelagem.

# COMMAND ----------

# MAGIC %run ./02_feature_engineering

# COMMAND ----------

start = time.time()

elapsed = round(time.time() - start, 1)

silver = spark.table("silver_ratings")
bronze = spark.table("bronze_ratings")

removidos  = bronze.count() - silver.count()
pct_remove = round(removidos / bronze.count() * 100, 1)

print(f"✅ Feature engineering concluído em {elapsed}s")
print(f"   Ratings Bronze  : {bronze.count():,}")
print(f"   Ratings Silver  : {silver.count():,}")
print(f"   Removidos       : {removidos:,} ({pct_remove}% cold-start)")
print(f"   Usuários ativos : {silver.select('userId').distinct().count():,}")
print(f"   Filmes ativos   : {silver.select('movieId').distinct().count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🤖 Etapa 3 — Treinamento ALS + MLflow
# MAGIC ### Collaborative Filtering
# MAGIC
# MAGIC O algoritmo **ALS (Alternating Least Squares)** aprende padrões de preferência
# MAGIC decompondo a matriz usuário×filme em vetores latentes.

# COMMAND ----------

# MAGIC %md
# MAGIC | Hiperparâmetro | Valores testados |
# MAGIC |---|---|
# MAGIC | rank | 10, 30 |
# MAGIC | regParam | 0.01, 0.1 |
# MAGIC | maxIter | 10 |
# MAGIC
# MAGIC > 💡 Todos os experimentos são rastreados automaticamente no **MLflow**.

# COMMAND ----------

# MAGIC %run ./03_train_als

# COMMAND ----------

start = time.time()

elapsed = round(time.time() - start, 1)

print(f"✅ Treinamento concluído em {elapsed}s")
print(f"   Melhor RMSE  : {best_rmse:.4f}")
print(f"   Rank         : {best_params['rank']}")
print(f"   RegParam     : {best_params['regParam']}")
print(f"   Experimento  : /mlflow-als-serving no MLflow UI")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 Etapa 4 — Model Registry
# MAGIC ### Registrar e Promover para Production
# MAGIC
# MAGIC O modelo treinado é registrado no **Unity Catalog Model Registry** com:
# MAGIC
# MAGIC - ✅ Signature (input/output schema)
# MAGIC - ✅ Alias `Production`
# MAGIC - ✅ Descrição e tags de rastreabilidade
# MAGIC
# MAGIC > 💡 O Model Registry garante governança, versionamento e auditoria do modelo.

# COMMAND ----------

# MAGIC %run ./04_model_registry

# COMMAND ----------

start = time.time()
elapsed = round(time.time() - start, 1)

print(f"✅ Model Registry concluído em {elapsed}s")
print(f"   Modelo   : workspace.default.movielens-als")
print(f"   Alias    : Production → v{latest_version}")
print(f"   Registry : Unity Catalog > workspace > default > movielens-als")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 Etapa 5 — Model Serving
# MAGIC ### API REST em produção
# MAGIC
# MAGIC O modelo é servido via **Databricks Model Serving** como endpoint REST,
# MAGIC permitindo inferência em tempo real para qualquer aplicação.

# COMMAND ----------

# MAGIC %md
# MAGIC > 💡 Com `scale_to_zero_enabled: true` o endpoint dorme quando não usado
# MAGIC > — essencial para não consumir cota no Free Edition.

# COMMAND ----------

# MAGIC %run ./05_model_serving

# COMMAND ----------

start = time.time()

elapsed = round(time.time() - start, 1)

print(f"✅ Model Serving concluído em {elapsed}s")
print(f"   Endpoint : movielens-als-endpoint")
print(f"   Estado   : READY")
print(f"   URL      : {DATABRICKS_HOST}/serving-endpoints/movielens-als-endpoint")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 📊 Visão Geral do Pipeline
# MAGIC ### Métricas, distribuições e recomendações geradas
# MAGIC
# MAGIC ---

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

title_color  = "#e6edf3"
accent_color = "#58a6ff"
bar_color    = "#1f6feb"
bg_color     = "#161b22"
grid_color   = "#30363d"

# ── Gráfico 1: Distribuição de ratings ──────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(bg_color)

dist = (
    spark.table("silver_ratings")
    .groupBy("rating").count()
    .orderBy("rating")
    .toPandas()
)

ax1.bar(dist["rating"].astype(str), dist["count"], color=bar_color, width=0.6)
ax1.set_title("⭐ Distribuição de Ratings", color=title_color, fontsize=13, pad=12)
ax1.set_xlabel("Rating", color=title_color, fontsize=10)
ax1.set_ylabel("Quantidade", color=title_color, fontsize=10)
ax1.tick_params(colors=title_color)
for spine in ax1.spines.values():
    spine.set_edgecolor(grid_color)
ax1.yaxis.grid(True, color=grid_color, linestyle="--", alpha=0.5)

# ── Gráfico 2: RMSE por combinação de hiperparâmetros ───────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(bg_color)

import mlflow
runs = mlflow.search_runs(
    experiment_names=["/mlflow-als-serving"],
    filter_string="tags.mlflow.runName != 'als_best_model'"
)
runs = runs.dropna(subset=["metrics.rmse", "params.rank", "params.regParam"])
runs["label"] = "rank=" + runs["params.rank"] + "\nreg=" + runs["params.regParam"]

colors = [accent_color if r == runs["metrics.rmse"].min() else bar_color
          for r in runs["metrics.rmse"]]

bars = ax2.bar(runs["label"], runs["metrics.rmse"], color=colors, width=0.5)
ax2.set_title("📉 RMSE por Hiperparâmetros", color=title_color, fontsize=13, pad=12)
ax2.set_ylabel("RMSE", color=title_color, fontsize=10)
ax2.tick_params(colors=title_color, labelsize=8)
for spine in ax2.spines.values():
    spine.set_edgecolor(grid_color)
ax2.yaxis.grid(True, color=grid_color, linestyle="--", alpha=0.5)

# Destacar melhor
for bar, rmse in zip(bars, runs["metrics.rmse"]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{rmse:.3f}", ha="center", color=title_color, fontsize=9)

# ── Gráfico 3: Top 10 usuários mais ativos ──────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(bg_color)

top_users = (
    spark.table("silver_ratings")
    .groupBy("userId").count()
    .orderBy("count", ascending=False)
    .limit(10)
    .toPandas()
)

ax3.barh(
    top_users["userId"].astype(str),
    top_users["count"],
    color=bar_color
)
ax3.invert_yaxis()
ax3.set_title("👤 Top 10 Usuários Mais Ativos", color=title_color, fontsize=13, pad=12)
ax3.set_xlabel("Nº de Ratings", color=title_color, fontsize=10)
ax3.set_ylabel("userId", color=title_color, fontsize=10)
ax3.tick_params(colors=title_color)
for spine in ax3.spines.values():
    spine.set_edgecolor(grid_color)
ax3.xaxis.grid(True, color=grid_color, linestyle="--", alpha=0.5)

# ── Gráfico 4: Top 10 recomendações usuário 1 ───────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(bg_color)

ax4.barh(
    results_df["title"].str[:30],
    results_df["prediction"],
    color=accent_color
)
ax4.invert_yaxis()
ax4.set_title("🎬 Top 10 Recomendações — Usuário 1", color=title_color, fontsize=13, pad=12)
ax4.set_xlabel("Score Previsto", color=title_color, fontsize=10)
ax4.tick_params(colors=title_color, labelsize=8)
for spine in ax4.spines.values():
    spine.set_edgecolor(grid_color)
ax4.xaxis.grid(True, color=grid_color, linestyle="--", alpha=0.5)

plt.suptitle(
    "mlflow-als-serving — Pipeline Dashboard",
    color=accent_color, fontsize=16, fontweight="bold", y=1.01
)

plt.savefig(
    "/Volumes/workspace/default/mlflow-als-serving/pipeline_dashboard.png",
    bbox_inches="tight", facecolor=fig.get_facecolor()
)
plt.show()
print("✅ Dashboard salvo no Volume!")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Pipeline Concluído com Sucesso!
# MAGIC
# MAGIC | Etapa | Descrição | Status |
# MAGIC |---|---|---|
# MAGIC | 🟤 Bronze | 100.836 ratings ingeridos | ✅ |
# MAGIC | 🥈 Silver | Filtros cold-start aplicados | ✅ |
# MAGIC | 🤖 Treino | Melhor ALS registrado no MLflow | ✅ |
# MAGIC | 📋 Registry | Modelo em Production no Unity Catalog | ✅ |
# MAGIC | 🚀 Serving | API REST respondendo recomendações | ✅ |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🔗 Links úteis
# MAGIC - **MLflow Experiments** → barra lateral > Experiments > `/mlflow-als-serving`
# MAGIC - **Model Registry** → Catalog > workspace > default > movielens-als
# MAGIC - **Serving Endpoint** → Serving > movielens-als-endpoint
# MAGIC
# MAGIC ---
# MAGIC > 🛠️ Projeto: `mlflow-als-serving` | Dataset: MovieLens Small | Plataforma: Databricks Free Edition