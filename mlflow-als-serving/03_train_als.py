# Databricks notebook source
# =============================================================
# mlflow-als-serving | 03_train_als
# Objetivo: treinar ALS, rastrear com MLflow, registrar modelo
# =============================================================

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import mlflow
import mlflow.spark

ratings = spark.table("silver_ratings")
print(f"Registros para treino: {ratings.count():,}")

# COMMAND ----------

# -------------------------------------------------------------
# 2. Split treino / teste
# (sem cache — não suportado no serverless)
# -------------------------------------------------------------
train, test = ratings.randomSplit([0.8, 0.2], seed=42)

print(f"Treino : {train.count():,}")
print(f"Teste  : {test.count():,}")

# COMMAND ----------

# -------------------------------------------------------------
# 3. Configurar ALS e avaliador
# -------------------------------------------------------------
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop",  # evita NaN no RMSE
    nonnegative=True
)

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

# COMMAND ----------

# -------------------------------------------------------------
# 4. Grid de hiperparâmetros
# -------------------------------------------------------------
# rank     = número de fatores latentes (dimensões do embedding)
# maxIter  = iterações do algoritmo
# regParam = regularização (evita overfitting)

param_grid = (
    ParamGridBuilder()
    .addGrid(als.rank,     [10, 30])
    .addGrid(als.maxIter,  [10])
    .addGrid(als.regParam, [0.01, 0.1])
    .build()
)

print(f"Combinações a testar: {len(param_grid)}")

# COMMAND ----------

# Descubra o caminho correto rodando isso primeiro
import mlflow

# No Free Edition usa caminho simples sem /Users/email
mlflow.set_experiment("/mlflow-als-serving")

print("Experimento criado com sucesso!")

# COMMAND ----------

# -------------------------------------------------------------
# 5. Treinar com loop manual + MLflow tracking
# -------------------------------------------------------------
import itertools

mlflow.set_experiment("/mlflow-als-serving")

ranks      = [10, 30]
reg_params = [0.01, 0.1]
max_iter   = 10

best_rmse   = float("inf")
best_model  = None
best_params = {}

for rank, reg in itertools.product(ranks, reg_params):

    with mlflow.start_run(run_name=f"als_rank{rank}_reg{reg}"):

        als = ALS(
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True,
            rank=rank,
            maxIter=max_iter,
            regParam=reg
        )

        model       = als.fit(train)
        predictions = model.transform(test)
        rmse        = evaluator.evaluate(predictions)

        mlflow.log_metric("rmse",    rmse)
        mlflow.log_param("rank",     rank)
        mlflow.log_param("regParam", reg)
        mlflow.log_param("maxIter",  max_iter)

        print(f"rank={rank:2d} | reg={reg:.2f} | RMSE={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse   = rmse
            best_model  = model
            best_params = {"rank": rank, "regParam": reg}

print(f"\nMelhor → rank={best_params['rank']} | reg={best_params['regParam']} | RMSE={best_rmse:.4f}")

# COMMAND ----------

# -------------------------------------------------------------
# 5b. Registrar o melhor modelo com signature (exigido no UC)
# -------------------------------------------------------------
import os
from mlflow.models.signature import infer_signature

os.environ["MLFLOW_DFS_TMP"] = "/Volumes/workspace/default/mlflow-als-serving"

mlflow.set_experiment("/mlflow-als-serving")

# Gerar input/output de exemplo para inferir a signature
sample_input  = train.select("userId", "movieId", "rating").limit(10).toPandas()
sample_output = best_model.transform(
    train.select("userId", "movieId", "rating").limit(10)
).select("prediction").toPandas()

signature = infer_signature(sample_input, sample_output)

with mlflow.start_run(run_name="als_best_model"):

    mlflow.log_metric("rmse",    best_rmse)
    mlflow.log_param("rank",     best_params["rank"])
    mlflow.log_param("regParam", best_params["regParam"])
    mlflow.log_param("maxIter",  max_iter)
    mlflow.log_param("dataset",  "movielens-small")

    mlflow.spark.log_model(
        best_model,
        artifact_path="als_model",
        registered_model_name="movielens-als",
        signature=signature
    )

    print(f"Modelo registrado com signature!")
    print(f"RMSE final: {best_rmse:.4f}")

# COMMAND ----------

# -------------------------------------------------------------
# 6. Validação — top 10 recomendações para usuário 1
# (evita higher-order functions do UC)
# -------------------------------------------------------------
import pandas as pd

# Pegar todos os filmes da Silver
movies_pd  = spark.table("bronze_movies").toPandas()
silver_pd  = spark.table("silver_ratings").toPandas()

# Montar DataFrame com usuário 1 × todos os filmes que ele NÃO avaliou
user_id     = 1
rated_ids   = silver_pd[silver_pd["userId"] == user_id]["movieId"].tolist()
unseen      = movies_pd[~movies_pd["movieId"].isin(rated_ids)][["movieId"]].copy()
unseen["userId"] = user_id

# Converter para Spark e pontuar com o modelo
unseen_spark = spark.createDataFrame(unseen)
scores       = best_model.transform(unseen_spark)

# Juntar com títulos e pegar top 10
scores_pd = scores.toPandas()
top10 = (
    scores_pd
    .merge(movies_pd, on="movieId")
    .dropna(subset=["prediction"])
    .sort_values("predicted_rating" if "predicted_rating" in scores_pd.columns else "prediction", ascending=False)
    [["title", "prediction"]]
    .head(10)
)

print("Top 10 recomendações para usuário 1:")
display(spark.createDataFrame(top10))

# COMMAND ----------

# MAGIC %md
# MAGIC