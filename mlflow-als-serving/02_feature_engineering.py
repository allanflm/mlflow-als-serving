# Databricks notebook source
# =============================================================
# mlflow-als-serving | 02_feature_engineering
# Objetivo: transformar Bronze → Silver
# =============================================================

from pyspark.sql.functions import col, count, avg, stddev, to_timestamp

ratings = spark.table("bronze_ratings")
movies  = spark.table("bronze_movies")

print(f"Bronze ratings: {ratings.count():,} linhas")

# COMMAND ----------

# -------------------------------------------------------------
# 1. Converter timestamp Unix → datetime legível
# -------------------------------------------------------------
ratings_clean = (
    ratings
    .withColumn("rated_at", to_timestamp(col("timestamp")))
    .drop("timestamp")
)

display(ratings_clean.limit(5))

# COMMAND ----------

# -------------------------------------------------------------
# 2. Filtrar cold-start: usuários e filmes com poucas avaliações
# -------------------------------------------------------------
# Usuários com menos de 5 ratings e filmes com menos de 10
# geram predições ruins no ALS — melhor remover agora

user_counts  = ratings_clean.groupBy("userId").agg(count("*").alias("n"))
movie_counts = ratings_clean.groupBy("movieId").agg(count("*").alias("n"))

active_users  = user_counts.filter(col("n")  >= 5).select("userId")
active_movies = movie_counts.filter(col("n") >= 10).select("movieId")

ratings_silver = (
    ratings_clean
    .join(active_users,  "userId",  "inner")
    .join(active_movies, "movieId", "inner")
)

removidos = ratings_clean.count() - ratings_silver.count()
print(f"Ratings removidos por cold-start : {removidos:,}")
print(f"Ratings Silver                   : {ratings_silver.count():,}")

# COMMAND ----------

# -------------------------------------------------------------
# 3. Estatísticas por usuário (útil para análise e debug)
# -------------------------------------------------------------
user_stats = (
    ratings_silver
    .groupBy("userId")
    .agg(
        count("*").alias("total_ratings"),
        avg("rating").alias("avg_rating"),
        stddev("rating").alias("std_rating")
    )
)

display(user_stats.orderBy("total_ratings", ascending=False).limit(10))

# COMMAND ----------

# -------------------------------------------------------------
# 4. Salvar camada Silver
# -------------------------------------------------------------
(ratings_silver
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("silver_ratings"))

(user_stats
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("silver_user_stats"))

print("Tabelas Silver criadas!")

# COMMAND ----------

# -------------------------------------------------------------
# 5. Validação final
# -------------------------------------------------------------
silver = spark.table("silver_ratings")

print(f"Total ratings  : {silver.count():,}")
print(f"Usuários únicos: {silver.select('userId').distinct().count():,}")
print(f"Filmes únicos  : {silver.select('movieId').distinct().count():,}")

# Distribuição de ratings — deve ser aproximadamente normal em torno de 3.5
display(
    silver.groupBy("rating")
    .count()
    .orderBy("rating")
)