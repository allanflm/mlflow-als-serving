# Databricks notebook source
# Célula de diagnóstico — rode antes de qualquer coisa
import os

# Verificar se o volume existe e qual o caminho exato
volume_base = "/Volumes/workspace/default"
print("Volumes disponíveis:")
print(os.listdir(volume_base))

# COMMAND ----------

# =============================================================
# mlflow-als-serving | 01_ingestion
# =============================================================

import urllib.request
import zipfile

# Caminho correto — hífen no nome do volume
VOLUME   = "/Volumes/workspace/default/mlflow-als-serving"
ZIP_PATH = f"{VOLUME}/movielens.zip"
EXT_PATH = f"{VOLUME}/raw/"

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

urllib.request.urlretrieve(URL, ZIP_PATH)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(EXT_PATH)

print("Download OK")

# COMMAND ----------

# Carregar CSVs
BASE = f"{EXT_PATH}ml-latest-small/"

ratings_raw = spark.read.csv(f"{BASE}ratings.csv", header=True, inferSchema=True)
movies_raw  = spark.read.csv(f"{BASE}movies.csv",  header=True, inferSchema=True)
tags_raw    = spark.read.csv(f"{BASE}tags.csv",    header=True, inferSchema=True)

ratings_raw.printSchema()

# COMMAND ----------

# Salvar Bronze
ratings_raw.write.format("delta").mode("overwrite").saveAsTable("bronze_ratings")
movies_raw.write.format("delta").mode("overwrite").saveAsTable("bronze_movies")
tags_raw.write.format("delta").mode("overwrite").saveAsTable("bronze_tags")

print("Tabelas Bronze criadas!")

# COMMAND ----------

# Validação
print(f"Ratings : {spark.table('bronze_ratings').count():>10,} linhas")
print(f"Filmes  : {spark.table('bronze_movies').count():>10,} linhas")
print(f"Tags    : {spark.table('bronze_tags').count():>10,} linhas")

display(spark.table("bronze_ratings").limit(5))