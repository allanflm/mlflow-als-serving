# Databricks notebook source
# =============================================================
# mlflow-als-serving | 04_model_registry
# =============================================================

import mlflow
from mlflow.tracking import MlflowClient

client     = MlflowClient()
model_name = "workspace.default.movielens-als"  # nome completo UC

# Listar versões registradas
versions = client.search_model_versions(f"name='{model_name}'")

for v in versions:
    print(f"Versão: {v.version} | Stage: {v.current_stage} | Run: {v.run_id[:8]}...")

# COMMAND ----------

# Versão mais recente
latest_version = max(int(v.version) for v in versions)
print(f"Versão mais recente: {latest_version}")

# COMMAND ----------

# Promover para Production via alias (padrão no Unity Catalog)
client.set_registered_model_alias(
    name=model_name,
    alias="Production",
    version=latest_version
)

print(f"Alias 'Production' → v{latest_version} definido!")

# COMMAND ----------

# Descrição e tags
client.update_registered_model(
    name=model_name,
    description="ALS Collaborative Filtering treinado com MovieLens Small. Pipeline: mlflow-als-serving."
)

client.update_model_version(
    name=model_name,
    version=latest_version,
    description="Melhor modelo do grid search — veja experimento /mlflow-als-serving"
)

client.set_model_version_tag(model_name, latest_version, "dataset",   "movielens-small")
client.set_model_version_tag(model_name, latest_version, "algorithm", "ALS")
client.set_model_version_tag(model_name, latest_version, "project",   "mlflow-als-serving")

print("Descrição e tags adicionadas!")

# COMMAND ----------

# Validar — carregar pelo alias e pontuar
import os
import pandas as pd

os.environ["MLFLOW_DFS_TMP"] = "/Volumes/workspace/default/mlflow-als-serving"

model_uri    = f"models:/{model_name}@Production"
model_loaded = mlflow.spark.load_model(model_uri)

print(f"Modelo carregado de: {model_uri}")

sample = spark.createDataFrame(
    pd.DataFrame({
        "userId":  [1, 1, 2, 2, 3],
        "movieId": [1, 50, 296, 318, 260]
    })
)

preds = model_loaded.transform(sample)
display(preds.select("userId", "movieId", "prediction"))

# COMMAND ----------

# Confirmar alias final
info = client.get_registered_model(model_name)
print(f"Modelo   : {info.name}")
print(f"Descrição: {info.description}")

versions = client.search_model_versions(f"name='{model_name}'")
for v in versions:
    print(f"v{v.version} | stage: {v.current_stage}")

print("\nNotebook 4 concluído!")