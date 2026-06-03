# Databricks notebook source
# =============================================================
# mlflow-als-serving | 05_model_serving
# Objetivo: criar endpoint REST e consultar recomendações
# =============================================================

import json

import requests

# Preencha com seus dados
DATABRICKS_HOST  = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
DATABRICKS_TOKEN = dbutils.secrets.get(scope="my-scope", key="databricks-token")


headers = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type":  "application/json"
}

# Testar conexão
resp = requests.get(f"{DATABRICKS_HOST}/api/2.0/serving-endpoints", headers=headers)
print(f"Status: {resp.status_code}")
print(f"Endpoints existentes: {[e['name'] for e in resp.json().get('endpoints', [])]}")

# COMMAND ----------

# -------------------------------------------------------------
# 2. Criar o endpoint de serving
# (rode apenas UMA vez — vai dar erro se o endpoint já existir)
# -------------------------------------------------------------
endpoint_config = {
    "name": "movielens-als-endpoint",
    "config": {
        "served_entities": [{
            "entity_name":           "workspace.default.movielens-als",
            "entity_version":        "1",
            "workload_size":         "Small",
            "scale_to_zero_enabled": True   # essencial no Free Edition!
        }]
    }
}

resp = requests.post(
    f"{DATABRICKS_HOST}/api/2.0/serving-endpoints",
    headers=headers,
    json=endpoint_config
)

print(f"Status: {resp.status_code}")
print(json.dumps(resp.json(), indent=2))

# COMMAND ----------

# Rode várias vezes para acompanhar o estado
import time

resp  = requests.get(
    f"{DATABRICKS_HOST}/api/2.0/serving-endpoints/movielens-als-endpoint",
    headers=headers
)
data  = resp.json()
state = data.get("state", {}).get("ready", "UNKNOWN")

print(f"Estado: {state}")

if state == "READY":
    print("Pronto! Pode rodar as células 4 e 5.")
else:
    print("Ainda subindo... aguarde e rode novamente.")

# COMMAND ----------

# -------------------------------------------------------------
# 4. Fazer chamadas de inferência via REST
# -------------------------------------------------------------
payload = {
    "dataframe_records": [
        {"userId": 1, "movieId": 1,   "rating": 0.0},
        {"userId": 1, "movieId": 50,  "rating": 0.0},
        {"userId": 2, "movieId": 296, "rating": 0.0},
        {"userId": 2, "movieId": 318, "rating": 0.0},
        {"userId": 3, "movieId": 260, "rating": 0.0}
    ]
}

resp = requests.post(
    f"{DATABRICKS_HOST}/serving-endpoints/movielens-als-endpoint/invocations",
    headers=headers,
    json=payload
)

print(f"Status: {resp.status_code}")
print(json.dumps(resp.json(), indent=2))

# COMMAND ----------

# -------------------------------------------------------------
# 5. Top 10 recomendações via API para usuário 1
# -------------------------------------------------------------
import pandas as pd

silver_pd = spark.table("silver_ratings").toPandas()
movies_pd = spark.table("bronze_movies").toPandas()

user_id = 1
rated   = silver_pd[silver_pd["userId"] == user_id]["movieId"].tolist()
unseen  = movies_pd[~movies_pd["movieId"].isin(rated)][["movieId"]].copy()
unseen["userId"] = user_id
unseen["rating"] = 0.0  # placeholder obrigatório pela signature

results    = []
batch_size = 50

for i in range(0, min(len(unseen), 500), batch_size):
    batch   = unseen.iloc[i:i+batch_size]
    payload = {"dataframe_records": batch.to_dict(orient="records")}

    resp = requests.post(
        f"{DATABRICKS_HOST}/serving-endpoints/movielens-als-endpoint/invocations",
        headers=headers,
        json=payload
    )

    if resp.status_code == 200:
        preds = resp.json().get("predictions", [])
        for j, pred in enumerate(preds):
            results.append({
                "movieId":    int(batch.iloc[j]["movieId"]),
                "prediction": pred
            })

results_df = (
    pd.DataFrame(results)
    .merge(movies_pd, on="movieId")
    .dropna(subset=["prediction"])
    .sort_values("prediction", ascending=False)
    [["title", "prediction"]]
    .head(10)
)

print(f"Top 10 recomendações para usuário {user_id} via API REST:")
display(spark.createDataFrame(results_df))