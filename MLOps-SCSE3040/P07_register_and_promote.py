"""P07 (MLflow part): train a model, register it, promote it to Production.

Run from this folder:
    python register_and_promote.py
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "delivery_times.csv"
FEATURES = ["distance_km", "prep_time_min", "traffic_level", "rain"]
MODEL_NAME = "delivery-time-model"

# 1. Tell MLflow where to store runs and the registry (a local SQLite file).
mlflow.set_tracking_uri(f"sqlite:///{(HERE / 'mlflow.db').as_posix()}")
mlflow.set_experiment("delivery-time-serving")

# 2. Load the data and split it.
orders = pd.read_csv(DATA)
X_train, X_test, y_train, y_test = train_test_split(
    orders[FEATURES], orders["delivery_min"], test_size=0.2, random_state=42
)

# 3. Train inside an MLflow run, log the metric, and register the model.
with mlflow.start_run(run_name="serving-model") as run:
    model = LinearRegression().fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mae_minutes", mae)

    mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=X_train.head(3),
        registered_model_name=MODEL_NAME,
    )
    run_id = run.info.run_id

# 4. Find the version we just registered and promote it to Production.
client = MlflowClient()
version = client.search_model_versions(f"run_id='{run_id}'")[0].version

client.transition_model_version_stage(
    name=MODEL_NAME, version=version, stage="Production",
    archive_existing_versions=True,
)
# Newer MLflow prefers aliases over stages, so set one as well.
client.set_registered_model_alias(MODEL_NAME, "production", version)

print("MAE           :", round(mae, 2), "minutes")
print("model name    :", MODEL_NAME)
print("version       :", version)
print("stage         :", client.get_model_version(MODEL_NAME, version).current_stage)
print("run id        :", run_id)
