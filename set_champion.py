
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5001")

client = MlflowClient()

client.set_registered_model_alias(
    name="cifar10_best_model",
    alias="champion",
    version="1",
)

print("Champion alias assigned to Version 1")
