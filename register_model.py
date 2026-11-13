
import mlflow
import mlflow.pytorch
import torch

from src.model import CIFAR10Model

# Connect to our CIFAR-10 MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Load the best model weights
model = CIFAR10Model()

state_dict = torch.load(
    "artifacts/model.pt",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state_dict)
model.eval()

# Create a new MLflow run for registration
mlflow.set_experiment("cifar10")

with mlflow.start_run(run_name="register_best_model"):

    # One example CIFAR-10 RGB image
    input_example = torch.randn(1, 3, 32, 32)

    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        name="cifar10_model",
        registered_model_name="cifar10_best_model",
        serialization_format="pickle",
        input_example=input_example,
    )

print("Model registered successfully!")
print("Model URI:", model_info.model_uri)

