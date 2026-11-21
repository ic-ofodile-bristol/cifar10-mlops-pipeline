
from contextlib import asynccontextmanager
import io
import os
import mlflow
import mlflow.pytorch
import torch
from fastapi import FastAPI, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms
from src.data import get_dataloaders, get_transform

# Store the loaded model in memory
ml_models = {}

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# Define the prediction response structure
class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    confidence: float


@asynccontextmanager
async def lifespan(app: FastAPI):

    # STARTUP: Load champion model once
   #mlflow.set_tracking_uri("http://127.0.0.1:5001")

    model_uri = os.getenv(
        "MODEL_URI",
        "models:/cifar10_best_model@champion"
    )

    if model_uri.startswith("models:/"):
        mlflow.set_tracking_uri(
            os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://127.0.0.1:5001"
            )
        )

    model = mlflow.pytorch.load_model(
        model_uri,
        map_location="cpu",
    )

    model.eval()
    ml_models["cifar10"] = model

    print("CIFAR-10 champion model loaded")

    yield

    # SHUTDOWN: Release model reference
    ml_models.clear()


app = FastAPI(
    title="CIFAR-10 Classification API",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if "cifar10" not in ml_models:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile):

    # Validate uploaded file type
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(
            status_code=400,
            detail="File must be PNG or JPEG",
        )

    # Read and preprocess image
    contents = await file.read()

    try:
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB").resize((32, 32))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image",
        )

    tensor = get_transform()(image).unsqueeze(0)

    # Model inference
    with torch.no_grad():
        logits = ml_models["cifar10"](tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)

    class_id = int(prediction.item())

    return PredictionResponse(
        prediction=class_id,
        class_name=CIFAR10_CLASSES[class_id],
        confidence=float(confidence.item()),
    )
