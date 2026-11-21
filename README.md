
# CIFAR-10 MLOps Pipeline

An independent end-to-end MLOps practice project developed to consolidate Sessions 1–5 of an MLOps mentorship programme.

The project covers modular PyTorch training, reproducibility, automated testing, MLflow experiment tracking and model registry, FastAPI inference, and Docker containerisation.

## Features

- Modular CIFAR-10 image classification pipeline
- YAML-based training configuration
- Reproducible training with random seeds
- Training and validation metrics
- Resumable model checkpoints
- MLflow experiment tracking and model registry
- Champion model selection using MLflow aliases
- FastAPI REST API for image classification
- Docker containerisation
- Automated testing with pytest

## Project Structure

```text
cifar10_mlops_pipeline/
├── configs/                 # Training configuration
├── src/                     # Data loading, model, training utilities
├── test/                    # Automated tests
├── samples/                 # Sample images
├── train.py                 # Training and MLflow tracking
├── predict.py               # Local inference
├── register_model.py        # MLflow model registration
├── set_champion.py          # Champion alias management
├── app.py                   # FastAPI application
├── Dockerfile               # Container definition
├── requirements.txt         # Development dependencies
└── requirements-serve.txt   # API serving dependencies
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Start the MLflow tracking server:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5001
```

In another terminal, run:

```bash
python train.py --config configs/train.yaml
```

Training metrics and parameters are logged to the `cifar10` experiment in MLflow.

Open the MLflow UI at http://127.0.0.1:5001.

## Model Registry

The registered model is named `cifar10_best_model`.

The `champion` alias identifies the selected model version for inference.

Model registration and alias management are handled using:

- `register_model.py`
- `set_champion.py`

## FastAPI Serving

Start the MLflow server before running the API locally.

```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```

API endpoints:

- `GET /health` — Application health
- `GET /ready` — Model readiness
- `POST /predict` — Image classification

Interactive API documentation:

http://127.0.0.1:8001/docs

## Docker

Build the image:

```bash
docker build -t cifar10-api:latest .
```

Run the container:

```bash
docker run --rm -p 8001:8000 --name cifar10-api cifar10-api:latest
```

The Docker image includes the model artifacts required by its configuration.

## Testing

Run automated tests:

```bash
python -m pytest test/ -v
```

All eight tests passed during local development.

## Results

The five-epoch CIFAR-10 training run achieved:

- Best validation accuracy: **71.52%**
- Batch size: 64
- Learning rate: 0.001
- Random seed: 42

## Next Steps

Deploy the containerised model to AWS using S3, ECR and the deployment infrastructure covered in subsequent mentorship sessions.
