
import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app import app, ml_models


@pytest.fixture
def client():
    # Enter FastAPI lifespan: loads the MLflow champion model
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert "cifar10" in ml_models


def test_predict(client):
    # Create a valid RGB image in memory
    image = Image.new("RGB", (32, 32), color="red")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.png",
                buffer.getvalue(),
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert isinstance(result["prediction"], int)
    assert 0 <= result["prediction"] <= 9
    assert result["class_name"] in [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    assert 0.0 <= result["confidence"] <= 1.0


def test_invalid_file(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "invalid.txt",
                b"This is not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
