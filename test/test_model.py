
import torch
from src.model import CIFAR10Model


def test_model_output_shape():
    model = CIFAR10Model()
    images = torch.randn(4, 3, 32, 32)

    outputs = model(images)

    assert outputs.shape == (4, 10)


def test_model_outputs_are_finite():
    model = CIFAR10Model()
    images = torch.randn(2, 3, 32, 32)

    outputs = model(images)

    assert torch.isfinite(outputs).all()
