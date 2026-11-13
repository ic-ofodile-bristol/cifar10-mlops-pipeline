
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import CIFAR10Model
from src.engine import train_one_epoch, evaluate


def test_training_and_evaluation():
    torch.manual_seed(42)

    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
    )

    model = CIFAR10Model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    train_loss, train_acc = train_one_epoch(
        model, loader, optimizer, criterion, "cpu"
    )

    val_loss, val_acc = evaluate(
        model, loader, criterion, "cpu"
    )

    assert train_loss >= 0
    assert val_loss >= 0
    assert 0 <= train_acc <= 1
    assert 0 <= val_acc <= 1

