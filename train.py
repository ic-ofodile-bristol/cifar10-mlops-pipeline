
import argparse
from pathlib import Path

import torch

from src.config import load_config
from src.data import get_dataloaders
from src.model import CIFAR10Model
from src.engine import train_one_epoch, evaluate
from src.utils import (
    set_seed,
    get_device,
    setup_logging,
    save_checkpoint,
    load_checkpoint,
)

def main(config_path):
    # Load configuration
    config = load_config(config_path)

    # Reproducibility
    set_seed(config["seed"])

    # Select device
    device = get_device()
    print(f"Training on: {device}")

    # Load CIFAR-10 data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        seed=config["seed"],
    )

    # Initialise model
    model = CIFAR10Model().to(device)

    # Loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    # Prepare model directory
    model_path = Path(config["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logging()
    checkpoint_path = Path(config["checkpoint_dir"]) / "latest.pt"
    start_epoch = 0
    best_val_accuracy = 0.0

    if checkpoint_path.exists():
        completed_epoch, best_val_accuracy = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            device,
        )
        start_epoch = completed_epoch + 1
        logger.info(
            "Resuming training from epoch %d",
            start_epoch + 1,
        )
    # Training loop
    for epoch in range(start_epoch, config["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Save best model based on validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved: {model_path}")

        save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_accuracy, config)

        logger.info(
            "Checkpoint saved after epoch %d",
            epoch + 1,
        )

    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
    )
    args = parser.parse_args()

    main(args.config)
