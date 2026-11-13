
import argparse

import mlflow
import mlflow.pytorch
import torch
from PIL import Image
from torchvision import transforms

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def predict(image_path):
    mlflow.set_tracking_uri("http://127.0.0.1:5001")

    model = mlflow.pytorch.load_model(
        "models:/cifar10_best_model@champion",
        map_location="cpu",
    )
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = probabilities.max(dim=1)

    class_id = predicted.item()

    print(f"Predicted class: {CIFAR10_CLASSES[class_id]}")
    print(f"Confidence: {confidence.item():.2%}")

    return class_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    predict(args.image)
