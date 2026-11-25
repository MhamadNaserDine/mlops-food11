"""
Measure the test accuracy of a registered food11 version and log it as
`test_accuracy` on the run that produced it.

Needed because the current champion was imported in Lab 4 without metrics,
so the quality gate (promote.py) would have nothing to compare against.

Usage:
    uv run python -m src.food11.evaluate_version --version 2
"""
import argparse

import mlflow.pytorch
import torch
from torch import nn

import mlflow
from mlflow import MlflowClient
from src.food11.train import evaluate, get_data_loaders

MODEL_NAME = "food11"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--dataset", choices=["mini", "processed"], default="mini")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    client = MlflowClient()
    mv = client.get_model_version(MODEL_NAME, args.version)

    device = torch.device("cpu")
    model = mlflow.pytorch.load_model(
        f"models:/{MODEL_NAME}/{args.version}", map_location="cpu"
    ).to(device)

    _, _, test_loader = get_data_loaders(args.dataset, args.batch_size)
    _loss, accuracy = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    n = len(test_loader.dataset)

    client.log_metric(mv.run_id, "test_accuracy", accuracy)
    client.set_model_version_tag(
        MODEL_NAME, args.version, "test_accuracy_source",
        f"measured on the {args.dataset} evaluation split ({n} images)",
    )
    print(f"{MODEL_NAME} v{args.version}: test_accuracy = {accuracy:.4f} on {n} images")
    print(f"Logged to run {mv.run_id}")


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    main()