"""
Quality gate for the food11 model (Lab 5).

Compares a candidate model version with the current `champion`:
    se = sqrt(p * (1 - p) / n)        p = champion accuracy, n = test split size
    promote if acc_candidate >= acc_champion + 2 * se
Otherwise the candidate gets the `challenger` alias and `champion` stays.
In both cases the decision and the numbers are written as tags on the candidate.

Usage:
    uv run python src/food11/promote.py --version 3
"""
import argparse
import math
import os
import sys

import mlflow
from mlflow import MlflowClient

MODEL_NAME = "food11"


def parse_args():
    parser = argparse.ArgumentParser(description="Promote a food11 version if it beats the champion.")
    parser.add_argument("--version", required=True, help="candidate model version number")
    parser.add_argument("--metric", default="test_accuracy")
    parser.add_argument("--n-test", type=int, default=1096, help="size of the test split")
    return parser.parse_args()


def run_metric(client: MlflowClient, run_id: str, metric: str) -> float:
    metrics = client.get_run(run_id).data.metrics
    if metric not in metrics:
        sys.exit(f"ERROR: run {run_id} has no metric '{metric}'")
    return metrics[metric]


def main():
    args = parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()

    candidate = client.get_model_version(MODEL_NAME, args.version)
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")

    if candidate.version == champion.version:
        sys.exit(f"Version {candidate.version} is already the champion, nothing to do.")

    acc_candidate = run_metric(client, candidate.run_id, args.metric)
    acc_champion = run_metric(client, champion.run_id, args.metric)

    se = math.sqrt(acc_champion * (1 - acc_champion) / args.n_test)
    threshold = acc_champion + 2 * se

    if acc_candidate >= threshold:
        client.set_registered_model_alias(MODEL_NAME, "champion", candidate.version)
        decision = "promoted"
    else:
        client.set_registered_model_alias(MODEL_NAME, "challenger", candidate.version)
        decision = "rejected"

    tags = {
        "gate_decision": decision,
        "candidate_accuracy": f"{acc_candidate:.4f}",
        "champion_accuracy": f"{acc_champion:.4f}",
        "champion_version": champion.version,
        "gate_se": f"{se:.4f}",
        "gate_threshold": f"{threshold:.4f}",
    }
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, candidate.version, key, value)

    print(f"Candidate v{candidate.version}: {args.metric} = {acc_candidate:.4f}")
    print(f"Champion  v{champion.version}: {args.metric} = {acc_champion:.4f}")
    print(f"se = sqrt(p(1-p)/n) = {se:.4f}  (n = {args.n_test})  ->  2*se = {2 * se:.4f}")
    print(f"Threshold = {acc_champion:.4f} + {2 * se:.4f} = {threshold:.4f}")
    if decision == "promoted":
        print(f"DECISION: PROMOTED -> champion is now v{candidate.version}")
    else:
        print(f"DECISION: REJECTED -> v{candidate.version} is 'challenger', champion stays v{champion.version}")


if __name__ == "__main__":
    main()