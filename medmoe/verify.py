import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch

from .data import Record, slide_id
from .metrics import classification
from .models import EXPERT_NAMES, MedMoE, parameter_count


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify(root):
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    manifest_path = root / "split.csv"
    require(hashlib.sha256(manifest_path.read_bytes()).hexdigest() == protocol["split_sha256"], "Manifest checksum differs")
    manifest = csv_rows(manifest_path)
    require(len({row["sha256"] for row in manifest}) == len(manifest), "Duplicate image hash in manifest")
    require(len({row["member"] for row in manifest}) == len(manifest), "Duplicate image identifier in manifest")
    for split, size in protocol["split_sizes"].items():
        require(sum(row["split"] == split for row in manifest) == size, f"Unexpected size for {split}")
    if protocol["slide_groups"]:
        slides = {}
        for row in manifest:
            group = slide_id(Record(row["member"], row["original_class"], row["sha256"]))
            require(group in protocol["slide_groups"][row["split"]], "Incorrect slide group assignment")
            slides.setdefault(group, set()).add(row["split"])
        require(all(len(splits) == 1 for splits in slides.values()), "Slide crosses splits")
    classes = len(protocol["class_names"])
    holdout = [(row["member"], int(row["label"])) for row in manifest if row["split"] == "test"]
    for seed in protocol["seeds"]:
        for name in protocol["models"]:
            model_directory = root / f"seed-{seed}" / name
            rows = csv_rows(model_directory / "predictions.csv")
            metrics = json.loads((model_directory / "test.json").read_text(encoding="utf-8"))
            require([(row["member"], int(row["label"])) for row in rows] == holdout, "Prediction rows differ from test manifest")
            route_counts = Counter()
            for row in rows:
                probabilities = [float(row[f"p_{index}"]) for index in range(classes)]
                require(all(0 <= score <= 1 for score in probabilities), "Invalid class probability")
                require(math.isclose(sum(probabilities), 1.0, abs_tol=1e-6), "Probabilities do not sum to 1")
                require(int(row["prediction"]) == max(range(classes), key=probabilities.__getitem__), "Prediction differs from argmax")
                if "selected_experts" in row:
                    experts = row["selected_experts"].split("|")
                    require(len(set(experts)) == len(experts) == protocol["top_k"], "Unexpected number of executed experts")
                    require(set(experts) <= set(metrics["utilization"]), "Route references an unknown expert")
                    route_counts.update(experts)
            actual = classification(torch.tensor([label for _, label in holdout]),
                                    torch.tensor([int(row["prediction"]) for row in rows]), classes)
            for key in ("accuracy", "sensitivity", "specificity", "macro_f1", "balanced_accuracy", "confusion_matrix", "n"):
                require(actual[key] == metrics[key], f"Metric {key} differs for seed {seed}, model {name}")
            if route_counts:
                kept = [index for index, expert in enumerate(EXPERT_NAMES) if expert in metrics["utilization"]]
                shape = MedMoE(classes, protocol["top_k"], kept)
                active = parameter_count(shape.stem) + parameter_count(shape.router)
                for index, module in zip(kept, shape.experts):
                    expert = EXPERT_NAMES[index]
                    utilization = route_counts[expert] / len(rows)
                    require(math.isclose(utilization, metrics["utilization"][expert]), "Expert utilization differs from routes")
                    active += utilization * parameter_count(module)
                require(math.isclose(active, metrics["mean_active_parameters"]), "Active parameter count differs from routes")
                require(parameter_count(shape) == metrics["total_parameters"], "Stored parameter count differs from architecture")
            history = json.loads((model_directory / "training.json").read_text(encoding="utf-8"))
            continuation = [epoch for epoch in history if epoch["phase"] == "train"] or history
            mean_payload = sum(epoch["gradient_bytes_sum"] for epoch in continuation) / sum(epoch["steps"] for epoch in continuation)
            require(mean_payload == metrics["mean_gradient_payload_bytes"], "Gradient tensor bytes differ from training log")
            require(sum(epoch["gradient_bytes_sum"] for epoch in history) == metrics["whole_run_gradient_payload_bytes"],
                    "Whole-run gradient bytes differ from training log")
            require(metrics["mean_active_parameters"] <= metrics["total_parameters"], "Active parameters exceed model size")
            require(metrics["mean_gradient_payload_bytes"] <= metrics["full_model_fp32_bytes"], "Gradient payload exceeds FP32 model size")
    print(f"Verified {root}: {len(holdout)} test images, {len(protocol['seeds'])} seeds, {len(protocol['models'])} models.")


def main():
    parser = argparse.ArgumentParser(description="Recompute metrics from predictions and audit the split")
    parser.add_argument("runs", type=Path, nargs="+")
    for run in parser.parse_args().runs:
        verify(run)


if __name__ == "__main__":
    main()
