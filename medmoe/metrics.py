import csv
import json
import statistics

import numpy as np
import torch


METRICS = (
    "accuracy", "sensitivity", "specificity", "balanced_accuracy", "macro_f1",
    "total_parameters", "mean_active_parameters", "mean_gradient_payload_bytes",
    "active_pct_dense", "communication_pct_dense", "active_pct_efficientnet",
    "communication_pct_efficientnet", "latency_ms_per_image",
)


def classification(targets, predictions, classes, allow_missing=False):
    matrix = torch.bincount(classes * targets + predictions, minlength=classes ** 2).reshape(classes, classes)
    true_positive = matrix.diag().double()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    negatives = matrix.sum() - support
    if not allow_missing and ((support == 0).any() or (negatives == 0).any()):
        raise ValueError("Every class needs positive and negative examples in the evaluated split")
    sensitivity = true_positive / support
    specificity = (negatives - (predicted - true_positive)) / negatives
    f1 = 2 * true_positive / (support + predicted)
    as_list = lambda values: [float(value) if value.isfinite() else None for value in values]
    return {
        "n": int(matrix.sum()), "accuracy": float(true_positive.sum() / matrix.sum()),
        "sensitivity": float(sensitivity[1] if classes == 2 else sensitivity.nanmean()),
        "specificity": float(specificity[1] if classes == 2 else specificity.nanmean()),
        "balanced_accuracy": float(sensitivity.nanmean()), "macro_f1": float(f1[support > 0].mean()),
        "confusion_matrix": matrix.tolist(), "support": support.tolist(),
        "per_class_sensitivity": as_list(sensitivity), "per_class_specificity": as_list(specificity),
    }


def paired_accuracy_interval(targets, candidate, reference, seed, draws=2000):
    delta = (candidate == targets).astype(np.int8) - (reference == targets).astype(np.int8)
    rng = np.random.default_rng(seed)
    # Stratify image bootstrap by class to preserve the evaluated class proportions.
    groups = [delta[targets == label] for label in np.unique(targets)]
    estimates = np.zeros(draws)
    for group in groups:
        estimates += rng.choice(group, size=(draws, len(group)), replace=True).sum(axis=1)
    low, high = np.quantile(estimates / len(delta), [0.025, 0.975])
    return {"accuracy_delta": float(delta.mean()), "image_bootstrap_ci95": [float(low), float(high)], "draws": draws}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(out, results):
    rows = [{"seed": seed, "model": model, **{key: metrics.get(key) for key in METRICS}}
            for seed, models in results.items() for model, metrics in models.items()]
    write_csv(out / "metrics_by_seed.csv", ["seed", "model", *METRICS], rows)
    summary = []
    for model in results[next(iter(results))]:
        for metric in METRICS:
            values = [models[model][metric] for models in results.values() if metric in models[model]]
            if values:
                summary.append({"model": model, "metric": metric, "seeds": len(values),
                                "mean": statistics.mean(values),
                                "sample_sd": statistics.stdev(values) if len(values) > 1 else None})
    write_csv(out / "summary.csv", ["model", "metric", "seeds", "mean", "sample_sd"], summary)
