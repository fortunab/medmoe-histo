import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


PAPER_MODELS = {"dense": "Dense CNN", "moe": "MoE", "pruned": "Usage-pruned", "random-pruned": "Random-pruned",
                "efficientnet": "EfficientNet-B0", "vit-small": "ViT-Small", "swin-tiny": "Swin-Tiny",
                "dinov2": "DINOv2-Small", "clip": "CLIP ViT-B/32"}
PAPER_FIELDS = ("accuracy", "sensitivity", "specificity", "mean_active_parameters", "mean_gradient_payload_bytes")
TEX_ESCAPE = str.maketrans({"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"})


def read_runs(specifications):
    groups = defaultdict(dict)
    signature = {}
    for specification in specifications:
        label, path = specification.split("=", maxsplit=1)
        root = Path(path)
        protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
        current = [protocol[key] for key in ("dataset", "task", "split_mode", "split_sha256", "epochs", "warmup",
                                              "batch_size", "prune_count", "top_k", "prune_threshold", "image_size", "lr",
                                              "balance", "weight_decay", "pretrained")]
        if label in signature and signature[label] != current:
            raise ValueError(f"Run settings differ within {label!r}")
        signature[label] = current
        with (root / "metrics_by_seed.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (row["model"], int(row["seed"]))
                if key in groups[label]:
                    raise ValueError(f"Duplicate model and seed for {label!r}: {key}")
                groups[label][key] = {field: float(row[field]) for field in PAPER_FIELDS}
    return groups


def mean(rows, field):
    return statistics.mean(row[field] for row in rows)


def latex_table(group, runs, index):
    caption = group.translate(TEX_ESCAPE)
    lines = [r"\begin{table}[!htbp]", r"\centering", f"\\caption{{{caption}. Results over $n$ training seeds.}}",
             f"\\label{{tab:run-{index}}}", r"\small", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Model & $n$ & Acc. & Sens. & Spec. & Act. (k) & Grad. (kB) \\", r"\midrule"]
    for model, name in PAPER_MODELS.items():
        rows = [row for (candidate, _), row in runs.items() if candidate == model]
        if not rows:
            continue
        scores = [f"{mean(rows, field):.3f}" for field in PAPER_FIELDS[:3]]
        if len(rows) > 1:
            sd = statistics.stdev(row["accuracy"] for row in rows)
            scores[0] = f"${scores[0]} \\pm {sd:.3f}$"
        active, gradient = (mean(rows, field) / 1000 for field in PAPER_FIELDS[3:])
        lines.append(" & ".join([name, str(len(rows)), *scores, f"{active:.1f}", f"{gradient:.1f}"]) + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from completed experiment outputs")
    parser.add_argument("--run", nargs="+", required=True, help="Repeated LABEL=directory pairs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    groups = read_runs(args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    tables = [latex_table(label, runs, index) for index, (label, runs) in enumerate(groups.items(), start=1)]
    (args.out / "tables.tex").write_text("\n\n".join(tables) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
