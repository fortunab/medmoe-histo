import argparse
import copy
import importlib.metadata
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from .data import DATASETS, ImageSource, Tiles, decode, read_index, slide_split, stratified_split, write_manifest
from .metrics import classification, paired_accuracy_interval, summarize, write_csv, write_json
from .models import BASELINES, EXPERT_NAMES, MedMoE, baseline, parameter_count


MOE_MODELS = ("moe", "pruned", "random-pruned")
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dense_or_sparse(model, images):
    if isinstance(model, MedMoE):
        return model(images)
    return model(images), None, None


def optimize(model, dataset, args, seed, epochs, model_name, warmup=False):
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    usage_total = np.zeros(len(model.experts), dtype=np.int64) if isinstance(model, MedMoE) else None
    history = []
    model.train()
    for epoch in range(epochs):
        stats = {"examples": 0, "steps": 0, "loss_sum": 0.0, "gradient_bytes_sum": 0}
        started = time.perf_counter()
        for images, targets in loader:
            images, targets = images.to(args.device), targets.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits, usage, balance = dense_or_sparse(model, images)
            loss = cross_entropy(logits, targets)
            if balance is not None:
                loss = loss + args.balance * balance
                usage_total += usage.sum(dim=0).detach().cpu().numpy()
            loss.backward()
            gradient_bytes = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            stats["gradient_bytes_sum"] += gradient_bytes
            stats["loss_sum"] += float(loss.detach()) * targets.numel()
            stats["examples"] += targets.numel()
            stats["steps"] += 1
        stats.update(epoch=epoch + 1, seconds=time.perf_counter() - started, phase="warmup" if warmup else "train")
        history.append(stats)
        print(f"{model_name} {stats['phase']} {epoch + 1}/{epochs} loss={stats['loss_sum'] / stats['examples']:.4f} "
              f"({stats['seconds']:.1f}s)", flush=True)
    return history, usage_total


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate(model, dataset, args, classes, missing=False):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size)
    totals = parameter_count(model)
    params = []
    usages = []
    if isinstance(model, MedMoE):
        shared = parameter_count(model.stem) + parameter_count(model.router)
        expert_sizes = torch.tensor([parameter_count(expert) for expert in model.experts])
    probabilities, targets, times = [], [], []
    for index, (images, labels) in enumerate(loader):
        images = images.to(args.device)
        if index == 0:
            for _ in range(3):
                dense_or_sparse(model, images)
        synchronize(args.device)
        started = time.perf_counter()
        logits, usage, _ = dense_or_sparse(model, images)
        synchronize(args.device)
        times.append(time.perf_counter() - started)
        probabilities.append(logits.softmax(dim=1).cpu())
        targets.append(labels)
        if usage is not None:
            active = usage.cpu()
            usages.append(active)
            params.append(shared + active @ expert_sizes)
    scores, labels = torch.cat(probabilities), torch.cat(targets)
    metrics = classification(labels, scores.argmax(dim=1), classes, allow_missing=missing)
    metrics.update(total_parameters=totals, mean_active_parameters=float(torch.cat(params).double().mean()) if params else float(totals),
                   latency_ms_per_image=1000 * sum(times) / len(dataset),
                   full_model_fp32_bytes=4 * totals)
    routes = None
    if usages:
        routes = torch.cat(usages)
        metrics.update(top_k=model.top_k, retained_experts=len(model.experts),
                       expert_sparsity=1 - model.top_k / len(model.experts),
                       utilization={EXPERT_NAMES[key]: value for key, value in zip(model.expert_ids, routes.double().mean(dim=0).tolist())})
    return metrics, scores.numpy(), labels.numpy(), routes


def send_to_device(model, device):
    return model.to(device=device, memory_format=torch.channels_last)


def normalization(name):
    if name == "clip":
        return (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    return IMAGENET


def tiles_for(source, splits, labels, size, name, cache):
    mean, std = normalization(name)
    return {split: Tiles(source, records, labels, size, cache, mean, std, split == "train")
            for split, records in splits.items()}


def retain_by_usage(utilization, prune_count, threshold=1.0):
    order = sorted(range(len(utilization)), key=lambda index: (utilization[index], index))
    removed = {index for index in order[:prune_count] if utilization[index] <= threshold}
    return [index for index in range(len(utilization)) if index not in removed]


def run_seed(source, splits, label_ids, args, seed, directory, cache, classes):
    directory.mkdir()
    data = tiles_for(source, splits, label_ids, args.image_size, "dense", cache)
    warm_history, utilization, initial = [], None, None
    if any(name in MOE_MODELS for name in args.models):
        set_seed(seed)
        initial = send_to_device(MedMoE(classes, args.top_k), args.device)
        warm_history, selections = optimize(initial, data["train"], args, seed, args.warmup, "moe", warmup=True)
        utilization = selections / sum(epoch["examples"] for epoch in warm_history)
        retain = retain_by_usage(utilization, args.prune_count, args.prune_threshold)
        random_keep = sorted(random.Random(seed + 2000).sample(range(len(EXPERT_NAMES)), len(retain)))
        write_json(directory / "warmup.json", {"history": warm_history, "utilization": dict(zip(EXPERT_NAMES, utilization.tolist())),
                                                "retained": [EXPERT_NAMES[i] for i in retain],
                                                "random_retained": [EXPERT_NAMES[i] for i in random_keep]})
        if args.pruning_sweep:
            sweep = []
            for removed in range(len(EXPERT_NAMES) - args.top_k + 1):
                sweep_model = initial.pruned(retain_by_usage(utilization, removed))
                scores, _, _, _ = evaluate(sweep_model, data["validation"], args, classes, missing=True)
                sweep.append({"removed_experts": removed, "retained_experts": len(sweep_model.experts),
                              "removed_parameters_pct": 100 * (1 - parameter_count(sweep_model) / parameter_count(initial)),
                              "validation_accuracy": scores["accuracy"], "active_parameters": scores["mean_active_parameters"]})
            write_csv(directory / "validation_pruning_sweep.csv", list(sweep[0]), sweep)
    output, predictions = {}, {}
    for name in args.models:
        model_directory = directory / name
        model_directory.mkdir()
        set_seed(seed)
        if name in MOE_MODELS:
            keep = {"moe": range(len(EXPERT_NAMES)), "pruned": retain, "random-pruned": random_keep}[name]
            model = copy.deepcopy(initial) if name == "moe" else initial.pruned(keep)
            history = list(warm_history)
            model_data = data
            train_seed, epochs = seed + 1, args.epochs - args.warmup
            if name != "moe":
                instant, _, _, _ = evaluate(model, model_data["validation"], args, classes, missing=True)
                write_json(model_directory / "validation_at_pruning.json", instant)
        else:
            model = send_to_device(baseline(name, classes, args.pretrained), args.device)
            size = 224 if name in ("vit-small", "swin-tiny", "clip", "dinov2") else args.image_size
            model_data = tiles_for(source, splits, label_ids, size, name, cache if size == args.image_size else None)
            history = []
            train_seed, epochs = seed, args.epochs
        updates, _ = optimize(model, model_data["train"], args, train_seed, epochs, name)
        history.extend(updates)
        validation, _, _, _ = evaluate(model, model_data["validation"], args, classes, missing=True)
        test, scores, labels, routes = evaluate(model, model_data["test"], args, classes)
        phase = updates or warm_history
        test["mean_gradient_payload_bytes"] = sum(e["gradient_bytes_sum"] for e in phase) / sum(e["steps"] for e in phase)
        test["whole_run_gradient_payload_bytes"] = sum(e["gradient_bytes_sum"] for e in history)
        output[name] = test
        predictions[name] = scores.argmax(axis=1)
        columns = ["member", "label", "prediction", *(f"p_{i}" for i in range(classes))]
        rows = [{"member": record.member, "label": label, "prediction": prediction,
                 **{f"p_{i}": value for i, value in enumerate(score)}}
                for record, label, prediction, score in zip(splits["test"], labels, predictions[name], scores)]
        if routes is not None:
            columns.append("selected_experts")
            for row, routed in zip(rows, routes):
                row["selected_experts"] = "|".join(EXPERT_NAMES[index] for index, active in zip(model.expert_ids, routed) if active)
        write_csv(model_directory / "predictions.csv", columns, rows)
        write_json(model_directory / "validation.json", validation)
        write_json(model_directory / "training.json", history)
        if args.checkpoints:
            torch.save({"state_dict": model.state_dict(), "model": name, "classes": classes,
                        "top_k": args.top_k, "expert_ids": getattr(model, "expert_ids", None)}, model_directory / "model.pt")
        print(f"{name} test accuracy={test['accuracy']:.4f} sensitivity={test['sensitivity']:.4f} "
              f"specificity={test['specificity']:.4f}", flush=True)
    for name, test in output.items():
        for reference in ("dense", "efficientnet"):
            if reference in output:
                test[f"active_pct_{reference}"] = 100 * test["mean_active_parameters"] / output[reference]["mean_active_parameters"]
                test[f"communication_pct_{reference}"] = 100 * test["mean_gradient_payload_bytes"] / output[reference]["mean_gradient_payload_bytes"]
        write_json(directory / name / "test.json", test)
    if "pruned" in predictions and args.split_mode == "image":
        pairs = {reference: paired_accuracy_interval(labels, predictions["pruned"], predictions[reference], seed)
                 for reference in ("moe", "random-pruned", "dense", "efficientnet") if reference in predictions}
        write_json(directory / "paired_accuracy.json", pairs)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate sparse MedMoE-Histo and dense baselines")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data", required=True, type=Path, help="Official ZIP or extracted image folder")
    parser.add_argument("--out", required=True, type=Path, help="New output directory")
    parser.add_argument("--task", choices=("multiclass", "binary"), default="multiclass")
    parser.add_argument("--models", nargs="+", choices=("dense", *MOE_MODELS, *BASELINES), default=("dense", *MOE_MODELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 29, 43))
    parser.add_argument("--split-seed", type=int, default=2027)
    parser.add_argument("--split-mode", choices=("image", "slide"), default="image")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--prune-count", type=int, default=2)
    parser.add_argument("--prune-threshold", type=float, default=1.0)
    parser.add_argument("--balance", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--checkpoints", action="store_true")
    parser.add_argument("--pruning-sweep", action="store_true", help="Evaluate every valid removal count on validation at warm-up")
    args = parser.parse_args()
    if not 1 <= args.top_k <= len(EXPERT_NAMES) or not 0 <= args.prune_count <= len(EXPERT_NAMES) - args.top_k:
        parser.error("top-k and pruning must retain at least k experts")
    if not 0 <= args.prune_threshold <= 1 or args.balance < 0 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("invalid optimizer or pruning settings")
    if args.epochs < 1 or min(args.batch_size, args.threads) < 1 or args.image_size < 32 or args.image_size % 32:
        parser.error("positive training sizes and an image size divisible by 32 are required")
    if any(name in MOE_MODELS for name in args.models) and not 1 <= args.warmup < args.epochs:
        parser.error("MoE requires 1 <= warmup < epochs")
    if args.pruning_sweep and not any(name in MOE_MODELS for name in args.models):
        parser.error("pruning sweep requires at least one MoE variant")
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        parser.error("models and seeds must be distinct")
    if min(args.seeds) < 0 or max(args.seeds) >= 2**32 - 1 or args.limit_per_class < 0:
        parser.error("invalid seeds or class limit")
    if args.split_mode == "slide" and (args.dataset != "kather" or args.limit_per_class):
        parser.error("slide mode requires Kather data and uses all indexed images")
    return args


def main():
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    source = ImageSource(args.data)
    try:
        records, stats = read_index(source, args.dataset, args.allow_subset)
        slide_groups = None
        if args.split_mode == "slide":
            splits, slide_groups = slide_split(records, args.split_seed)
        else:
            splits = stratified_split(records, args.split_seed, args.limit_per_class)
        original = sorted({record.label for record in records})
        labels = {name: index for index, name in enumerate(original)}
        names = original
        if args.task == "binary":
            positive = {name.casefold() for name in DATASETS[args.dataset]["positive"]}
            labels = {name: int(name.casefold() in positive) for name in original}
            if set(labels.values()) != {0, 1}:
                raise ValueError(f"Binary mapping for {args.dataset} requires positive class {sorted(positive)} and negative classes")
            names = ["rest", "|".join(name for name in original if labels[name])]
        args.out.mkdir(parents=True, exist_ok=False)
        manifest_sha = write_manifest(args.out / "split.csv", splits, labels)
        metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        versions = {dep: importlib.metadata.version(dep) for dep in ("numpy", "Pillow", "torch")}
        if any(name in BASELINES for name in args.models):
            versions.update(timm=importlib.metadata.version("timm"), torchvision=importlib.metadata.version("torchvision"))
        metadata.update(dataset_index=stats, class_names=names, input_label_mapping=labels, split_sha256=manifest_sha,
                        split_sizes={name: len(split) for name, split in splits.items()}, slide_groups=slide_groups,
                        python=platform.python_version(), platform=platform.platform(), dependencies=versions,
                        model_identifiers={name: BASELINES[name] for name in args.models if name in BASELINES})
        write_json(args.out / "protocol.json", metadata)
        print(f"{args.dataset} {args.task}: {metadata['split_sizes']}, manifest={manifest_sha}", flush=True)
        cache = None
        if not args.no_cache:
            cache = {record.member: decode(source, record, args.image_size) for split in splits.values() for record in split}
        results = {seed: run_seed(source, splits, labels, args, seed, args.out / f"seed-{seed}", cache, len(names))
                   for seed in args.seeds}
        summarize(args.out, results)
    finally:
        source.close()


if __name__ == "__main__":
    main()
