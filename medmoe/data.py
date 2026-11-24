import csv
import hashlib
import io
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DATASETS = {
    "kather": {
        "images": 5000, "classes": 8, "positive": ("01_TUMOR",),
        "url": "https://zenodo.org/records/53169/files/Kather_texture_2016_image_tiles_5000.zip",
    },
    "hyper-kvasir": {
        "images": 10662, "classes": 23, "positive": ("polyps",),
        "url": "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip",
    },
    "kvasir-v2": {
        "images": 8000, "classes": 8, "positive": ("polyps",),
        "url": "https://datasets.simula.no/downloads/kvasir/kvasir-dataset-v2.zip",
    },
}
IMAGE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Record:
    member: str
    label: str
    sha256: str


class ImageSource:
    def __init__(self, path):
        self.path = Path(path)
        self.archive = ZipFile(path) if self.path.is_file() else None
        if self.archive is None and not self.path.is_dir():
            raise FileNotFoundError(path)

    def names(self):
        if self.archive is not None:
            return self.archive.namelist()
        return [path.relative_to(self.path).as_posix() for path in self.path.rglob("*") if path.is_file()]

    def read(self, name):
        if self.archive is not None:
            return self.archive.read(name)
        return (self.path / name).read_bytes()

    def close(self):
        if self.archive is not None:
            self.archive.close()


def read_index(source, dataset, allow_subset=False):
    groups = defaultdict(list)
    observed = 0
    for name in sorted(source.names()):
        path = PurePosixPath(name)
        if path.suffix.lower() not in IMAGE_EXTENSIONS or any(part.startswith((".", "__MACOSX")) for part in path.parts):
            continue
        if len(path.parts) < 2:
            raise ValueError(f"Image needs a class directory: {name}")
        observed += 1
        digest = hashlib.sha256(source.read(name)).hexdigest()
        groups[digest].append(Record(name, path.parent.name, digest))
    if not groups:
        raise ValueError("No supported images were found")
    labels = {record.label for records in groups.values() for record in records}
    expected = DATASETS[dataset]
    if not allow_subset and (observed != expected["images"] or len(labels) != expected["classes"]):
        raise ValueError(f"{dataset}: found {observed} images and {len(labels)} classes, expected {expected}. "
                         "Use the designated archive or explicitly pass --allow-subset for a pilot.")
    conflicts = {digest for digest, records in groups.items() if len({record.label for record in records}) > 1}
    records = sorted((records[0] for digest, records in groups.items() if digest not in conflicts), key=lambda r: r.member)
    stats = {
        "images_found": observed,
        "raw_classes": sorted(labels),
        "conflicting_duplicate_images_dropped": sum(len(groups[digest]) for digest in conflicts),
        "same_label_duplicate_images_dropped": sum(len(records) - 1 for key, records in groups.items() if key not in conflicts),
    }
    return records, stats


def stratified_split(records, seed, per_class_limit=0):
    by_class = defaultdict(list)
    for record in records:
        by_class[record.label].append(record)
    if len(by_class) < 2:
        raise ValueError("At least two classes are required")
    result = {split: [] for split in SPLITS}
    rng = random.Random(seed)
    for label, members in sorted(by_class.items()):
        sorted_members = sorted(members, key=lambda record: record.member)
        rng.shuffle(sorted_members)
        if per_class_limit:
            sorted_members = sorted_members[:per_class_limit]
        if len(sorted_members) < 3:
            raise ValueError(f"{label!r} needs at least three distinct images")
        holdout = max(1, round(len(sorted_members) * 0.15))
        result["test"].extend(sorted_members[:holdout])
        result["validation"].extend(sorted_members[holdout:holdout * 2])
        result["train"].extend(sorted_members[holdout * 2:])
    return {name: sorted(members, key=lambda r: r.member) for name, members in result.items()}


def slide_id(record):
    match = re.search(r"CRC-Prim-HE-(\d{2})", PurePosixPath(record.member).name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not recover Kather slide ID from {record.member}")
    return match.group(1)


def slide_split(records, seed):
    classes = sorted({record.label for record in records})
    groups = defaultdict(list)
    for record in records:
        groups[slide_id(record)].append(record)
    group_ids = sorted(groups)
    counts = {group: np.array([sum(r.label == label for r in groups[group]) for label in classes])
              for group in group_ids}
    totals = sum(counts.values())
    candidates = []
    for test in combinations(group_ids, 2):
        test_counts = sum(counts[group] for group in test)
        if (test_counts == 0).any():
            continue
        for validation in combinations(sorted(set(group_ids) - set(test)), 2):
            val_counts = sum(counts[group] for group in validation)
            if (totals - test_counts - val_counts == 0).any():
                continue
            deviation = abs(test_counts / totals - 0.15).sum() + abs(val_counts / totals - 0.15).sum()
            candidates.append((round(float(deviation), 10), test, validation))
    if not candidates:
        raise ValueError("Unable to put each Kather class in both slide-disjoint training and test sets")
    candidates.sort()
    best = [candidate for candidate in candidates if candidate[0] == candidates[0][0]]
    _, test, validation = random.Random(seed).choice(best)
    split_groups = {"train": sorted(set(group_ids) - set(test) - set(validation)),
                    "validation": list(validation), "test": list(test)}
    result = {name: sorted((record for group in ids for record in groups[group]), key=lambda r: r.member)
              for name, ids in split_groups.items()}
    return result, split_groups


def write_manifest(path, splits, label_ids):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "member", "original_class", "label", "sha256"])
        for split, records in splits.items():
            writer.writerows((split, r.member, r.label, label_ids[r.label], r.sha256) for r in records)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(source, record, size):
    with Image.open(io.BytesIO(source.read(record.member))) as image:
        rgb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return np.array(rgb)


class Tiles(Dataset):
    def __init__(self, source, records, label_ids, size, cache, mean, std, training):
        self.source = source
        self.records = records
        self.labels = [label_ids[record.label] for record in records]
        self.size = size
        self.cache = cache
        self.mean = torch.tensor(mean).reshape(3, 1, 1)
        self.std = torch.tensor(std).reshape(3, 1, 1)
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        pixels = self.cache[record.member] if self.cache is not None else decode(self.source, record, self.size)
        image = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255)
        if self.training and random.random() < 0.5:
            image = image.flip(-1)
        return (image - self.mean) / self.std, self.labels[index]
