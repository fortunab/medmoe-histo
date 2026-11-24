import hashlib
import io
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image

from medmoe.data import ImageSource, read_index, stratified_split
from medmoe.metrics import classification, paired_accuracy_interval
from medmoe.models import MedMoE, parameter_count
from medmoe.run import retain_by_usage
from medmoe.verify import verify


def test_sparse_dispatch_and_gradient():
    torch.manual_seed(17)
    model = MedMoE(classes=3, top_k=2)
    with torch.no_grad():
        model.router.weight.zero_()
        model.router.bias.copy_(torch.tensor([4.0, 3.0, 0.0, -1.0, -2.0, -3.0]))
    hits = {}
    hooks = [expert.register_forward_hook(lambda _, inputs, output, index=index: hits.update({index: len(inputs[0])}))
             for index, expert in enumerate(model.experts)]
    images = torch.randn(3, 3, 32, 32)
    logits, usage, balance = model(images)
    (logits.square().sum() + balance).backward()
    assert hits == {0: 3, 1: 3}
    assert usage.sum(dim=1).tolist() == [2, 2, 2]
    assert all(p.grad is not None for p in model.experts[0].parameters())
    assert all(p.grad is None for p in model.experts[2:].parameters())
    for hook in hooks:
        hook.remove()
    with torch.no_grad():
        features = model.stem(images)
        weights = torch.tensor([4.0, 3.0]).softmax(dim=0)
        torch.testing.assert_close(logits, weights[0] * model.experts[0](features) + weights[1] * model.experts[1](features))


def test_pruning_reuses_weights_and_removes_capacity():
    torch.manual_seed(17)
    original = MedMoE(classes=3, top_k=2).eval()
    retained = [0, 2, 5]
    pruned = original.pruned(retained).eval()
    images = torch.randn(4, 3, 32, 32)
    with torch.inference_mode():
        original.router.bias[[1, 3, 4]] = -1e5
        expected, expected_usage, _ = original(images)
        actual, usage, _ = pruned(images)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(usage, expected_usage[:, retained])
    assert pruned.expert_ids == tuple(retained)
    assert parameter_count(pruned) < parameter_count(original)
    assert retain_by_usage([0.4, 0.6, 0.1, 0.1], 2) == [0, 1]
    assert retain_by_usage([0.4, 0.6, 0.1, 0.2], 2, threshold=0.15) == [0, 1, 3]


def test_experts_receive_their_own_rows_in_a_mixed_batch():
    model = MedMoE(classes=3, top_k=1)
    model.stem = torch.nn.Identity()
    with torch.no_grad():
        model.router.weight.zero_()
        model.router.bias.zero_()
        model.router.weight[0, 0] = 1
        model.router.weight[1, 0] = -1
    features = torch.zeros(3, 32, 4, 4)
    features[:, 0, :, :] = torch.tensor([2.0, -1.0, -3.0])[:, None, None]
    hits = {}
    for index, expert in enumerate(model.experts):
        expert.register_forward_pre_hook(lambda _, inputs, index=index: hits.update({index: inputs[0][:, 0, 0, 0].tolist()}))
    _, usage, _ = model(features)
    assert hits == {0: [2.0], 1: [-1.0, -3.0]}
    assert usage.argmax(dim=1).tolist() == [0, 1, 1]


def test_binary_metrics_and_paired_interval():
    target = torch.tensor([0, 0, 0, 0, 1, 1])
    prediction = torch.tensor([0, 0, 0, 1, 0, 1])
    metrics = classification(target, prediction, 2)
    assert metrics["confusion_matrix"] == [[3, 1], [1, 1]]
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.75
    assert metrics["accuracy"] == 4 / 6
    comparison = paired_accuracy_interval(target.numpy(), prediction.numpy(), prediction.numpy(), 17)
    assert comparison["image_bootstrap_ci95"] == [0.0, 0.0]


def test_multiclass_macro_and_missing_validation_class():
    true = torch.tensor([0, 0, 1, 1, 2, 2])
    predicted = torch.tensor([0, 1, 1, 2, 2, 2])
    metrics = classification(true, predicted, 3)
    assert np.isclose(metrics["sensitivity"], 2 / 3)
    assert np.isclose(metrics["specificity"], 5 / 6)
    validation = classification(true[:4], predicted[:4], 3, allow_missing=True)
    assert validation["per_class_sensitivity"] == [0.5, 0.5, None]


def test_content_deduplication_and_split(tmp_path):
    path = tmp_path / "tiles.zip"
    with ZipFile(path, "w") as archive:
        for label in range(2):
            for index in range(7):
                image = Image.new("RGB", (8, 8), (label * 127, index, 0))
                with io.BytesIO() as stream:
                    image.save(stream, format="PNG")
                    archive.writestr(f"class{label}/{index}.png", stream.getvalue())
                    if index == 0 and label == 0:
                        duplicate = stream.getvalue()
        archive.writestr("class0/dup.png", duplicate)
    source = ImageSource(path)
    try:
        records, stats = read_index(source, "kather", allow_subset=True)
    finally:
        source.close()
    assert stats["same_label_duplicate_images_dropped"] == 1
    groups = stratified_split(records, 2027)
    assert [len(group) for group in groups.values()] == [10, 2, 2]
    hashes = [record.sha256 for group in groups.values() for record in group]
    assert len(set(hashes)) == len(hashes)
    assert hashlib.sha256(duplicate).hexdigest() in hashes
    assert stratified_split(records[::-1], 2027) == groups


def test_train_and_verify_public_entry_points(tmp_path):
    data = tmp_path / "images"
    for label in ("01_TUMOR", "02_STROMA"):
        folder = data / label
        folder.mkdir(parents=True)
        for index in range(8):
            Image.new("RGB", (32, 32), (index, 100 * (label == "01_TUMOR"), 0)).save(folder / f"{index}.png")
    out = tmp_path / "run"
    subprocess.run([sys.executable, "-m", "medmoe.run", "--dataset", "kather", "--data", str(data),
                    "--out", str(out), "--task", "binary", "--allow-subset", "--seeds", "17", "--epochs", "2",
                    "--warmup", "1", "--image-size", "32", "--batch-size", "4", "--threads", "1"], check=True)
    verify(out)
