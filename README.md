# MedMoE-Histo

Reference implementation of sparse mixture-of-experts classification on colorectal histology and gastrointestinal endoscopy. The model routes each image to a subset of six heterogeneous heads. It counts assignments during ordinary warm-up, removes low-use experts and the corresponding router rows, and continues training within the same epoch budget. Unpruned and random-pruned controls start from the same warm-up state.

The experiment suite contains a small dense CNN and configurable EfficientNet-B0, ViT-Small, Swin-Tiny, DINOv2-Small and CLIP ViT-B/32 baselines. See [results](results/README.md) for the supplied runs and the [source audit](SOURCE-AUDIT.md) for the mapping to the earlier manuscript.

## Install

Run the commands from the repository root with Python 3.12. For CPU runs, install PyTorch from its CPU wheel index before installing the pinned dependencies.

```shell
python -m venv .venv
# On PowerShell: .venv\Scripts\Activate.ps1
# On bash: source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0
python -m pip install -r requirements-vision.txt -r requirements-test.txt
python -m pytest -q
```

The core models need only `requirements.txt` and the CPU torch wheel. The `timm` baselines use `requirements-vision.txt`. For a CUDA run, install matching torch and torchvision builds from the [PyTorch wheel index](https://pytorch.org/get-started/locally/) and pass `--device cuda`. The bundled evidence was produced on CPU. Deterministic operations are enabled, and exact numerical agreement can vary across hardware and library builds.

## Get the data

```shell
python -m medmoe.download kather
python -m medmoe.download hyper-kvasir
python -m medmoe.download kvasir-v2
```

The runner reads a ZIP directly or accepts an extracted directory with images inside class directories. It checks the expected raw class and image counts before deduplication. `--allow-subset` permits a separately labeled pilot and `--limit-per-class` takes a deterministic per-class subsample. The three public sources are [Kather CRC-2016](https://doi.org/10.5281/zenodo.53169), [HyperKvasir](https://doi.org/10.1038/s41597-020-00622-y) and [Kvasir v2](https://datasets.simula.no/kvasir/). `data_sha256.json` records the checksums of the exact publisher archives used. HyperKvasir is the 23-class endoscopy collection linked from the manuscript. Kvasir v2 is a separate eight-class archive. The run reports identify each dataset explicitly.

## Train and evaluate

These commands use the dataset image split, train for six epochs, and evaluate the final epoch on the held-out test set. Set `--out` to a fresh path for each run.

```shell
python -m medmoe.run --dataset kather --data data/Kather_texture_2016_image_tiles_5000.zip --out runs/kather-image --models dense moe pruned random-pruned efficientnet --seeds 17 --epochs 6 --warmup 1 --batch-size 64 --threads 3
python -m medmoe.run --dataset kather --data data/Kather_texture_2016_image_tiles_5000.zip --out runs/kather-image-extra-seeds --models dense moe pruned random-pruned --seeds 29 43 --epochs 6 --warmup 1 --batch-size 64 --threads 2
python -m medmoe.run --dataset kather --data data/Kather_texture_2016_image_tiles_5000.zip --out runs/kather-slide --split-mode slide --seeds 17 --epochs 6 --warmup 1 --batch-size 64 --threads 2
python -m medmoe.run --dataset kather --data data/Kather_texture_2016_image_tiles_5000.zip --out runs/kather-top2 --models moe pruned random-pruned --top-k 2 --seeds 17 --epochs 6 --warmup 1 --batch-size 64 --threads 2
python -m medmoe.run --dataset hyper-kvasir --data data/hyper-kvasir-labeled-images.zip --out runs/hyper-multiclass --seeds 17 --epochs 6 --warmup 1 --batch-size 64 --threads 3
python -m medmoe.run --dataset kvasir-v2 --data data/kvasir-dataset-v2.zip --out runs/kvasir-v2-multiclass --seeds 17 --epochs 6 --warmup 1 --batch-size 64 --threads 2
python -m medmoe.run --dataset kather --data data/Kather_texture_2016_image_tiles_5000.zip --out runs/kather-binary --task binary --seeds 17 --epochs 6 --warmup 1 --batch-size 64 --threads 2
python -m medmoe.run --dataset hyper-kvasir --data data/hyper-kvasir-labeled-images.zip --out runs/hyper-binary --task binary --models dense moe pruned --seeds 17 --epochs 6 --warmup 1 --batch-size 64 --threads 2
```

Use `--task binary` for Kather tumor versus other tissue or endoscopy polyps versus all other classes. The split is assigned before remapping, so both tasks share image assignments. Use `--models efficientnet vit-small swin-tiny dinov2 clip --pretrained` for pretrained baselines. Vision transformers use 224-pixel inputs, and the compact models and EfficientNet use `--image-size` (64 by default, must be divisible by 32). The pretrained and resolution settings belong in the comparison protocol. The default experiment uses three training seeds and 15 epochs when these values are omitted. `--checkpoints` saves trained states and their model identifiers. `--no-cache` decodes each batch from the input archive to reduce RAM usage.

Pass `--pruning-sweep` to evaluate each feasible removal count immediately after warm-up on validation. The sweep uses a fixed $k$, removes experts in increasing utilization order, and records both predictive and parameter metrics without further training between points. It does not change the removal count used for the test run.

## Validate the runs

```shell
python -m medmoe.verify runs/kather-image runs/kather-slide
python -m medmoe.paper_tables --run "Kather image split=runs/kather-image" "Kather image split=runs/kather-image-extra-seeds" --out generated
```

Each run saves a content-hashed split manifest, protocol and dependency versions, warm-up assignments and retained experts, train history, validation and test metrics, per-image probabilities and route IDs, per-seed metrics, and across-seed summaries. The verification command checks image and slide disjointness and recomputes scores from saved test predictions. For image-level splits, paired accuracy intervals use a fixed-seed, class-stratified image bootstrap. Those intervals describe uncertainty at image level and do not account for correlated images from the same slide or procedure.

The image split holds out approximately 15% per class for validation and 15% for test after byte-level deduplication. The slide robustness split uses the `CRC-Prim-HE-XX` prefix in Kather filenames. The `08_EMPTY` class appears on only two slides. The split requires all eight classes in both training and test, so this class is absent from slide-disjoint validation. Validation reports its unavailable sensitivity as null and averages sensitivity over represented classes. Test metrics require every class to be represented. The image-level endoscopy split has no patient or procedure grouping, and byte-deduplication cannot detect all near-duplicates.

## Read the metrics

- Accuracy uses all test images. Multiclass sensitivity and specificity use unweighted one-vs-rest class averages. Binary sensitivity and specificity use the declared positive class.
- Active parameters count the shared stem and router plus the parameters of the selected experts for each image. `total_parameters` includes all retained weights. At `E=6, k=4, E'=4`, all four retained experts run per image after pruning. At `k=2, E'=4`, pruning preserves sparse per-image execution.
- Gradient payload is the FP32 byte count of gradient tensors present in a training update. The headline mean covers the continuation phase for MoE variants and the full training phase for dense baselines. `whole_run_gradient_payload_bytes` includes the warm-up. These bytes assume selective transmission and exclude network protocol overhead and distributed synchronization. At batch size 64, many routes can activate almost every expert across the batch. Dense communication budgets can use `full_model_fp32_bytes`.
- Local inference timing uses model forward time after three warm-up forwards on the first batch. It excludes data decoding and host-to-device transfer and depends on batch size, device, and contention. Compare predictive metrics and parameter counts across runs independently of timing variation.
