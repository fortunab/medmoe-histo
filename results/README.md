# Verified experiment results

All main experiments use six epochs, including one MoE warm-up epoch, batch size 64, and 64-pixel inputs. The Kather image split has three training seeds for the compact models. The other comparisons, including EfficientNet-B0 from scratch, use seed 17. All results were produced on a CPU.

Metrics are test scores. Active weights are per image. Gradient payload is a selective FP32 tensor-byte estimate per update in the reported training phase. Values are decimal thousands in the last two columns.

## Kather multiclass, image split, top-4

| Model | Seeds | Accuracy | Sensitivity | Specificity | Active weights (k) | Gradient payload (kB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense CNN | 3 | 0.813 ± 0.016 | 0.813 | 0.973 | 67.0 | 268.1 |
| MoE | 3 | 0.811 ± 0.002 | 0.811 | 0.973 | 66.0 | 331.6 |
| Usage-pruned | 3 | 0.808 ± 0.006 | 0.808 | 0.973 | 76.5 | 305.9 |
| Random-pruned | 3 | 0.808 ± 0.006 | 0.808 | 0.973 | 63.0 | 251.9 |
| EfficientNet-B0 | 1 | 0.703 | 0.703 | 0.958 | 4017.8 | 16071.2 |

## Kather multiclass, slide split, top-4

| Model | Seeds | Accuracy | Sensitivity | Specificity | Active weights (k) | Gradient payload (kB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense CNN | 1 | 0.645 | 0.679 | 0.947 | 67.0 | 268.1 |
| MoE | 1 | 0.717 | 0.750 | 0.958 | 69.0 | 331.6 |
| Usage-pruned | 1 | 0.729 | 0.759 | 0.960 | 60.9 | 243.5 |
| Random-pruned | 1 | 0.712 | 0.748 | 0.957 | 73.0 | 292.1 |

## Kather multiclass, image split, top-2

| Model | Seeds | Accuracy | Sensitivity | Specificity | Active weights (k) | Gradient payload (kB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MoE | 1 | 0.801 | 0.801 | 0.972 | 40.4 | 331.0 |
| Usage-pruned | 1 | 0.818 | 0.818 | 0.974 | 43.6 | 288.8 |
| Random-pruned | 1 | 0.791 | 0.791 | 0.970 | 42.9 | 292.1 |

## HyperKvasir multiclass, top-4

| Model | Seeds | Accuracy | Sensitivity | Specificity | Active weights (k) | Gradient payload (kB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense CNN | 1 | 0.781 | 0.447 | 0.990 | 68.0 | 272.0 |
| MoE | 1 | 0.771 | 0.426 | 0.989 | 68.0 | 348.3 |
| Usage-pruned | 1 | 0.768 | 0.424 | 0.989 | 76.2 | 304.8 |
| Random-pruned | 1 | 0.768 | 0.424 | 0.989 | 76.2 | 304.8 |

## Kvasir v2 multiclass, top-4

| Model | Seeds | Accuracy | Sensitivity | Specificity | Active weights (k) | Gradient payload (kB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense CNN | 1 | 0.700 | 0.700 | 0.957 | 67.0 | 268.1 |
| MoE | 1 | 0.696 | 0.695 | 0.957 | 67.7 | 331.6 |
| Usage-pruned | 1 | 0.696 | 0.695 | 0.957 | 73.0 | 292.1 |
| Random-pruned | 1 | 0.696 | 0.695 | 0.957 | 73.0 | 292.1 |

## Kather binary, top-4

| Model | Seeds | Accuracy | Sensitivity | Specificity | Active weights (k) | Gradient payload (kB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense CNN | 1 | 0.959 | 0.787 | 0.983 | 66.6 | 266.5 |
| MoE | 1 | 0.963 | 0.851 | 0.979 | 58.4 | 324.9 |
| Usage-pruned | 1 | 0.964 | 0.894 | 0.974 | 55.4 | 221.5 |
| Random-pruned | 1 | 0.961 | 0.894 | 0.971 | 71.8 | 287.0 |

## HyperKvasir binary, top-4

| Model | Seeds | Accuracy | Sensitivity | Specificity | Active weights (k) | Gradient payload (kB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense CNN | 1 | 0.994 | 0.935 | 1.000 | 66.6 | 266.5 |
| MoE | 1 | 0.985 | 0.955 | 0.988 | 52.4 | 324.9 |
| Usage-pruned | 1 | 0.977 | 0.961 | 0.979 | 55.4 | 221.5 |

## Validation evidence

A repeated Kather seed-17 run produced byte-identical test prediction files for the unpruned and usage-pruned MoE. Per-run JSON files include class support, confusion matrices, routing utilization, and parameter denominators. The slide-disjoint split used slides 09 and 10 for test and 02 and 07 for validation.

The two endoscopy multiclass random draws selected the same expert subset as the utilization rule. Kvasir v2 had 13 duplicate images removed, including two with conflicting labels. Fourteen distinct, byte-identical files remain common to HyperKvasir and Kvasir v2. Their models and splits were evaluated separately.

- Seed-17 `moe` prediction SHA-256 `2acc83fd5311877baad0c725d7418794f4ab6f43b4607b7bb5b3b40ee5838b43`
- Seed-17 `pruned` prediction SHA-256 `1aff14023d9a678918e3ae2a5ffe04dc87b95232074c5e7ca9fcef19fd7e3485`

The held-out labels and reported predictive metrics can be rechecked from the bundled outputs.

```shell
python -m medmoe.verify results/kather-image results/kather-image-extra-seeds results/kather-slide results/kather-top2 results/hyper-multiclass results/kvasir-v2-multiclass results/kather-binary results/hyper-binary results/kather-pruning-sweep results/kather-repeat
```
