# Mapping to the original manuscript

| Earlier manuscript | Reconstructed experiment |
| --- | --- |
| DS1 colorectal histology | Original Kather archive with all 5,000 tiles. A stratified image split is paired with a fixed slide-disjoint robustness split. |
| DS2 named Kvasir v2 and linked to HyperKvasir | HyperKvasir is the primary endoscopy benchmark. Kvasir v2 is run separately with its own class mapping. |
| $E=6$ and top-four routing | Image-level gather, expert execution and scatter for $k=4$. A Kather top-two ablation tests sparsity after two experts have been removed. |
| Early expert utilization | Selection frequency during ordinary warm-up. Structural removal takes effect in the expert module list and router output. The control removes the same number of experts using a seeded random draw. |
| Curriculum, heterogeneous experts and load balancing | One shared warm-up followed by paired continuation. Heads are CNNs, self-attention, a linear head and a channel MLP. The load-balancing loss is specified in the manuscript and code. |
| Accuracy, sensitivity and specificity | Predictions and confusion matrices per dataset and task. Multiclass rates use macro one-vs-rest aggregation. Binary rates use the positive-class mapping in each protocol file. |
| Relative active parameters and communication in Table 4 | Exact per-image active parameter counts and an explicit per-training-step selective FP32 gradient payload. The same dense denominators are exported where available. Real distributed traffic is outside this experiment. |
| Pruning and utilization in Figures 4 and 5 | `validation_pruning_sweep.csv` records feasible removals at fixed $k$. `warmup.json`, predicted routes and test utilization give the source data for routing plots. With $k=4$, the sum of expert selection probabilities is four. Share of all assignments is each selection probability divided by four. |
| Baseline tables | EfficientNet-B0 implemented and trained on Kather in the supplied run. ViT-Small, Swin-Tiny, DINOv2-Small and CLIP ViT-B/32 have executable model configurations. Their forward shapes were verified. The pretrained runs have their own budget and need separate evidence. |

The Overleaf source contains literal values in Tables 1–5 and literal plot coordinates in Figures 4 and 5. It has no run logs, image assignments, executable models, seeds, checkpoints, or script that connects those values to experiments. The reported multiclass accuracy is 0.930 in Table 1 and 0.900 in Tables 3 and 5, with no run identifiers explaining the difference. The 98.3% accuracy in Table 2 has no task label that establishes comparability with its dense and unpruned rows. The reconstructed results use the supplied run records as evidence and retain dataset and task labels throughout.
