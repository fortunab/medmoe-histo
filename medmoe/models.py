import copy

import torch
from torch import nn


BASELINES = {
    "efficientnet": "efficientnet_b0.ra_in1k",
    "vit-small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "swin-tiny": "swin_tiny_patch4_window7_224.ms_in1k",
    "dinov2": "vit_small_patch14_dinov2.lvd142m",
    "clip": "vit_base_patch32_clip_224.openai",
}

EXPERT_NAMES = ("texture_3x3", "texture_5x5", "context", "wide_context", "lightweight", "channel_mlp")


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def conv(in_channels, out_channels, kernel=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.GELU(),
    )


def backbone():
    return nn.Sequential(conv(3, 16, stride=2), conv(16, 24, stride=2), conv(24, 32, stride=2))


def pooled_head(width, classes):
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, classes))


class ContextExpert(nn.Module):
    def __init__(self, width, classes):
        super().__init__()
        self.project = nn.Linear(32, width)
        self.position = nn.Parameter(torch.zeros(1, 16, width))
        self.encoder = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0, batch_first=True)
        self.classify = nn.Linear(width, classes)

    def forward(self, features):
        pool_size = (features.shape[-2] // 4, features.shape[-1] // 4)
        pooled = (nn.functional.avg_pool2d(features, pool_size) if features.is_cuda
                  else nn.functional.adaptive_avg_pool2d(features, (4, 4)))
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.encoder(self.project(tokens) + self.position)
        return self.classify(tokens.mean(dim=1))


def make_experts(classes):
    return nn.ModuleList([
        nn.Sequential(conv(32, 32), pooled_head(32, classes)),
        nn.Sequential(conv(32, 32, kernel=5), pooled_head(32, classes)),
        ContextExpert(32, classes),
        ContextExpert(48, classes),
        pooled_head(32, classes),
        nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 96), nn.GELU(), nn.Linear(96, classes)),
    ])


class MedMoE(nn.Module):
    def __init__(self, classes, top_k, expert_ids=tuple(range(len(EXPERT_NAMES)))):
        super().__init__()
        self.stem = backbone()
        self.router = nn.Linear(32, len(expert_ids))
        self.classes = classes
        self.top_k = top_k
        self.expert_ids = tuple(expert_ids)
        if not 1 <= top_k <= len(expert_ids):
            raise ValueError("top_k must be between 1 and the number of retained experts")
        experts = make_experts(classes)
        self.experts = nn.ModuleList(experts[index] for index in expert_ids)

    def forward(self, images):
        features = self.stem(images)
        scores = self.router(features.mean(dim=(2, 3)))
        probabilities = scores.softmax(dim=1)
        selected = scores.argsort(dim=1, descending=True, stable=True)[:, :self.top_k]
        weights = scores.gather(1, selected).softmax(dim=1)
        logits = features.new_zeros(images.shape[0], self.classes)
        for expert_index, expert in enumerate(self.experts):
            rows, slots = torch.where(selected == expert_index)
            if rows.numel():
                output = expert(features.index_select(0, rows))
                weighted = output * weights[rows, slots, None]
                # Rows are unique for each expert. Each expert receives only its assigned samples.
                logits = logits + torch.zeros_like(logits).index_copy(0, rows, weighted.to(logits.dtype))
        usage = torch.nn.functional.one_hot(selected, len(self.experts)).sum(dim=1)
        balance = len(self.experts) * ((usage.float().mean(dim=0) / self.top_k) * probabilities.mean(dim=0)).sum()
        return logits, usage, balance

    def pruned(self, keep):
        keep = sorted(keep)
        if len(keep) < self.top_k or len(keep) != len(set(keep)) or any(i not in range(len(self.experts)) for i in keep):
            raise ValueError("retain distinct existing experts and at least top_k experts")
        model = copy.deepcopy(self)
        model.experts = nn.ModuleList(model.experts[index] for index in keep)
        model.expert_ids = tuple(self.expert_ids[index] for index in keep)
        last = self.router
        model.router = nn.Linear(last.in_features, len(keep), device=last.weight.device, dtype=last.weight.dtype)
        with torch.no_grad():
            model.router.weight.copy_(last.weight[keep])
            model.router.bias.copy_(last.bias[keep])
        return model


def baseline(name, classes, pretrained):
    if name == "dense":
        return nn.Sequential(backbone(), conv(32, 64), conv(64, 64), pooled_head(64, classes))
    import timm

    options = {"num_classes": classes, "pretrained": pretrained}
    if name == "dinov2":
        options["img_size"] = 224
    return timm.create_model(BASELINES[name], **options)
