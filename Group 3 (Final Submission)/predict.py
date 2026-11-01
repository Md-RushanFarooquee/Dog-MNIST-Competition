"""Dog-MNIST submission: 12-member logit-averaged ensemble of ResNet(32, 64, 128).

Members: 4 training recipes (none, mixup, Cutout, CutMix) x seeds 0-2, each trained from
scratch on the Dog-MNIST train set only (480 epochs, dropout 0.2 during training).
Dropout has no parameters and is inactive at inference, so it is not part of this file.
Each member keeps the mean/std computed on its own training data; predictions average the
logits of the original and the horizontally flipped image, then average over the members.
CPU only, no downloads, relative paths only.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10
IMAGE_SHAPE = (32, 32, 3)
WIDTHS = (32, 64, 128)
BATCH_SIZE = 128
MEMBERS_DIR = "members"
MEMBER_NAMES = ("DROP_EP480", "DROP_EP480_MIX", "DROP_EP480_CUTOUT", "DROP_EP480_CUTMIX")
MEMBER_SEEDS = (0, 1, 2)
FLIP_AVERAGE = True
PARAMS_PER_MEMBER = 308650


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + (x if self.short is None else self.short(x)))


class ResNet(nn.Module):
    def __init__(self, widths, num_classes):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        blocks, cin = [], widths[0]
        for s, w in enumerate(widths):
            blocks.append(Block(cin, w, 1 if s == 0 else 2))
            cin = w
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        return self.fc(self.blocks(self.stem(x)).mean(dim=(2, 3)))


class Ensemble(nn.Module):
    """Holds all members in one nn.ModuleList so evaluate.py counts every parameter."""

    def __init__(self, members, means, stds):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.means = list(means)  # plain tensors, per-member normalisation constants (not parameters)
        self.stds = list(stds)

    def forward(self, x):
        # x: float32 NCHW in [0, 1]
        total = None
        for m, mean, std in zip(self.members, self.means, self.stds):
            xn = (x - mean) / std
            logits = m(xn)
            if FLIP_AVERAGE:
                logits = (logits + m(xn.flip(3))) / 2
            total = logits if total is None else total + logits
        return total / len(self.members)


def _member_path(name, seed):
    return os.path.join(MEMBERS_DIR, f"final_{name}_s{seed}.pt")


def load_model():
    members, means, stds = [], [], []
    for name in MEMBER_NAMES:
        for seed in MEMBER_SEEDS:
            path = _member_path(name, seed)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"missing member weights: {path}")
            ck = torch.load(path, map_location="cpu", weights_only=True)
            if ck["model_key"] != "resnet_32_64_128":
                raise ValueError(f"{path}: unexpected model_key {ck['model_key']}")
            net = ResNet(WIDTHS, NUM_CLASSES)
            net.load_state_dict(ck["state_dict"])  # strict: keys and shapes must match exactly
            net.eval()
            members.append(net)
            means.append(ck["mean"].float().reshape(1, 3, 1, 1))
            stds.append(ck["std"].float().reshape(1, 3, 1, 1))
    model = Ensemble(members, means, stds)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    expected = PARAMS_PER_MEMBER * len(MEMBER_NAMES) * len(MEMBER_SEEDS)
    if n_params != expected:
        raise ValueError(f"parameter count {n_params} != expected {expected}")
    return model


@torch.no_grad()
def predict(model, images):
    x = np.asarray(images)
    if x.ndim != 4 or tuple(x.shape[1:]) != IMAGE_SHAPE:
        raise ValueError(f"expected images of shape (N, 32, 32, 3), got {x.shape}")
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float().div(255.0)
    model.eval()
    out = [model(x[i:i + BATCH_SIZE]).argmax(dim=1) for i in range(0, x.shape[0], BATCH_SIZE)]
    return torch.cat(out).numpy().astype(np.int64)