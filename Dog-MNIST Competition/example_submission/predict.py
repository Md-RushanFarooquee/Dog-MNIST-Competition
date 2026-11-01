"""Example Dog-MNIST submission: a small CNN trained from scratch."""
import numpy as np
import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        def block(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1),
                                 nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def preprocess(images):
    """uint8 (N, 32, 32, 3) RGB -> float tensor (N, 3, 32, 32) in [0, 1]."""
    return torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0


def load_model():
    model = SmallCNN()
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    return model.eval()


@torch.no_grad()
def predict(model, images):
    x = preprocess(images)
    return np.concatenate([model(b).argmax(1).numpy() for b in x.split(256)])
