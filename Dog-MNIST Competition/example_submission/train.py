"""Train the example model from scratch on the Dog-MNIST train set only."""
import sys

import torch
from torch import nn

from predict import SmallCNN, preprocess

sys.path.insert(0, "..")
from evaluate import load_dog_mnist  # noqa: E402

torch.manual_seed(0)
X, y = load_dog_mnist("../train-images-idx3-ubyte", "../train-labels-idx1-ubyte")
X = preprocess(X.reshape(-1, 32, 32, 3))
y = torch.from_numpy(y).long()

model = SmallCNN()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(15):
    model.train()
    perm = torch.randperm(len(X))
    total = 0.0
    for i in range(0, len(X), 64):
        idx = perm[i:i + 64]
        xb = X[idx]
        flip = torch.rand(len(xb)) < 0.5            # random horizontal flip
        xb[flip] = xb[flip].flip(3)
        loss = loss_fn(model(xb), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    print(f"epoch {epoch + 1:2d}  loss {total / len(X):.4f}")

torch.save(model.state_dict(), "model.pt")
print("saved model.pt")
