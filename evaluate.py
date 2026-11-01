import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import importlib.util
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROWS, COLS, CH = 32, 32, 3
NUM_CLASSES = 10


def load_dog_mnist(images_path, labels_path):
    with open(labels_path, "rb") as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    if magic != 2049 or labels.size != n_labels:
        raise ValueError(f"Bad label file {labels_path}: magic={magic}, header count={n_labels}, bytes={labels.size}")
    with open(images_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        pixels = np.fromfile(f, dtype=np.uint8)
    if magic != 2051 or n != n_labels or (rows, cols) != (ROWS, COLS) or pixels.size != n * ROWS * COLS * CH:
        raise ValueError(f"Bad image file {images_path}: magic={magic}, n={n}, rows={rows}, cols={cols}, bytes={pixels.size}, labels={n_labels}")
    return pixels.reshape(n, ROWS * COLS * CH), labels


def import_predict(folder):
    spec = importlib.util.spec_from_file_location("submission_predict", folder / "predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameter_info(model, module):
    try:
        import torch.nn as nn
    except ImportError:
        nn = None
    if nn is not None and isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters()), "PyTorch, counted by evaluator"
    if hasattr(module, "count_parameters"):
        return int(module.count_parameters(model)), "from count_parameters(), to be checked by hand"
    raise RuntimeError("model is not an nn.Module and predict.py defines no count_parameters()")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()

    folder = Path(args.submission).resolve()
    images_path = Path(args.images).resolve()
    labels_path = Path(args.labels).resolve()
    flat, labels = load_dog_mnist(images_path, labels_path)
    images = flat.reshape(-1, ROWS, COLS, CH)
    n = len(labels)

    os.chdir(folder)
    sys.path.insert(0, str(folder))
    module = import_predict(folder)
    model = module.load_model()

    t0 = time.perf_counter()
    preds = np.asarray(module.predict(model, images))
    elapsed = time.perf_counter() - t0

    if preds.shape != (n,):
        raise ValueError(f"predict() returned shape {preds.shape}, expected ({n},)")
    if preds.dtype.kind not in "iu":
        raise ValueError(f"predict() returned dtype {preds.dtype}, expected integer labels")
    if preds.min() < 0 or preds.max() >= NUM_CLASSES:
        raise ValueError(f"predict() returned labels outside 0-{NUM_CLASSES - 1}: min={preds.min()}, max={preds.max()}")

    correct = int((preds == labels).sum())
    present = np.unique(labels)
    mean_class_acc = float(np.mean([(preds[labels == c] == c).mean() for c in present]))
    n_params, how = parameter_info(model, module)

    print(f"Submission  : {folder.name}")
    print(f"Data        : {images_path.name} ({n} images)")
    print(f"Accuracy    : {correct / n:.4f} ({correct}/{n})")
    print(f"Mean class accuracy : {mean_class_acc:.4f}")
    print(f"Parameters  : {n_params:,} ({how})")
    print(f"Predict time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
