# Dog-MNIST Competition

- Dog-MNIST is a small **colour** image-classification dataset in the classic MNIST file
  format. 
- Each image is a 32×32 RGB photo of a dog, and the task is to predict which of
  **10 breeds** it shows.

## What you receive

| Split | Images | Purpose |
|---|---|---|
| `train` | 1,979 | Train your model (you may split it further for validation) |
| `sample-test` | 48 | Only for checking that your code runs; **not** for training or tuning |

### Files

```
dog-mnist/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── sample-test-images-idx3-ubyte
├── sample-test-labels-idx1-ubyte
├── labels.txt                     # label index -> breed name
└── example_submission/           # a working submission to copy from
    ├── predict.py
    ├── train.py
    └── model.pt
```

## Classes

| Label | Breed | Train | Sample-test |
|---|---|---|---|
| 0 | Bedlington terrier | 217 | 5 |
| 1 | Boston bull | 216 | 5 |
| 2 | English setter | 195 | 5 |
| 3 | Italian greyhound | 220 | 5 |
| 4 | Labrador retriever | 204 | 5 |
| 5 | Norfolk terrier | 204 | 5 |
| 6 | Walker hound | 178 | 4 |
| 7 | boxer | 180 | 5 |
| 8 | golden retriever | 174 | 4 |
| 9 | wire-haired fox terrier | 191 | 5 |
| | **Total** | **1,979** | **48** |

## File format

The files follow the MNIST IDX layout, with one change: **each pixel has 3 bytes (R, G, B) instead of 1**.

**Image files** (`*-images-idx3-ubyte`)

| Offset | Type | Value |
|---|---|---|
| 0 | 32-bit int, big-endian | magic number `2051` |
| 4 | 32-bit int, big-endian | number of images *N* |
| 8 | 32-bit int, big-endian | rows = `32` |
| 12 | 32-bit int, big-endian | columns = `32` |
| 16 | unsigned byte × N·32·32·3 | pixel data |

Each image takes 3,072 bytes. The pixels are stored row by row, with the R, G and B values of each pixel
next to each other (height × width × channel order). Values range from 0 to 255.

**Label files** (`*-labels-idx1-ubyte`)

| Offset | Type | Value |
|---|---|---|
| 0 | 32-bit int, big-endian | magic number `2049` |
| 4 | 32-bit int, big-endian | number of labels *N* |
| 8 | unsigned byte × N | labels 0–9 |

### Loading the data

> **Note:** `mlxtend.data.loadlocal_mnist` does **not** work with this dataset, because it
> assumes 28×28 greyscale images (784 values). Use the loader below instead.

```python
import struct
import numpy as np

def load_dog_mnist(images_path, labels_path):
    with open(labels_path, "rb") as f:
        struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    with open(images_path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(n, rows * cols * 3)
    return images, labels

X_train, y_train = load_dog_mnist("dog-mnist/train-images-idx3-ubyte",
                                  "dog-mnist/train-labels-idx1-ubyte")
print(X_train.shape, y_train.shape)        # (1979, 3072) (1979,)

images = X_train.reshape(-1, 32, 32, 3)    # (N, H, W, C) uint8 RGB
# For PyTorch (N, C, H, W): images.transpose(0, 3, 1, 2)
```

## Rules

1. **No pretrained models.** Every model must be built and trained **from scratch** on the data provided. You may not use pretrained weights, transfer learning, fine-tuning, distillation from a pretrained model, or embeddings from any pretrained network. **Using a pretrained model in any form means you forfeit.**
2. **No external data.** Train only on the Dog-MNIST **train** set. You may not use any other images or datasets, including images made by generative models. Augmenting the training images (flips, crops, colour jitter and so on) is allowed. The **sample-test** set is only for checking that your code runs. Do not train, validate or tune on it. To get a validation set, split one off from `train`.
3. **Deadline.** Submissions must arrive by **26 October 2026, 23:59 IST**. Late submissions will not be evaluated.
4. **One final run.** The organiser runs your testing code on the hidden test set **once**. Nothing can be re-run, fixed or resubmitted after that run, so make sure your code works on `sample-test` before you submit.
5. **Tie-break.** If two or more submissions get the same score, the model with the **fewest parameters** wins.

## What to submit

Submit one folder, named after your team, containing:

| File | Contents |
|---|---|
| `predict.py` | The testing code, following the interface below |
| Model weights | Any file name or format; `predict.py` loads it |
| `train.py` (or a notebook) | The code that trained your model from scratch, so the no-pretrained-models and no-external-data rules can be checked |
| `requirements.txt` | Every library your code needs, with versions |

### `predict.py` interface

The organiser does not run your own script. Instead, `evaluate.py` imports your `predict.py`
and calls these functions:

```python
def load_model():
    """Build the model and load its weights. Return the model."""

def predict(model, images):
    """images: numpy uint8 array, shape (N, 32, 32, 3), RGB, values 0-255.
    Return N predicted labels (ints 0-9) as a list or numpy array."""

def count_parameters(model):   # only needed if the model is not PyTorch or Keras
    """Return the total number of parameters in the model."""
```

- Do all preprocessing (normalisation, reordering channels, etc.) inside `predict()`. You receive raw pixels only. The labels are never passed to your code.
- `evaluate.py` changes into your folder before calling `load_model()`, so load weights with a relative path such as `"model.pt"`.
- Your code must not download anything. The evaluation machine runs offline, on the CPU. `example_submission/` is a complete working example (a small PyTorch CNN), and you can copy its layout. You can use any other 

### Check your submission before sending it

Run the official script on `sample-test`:

```bash
python evaluate.py --submission path/to/your_team \
                   --images sample-test-images-idx3-ubyte \
                   --labels sample-test-labels-idx1-ubyte
```

Expected output (from the example submission):

```
Submission  : example_submission
Data        : sample-test-images-idx3-ubyte (48 images)
Accuracy    : 0.1875 (9/48)
Mean class accuracy : 0.2050
Parameters  : 24,458 (PyTorch, counted by evaluator)
Predict time: 0.01 s
```

- **If this command fails on your submission, your entry will fail in the final evaluation too.**

- The final evaluation runs exactly this command, with the hidden test files instead of
  `sample-test`.

## Evaluation

- Your model is ranked by its **accuracy** on the hidden test set (445 images), as printed
  by `evaluate.py`.
- If scores tie, the model with **fewer parameters** ranks higher. Every parameter counts,
  whether it is trainable or frozen. For PyTorch (`nn.Module`) and Keras models, `evaluate.py` counts the parameters itself. For any other kind of model, the count from your `count_parameters()` is checked by hand.
- Each submission is evaluated exactly once. If it crashes, there is no second run.
