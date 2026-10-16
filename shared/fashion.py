"""Fashion-MNIST dataset contract and loader.

Why Fashion-MNIST over CIFAR-10: both are offered by the brief, but
Fashion-MNIST is 28x28 single-channel against CIFAR-10's 32x32 RGB -- roughly a
third of the data per image. On a CPU that is the difference between a training
run you wait minutes for and one you wait an hour for, and this project needs
SIX runs (baseline, improved, and four ablations) under a matched budget. The
classes are also genuinely confusable in a way that gives Grad-CAM something to
explain: shirt / coat / pullover share a silhouette, so the failures are
interesting rather than random.

Where the data lives: on disk, downloaded by torchvision at training time, and
afterwards only inside the committed checkpoint's weights. It is NEVER written
to Supabase. That is a deliberate choice, not an oversight -- see the
"Data persisted / not persisted" section of the README.

IMPORT DISCIPLINE: this module is imported by the Streamlit tier for CLASS_NAMES
alone, and Streamlit Cloud must not install torch or torchvision. So the heavy
imports live INSIDE the loader functions, never at module scope. Anything above
the first function must stay dependency-free.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# The contract. Pure data -- safe for the thin client to import.
# ---------------------------------------------------------------------------
IMG_SIZE = 28
N_CHANNELS = 1

CLASS_NAMES: List[str] = [
    "T-shirt/top",   # 0
    "Trouser",       # 1
    "Pullover",      # 2
    "Dress",         # 3
    "Coat",          # 4
    "Sandal",        # 5
    "Shirt",         # 6
    "Sneaker",       # 7
    "Bag",           # 8
    "Ankle boot",    # 9
]
N_CLASSES = len(CLASS_NAMES)

# Fashion-MNIST's per-channel statistics, used to normalize inputs. Computed
# over the full 60k training split; hard-coded so train and serve normalize
# identically without the API ever needing the dataset.
MEAN = 0.2860
STD = 0.3530

# ---------------------------------------------------------------------------
# The matched compute budget. Every run in the comparison uses these exact
# values, so "the improved model is better" is a claim about the architecture
# and the regularization, not about one model having seen more data.
# ---------------------------------------------------------------------------
N_TRAIN = 20_000      # subsampled from the official 60k training split
VAL_FRACTION = 0.10   # carved out of N_TRAIN, never out of test
SEED = 42
DATA_DIR = Path("data")

# Classes that are genuinely hard to separate. Called out in the report's
# qualitative analysis and used to pick interesting gallery examples.
CONFUSABLE: Dict[str, List[str]] = {
    "Shirt": ["T-shirt/top", "Pullover", "Coat"],
    "Pullover": ["Coat", "Shirt"],
    "Coat": ["Pullover", "Shirt"],
    "Sneaker": ["Ankle boot", "Sandal"],
}


def class_payload() -> dict:
    """What GET /classes returns. Drives the UI's label rendering."""
    return {
        "classes": CLASS_NAMES,
        "n_classes": N_CLASSES,
        "img_size": IMG_SIZE,
        "channels": N_CHANNELS,
        "dataset": "Fashion-MNIST",
    }


def sha256_bytes(raw: bytes) -> str:
    """Hash of the raw uploaded bytes. This -- not the pixels -- is what gets
    logged, so the same image uploaded twice is recognisable without the
    database ever holding the image."""
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Loading. Everything below pulls in torch/torchvision and is API/CLI only.
# ---------------------------------------------------------------------------
def load_splits(
    n_train: int = N_TRAIN,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
    data_dir: Path = DATA_DIR,
):
    """Return ``(x_train, y_train, x_val, y_val, x_test, y_test)`` as uint8
    numpy arrays of shape [N, 28, 28] and int64 label arrays.

    Downloads Fashion-MNIST on first call (~30 MB) and caches it under
    ``data/``, which is git-ignored. The official 10k test split is used whole
    and untouched; the validation set is carved out of the training subsample,
    so no test image influences early stopping or model selection.
    """
    import numpy as np
    from torchvision import datasets

    data_dir.mkdir(exist_ok=True)
    train_ds = datasets.FashionMNIST(root=str(data_dir), train=True, download=True)
    test_ds = datasets.FashionMNIST(root=str(data_dir), train=False, download=True)

    x_full = train_ds.data.numpy().astype("uint8")       # [60000, 28, 28]
    y_full = train_ds.targets.numpy().astype("int64")
    x_test = test_ds.data.numpy().astype("uint8")        # [10000, 28, 28]
    y_test = test_ds.targets.numpy().astype("int64")

    # Deterministic subsample. Same seed for every config, so every run in the
    # comparison trains on the identical images in the identical order.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_full))[:n_train]
    x_sub, y_sub = x_full[idx], y_full[idx]

    n_val = int(len(x_sub) * val_fraction)
    x_val, y_val = x_sub[:n_val], y_sub[:n_val]
    x_train, y_train = x_sub[n_val:], y_sub[n_val:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def to_tensor(images_uint8, normalize: bool = True):
    """[N, 28, 28] uint8 -> [N, 1, 28, 28] float32, optionally standardized.

    The same transform runs at training time and at serve time. If these ever
    diverge the model still returns confident predictions -- they are just
    wrong -- which is the quietest possible bug, so both paths call this one
    function.
    """
    import numpy as np
    import torch

    arr = np.asarray(images_uint8, dtype="float32") / 255.0
    if arr.ndim == 2:
        arr = arr[None, :, :]
    t = torch.from_numpy(arr).unsqueeze(1)
    if normalize:
        t = (t - MEAN) / STD
    return t


def export_gallery(
    out_path: Path = Path("data/gallery.npz"),
    n_per_bucket: int = 12,
    seed: int = SEED,
    data_dir: Path = DATA_DIR,
) -> Path:
    """Save a small set of held-out test images for the 'Compare Models' tab.

    The tab needs to SHOW images, and Supabase must never hold image bytes, so
    the gallery is committed to the repository instead: a handful of test
    examples, stratified across all ten classes, in a ~20 KB npz. That keeps the
    privacy invariant ("no raw images in Postgres") intact while still letting
    the deployed UI render real pictures.
    """
    import numpy as np

    _, _, _, _, x_test, y_test = load_splits(seed=seed, data_dir=data_dir)
    rng = np.random.default_rng(seed)

    picks: List[int] = []
    for c in range(N_CLASSES):
        where = np.flatnonzero(y_test == c)
        picks.extend(rng.choice(where, size=n_per_bucket, replace=False).tolist())

    picks_arr = np.asarray(sorted(picks))
    out_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        out_path,
        images=x_test[picks_arr],
        labels=y_test[picks_arr],
        test_index=picks_arr,
    )
    print(f"Wrote {out_path} with {len(picks_arr)} images "
          f"({n_per_bucket} per class, from the held-out test split)")
    return out_path


if __name__ == "__main__":
    xt, yt, xv, yv, xs, ys = load_splits()
    print(f"train {xt.shape}  val {xv.shape}  test {xs.shape}")
    print("class balance (train):",
          {CLASS_NAMES[c]: int((yt == c).sum()) for c in range(N_CLASSES)})
    export_gallery()