"""The two CNN architectures, plus the receptive-field arithmetic the report needs.

Both classes expose ``.features`` (the convolutional trunk) and ``.classifier``
(the head). That is not stylistic -- the template's Grad-CAM in
``api/training.py`` registers its hooks on ``model.features``, so keeping that
attribute name means the existing, working Grad-CAM code runs unchanged against
either model.

BaselineCNN is deliberately plain: two conv blocks, ReLU, max-pool, linear head.
No normalization, no dropout, no augmentation at train time. It exists to be
beaten, and to be beaten under a matched budget so the improvement is
attributable to the techniques rather than to extra compute.

ImprovedCNN adds depth plus switchable regularization. Each switch is a
constructor flag so an ablation is a config file, not a code edit: turn one
technique off, retrain, and the runs table shows what that technique was worth.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from shared.fashion import IMG_SIZE, N_CHANNELS, N_CLASSES

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
class BaselineCNN(nn.Module):
    """Two 3x3 conv blocks -> flatten -> linear. ~28x28 -> 14x14 -> 7x7."""

    def __init__(self, img_size: int = IMG_SIZE, n_classes: int = N_CLASSES,
                 channels: int = N_CHANNELS, activation: str = "relu"):
        super().__init__()
        act = ACTIVATIONS[activation]
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            act(),
            nn.MaxPool2d(2),                      # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            act(),
            nn.MaxPool2d(2),                      # 14 -> 7
        )
        feat = img_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * feat * feat, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Improved
# ---------------------------------------------------------------------------
class ImprovedCNN(nn.Module):
    """Three conv blocks with optional batch norm and dropout.

    Depth is the structural change: a third 3x3 block widens the receptive
    field enough to cover most of a 28x28 garment, which matters for classes
    distinguished by overall silhouette (coat vs pullover) rather than by local
    texture.

    ``batch_norm`` and ``dropout`` are flags rather than hard-coded so the
    ablation runs can disable exactly one at a time. Augmentation, weight decay
    and the LR schedule are training-time concerns and live in the config, not
    here.

    Dropout sits only in the classifier head, not between conv layers:
    dropping whole activations inside a convolutional feature map tends to hurt
    more than it regularizes at this scale, since neighbouring pixels are
    highly correlated and the signal survives anyway.
    """

    def __init__(self, img_size: int = IMG_SIZE, n_classes: int = N_CLASSES,
                 channels: int = N_CHANNELS, activation: str = "relu",
                 batch_norm: bool = True, dropout: float = 0.3):
        super().__init__()
        act = ACTIVATIONS[activation]

        def block(cin: int, cout: int, pool: bool) -> List[nn.Module]:
            layers: List[nn.Module] = [nn.Conv2d(cin, cout, kernel_size=3, padding=1)]
            if batch_norm:
                # Before the activation: normalizing the pre-activations is what
                # keeps their distribution stable as the layers below update,
                # which is the whole point.
                layers.append(nn.BatchNorm2d(cout))
            layers.append(act())
            if pool:
                layers.append(nn.MaxPool2d(2))
            return layers

        self.features = nn.Sequential(
            *block(channels, 32, pool=True),      # 28 -> 14
            *block(32, 64, pool=True),            # 14 -> 7
            *block(64, 128, pool=False),          # 7 -> 7, deeper not smaller
        )
        feat = img_size // 4
        head: List[nn.Module] = [nn.Flatten()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(128 * feat * feat, 128))
        head.append(act())
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(128, n_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(config: Dict) -> nn.Module:
    """Construct the model a config describes. Used by the training CLI and by
    the API when rebuilding a model from its checkpoint."""
    arch = config.get("arch", "baseline").lower()
    if arch == "baseline":
        return BaselineCNN(activation=config.get("activation", "relu"))
    if arch == "improved":
        return ImprovedCNN(
            activation=config.get("activation", "relu"),
            batch_norm=bool(config.get("batch_norm", True)),
            dropout=float(config.get("dropout", 0.3)),
        )
    raise ValueError(f"unknown arch '{arch}' (expected 'baseline' or 'improved')")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(model: nn.Module) -> str:
    """One-line architecture summary, persisted to runs.architecture so the
    comparison table can show what was actually trained."""
    parts: List[str] = []
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            parts.append(f"conv{layer.kernel_size[0]}x{layer.kernel_size[0]}-{layer.out_channels}")
        elif isinstance(layer, nn.BatchNorm2d):
            parts.append("bn")
        elif isinstance(layer, nn.MaxPool2d):
            parts.append("pool2")
    head = [l for l in model.classifier if isinstance(l, (nn.Linear, nn.Dropout))]
    for layer in head:
        if isinstance(layer, nn.Linear):
            parts.append(f"fc-{layer.out_features}")
        else:
            parts.append(f"drop{layer.p}")
    return " → ".join(parts)


# ---------------------------------------------------------------------------
# Receptive field
# ---------------------------------------------------------------------------
def receptive_field(layers: List[Tuple[int, int]]) -> List[Dict]:
    """Walk a stack of (kernel_size, stride) layers and report the receptive
    field after each one.

    The recurrence, applied layer by layer:

        j_out = j_in * s
        r_out = r_in + (k - 1) * j_in

    where j is the jump (how many input pixels one step in this feature map
    corresponds to) and r is the receptive field. Both start at 1.

    The result worth quoting: three stacked 3x3 convolutions, stride 1, reach
    7x7 -- the same window as a single 7x7 kernel, but with 3*(3*3)=27 weights
    per channel pair instead of 49, and two extra non-linearities in between.
    That is why modern CNNs stack small kernels rather than using one large one,
    and it is the derivation the Concepts tab walks through.
    """
    r, j = 1, 1
    out: List[Dict] = []
    for i, (k, s) in enumerate(layers, start=1):
        r = r + (k - 1) * j
        j = j * s
        out.append({"layer": i, "kernel": k, "stride": s, "jump": j, "receptive_field": r})
    return out


if __name__ == "__main__":
    for name in ("baseline", "improved"):
        m = build_model({"arch": name})
        print(f"\n{name}: {count_params(m):,} params")
        print(f"  {describe(m)}")
        with torch.no_grad():
            out = m(torch.zeros(2, 1, IMG_SIZE, IMG_SIZE))
        print(f"  forward ok -> {tuple(out.shape)}")

    print("\nThree stacked 3x3 convs (stride 1):")
    for row in receptive_field([(3, 1), (3, 1), (3, 1)]):
        print(f"  after layer {row['layer']}: RF = {row['receptive_field']}x{row['receptive_field']}")