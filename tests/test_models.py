"""Architecture and checkpoint tests. No network, no database."""
from pathlib import Path

import joblib
import pytest
import torch

from api.models import BaselineCNN, ImprovedCNN, build_model, count_params, receptive_field
from shared.fashion import CLASS_NAMES, IMG_SIZE, N_CLASSES

MODELS_DIR = Path("models")


def test_both_architectures_forward():
    x = torch.zeros(4, 1, IMG_SIZE, IMG_SIZE)
    for cfg in ({"arch": "baseline"}, {"arch": "improved"}):
        model = build_model(cfg)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, N_CLASSES)


def test_both_expose_features_for_gradcam():
    """Grad-CAM hooks `model.features`. If an architecture renamed that
    attribute the explanations would break silently at serve time."""
    for cfg in ({"arch": "baseline"}, {"arch": "improved"}):
        model = build_model(cfg)
        assert hasattr(model, "features")
        assert hasattr(model, "classifier")


def test_improved_has_more_capacity_than_baseline():
    assert count_params(build_model({"arch": "improved"})) > \
           count_params(build_model({"arch": "baseline"}))


def test_ablation_flags_change_the_model():
    """Turning batch norm off must actually remove the BatchNorm layers,
    otherwise every ablation run would be measuring nothing."""
    with_bn = ImprovedCNN(batch_norm=True, dropout=0.3)
    without_bn = ImprovedCNN(batch_norm=False, dropout=0.3)
    n_bn = sum(1 for m in with_bn.modules() if isinstance(m, torch.nn.BatchNorm2d))
    n_none = sum(1 for m in without_bn.modules() if isinstance(m, torch.nn.BatchNorm2d))
    assert n_bn == 3 and n_none == 0

    with_drop = ImprovedCNN(batch_norm=True, dropout=0.3)
    without_drop = ImprovedCNN(batch_norm=True, dropout=0.0)
    assert sum(1 for m in with_drop.modules() if isinstance(m, torch.nn.Dropout)) > 0
    assert sum(1 for m in without_drop.modules() if isinstance(m, torch.nn.Dropout)) == 0


def test_unknown_architecture_raises():
    with pytest.raises(ValueError):
        build_model({"arch": "transformer"})


def test_receptive_field_three_stacked_3x3():
    rf = receptive_field([(3, 1), (3, 1), (3, 1)])
    assert [r["receptive_field"] for r in rf] == [3, 5, 7]


def test_receptive_field_grows_with_stride():
    """A stride-2 layer doubles the jump, so later layers see much more."""
    rf = receptive_field([(3, 2), (3, 1)])
    assert rf[0]["jump"] == 2
    assert rf[1]["receptive_field"] == 7


@pytest.mark.parametrize("name", ["baseline", "improved_no_aug"])
def test_checkpoints_are_loadable_and_complete(name):
    """The two checkpoints the API and the gallery depend on must carry
    everything needed to rebuild and describe the model."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        pytest.skip(f"{path} not present")
    ckpt = joblib.load(path)

    for key in ("state_dict", "config", "metrics", "history", "confusion",
                "per_class", "class_names", "arch", "n_params"):
        assert key in ckpt, f"checkpoint is missing '{key}'"

    assert ckpt["class_names"] == CLASS_NAMES
    assert len(ckpt["confusion"]) == N_CLASSES
    assert len(ckpt["per_class"]) == N_CLASSES
    assert 0.0 < ckpt["metrics"]["test_accuracy"] < 1.0

    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])     # raises if the shapes disagree


def test_confusion_matrix_totals_the_test_split():
    path = MODELS_DIR / "improved_no_aug.joblib"
    if not path.exists():
        pytest.skip("checkpoint not present")
    cm = joblib.load(path)["confusion"]
    assert sum(sum(row) for row in cm) == 10_000, "confusion matrix must cover every test image"