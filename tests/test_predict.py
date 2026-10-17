"""The three prediction tests the brief names:

  * a multipart upload smoke test for /predict
  * a shape test that the Grad-CAM overlay matches the input dimensions
  * a regression test that a fixed reference image gives a stable probability

The reference test is the one that earns its keep over time. Retraining, a
library upgrade, or an accidental change to the normalization constants would
all shift the output on a known input, and a classifier that silently drifts is
far worse than one that loudly fails.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from shared.fashion import CLASS_NAMES, IMG_SIZE

GALLERY = Path("data/compare_gallery.json")
FIXTURE = Path(__file__).parent / "reference_prediction.json"
TOLERANCE = 1e-3


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def reference_png_bytes() -> bytes:
    """A fixed image: the first gallery example, which is committed to the repo
    and therefore identical on every machine and every CI run."""
    if not GALLERY.exists():
        pytest.skip("data/compare_gallery.json missing -- run `python -m api.gallery`")
    gal = json.loads(GALLERY.read_text())
    return base64.b64decode(gal["examples"][0]["input_png"])


def synthetic_png(size=(64, 48)) -> bytes:
    """An image that is NOT 28x28 and not Fashion-MNIST, to prove the endpoint
    handles arbitrary uploads rather than only well-formed ones."""
    arr = (np.random.default_rng(0).random((size[1], size[0])) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr, "L").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Multipart upload smoke test
# ---------------------------------------------------------------------------
def test_predict_accepts_a_multipart_upload(client):
    resp = client.post("/predict", files={"file": ("ref.png", reference_png_bytes(), "image/png")})
    if resp.status_code == 503:
        pytest.skip("no model checkpoint loaded")
    assert resp.status_code == 200

    body = resp.json()
    assert body["class_name"] in CLASS_NAMES
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probs"]) == len(CLASS_NAMES)
    assert abs(sum(body["probs"]) - 1.0) < 1e-4, "softmax output must sum to 1"
    assert len(body["top5"]) == 5
    assert len(body["image_hash"]) == 64, "sha256 is 64 hex characters"


def test_predict_handles_an_arbitrary_sized_image(client):
    """A 64x48 random image is nothing like a garment. The model will be wrong,
    and that is fine -- what matters is that the endpoint resizes, scores and
    returns cleanly rather than raising."""
    resp = client.post("/predict", files={"file": ("odd.png", synthetic_png(), "image/png")})
    if resp.status_code == 503:
        pytest.skip("no model checkpoint loaded")
    assert resp.status_code == 200
    body = resp.json()
    assert body["width"] == 64 and body["height"] == 48, "original dimensions are reported"


def test_predict_rejects_a_non_image(client):
    resp = client.post("/predict", files={"file": ("notes.txt", b"this is not an image", "text/plain")})
    assert resp.status_code == 422


def test_predict_rejects_an_empty_upload(client):
    resp = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 2. Grad-CAM shape test
# ---------------------------------------------------------------------------
def test_gradcam_overlay_matches_input_dimensions(client):
    """The overlay must be the same size as the image it explains.

    Note what "input" means here: the model's input is always 28x28, because
    every upload is resized before scoring. So the overlay matches the 28x28
    tensor the network actually saw, not the original upload -- displaying a
    heatmap at the uploader's resolution would imply a spatial precision the
    7x7 feature map does not have.
    """
    resp = client.post("/predict", files={"file": ("ref.png", reference_png_bytes(), "image/png")})
    if resp.status_code == 503:
        pytest.skip("no model checkpoint loaded")
    body = resp.json()

    inp = Image.open(io.BytesIO(base64.b64decode(body["input_png"])))
    cam = Image.open(io.BytesIO(base64.b64decode(body["cam_png"])))

    assert inp.size == (IMG_SIZE, IMG_SIZE)
    assert cam.size == inp.size, f"overlay {cam.size} != input {inp.size}"
    assert cam.mode == "RGB", "the overlay is colour; the input is grayscale"
    assert inp.mode == "L"


def test_gradcam_is_not_blank(client):
    """A map of all zeros would pass a shape test and explain nothing."""
    resp = client.post("/predict", files={"file": ("ref.png", reference_png_bytes(), "image/png")})
    if resp.status_code == 503:
        pytest.skip("no model checkpoint loaded")
    cam = np.asarray(Image.open(io.BytesIO(base64.b64decode(resp.json()["cam_png"]))))
    assert cam[..., 0].max() > 0, "the red (heat) channel is entirely empty"
    assert cam.std() > 1.0, "the overlay has no variation -- it is not localizing anything"


# ---------------------------------------------------------------------------
# 3. Regression test on a frozen reference image
# ---------------------------------------------------------------------------
def test_reference_image_probability_is_stable(client):
    """First run records the probability and skips; every run after compares.
    Commit tests/reference_prediction.json so the pinned value travels with the
    repository."""
    resp = client.post("/predict", files={"file": ("ref.png", reference_png_bytes(), "image/png")})
    if resp.status_code == 503:
        pytest.skip("no model checkpoint loaded")
    body = resp.json()

    if not FIXTURE.exists():
        FIXTURE.write_text(json.dumps({
            "class_name": body["class_name"],
            "confidence": body["confidence"],
            "tolerance": TOLERANCE,
        }, indent=2))
        pytest.skip(f"recorded reference: {body['class_name']} at "
                    f"{body['confidence']:.6f} -- commit {FIXTURE.name} and re-run")

    expected = json.loads(FIXTURE.read_text())
    assert body["class_name"] == expected["class_name"], (
        f"the model now calls this a {body['class_name']}, not a {expected['class_name']}"
    )
    assert abs(body["confidence"] - expected["confidence"]) < TOLERANCE, (
        f"confidence drifted: expected {expected['confidence']:.6f}, "
        f"got {body['confidence']:.6f}. If this is an intentional retrain, "
        f"delete {FIXTURE.name} and re-run."
    )


def test_prediction_is_deterministic(client):
    """Two identical requests must agree exactly. If they do not, dropout is
    still active at inference or the model is being rebuilt per request."""
    img = reference_png_bytes()
    r1 = client.post("/predict", files={"file": ("ref.png", img, "image/png")})
    if r1.status_code == 503:
        pytest.skip("no model checkpoint loaded")
    r2 = client.post("/predict", files={"file": ("ref.png", img, "image/png")})
    assert r1.json()["probs"] == r2.json()["probs"]
    assert r1.json()["image_hash"] == r2.json()["image_hash"], "same bytes, same hash"