"""Liveness, version, and the class contract."""
from fastapi.testclient import TestClient

from api.main import app
from shared.fashion import CLASS_NAMES, IMG_SIZE, N_CLASSES


def test_healthz_always_answers():
    """Render polls this endpoint to decide whether the service is alive, so it
    must return 200 even when a dependency is down -- a 500 here would make
    Render kill and restart the container in a loop."""
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert "model_loaded" in body and "supabase" in body


def test_version_reports_the_stack():
    with TestClient(app) as client:
        body = client.get("/version").json()
    assert "torch" in body
    assert body["dataset"] == "Fashion-MNIST"
    assert body["n_classes"] == N_CLASSES


def test_classes_matches_the_contract():
    """The UI renders labels from this endpoint rather than hard-coding them,
    so a drift here would mislabel every prediction in the interface."""
    with TestClient(app) as client:
        body = client.get("/classes").json()
    assert body["classes"] == CLASS_NAMES
    assert body["n_classes"] == N_CLASSES
    assert body["img_size"] == IMG_SIZE
    assert body["channels"] == 1


def test_receptive_field_arithmetic():
    """Three stacked 3x3 convolutions reach 7x7. The Concepts tab quotes this
    endpoint, so the number shown to a reader is computed, not typed."""
    with TestClient(app) as client:
        body = client.get("/receptive_field", params={"kernels": "3,3,3",
                                                      "strides": "1,1,1"}).json()
    assert [l["receptive_field"] for l in body["layers"]] == [3, 5, 7]


def test_receptive_field_rejects_mismatched_input():
    with TestClient(app) as client:
        resp = client.get("/receptive_field", params={"kernels": "3,3", "strides": "1"})
    assert resp.status_code == 422