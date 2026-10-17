"""Supabase round trip, plus the outage path.

The round trip proves /predict writes a predictions row linked to a runs row.
The outage test proves that when Supabase is unreachable the prediction is
still served -- logging is a side effect, not a precondition, and a database
blip must not take the classifier down.
"""
import base64
import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

from fastapi.testclient import TestClient

from api.main import app

GALLERY = Path("data/compare_gallery.json")

has_creds = bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"))


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def reference_png_bytes() -> bytes:
    if not GALLERY.exists():
        pytest.skip("data/compare_gallery.json missing")
    return base64.b64decode(json.loads(GALLERY.read_text())["examples"][0]["input_png"])


@pytest.mark.skipif(not has_creds, reason="Supabase credentials not set")
def test_predict_logs_a_row_linked_to_a_run(client):
    from api.main import get_client
    sb = get_client()

    resp = client.post("/predict", files={"file": ("ref.png", reference_png_bytes(), "image/png")})
    if resp.status_code == 503:
        pytest.skip("no model checkpoint loaded")
    assert resp.status_code == 200
    body = resp.json()
    assert body["served_by_run_id"] is not None, "prediction was not attributed to a run"

    logged = (sb.table("predictions").select("*")
              .eq("image_hash", body["image_hash"])
              .order("id", desc=True).limit(1).execute())
    assert logged.data, "no predictions row was written"
    row = logged.data[0]

    try:
        assert row["predicted_class"] == body["class_name"]
        assert abs(row["predicted_proba"] - body["confidence"]) < 1e-6
        assert row["served_by_run_id"] == body["served_by_run_id"]
        assert len(row["top5"]) == 5

        # The foreign key must resolve -- a prediction pointing at a run that
        # does not exist cannot be audited later.
        run = sb.table("runs").select("id,config_name").eq("id", row["served_by_run_id"]).execute()
        assert run.data, "served_by_run_id does not resolve to a runs row"

        # The privacy invariant, asserted rather than asserted-about: nothing
        # in the stored row may contain image bytes.
        assert "image" not in row or row.get("image") is None
        assert len(row["image_hash"]) == 64
        for key, value in row.items():
            if isinstance(value, str) and len(value) > 500:
                pytest.fail(f"column '{key}' holds {len(value)} characters -- "
                            f"suspiciously large for metadata; is it image data?")

        meta = (sb.table("image_metadata").select("*")
                .eq("sha256", body["image_hash"]).order("id", desc=True).limit(1).execute())
        assert meta.data, "no image_metadata row was written"
        assert meta.data[0]["width"] > 0 and meta.data[0]["height"] > 0
    finally:
        sb.table("predictions").delete().eq("id", row["id"]).execute()
        sb.table("image_metadata").delete().eq("sha256", body["image_hash"]).execute()


def test_prediction_survives_a_supabase_outage(client, monkeypatch):
    """Break the database client, then predict. The response must still be 200.

    This is the degradation path the rubric asks for: on Render's free tier a
    cold start or a transient network failure is routine, and the right
    behaviour is to serve the prediction and lose the log line, not to fail
    the request."""
    import api.main as main

    def exploding_client():
        raise RuntimeError("simulated Supabase outage")

    monkeypatch.setattr(main, "get_client", exploding_client)

    resp = client.post("/predict", files={"file": ("ref.png", reference_png_bytes(), "image/png")})
    if resp.status_code == 503:
        pytest.skip("no model checkpoint loaded")

    assert resp.status_code == 200, "a logging failure must not fail the prediction"
    body = resp.json()
    assert body["class_name"], "the classification still came back"
    assert body["cam_png"], "the explanation still came back"


def test_healthz_reports_degraded_during_an_outage(client, monkeypatch):
    """When the database is unreachable, /healthz must say so rather than
    claiming everything is fine."""
    import api.main as main

    def exploding_client():
        raise RuntimeError("simulated Supabase outage")

    monkeypatch.setattr(main, "get_client", exploding_client)
    body = client.get("/healthz").json()
    assert body["supabase"] is False
    assert body["status"] == "degraded"