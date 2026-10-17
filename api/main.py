"""FastAPI vision service -- Cloud #2 (Render.com).

Owns the CNN, the preprocessing transform, and the Grad-CAM hook. Streamlit
never imports torch; it posts an image and renders the JSON that comes back.

PRIVACY INVARIANT, enforced here rather than merely documented: the raw bytes
of an uploaded image are hashed, scored, and discarded. Nothing writes pixels
to Supabase. `predictions` gets the hash and the model's output;
`image_metadata` gets the hash, the dimensions and the byte count. If you need
to know whether the same image was uploaded twice, the hash answers that. If
you need to see the image again, you cannot, and that is the point.
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import base64
import hashlib
import io
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from supabase import create_client

from api.models import build_model, receptive_field
from api.training import _grad_cam, _overlay_png, _png_b64
from shared.fashion import CLASS_NAMES, IMG_SIZE, N_CLASSES, class_payload, to_tensor

MODELS_DIR = Path("models")
GALLERY_JSON = Path("data/compare_gallery.json")
DEFAULT_MODEL = os.environ.get("MODEL_NAME", "improved_no_aug")
BASELINE_MODEL = os.environ.get("BASELINE_MODEL", "baseline")

TAGS = [
    {"name": "prediction", "description": "Classify an image and explain the decision."},
    {"name": "models", "description": "Training runs and the baseline/improved comparison."},
    {"name": "meta", "description": "Class labels and the feature contract."},
    {"name": "ops", "description": "Liveness, version, and build information."},
]

app = FastAPI(
    title="See-Sense API",
    version="1.0.0",
    description=(
        "CNN image classifier for Fashion-MNIST with Grad-CAM explanations. "
        "Upload an image to /predict and get back a label, a probability vector, "
        "and a heatmap showing which pixels drove the decision. Only the SHA-256 "
        "hash of an uploaded image is ever stored."
    ),
    openapi_tags=TAGS,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: Dict = {"model": None, "ckpt": None, "run_id": None, "client": None}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ClassProb(BaseModel):
    class_name: str
    proba: float


class PredictResponse(BaseModel):
    label: int
    class_name: str
    confidence: float
    top5: List[ClassProb]
    probs: List[float] = Field(..., description="Full probability vector, class order")
    input_png: str = Field(..., description="base64 PNG of the 28x28 model input")
    cam_png: str = Field(..., description="base64 PNG of the Grad-CAM overlay")
    image_hash: str
    width: int
    height: int
    served_by_run_id: Optional[int]
    latency_ms: int


class Health(BaseModel):
    status: str
    model_loaded: bool
    supabase: bool
    model_name: str
    run_id: Optional[int]


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
def get_client():
    if _state["client"] is None:
        url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY not set")
        _state["client"] = create_client(url, key)
    return _state["client"]


def load_model(name: str = DEFAULT_MODEL) -> None:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path} -- run api/train.py first")
    ckpt = joblib.load(path)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _state.update(model=model, ckpt=ckpt)

    # Every logged prediction names the run that served it, so a fairness or
    # accuracy review six months from now can be tied to a specific model.
    try:
        resp = (get_client().table("runs").select("id")
                .eq("config_name", name).order("id", desc=True).limit(1).execute())
        _state["run_id"] = resp.data[0]["id"] if resp.data else None
    except Exception as exc:
        print(f"could not resolve run_id for '{name}': {exc}")
        _state["run_id"] = None


@app.on_event("startup")
def _startup() -> None:
    try:
        load_model()
        print(f"loaded {DEFAULT_MODEL} (run_id={_state['run_id']})")
    except Exception as exc:
        # Stay up so /healthz can report the problem instead of Render seeing a
        # dead port and restarting in a loop.
        print(f"startup: model load failed: {exc}")


def _require_model():
    if _state["model"] is None:
        raise HTTPException(503, "model not loaded")


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------
def preprocess(raw: bytes) -> tuple[np.ndarray, int, int]:
    """Decode arbitrary uploaded bytes into the 28x28 grayscale array the model
    expects. Returns (array, original_width, original_height).

    Fashion-MNIST is light garment on a black background. A photo or a drawing
    from a user is usually the opposite -- dark subject on white paper -- and
    feeding that in unchanged gives confident nonsense, because every learned
    feature has its polarity flipped. So if the image is mostly bright, we
    invert it. The heuristic is crude and it is stated in the model card rather
    than hidden: an unusual input (a dark photo of a light garment) will be
    inverted wrongly, and the Grad-CAM overlay is what lets a user notice.
    """
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(422, f"could not decode image: {exc}")

    w, h = img.size
    gray = img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.uint8)

    if arr.mean() > 127:
        arr = 255 - arr

    return arr, w, h


def build_response(arr: np.ndarray, raw_len: int, image_hash: str,
                   width: int, height: int, t0: float) -> PredictResponse:
    model = _state["model"]
    x = to_tensor(arr)                       # normalized, exactly as in training

    with torch.no_grad():
        probs_t = torch.softmax(model(x), dim=1)[0]
    probs = [float(p) for p in probs_t.tolist()]
    label = int(np.argmax(probs))

    cam = _grad_cam(model, x, label)
    order = np.argsort(probs)[::-1][:5]

    return PredictResponse(
        label=label,
        class_name=CLASS_NAMES[label],
        confidence=probs[label],
        top5=[ClassProb(class_name=CLASS_NAMES[int(i)], proba=round(probs[int(i)], 4))
              for i in order],
        probs=[round(p, 6) for p in probs],
        input_png=_png_b64(Image.fromarray(arr, "L")),
        cam_png=_overlay_png(arr, cam),
        image_hash=image_hash,
        width=width,
        height=height,
        served_by_run_id=_state["run_id"],
        latency_ms=int((time.time() - t0) * 1000),
    )


def log_prediction(resp: PredictResponse, byte_size: int, source: str,
                   true_label: Optional[int] = None) -> None:
    """Write the prediction and the image metadata. Never raises: a logging
    outage must not turn a working prediction into a 500 for the user. The
    Supabase-outage test exercises exactly this path."""
    if resp.served_by_run_id is None:
        return
    try:
        client = get_client()
        client.table("predictions").insert({
            "served_by_run_id": resp.served_by_run_id,
            "image_hash":       resp.image_hash,
            "predicted_label":  resp.label,
            "predicted_class":  resp.class_name,
            "predicted_proba":  resp.confidence,
            "top5":             [t.model_dump() for t in resp.top5],
            "true_label":       true_label,
            "true_class":       CLASS_NAMES[true_label] if true_label is not None else None,
            "correct":          (true_label == resp.label) if true_label is not None else None,
            "source":           source,
            "latency_ms":       resp.latency_ms,
        }).execute()
        client.table("image_metadata").insert({
            "run_id":     resp.served_by_run_id,
            "sha256":     resp.image_hash,
            "width":      resp.width,
            "height":     resp.height,
            "channels":   1,
            "byte_size":  byte_size,
            "label":      resp.label,
            "class_name": resp.class_name,
            "confidence": resp.confidence,
            "source":     source,
        }).execute()
    except Exception as exc:
        print(f"logging failed (prediction still served): {exc}")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(file: UploadFile = File(..., description="Any common image format")):
    """Classify an uploaded image and return a Grad-CAM explanation."""
    _require_model()
    t0 = time.time()
    raw = await file.read()
    if not raw:
        raise HTTPException(422, "empty upload")

    image_hash = hashlib.sha256(raw).hexdigest()
    arr, w, h = preprocess(raw)
    resp = build_response(arr, len(raw), image_hash, w, h, t0)
    log_prediction(resp, len(raw), source="upload")
    return resp


@app.post("/predict_sample", response_model=PredictResponse, tags=["prediction"])
def predict_sample(index: Optional[int] = Query(None, ge=0, description="Gallery index; random if omitted")):
    """Classify a held-out gallery image. Unlike an upload, this one has ground
    truth, so the prediction is logged as correct or incorrect."""
    _require_model()
    if not GALLERY_JSON.exists():
        raise HTTPException(503, "gallery not available -- run api/gallery.py")

    gallery = json.loads(GALLERY_JSON.read_text())
    examples = gallery["examples"]
    i = int(np.random.randint(len(examples))) if index is None else index % len(examples)
    ex = examples[i]

    t0 = time.time()
    raw = base64.b64decode(ex["input_png"])
    image_hash = hashlib.sha256(raw).hexdigest()
    arr = np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.uint8)
    resp = build_response(arr, len(raw), image_hash, IMG_SIZE, IMG_SIZE, t0)
    log_prediction(resp, len(raw), source="sample", true_label=ex["true_label"])
    return resp


# ---------------------------------------------------------------------------
# Models and runs
# ---------------------------------------------------------------------------
@app.get("/runs", tags=["models"])
def list_runs(limit: int = Query(50, ge=1, le=200)):
    """Every training run, newest first. Powers the comparison table."""
    try:
        resp = get_client().table("runs").select("*").order("id", desc=True).limit(limit).execute()
        return {"runs": resp.data or []}
    except Exception as exc:
        raise HTTPException(503, f"could not read runs: {exc}")


@app.get("/runs/{run_id}", tags=["models"])
def get_run(run_id: int):
    try:
        resp = get_client().table("runs").select("*").eq("id", run_id).limit(1).execute()
    except Exception as exc:
        raise HTTPException(503, f"could not read run: {exc}")
    if not resp.data:
        raise HTTPException(404, "run not found")
    return resp.data[0]


@app.get("/predictions", tags=["models"])
def list_predictions(limit: int = Query(100, ge=1, le=500)):
    """Recent predictions joined to the run that served them. This is the
    'Recent Classifications' wall -- note that it exposes hashes and labels,
    never images."""
    try:
        resp = (get_client().table("predictions")
                .select("id,image_hash,predicted_class,predicted_proba,true_class,"
                        "correct,source,latency_ms,created_at,served_by_run_id,"
                        "runs(config_name,test_accuracy)")
                .order("id", desc=True).limit(limit).execute())
        return {"predictions": resp.data or []}
    except Exception as exc:
        raise HTTPException(503, f"could not read predictions: {exc}")


@app.get("/compare_gallery", tags=["models"])
def compare_gallery():
    """The precomputed baseline-vs-improved gallery with Grad-CAM overlays.

    Served from a committed JSON rather than computed per request: 24 forward
    and 24 backward passes behind a free-tier cold start would make the tab
    unusable, and the inputs never change."""
    if not GALLERY_JSON.exists():
        raise HTTPException(503, "gallery not generated -- run `python -m api.gallery`")
    return json.loads(GALLERY_JSON.read_text())


@app.get("/performance", tags=["models"])
def performance(name: str = Query(DEFAULT_MODEL)):
    """Curves, confusion matrix and per-class metrics for one config, read from
    its checkpoint so the plots show exactly the numbers it was evaluated on."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise HTTPException(404, f"no checkpoint for '{name}'")
    ckpt = joblib.load(path)
    return {
        "config": ckpt["config"],
        "architecture": ckpt["arch"],
        "n_params": ckpt["n_params"],
        "metrics": ckpt["metrics"],
        "history": ckpt["history"],
        "confusion": ckpt["confusion"],
        "per_class": ckpt["per_class"],
        "class_names": ckpt["class_names"],
    }


# ---------------------------------------------------------------------------
# Meta and ops
# ---------------------------------------------------------------------------
@app.get("/classes", tags=["meta"])
def classes():
    """Label list and input contract. The UI renders names from this rather
    than hard-coding them."""
    return class_payload()


@app.get("/receptive_field", tags=["meta"])
def rf(kernels: str = Query("3,3,3"), strides: str = Query("1,1,1")):
    """Receptive field after each layer of a conv stack. Backs the Concepts tab
    so the arithmetic shown to the reader is computed, not typed."""
    try:
        ks = [int(k) for k in kernels.split(",")]
        ss = [int(s) for s in strides.split(",")]
    except ValueError:
        raise HTTPException(422, "kernels and strides must be comma-separated integers")
    if len(ks) != len(ss):
        raise HTTPException(422, "kernels and strides must have the same length")
    return {"layers": receptive_field(list(zip(ks, ss)))}


@app.get("/healthz", response_model=Health, tags=["ops"])
def healthz():
    ok_model = _state["model"] is not None
    ok_db = True
    try:
        get_client().table("runs").select("id").limit(1).execute()
    except Exception:
        ok_db = False
    return Health(
        status="ok" if (ok_model and ok_db) else "degraded",
        model_loaded=ok_model,
        supabase=ok_db,
        model_name=DEFAULT_MODEL,
        run_id=_state["run_id"],
    )


@app.get("/version", tags=["ops"])
def version():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = os.environ.get("RENDER_GIT_COMMIT", "unknown")[:7]
    url = os.environ.get("SUPABASE_URL", "")
    ckpt = _state["ckpt"] or {}
    return {
        "git_sha": sha,
        "torch": torch.__version__,
        "supabase_project_ref": url.split("//")[-1].split(".")[0] if url else None,
        "deployed_model": DEFAULT_MODEL,
        "baseline_model": BASELINE_MODEL,
        "architecture": ckpt.get("arch"),
        "n_params": ckpt.get("n_params"),
        "dataset": "Fashion-MNIST",
        "n_classes": N_CLASSES,
    }