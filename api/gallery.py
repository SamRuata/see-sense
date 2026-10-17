"""Precompute the 'Compare Models' gallery: Grad-CAM overlays for both models.

    python -m api.gallery --baseline baseline --improved improved_no_aug

Scores the 120 held-out images in data/gallery.npz with both checkpoints,
computes a Grad-CAM map for each model on each image, picks >=5 correctly and
>=5 incorrectly classified examples, and writes everything to
data/compare_gallery.json for the API to serve.

WHY PRECOMPUTE. Rendering this tab live would mean 24 forward passes and 24
backward passes per page load on Render's free tier, behind a cold start. The
images are fixed and the checkpoints are fixed, so the answer is fixed too --
compute it once here, commit the ~50 KB of JSON, and the deployed tab loads
instantly. It also makes the figures in the report reproducible: anyone can
rerun this command and get byte-identical overlays.

WHY THIS DOES NOT BREAK THE PRIVACY INVARIANT. The JSON holds base64 PNGs of
twelve public Fashion-MNIST test images. It lives in the git repository, not in
Postgres. Supabase still never receives image bytes -- only hashes. The
distinction matters: the database is the thing that accumulates a record of
what real users uploaded, and that is what must stay free of pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from PIL import Image

# Reuse the template's working Grad-CAM rather than reimplementing it.
from api.training import _grad_cam, _overlay_png, _png_b64
from api.models import build_model
from shared.fashion import CLASS_NAMES, CONFUSABLE, to_tensor

GALLERY_NPZ = Path("data/gallery.npz")
OUT_JSON = Path("data/compare_gallery.json")
MODELS_DIR = Path("models")

N_CORRECT = 6
N_INCORRECT = 6


def load_checkpoint(name: str):
    """Rebuild a model from its joblib checkpoint.

    NOTE the normalization trap: api/training.py's own `_to_tensor` only scales
    to [0,1], but these models were trained on inputs standardized with
    Fashion-MNIST's mean and std. Scoring with the wrong transform produces
    confident, wrong predictions and no error message, so everything here goes
    through shared.fashion.to_tensor -- the same function training used.
    """
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path}")
    ckpt = joblib.load(path)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def score_all(model, images_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_labels, probabilities) for every gallery image."""
    x = to_tensor(images_u8)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).numpy()
    return probs.argmax(axis=1), probs


def explain_one(model, img_u8: np.ndarray, class_idx: int) -> str:
    """Grad-CAM overlay for one image, as a base64 PNG."""
    x = to_tensor(img_u8)                      # [1, 1, 28, 28], normalized
    cam = _grad_cam(model, x, class_idx)
    return _overlay_png(img_u8, cam)


def top5(probs_row: np.ndarray) -> List[Dict]:
    order = np.argsort(probs_row)[::-1][:5]
    return [{"class_name": CLASS_NAMES[int(i)], "proba": round(float(probs_row[i]), 4)}
            for i in order]


def interest_score(true_label: int) -> int:
    """Rank examples so the gallery leans toward the confusable garment classes.

    A gallery of twelve easy Bags and Trousers would show Grad-CAM working and
    teach nothing. The classes that actually separate a good model from a bad
    one here are the upper-body garments, so those sort first.
    """
    return 0 if CLASS_NAMES[true_label] in CONFUSABLE else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--improved", default="improved_no_aug")
    args = ap.parse_args()

    if not GALLERY_NPZ.exists():
        raise SystemExit(f"{GALLERY_NPZ} missing -- run `python -m shared.fashion` first")

    data = np.load(GALLERY_NPZ)
    images, labels, test_index = data["images"], data["labels"], data["test_index"]
    print(f"Loaded {len(images)} gallery images")

    base_model, base_ckpt = load_checkpoint(args.baseline)
    imp_model, imp_ckpt = load_checkpoint(args.improved)
    print(f"baseline  = {args.baseline}  (test acc {base_ckpt['metrics']['test_accuracy']:.4f})")
    print(f"improved  = {args.improved}  (test acc {imp_ckpt['metrics']['test_accuracy']:.4f})")

    base_pred, base_probs = score_all(base_model, images)
    imp_pred, imp_probs = score_all(imp_model, images)

    base_ok = base_pred == labels
    imp_ok = imp_pred == labels
    print(f"On the gallery: baseline {base_ok.mean():.1%} correct, "
          f"improved {imp_ok.mean():.1%} correct")
    print(f"  improved fixed {int((~base_ok & imp_ok).sum())} that baseline got wrong")
    print(f"  improved broke {int((base_ok & ~imp_ok).sum())} that baseline got right")

    # --- selection -------------------------------------------------------
    # Correct bucket: the improved model is right. Prefer cases where the
    # baseline was WRONG, because those are the ones worth looking at -- the
    # two saliency maps side by side show what changed.
    correct_pool = np.flatnonzero(imp_ok)
    correct_pool = sorted(
        correct_pool,
        key=lambda i: (base_ok[i], interest_score(int(labels[i])), -float(imp_probs[i].max())),
    )
        # One per class. Six near-identical Sneakers at 100% confidence would show
    # Grad-CAM working and teach nothing; variety is the point of the gallery.
    chosen_correct, seen = [], set()
    for i in correct_pool:
        c = int(labels[i])
        if c in seen:
            continue
        seen.add(c)
        chosen_correct.append(i)
        if len(chosen_correct) == N_CORRECT:
            break

    # Incorrect bucket: the improved model is wrong. Confusable classes first,
    # then the most confident mistakes -- a wrong answer given at 0.95 is more
    # revealing than one given at 0.35.
    wrong_pool = np.flatnonzero(~imp_ok)
    wrong_pool = sorted(
        wrong_pool,
        key=lambda i: (interest_score(int(labels[i])), -float(imp_probs[i].max())),
    )
    chosen_wrong = wrong_pool[:N_INCORRECT]

    if len(chosen_wrong) < 5:
        print(f"WARNING: only {len(chosen_wrong)} misclassified gallery images; "
              f"the rubric wants at least 5. Re-run shared.fashion with a larger "
              f"n_per_bucket to widen the pool.")

    examples: List[Dict] = []
    for bucket, idxs in (("correct", chosen_correct), ("incorrect", chosen_wrong)):
        for i in idxs:
            i = int(i)
            img = images[i]
            true_label = int(labels[i])
            examples.append({
                "bucket": bucket,
                "test_index": int(test_index[i]),
                "true_label": true_label,
                "true_class": CLASS_NAMES[true_label],
                "input_png": _png_b64(Image.fromarray(img, "L")),
                "baseline": {
                    "label": int(base_pred[i]),
                    "class_name": CLASS_NAMES[int(base_pred[i])],
                    "confidence": round(float(base_probs[i].max()), 4),
                    "correct": bool(base_ok[i]),
                    "top5": top5(base_probs[i]),
                    "cam_png": explain_one(base_model, img, int(base_pred[i])),
                },
                "improved": {
                    "label": int(imp_pred[i]),
                    "class_name": CLASS_NAMES[int(imp_pred[i])],
                    "confidence": round(float(imp_probs[i].max()), 4),
                    "correct": bool(imp_ok[i]),
                    "top5": top5(imp_probs[i]),
                    "cam_png": explain_one(imp_model, img, int(imp_pred[i])),
                },
            })
            flag = "ok " if bucket == "correct" else "ERR"
            print(f"  [{flag}] true={CLASS_NAMES[true_label]:<12} "
                  f"baseline={CLASS_NAMES[int(base_pred[i])]:<12} "
                  f"improved={CLASS_NAMES[int(imp_pred[i])]:<12} "
                  f"conf={imp_probs[i].max():.3f}")

    payload = {
        "baseline_run": args.baseline,
        "improved_run": args.improved,
        "baseline_test_accuracy": base_ckpt["metrics"]["test_accuracy"],
        "improved_test_accuracy": imp_ckpt["metrics"]["test_accuracy"],
        "class_names": CLASS_NAMES,
        "n_correct": sum(1 for e in examples if e["bucket"] == "correct"),
        "n_incorrect": sum(1 for e in examples if e["bucket"] == "incorrect"),
        "examples": examples,
        "note": ("Images are public Fashion-MNIST test examples, committed to the "
                 "repository so the UI can render them. No image bytes are stored "
                 "in Supabase."),
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload))
    size_kb = OUT_JSON.stat().st_size / 1024
    print(f"\nWrote {OUT_JSON} ({size_kb:.0f} KB, "
          f"{payload['n_correct']} correct + {payload['n_incorrect']} incorrect)")


if __name__ == "__main__":
    main()