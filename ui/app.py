"""Streamlit UI -- Cloud #1. A thin client.

No torch, no torchvision, no PIL, no model code. Every prediction and every
Grad-CAM overlay arrives from the FastAPI service as a base64 PNG; this file
decodes and displays them. The only direct database access is a read-only
Supabase query with the ANON key, used for the Recent Classifications wall.
"""
from __future__ import annotations

import base64
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from supabase import create_client

API_URL = st.secrets["API_URL"].rstrip("/")
SUPABASE_URL = st.secrets["SUPABASE_URL"].rstrip("/")
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

st.set_page_config(page_title="See-Sense", page_icon="🔍", layout="wide")
st.title("🔍 See-Sense — Image Classification with Explanations")
st.caption("Streamlit (this UI) → FastAPI + CNN (Render) → Supabase (runs, predictions, hashes). "
           "No image is ever stored — only its SHA-256 hash.")


@st.cache_resource
def sb():
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def api_get(path: str, **params):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def png(b64: str) -> BytesIO:
    """base64 string -> something st.image can render."""
    return BytesIO(base64.b64decode(b64))


tabs = st.tabs(["Concepts", "Try It", "Compare Models", "Recent Classifications", "Model Card"])

# ===========================================================================
# 1. Concepts
# ===========================================================================
with tabs[0]:
    st.header("How a CNN sees")

    st.subheader("The convolution operation")
    st.markdown(
        "A convolutional layer slides a small learned kernel across the image. "
        "For input $X$, a $k \\times k$ kernel $W$, and output position $(i, j)$:"
    )
    st.latex(r"Z_{i,j} = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} W_{m,n} \, X_{i\cdot s + m - p,\; j\cdot s + n - p} \;+\; b")
    st.markdown(
        "with stride $s$ and padding $p$. Two properties follow directly, and they "
        "are why convolutions beat fully connected layers on images: the same $W$ is "
        "used at every position (**weight sharing** — a vertical edge is a vertical edge "
        "wherever it appears), and each output depends on only a $k \\times k$ patch "
        "(**locality** — nearby pixels are related, distant ones usually are not)."
    )

    st.subheader("Output size")
    st.latex(r"H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - k}{s} \right\rfloor + 1")
    st.markdown(
        "In this model, every conv uses $k=3$, $s=1$, $p=1$, which gives "
        "$H_{out} = H_{in}$ — the convolutions preserve size, and all the downsampling "
        "is done by the $2\\times2$ max-pools. So $28 \\to 14 \\to 7$ across the network."
    )

    st.subheader("Receptive field")
    st.markdown(
        "The receptive field is how much of the original image one unit in a feature "
        "map can see. Walking up the stack, with $j$ the jump (input pixels per step "
        "at this depth) and $r$ the receptive field, both starting at 1:"
    )
    st.latex(r"j_{\text{out}} = j_{\text{in}} \cdot s, \qquad r_{\text{out}} = r_{\text{in}} + (k - 1)\cdot j_{\text{in}}")

    st.markdown("**Worked example — three stacked $3\\times3$ convolutions, stride 1:**")
    try:
        rf = api_get("/receptive_field", kernels="3,3,3", strides="1,1,1")["layers"]
        st.table(pd.DataFrame(rf)[["layer", "kernel", "stride", "jump", "receptive_field"]])
        st.caption("Computed by the API's /receptive_field endpoint, not typed by hand.")
    except Exception:
        st.table(pd.DataFrame([
            {"layer": 1, "kernel": 3, "stride": 1, "jump": 1, "receptive_field": 3},
            {"layer": 2, "kernel": 3, "stride": 1, "jump": 1, "receptive_field": 5},
            {"layer": 3, "kernel": 3, "stride": 1, "jump": 1, "receptive_field": 7},
        ]))

    st.markdown(
        "Three $3\\times3$ layers reach $7\\times7$ — the same window as one $7\\times7$ "
        "kernel, but with $3 \\times 3^2 = 27$ weights per channel pair instead of $49$, "
        "and two extra non-linearities in between. More expressive, fewer parameters. "
        "That is the whole argument for stacking small kernels, and it is why depth "
        "buys capacity: a 28-pixel-wide garment needs a receptive field wide enough to "
        "see its silhouette, not just its texture."
    )

    st.subheader("Grad-CAM")
    st.markdown(
        "A CNN's prediction is a number; Grad-CAM turns it into a picture of *where* "
        "that number came from. Take the score $y^c$ for class $c$ before the softmax, "
        "and the activations $A^k$ of the last convolutional block. First, how much "
        "does each channel matter? Average its gradient over all spatial positions:"
    )
    st.latex(r"\alpha_k^c = \frac{1}{Z}\sum_{i}\sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}")
    st.markdown("Then weight the channels by that importance and keep only positive evidence:")
    st.latex(r"L^c_{\text{Grad-CAM}} = \mathrm{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)")
    st.markdown(
        "The ReLU matters: without it the map would mix pixels that *support* class $c$ "
        "with pixels that argue *against* it, and the result would be unreadable. "
        "The map comes out at the conv block's resolution ($7\\times7$ here) and is "
        "upsampled to $28\\times28$ for display — which is why the overlays look blocky. "
        "That blockiness is honest: the network genuinely does not localize more "
        "finely than its last feature map."
    )

# ===========================================================================
# 2. Try It
# ===========================================================================
with tabs[1]:
    st.header("Classify an image")

    try:
        meta = api_get("/classes")
        st.caption(f"Model predicts over {meta['n_classes']} classes: "
                   f"{', '.join(meta['classes'])}. Inputs are resized to "
                   f"{meta['img_size']}×{meta['img_size']} grayscale.")
    except Exception as exc:
        st.error(f"Could not reach the API: {exc}")
        meta = None

    left, right = st.columns([1, 1])
    with left:
        upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])
        go_upload = st.button("Classify upload", type="primary", disabled=upload is None)
    with right:
        st.markdown("No image handy? Score one of the held-out test images instead.")
        go_sample = st.button("Classify a held-out sample")

    result = None
    if go_upload and upload is not None:
        try:
            r = requests.post(f"{API_URL}/predict",
                              files={"file": (upload.name, upload.getvalue(), upload.type)},
                              timeout=180)
            r.raise_for_status()
            result = r.json()
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    elif go_sample:
        try:
            r = requests.post(f"{API_URL}/predict_sample", timeout=180)
            r.raise_for_status()
            result = r.json()
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if result:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.image(png(result["input_png"]), caption="Model input (28×28)", width=200)
        with c2:
            st.image(png(result["cam_png"]), caption="Grad-CAM — red is what drove it", width=200)
        with c3:
            st.metric("Predicted class", result["class_name"])
            st.metric("Confidence", f"{result['confidence']:.1%}")
            st.caption(f"Served by run {result['served_by_run_id']} · "
                       f"{result['latency_ms']} ms · hash `{result['image_hash'][:16]}…`")

        st.subheader("Top-5 probabilities")
        top5 = pd.DataFrame(result["top5"]).set_index("class_name")
        st.bar_chart(top5["proba"])

        if result["confidence"] < 0.6:
            st.warning("Low confidence. The model is close to indifferent here — "
                       "check the Grad-CAM: if it is attending to background rather "
                       "than to the garment, the input is outside what it was trained on.")
        st.info("Only the SHA-256 hash of this image was written to the database. "
                "The pixels were discarded after scoring.")

# ===========================================================================
# 3. Compare Models
# ===========================================================================
with tabs[2]:
    st.header("Baseline vs. improved")

    try:
        runs = api_get("/runs")["runs"]
    except Exception as exc:
        runs = []
        st.error(f"Could not load runs: {exc}")

    if runs:
        st.subheader("All runs, matched budget")
        df = pd.DataFrame(runs)
        cols = ["config_name", "test_accuracy", "macro_f1", "best_epoch", "train_seconds",
                "batch_norm", "dropout", "augmentation", "weight_decay", "lr_schedule", "n_params"]
        table = df[[c for c in cols if c in df.columns]].sort_values(
            "test_accuracy", ascending=False).reset_index(drop=True)
        st.dataframe(table, use_container_width=True)
        st.caption("Every run: same seed, same 18,000 training images in the same order, "
                   "same 12 epochs, same batch size. Sourced live from the Supabase "
                   "`runs` table, not hard-coded.")

        st.subheader("What each technique was worth")
        by_name = {r["config_name"]: r for r in runs if r.get("config_name")}

        def delta(a: str, b: str):
            """Accuracy of a minus accuracy of b, in percentage points."""
            if a in by_name and b in by_name:
                return (by_name[a]["test_accuracy"] - by_name[b]["test_accuracy"]) * 100
            return None

        contributions = [
            ("Batch normalization", delta("improved", "improved_no_bn")),
            ("Cosine LR schedule", delta("improved", "improved_no_sched")),
            ("Dropout", delta("improved", "improved_no_dropout")),
            ("Augmentation", delta("improved", "improved_no_aug")),
            ("Extra capacity (depth + width)", delta("improved_plain", "baseline")),
        ]
        contributions = [(k, v) for k, v in contributions if v is not None]
        if contributions:
            cdf = pd.DataFrame(contributions, columns=["technique", "accuracy_points"])
            cdf = cdf.sort_values("accuracy_points", ascending=False)
            st.bar_chart(cdf.set_index("technique")["accuracy_points"])
            st.caption("Each bar is `improved` minus the ablation with that technique "
                       "removed, in accuracy points on the 10,000-image test split. "
                       "Positive means the technique helped; negative means removing "
                       "it made the model better.")

    st.divider()
    st.subheader("Where the two models look")
    try:
        gal = api_get("/compare_gallery")
    except Exception as exc:
        gal = None
        st.error(f"Could not load the comparison gallery: {exc}")

    if gal:
        a, b = st.columns(2)
        a.metric(f"Baseline — {gal['baseline_run']}", f"{gal['baseline_test_accuracy']:.2%}")
        b.metric(f"Improved — {gal['improved_run']}", f"{gal['improved_test_accuracy']:.2%}")

        view = st.radio("Show", ["Correctly classified", "Misclassified"], horizontal=True)
        want = "correct" if view.startswith("Correct") else "incorrect"
        shown = [e for e in gal["examples"] if e["bucket"] == want]
        st.caption(f"{len(shown)} examples, all from the held-out test split.")

        for ex in shown:
            st.markdown(f"**True class: {ex['true_class']}**")
            c = st.columns([1, 1, 1, 2])
            c[0].image(png(ex["input_png"]), caption="Input", width=140)
            c[1].image(png(ex["baseline"]["cam_png"]),
                       caption=f"Baseline → {ex['baseline']['class_name']} "
                               f"({ex['baseline']['confidence']:.0%})", width=140)
            c[2].image(png(ex["improved"]["cam_png"]),
                       caption=f"Improved → {ex['improved']['class_name']} "
                               f"({ex['improved']['confidence']:.0%})", width=140)
            with c[3]:
                marks = []
                marks.append("✅ baseline correct" if ex["baseline"]["correct"] else "❌ baseline wrong")
                marks.append("✅ improved correct" if ex["improved"]["correct"] else "❌ improved wrong")
                st.write("  \n".join(marks))
                st.caption("Improved top-5: " + ", ".join(
                    f"{t['class_name']} {t['proba']:.2f}" for t in ex["improved"]["top5"][:3]))
            st.divider()

# ===========================================================================
# 4. Recent Classifications
# ===========================================================================
with tabs[3]:
    st.header("Recent classifications")
    st.caption("Read directly from Supabase with the read-only anon key — no API call. "
               "Row-level security permits anonymous SELECT on `runs` and `predictions` "
               "and nothing else.")

    try:
        rows = (sb().table("predictions")
                .select("id,image_hash,predicted_class,predicted_proba,true_class,"
                        "correct,source,latency_ms,created_at,served_by_run_id")
                .order("id", desc=True).limit(300).execute())
        log = pd.DataFrame(rows.data)
    except Exception as exc:
        log = pd.DataFrame()
        st.error(f"Supabase read failed: {exc}")

    if log.empty:
        st.info("No classifications logged yet. Try the **Try It** tab.")
    else:
        m = st.columns(4)
        m[0].metric("Total logged", len(log))
        m[1].metric("Mean confidence", f"{log['predicted_proba'].mean():.1%}")
        scored = log[log["correct"].notna()]
        m[2].metric("Accuracy (where known)",
                    f"{scored['correct'].mean():.1%}" if len(scored) else "—")
        m[3].metric("Median latency",
                    f"{int(log['latency_ms'].dropna().median())} ms"
                    if log["latency_ms"].notna().any() else "—")

        st.subheader("Class distribution")
        counts = log["predicted_class"].value_counts().sort_index()
        st.bar_chart(counts)
        st.caption("How often the model predicts each class across everything it has "
                   "served. A spike here against a flat input distribution is an early "
                   "warning that the model has collapsed toward a favourite answer.")

        st.subheader("The log")
        show = log.copy()
        show["image_hash"] = show["image_hash"].str.slice(0, 16) + "…"
        st.dataframe(show, use_container_width=True, height=380)
        st.caption("`image_hash` is a SHA-256 digest of the uploaded bytes. The image "
                   "itself was never stored, so no row here can be turned back into a "
                   "picture.")

# ===========================================================================
# 5. Model Card
# ===========================================================================
with tabs[4]:
    st.header("Model Card")
    try:
        st.markdown(open("MODEL_CARD.md", encoding="utf-8").read())
    except FileNotFoundError:
        st.error("MODEL_CARD.md not found in the repository root.")