# See-Sense — A Vision-Classification Service with Explanations

**Name:** Lal Ruata
**Course / Section:** CST-435 [SECTION]
**GitHub:** https://github.com/SamRuata/see-sense
**Streamlit Cloud (UI):** https://see-sense-wa4u2xnbp5mmxsrvc42esn.streamlit.app/
**Render.com (API):** https://see-sense-api-ojye.onrender.com, health check at [`/healthz`](https://see-sense-api-ojye.onrender.com/healthz)
**Supabase project ref:** `utylyaxgubafjhyqcvnk`
**Product presentation (video):** [https://youtu.be/CKwVW97GLkM]
**Engineering report:** below, in this README
**Model card:** [`MODEL_CARD.md`](MODEL_CARD.md) · **Dataset card:** [`DATASET_CARD.md`](DATASET_CARD.md)

> The Render API runs on the free tier and sleeps when idle. The first request after a quiet period can take up to 60 seconds.

## Product pitch

See-Sense is an image-recognition demo built for an undergraduate AI literacy
course. A visitor uploads an image and gets back a predicted garment class,
calibrated probabilities across all ten classes, and — the part that matters —
a Grad-CAM heatmap showing which pixels drove the decision. A public Recent
Classifications wall shows what the model has been asked recently and how
confidently it answered, and a Compare Models tab puts a baseline CNN and an
improved one side by side on the same images so visitors can see not just that
one is better but *where* they look differently. The pedagogy is the product:
a visitor should leave understanding that a CNN does not see the way a person
does. No image a visitor uploads is ever stored — only its SHA-256 hash.

**Dataset choice: Fashion-MNIST**, not CIFAR-10. Two reasons. Practically, this
project trains eight models under a matched budget; at 28×28 grayscale that
takes twenty minutes on a CPU, where CIFAR-10's 32×32 RGB would take hours and
the ablations would have been cut. Substantively, Grad-CAM needs something
legible to localize — on a clean garment silhouette you can see whether the
model is reading the sleeve or the outline, where on a cluttered colour photo
the heatmap competes with background. Fashion-MNIST also has a genuinely hard
sub-problem (Shirt / T-shirt / Pullover / Coat are near-identical at this
resolution), which gives the failure analysis real material.

---

## Architecture

```
┌───────────────────────┐  multipart image  ┌──────────────────────────┐  service-role  ┌───────────────────┐
│  Streamlit Cloud      │ ────────────────▶ │  FastAPI on Render        │ ─────────────▶ │  Supabase         │
│  ui/app.py            │   POST /predict   │  api/main.py              │   full access  │  Postgres         │
│  thin client:         │                   │  CNN + Grad-CAM           │                │  runs             │
│  no torch, no PIL,    │ ◀──────────────── │  hashes the bytes,        │ ◀───────────── │  predictions      │
│  no model code        │  label + probs +  │  discards the pixels      │                │  image_metadata   │
│  decodes base64 PNGs  │  base64 Grad-CAM  │                           │                │  (hashes only)    │
└──────────┬────────────┘                   └──────────────────────────┘                └─────────▲─────────┘
           │                                                                                      │
           └─────────────── anon key, read-only: Recent Classifications wall ─────────────────────┘
```

The Streamlit tier imports no model code. Every prediction and every heatmap
arrives from FastAPI as a base64 PNG. Its only direct database access is a
read-only `SELECT` with the **anon** key, permitted by row-level security. The
**service-role** key exists only in Render's environment and never reaches a
browser.

## Data persisted / not persisted

**Persisted in Supabase**
- `runs` — one row per training run: architecture, every hyperparameter and regularization switch, test accuracy, macro-F1, per-epoch curves, the 10×10 confusion matrix, and per-class precision/recall/F1.
- `predictions` — one row per served request: SHA-256 of the uploaded bytes, predicted class and probability, the top-5 vector, the run that served it, latency, and ground truth where it exists.
- `image_metadata` — SHA-256, original width, height, byte size, channels.

**Deliberately NOT persisted**
- **Raw image bytes, in any form or encoding.** This is the project's central privacy decision. The database is the component that accumulates a permanent record of what real people uploaded; a hash lets you recognise a repeat upload, reproduce an audit, and detect a replay, without the database ever being able to reconstruct a picture. Storing the pixels would buy convenience for debugging and create a growing archive of other people's images with no retention policy and no consent.
- The trained weights themselves — checkpoints live in `models/` in git, not in Postgres.
- The Fashion-MNIST dataset, which is downloaded on demand and git-ignored.

**Committed to the repository (not the database):** `data/gallery.npz` (120 public test images) and `data/compare_gallery.json` (12 with precomputed overlays), so the Compare Models tab can render real pictures. These are public dataset images in a public repo — not user data in a live database.

## Repository layout

```
see-sense/
├── api/
│   ├── main.py            # 11 endpoints, OpenAPI-tagged
│   ├── models.py          # BaselineCNN, ImprovedCNN, receptive-field arithmetic
│   ├── train.py           # CLI: trains one config, checkpoints, writes a runs row
│   ├── gallery.py         # CLI: precomputes the Grad-CAM comparison gallery
│   ├── training.py        # Grad-CAM hooks + overlay rendering (from the template)
│   └── configs/           # 8 YAML configs: baseline, improved, 5 ablations, improved_v2
├── db/migrations/         # 001 (template) + 002 (predictions, run columns, RLS)
├── shared/fashion.py      # dataset contract, loader, gallery export
├── models/*.joblib        # 8 checkpoints
├── ui/app.py              # 5-tab thin client
└── tests/                 # pytest suite
```

## API endpoints

`POST /predict` (multipart) · `POST /predict_sample` · `GET /runs` · `GET /runs/{id}` ·
`GET /predictions` · `GET /compare_gallery` · `GET /performance` · `GET /classes` ·
`GET /receptive_field` · `GET /healthz` · `GET /version`

Grouped under four OpenAPI tags — prediction, models, meta, ops — so `/docs` reads as documentation.

## Configuration comparison

Generated by SQL against `runs`, not typed by hand:

```sql
select config_name, test_accuracy, macro_f1, best_epoch, train_seconds,
       batch_norm, dropout, augmentation, weight_decay, lr_schedule, n_params
from runs order by test_accuracy desc;
```

| config | test acc | macro F1 | best epoch | train s | BN | dropout | aug | wd | sched | params |
|---|---|---|---|---|---|---|---|---|---|---|
| **improved_no_aug** | **0.9142** | 0.9139 | 12 | 139.4 | ✓ | 0.3 | — | 1e-4 | cosine | 897,354 |
| improved_v2 | 0.9107 | 0.9106 | 10 | 165.7 | ✓ | 0 | — | 1e-4 | cosine | 897,354 |
| improved_no_dropout | 0.9063 | 0.9056 | 11 | 156.2 | ✓ | 0 | ✓ | 1e-4 | cosine | 897,354 |
| improved | 0.9026 | 0.9016 | 12 | 150.7 | ✓ | 0.3 | ✓ | 1e-4 | cosine | 897,354 |
| baseline | 0.8979 | 0.8954 | 10 | 72.9 | — | 0 | — | 0 | — | 50,186 |
| improved_plain | 0.8963 | 0.8957 | 9 | 118.4 | — | 0 | — | 0 | — | 896,906 |
| improved_no_sched | 0.8960 | 0.8947 | 12 | 159.3 | ✓ | 0.3 | ✓ | 1e-4 | — | 897,354 |
| improved_no_bn | 0.8862 | 0.8847 | 10 | 159.3 | — | 0.3 | ✓ | 1e-4 | cosine | 896,906 |

Every run: seed 42, the same 18,000 images in the same order, 12 epochs, batch 128.

## Tests

`pytest -q` — covers a multipart upload smoke test for `/predict`, a shape test
that the Grad-CAM overlay matches the model input's dimensions, a regression
test pinning a frozen reference image's probability to ±1e-3, a Supabase round
trip confirming `/predict` writes a `predictions` row whose foreign key
resolves to a `runs` row, and a **Supabase-outage path** that breaks the
database client and asserts `/predict` still returns 200.

## Running locally

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install torchvision pyyaml joblib "scikit-learn==1.5.2" "scipy==1.13.1"
cp .env.example .env                                    # SUPABASE_URL + SERVICE key
python -m shared.fashion                                # download + gallery
python -m api.train --config api/configs/baseline.yaml
python -m api.gallery --baseline baseline --improved improved_no_aug
uvicorn api.main:app --reload --port 8000               # terminal 1
streamlit run ui/app.py                                 # terminal 2
```

---

# Engineering Report

## Quantitative analysis

Both models trained on an identical budget: seed 42, the same 18,000 images in
the same order, 12 epochs, batch 128. The baseline is two 3×3 conv blocks with
max-pooling and a linear head (50,186 parameters, no regularization). The
deployed model adds a third block, batch normalization, dropout 0.3, weight
decay and cosine annealing (897,354 parameters). **Top-1 accuracy on the
10,000-image held-out split rose from 0.8979 to 0.9142**; macro-F1 0.8954 →
0.9139.

The curves show why regularization mattered: the baseline's validation loss
bottoms out at epoch 10 then rises while training accuracy climbs to 0.9383,
and a variant without dropout hit 0.9997 training accuracy with validation loss
turning upward after epoch 7 — memorisation.

Six ablations, each differing in exactly one switch, isolate the contributions:

| technique removed | accuracy points |
|---|---|
| Batch normalization | **−1.64** |
| Cosine LR schedule | −0.66 |
| Dropout | +0.37 |
| Augmentation | +1.16 |
| Extra depth and width | +0.16 |

![Baseline training curves](docs/curves_baseline.png)
![Deployed model training curves](docs/curves_improved_no_aug.png)
![Baseline confusion matrix](docs/confusion_baseline.png)
![Deployed model confusion matrix](docs/confusion_improved_no_aug.png)


Two results contradict the design's assumptions. **The architecture change
bought nothing**: `improved_plain` — the deeper network with every technique
off — scored 0.8963 against the baseline's 0.8979, slightly worse with
eighteen times the parameters. The gain came entirely from regularization;
without that run it would have been misattributed to capacity. And **the
configuration labelled "improved" loses to one of its own ablations.** Four
runs form a 2×2 factorial on augmentation and dropout: dropout helps (+0.35,
+0.37), augmentation hurts (−0.44, −0.79), because augmentation makes each
epoch harder and repays that only over longer training. The deployed model is
therefore `improved_no_aug`.

Per-class results locate the difficulty: both models are near ceiling on
Trouser, Bag, Sandal, Ankle boot and Sneaker (F1 0.956–0.987) and weak on the
four upper-body garments. **Shirt is hardest in both** — recall rises 0.585 →
0.733 while precision *falls* 0.798 → 0.779. The deployed model answers
"Shirt" more readily, catching more true shirts and misfiling more coats.

## Qualitative analysis

Overlays are computed at the last conv block's 7×7 resolution and upsampled, so
they are blocky — accurate, not a rendering flaw: the network does not localize
more finely than its final feature map.

On easy predictions — Bag, Sneaker, Trouser near 1.00 — both models
concentrate on the silhouette and the heatmaps are nearly identical. Two
failures fall on opposite sides of the dataset/model line.

**`Coat → Shirt` at 0.876 — a dataset failure.** Both models make it, and both
attend to the outer silhouette and shoulder line. That is the right region; at
28×28 the distinguishing information — button placket, collar, fabric weight —
has been destroyed by downsampling. The map shows a model doing the right thing
with insufficient data, which is why both fail identically.

**`Sneaker → Ankle boot` at 0.997 — a model failure.** The overlay concentrates
on the upper and heel while ignoring the ankle opening, the feature that
defines the distinction. The information is in the image; the model weights the
wrong part of it, with near-total confidence. Being wrong at 0.997 is worse
than at 0.6, because the confidence invites a downstream system to act.

The gallery shows that trade-off case by case: both images where the deployed
model regressed are `→ Shirt`. One caveat — on 120 gallery images both models
score 91.7%, since 1.6 points is about two images. The gallery shows *how* they
differ; the test split shows *whether*.

![Grad-CAM on correctly classified images](docs/gradcam_correct.png)
![Grad-CAM on misclassified images](docs/gradcam_incorrect.png)

## Ethical reflection

**Harms.** In **medical triage**, a screening CNN carries this project's exact
failure mode: nothing distinguishes a confidently wrong answer from a
confidently right one, so a mis-ranked patient gives the clinician no signal to
look twice. In **content moderation**, the per-class pattern transfers —
accuracy collapses on classes resembling other classes, meaning satire,
reclaimed language and medical imagery, precisely where being wrong costs
someone their account.

**Technical safeguards.** An abstain path — a per-class confidence threshold
below which the system returns "uncertain", calibrated separately because Shirt
and Trouser are not equally reliable. And mandatory explanation at the point of
decision: the overlay beside every prediction, never behind a request, so a
reviewer sees a model attending to the wrong region before acting.

**Governance safeguards.** A named human with authority to override, logged; a
model nobody can overrule is making the decision whatever the policy says. And
a scheduled review against the logged predictions — which this design enables,
since every prediction records its serving run — with authority to withdraw the
model, not merely to file a report.

**What we deliberately did not log: image bytes, in any encoding.** Only a
SHA-256 hash, dimensions, byte size and the model's output — and the test suite
enforces it, failing if any stored string exceeds 500 characters. The database
is what accumulates a permanent record of what real people uploaded. A hash
supports every legitimate audit — recognising a repeat, reproducing which model
decided what, detecting a replay — and nothing an illegitimate one wants.
Storing pixels would have eased debugging and created an archive of other
people's images with no retention policy and no consent.

## Worldview reflection

"So God created man in his own image" (Genesis 1:27, ESV). A classifier pointed
at a person exercises power over an image-bearer: it sorts them into a category
someone else chose, attaches a number to that judgment, and hands it to a
system that acts. My model sorts garments. The architecture is the same one.

**I would refuse to build a classifier that infers a protected characteristic
from appearance** — ethnicity, religion, sexual orientation, immigration status
— whatever the stated purpose. Not squeamishness about a hard problem: such a
system cannot be built correctly, because what it learns is the correlation
between surface features and a label some human assigned, and those most often
misclassified are the people who least resemble whoever the training data
treated as typical. The harm lands where there is least recourse.

That refusal is engineering, not a layer added afterward. Every choice here
already encoded one: hashing uploads rather than storing them decided who can
be re-identified later; putting Grad-CAM in the interface rather than behind an
API decided whether anyone would look; running the ablation that showed my
architecture change was worthless decided whether I would find out. No
requirements document adds honesty after the schema is written.