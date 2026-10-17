# Dataset Card — Fashion-MNIST (See-Sense)

**Dataset:** Fashion-MNIST
**Used by:** See-Sense, CST-435 Topic 3
**Maintainer of this card:** Lal Ruata

---

## Why this dataset

The brief offered CIFAR-10 or Fashion-MNIST. Fashion-MNIST was chosen for two
reasons, one practical and one about what the project needed to show.

**Practical.** Fashion-MNIST is 28×28 single-channel; CIFAR-10 is 32×32 RGB,
roughly 3.9× the input volume. This project trains **eight** models — a
baseline, an improved network, and six ablations — on CPU under a matched
budget. At 73–166 seconds per run on Fashion-MNIST, the full comparison takes
under twenty minutes; the same comparison on CIFAR-10 would take hours and the
ablations would likely have been cut.

**Substantive.** Grad-CAM needs something legible to localize. On a 32×32
colour photograph the heatmap competes with background clutter and texture; on
a clean garment silhouette the overlay clearly shows whether the model is
looking at the sleeve, the sole, or the outline. Fashion-MNIST also contains a
genuinely hard sub-problem — Shirt, T-shirt/top, Pullover and Coat are
near-identical at this resolution — which gives the qualitative failure
analysis real material instead of random errors.

## Source and provenance

- **Origin:** Zalando Research, released 2017 as a drop-in replacement for the
  original MNIST digits, which modern classifiers had saturated.
- **Composition:** 70,000 grayscale images, 28×28, across 10 balanced classes.
  60,000 train / 10,000 test, split by the publisher.
- **Content:** product photographs from Zalando's catalogue, converted to
  grayscale, trimmed, resized and centred.
- **Licence:** MIT.
- **Access path:** downloaded by `torchvision.datasets.FashionMNIST` on the
  first run of `python -m shared.fashion`, cached under `data/`, which is
  git-ignored.

## Classes

| id | class | id | class |
|---|---|---|---|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

Classes 0, 2, 4 and 6 are all upper-body garments and account for most of the
model's errors. Classes 5, 7 and 9 are all footwear and account for most of the
rest.

## What this project actually used

The full 60,000-image training split was **not** used. Every run in the
comparison trains on the same deterministic subsample so that differences
between runs are attributable to the models rather than to the data:

| Split | Images | Source |
|---|---|---|
| Train | 18,000 | 20,000 sampled from the official train split (seed 42), minus validation |
| Validation | 2,000 | 10% carved out of that subsample — never from test |
| Test | 10,000 | the official test split, whole and untouched |

The validation set is taken from training rows, not test rows, so no test image
influences model selection. The same seed produces the same subsample in the
same order on every run and on every machine: two separate invocations of
`improved_plain` produced results identical to fourteen decimal places, which
is the evidence for the matched-seed claim.

Class balance in the 18,000-image training subsample ranges from 1,754 (Shirt)
to 1,845 (Pullover) — within about 5% of even, so no class weighting was
applied.

## Preprocessing

1. Decode to a single grayscale channel.
2. Scale to [0, 1], then standardize with Fashion-MNIST's training statistics,
   mean **0.2860** and standard deviation **0.3530**. These are hard-coded in
   `shared/fashion.py` so the deployed API normalizes identically to training
   without ever needing the dataset.
3. For `augmentation: true` runs only, at training time: random horizontal flip
   and a random 28×28 crop from a 2-pixel zero-padded image.

Horizontal flip is safe here because a mirrored garment is still the same
garment. It would be unsafe on data with chirality — text, some digits, left
and right medical scans — and that limit is worth stating rather than
inheriting by accident.

**At serve time**, an uploaded image is resized to 28×28 with Lanczos
resampling and **inverted if its mean pixel value exceeds 127**. Fashion-MNIST
is a light garment on a black background; a user's photo or drawing is usually
dark subject on white paper, and feeding that in unflipped produces confident
nonsense because every learned feature has its polarity reversed. The heuristic
is crude and will invert wrongly on an unusual input (a dark garment
photographed against a dark background). The Grad-CAM overlay is what lets a
user notice.

## Known biases and limitations

- **Catalogue photographs, not the world.** Every image is a professionally
  shot, centred, evenly lit product photo on a plain background. A model
  trained here has seen nothing resembling a garment on a person, at an angle,
  partly occluded, or in uneven light. Accuracy on this test split says
  essentially nothing about accuracy on a phone camera.
- **Single retailer, single market.** The catalogue is Zalando's, a European
  retailer. Garment styles outside that catalogue — traditional dress from any
  culture it does not stock — are not represented in any class and will be
  forced into whichever of the ten the model finds nearest.
- **28×28 destroys the distinguishing detail.** Buttons, collars, fabric
  texture and print are exactly what separates a shirt from a coat, and they
  are the first thing lost at this resolution. The model's confusion between
  those classes is partly a property of the data, not solely a failure of the
  network — an argument the Grad-CAM overlays support.
- **The ten classes are not exhaustive or mutually obvious.** A garment that is
  arguably both a pullover and a coat has one label in the data, and the model
  is penalized for the other answer.

## Privacy and what is stored

Fashion-MNIST contains no people and no personal data, so this project's
privacy design is not about protecting these images — it is about the images a
**user** uploads at serve time.

**Never written to Supabase:** image bytes, in any form or encoding.

**Written to Supabase:** a SHA-256 hash of the uploaded bytes, the image's
original width and height, byte size, the predicted class and probability, and
the run that served it.

**Committed to the git repository:** `data/gallery.npz` (120 public
Fashion-MNIST test images) and `data/compare_gallery.json` (12 of those with
Grad-CAM overlays). These exist so the Compare Models tab can display real
pictures. They are public dataset images in a public repository, not user data
in a growing database — the distinction that matters is that the database is
the thing which accumulates a record of what real people uploaded, and that is
what must stay free of pixels.

**Not committed:** the raw 70,000-image dataset. It is downloaded on demand and
git-ignored.

## Reproducing

```bash
python -m shared.fashion        # downloads, splits, writes data/gallery.npz
python -m api.train --config api/configs/baseline.yaml
python -m api.gallery --baseline baseline --improved improved_no_aug
```