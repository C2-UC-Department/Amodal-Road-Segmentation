# Amodal Road Segmentation — Dataset Pipeline

Build your own **amodal road segmentation** dataset from the public
[KITTI Road](https://www.cvlibs.net/datasets/kitti/eval_road.php) dataset, as a
free stand-in for the non-public KITTI **OFRS** dataset.

"Amodal" road = the road surface **including the parts hidden behind foreground
objects** (cars, pedestrians, cyclists, …). KITTI only labels the *visible*
road, so we:

1. decode KITTI's visible-road ground truth,
2. auto-detect foreground objects with **Mask2Former** (trained on **Mapillary
   Vistas**) to find *where* the road is occluded,
3. hand-complete the road under those objects with an **interactive annotator**,
4. save the corrected mask as the **amodal road ground truth**, keeping KITTI's
   original file names.

---

## Pipeline at a glance

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1  | `src/s1_download_kitti.py` | — | `data/raw/data_road/…` |
| 1b | `src/s2_prepare_visible_masks.py` | KITTI GT | `data/processed/visible_road/*.png` |
| 2  | `src/s3_detect_foreground.py` | KITTI images | `data/processed/foreground/*.png` |
| 3  | `src/s4_build_incomplete.py` | visible + foreground | `data/processed/occlusion/*.png`, seeded `data/amodal_road/*.png` |
| 5  | `src/s5_export_semantics.py` | KITTI images (or `--source footage`) | `data/processed/semantic_ofrs/*.png` (**OFRSNet input**) |
| 4  | `src/annotator.py` | all of the above | **`data/amodal_road/*.png`** (final GT) |
| 6a | `src/extract_frames.py` | `data/footage/*.mov,.mp4,.heic` | `data/footage_frames/*.jpg` |
| 6b | `src/s6_prepare_footage.py` | footage frames + checkpoint | model-seeded `data/amodal_road/*.png` |
| 6c | `src/s7_export_geometry.py` | images + semantic maps | `data/processed/geometry/*.npz` (**ground manifold**) |
| 7  | `src/ofrs/train.py` | semantic + geometry + amodal (KITTI + footage, pooled) | `checkpoints/ofrsnet_best.pt` |
| 8  | `src/predict.py` | any image folder | predicted amodal road + visualization |

All masks are single-channel **0/255 PNG**, one per image, sharing the source's
base name (`um_000000.png`, `umm_000012.png`, `uu_000003.png` for KITTI;
`IMG_0056_f00002.png` for footage frames).

---

## Setup

Dependencies are already installed in `.venv`. To reproduce elsewhere:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Runs on Apple Silicon (MPS), CUDA, or CPU — auto-selected (see
`config.DEVICE_PREFERENCE`).

---

## Run the automated part (Steps 1–3)

```bash
# Full run: download KITTI, decode masks, detect foreground, build occlusion hints
python run_pipeline.py --preview          # --preview writes QA overlays

# Handy variants
python run_pipeline.py --skip-download     # KITTI already extracted
python run_pipeline.py --limit 10          # only first 10 images (quick trial)
```

Or run steps individually:

```bash
python -m src.s1_download_kitti
python -m src.s2_prepare_visible_masks
python -m src.s3_detect_foreground --preview
python -m src.s4_build_incomplete --preview
```

The first foreground run downloads the Mask2Former weights (~1 GB) from
Hugging Face.

---

## Annotate (Step 4)

```bash
python -m src.annotator            # start at the first image
python -m src.annotator --start 40 # jump to index 40
```

The tool opens an OpenCV window showing the image with four overlays:

| Overlay | Colour | Meaning |
|---------|--------|---------|
| Visible road | blue | KITTI ground truth |
| Foreground | red | detected occluding objects |
| Occlusion hint | yellow | foreground pixels sitting on/near the road |
| **Amodal road** | **green** | **what you are editing** |

Your job: **paint green road underneath the red/yellow objects** so the amodal
mask represents the full road surface, then move on. The amodal mask starts as
a copy of the visible road (or resumes from your last save).

### Controls

```
Left drag     paint amodal road          Middle drag / h j k l   pan
Right drag    erase amodal road          Mouse wheel / + -       zoom
[ ]           brush size                 , .                     overlay opacity
1 2 3 4       toggle V / F / O / A        z / y                   undo / redo
f             auto-fill (visible+occl.)   c                       reset to visible road
r             fit view      s  save       n / p                   next / prev (autosaves)
q / Esc       quit (autosaves)
```

Suggested workflow per image: press **`f`** to pre-fill road under the occlusion
hints, then refine edges by hand with brush/erase, zooming in (`wheel`) for
precision. Saving is automatic on `n`/`p`/`q`; `s` saves on demand. Re-opening
an image loads your saved amodal mask, so you can stop and resume anytime.

---

## Training OFRSNet (Steps 5, 7)

Per the paper (*Occlusion-Free Road Segmentation Leveraging Semantics*, Sensors
2019), **OFRSNet's input is a semantic map, not RGB**:

* **Input** — an 11-class one-hot semantic map (road, sidewalk, building, wall,
  fence, pole, traffic_sign, vegetation, person, vehicle, unlabeled). We produce
  it from Mask2Former in `s5_export_semantics.py` (Mapillary's 65 classes are
  remapped to these 11). Person/vehicle pixels are the occlusion cue.
* **Target** — the binary occlusion-free road mask you painted in Step 4.
* **Loss** — spatially-weighted cross-entropy (heavier on road edges and away
  from the image centre), exactly as the paper describes. No GAN.

```bash
python -m src.s5_export_semantics          # build the 11-class inputs (all images)
# ... annotate with src.annotator ...
python -m src.ofrs.train                   # train; writes checkpoints/ofrsnet_best.pt
python -m src.ofrs.train --epochs 100 --batch-size 4 --lr 1e-3
```

Key differences from the paper (intentional, and generally *better* for your
goal of deploying with Mask2Former):

* The paper built KITTI-OFRS on the KITTI **semantic** set (200 imgs) with
  **human** semantics for training, then swapped to DeepLabv3+ at inference.
  Here the semantic input comes from **Mask2Former at both train and inference**,
  so there is no train/inference domain shift.
* The paper resizes every image to a fixed 384×1248 canvas. **We don't.**
  OFRSNet is fully convolutional (the global context block uses adaptive
  pooling; every up-sampling stage targets its skip connection's own shape),
  so it accepts any resolution or aspect ratio natively. Each image is only
  (a) uniformly downscaled — same factor on both axes, aspect ratio exactly
  preserved — if it exceeds `config.OFRS_MAX_SIDE` (1024px, purely to bound
  compute), then (b) padded with the `unlabeled` class up to a multiple of
  `config.OFRS_NET_STRIDE` (8) so the /8 down/up-sampling lines up, and the
  padding is cropped back off afterward. This matters because the old fixed
  384×1248 (aspect 3.25) canvas was fine for KITTI (aspect ≈3.31, negligible
  distortion) but **stretched portrait phone photos by ~5.8×** — a real
  contributor to poor generalization on real-world footage, not just a
  difference in camera angle. Training batches mix images of different native
  shapes via a custom collate (`ofrs_collate` in `src/ofrs/dataset.py`) that
  pads each batch only up to its own max size; padded pixels get weight 0 and
  never affect the loss or the reported IoU.
* **If you have an existing checkpoint trained under the old fixed-canvas
  behavior, retrain.** Its weights were fit entirely on squashed/stretched
  geometry; running it through the new native-resolution pipeline puts it
  further out of distribution, not closer.

All OFRS knobs (class scheme, Mapillary→11 mapping, loss weights, label
smoothing, split, epochs) live in `config.py`.

---

## The ground manifold: geometry-guided message passing (Step 6c)

Road completion is reframed as **graph completion on a ground manifold**:

* **Mask2Former** says *what* each visible pixel is (semantic layout).
* **GroundNet's depth stream** says *where the road surface is in 3D*.
* The context module propagates semantic evidence preferentially **between
  pixels that lie on the same ground surface**.

### Why the original context module could not do this

The paper's global-context block computes `z_i = x_i + W_v Σ_j softmax_j(W_k x_j) x_j`.
Its attention weights do not depend on the query pixel `i` at all — it pools
**one** global vector and broadcasts it identically everywhere, so no two
pixels ever exchange information specifically. `MultimodalContextModule`
replaces that with genuinely query-dependent, geometrically-gated aggregation.

### The two dense fields (`src/geometry.py`)

From monocular metric depth + a RANSAC ground-plane fit (GroundNet Eq. 1–5),
using our **existing** Mask2Former `road` class to isolate ground pixels:

| Field | Meaning |
|---|---|
| **G** `(3,H,W)` | **Ground footprint** — where each pixel's viewing ray meets the plane. Depends only on intrinsics + plane, **not depth**, so it is defined even for occluded pixels. |
| **h** `(1,H,W)` | **Signed height above the plane** of the pixel's own 3D point. `\|h\|≈0` ⇔ the pixel is genuinely *on* the manifold. |
| **gvalid** `(1,H,W)` | Ray meets the plane in front of the camera (false above the horizon). |

Measured on real footage, `|h|` separates the manifold cleanly:
road **0.05–0.09 m** vs vehicles **2.4–3.0 m** — a 34–50× margin.

### The module (`MultimodalContextModule`)

**Tier 1 — ground-weighted pooling.** Weight each pixel by `exp(-|h|/τ_h)`
before pooling, so the global summary describes the *road surface* instead of
averaging in cars, buildings and sky.

**Tier 2 — geometry-gated non-local attention.**
```
logit(i,j) = <q_i,k_j>/√d  −  α·( ‖G_i − G_j‖² / τ_d  +  |h_j| / τ_h )
```
Two deliberate asymmetries, both essential for *amodal* completion:

* The manifold gate `|h|` applies to **keys only**. A query on a car body is
  off-manifold — but that is precisely the pixel we need to fill in, so it must
  stay free to gather. What we constrain is the *source* of evidence.
* Distance uses the **ground footprint G**, not the pixel's own 3D point. A car
  roof is metres off the manifold, yet its footprint is exactly the road it
  hides, so it still gathers from the right neighbourhood. Using the raw 3D
  point would attract occluders to *other elevated things* — the opposite of
  what is wanted.

`τ_h`, `τ_d`, `α` and the residual scale `γ` are all learnable, so the network
tunes its own geometric receptive field. `γ` is **zero-initialised**: Tier 2
starts as an exact no-op and switches itself on during training (Tier 1 is
active from step one — it is only a reweighted average, so it needs no warm-up).

```bash
python -m src.s7_export_geometry                  # KITTI
python -m src.s7_export_geometry --source footage --preview
```

Geometry is **optional per sample**: images without a cached plane (or where
the fit failed for lack of visible road) fall back to the original
semantic-only behaviour, so a partially-precomputed dataset still trains.
Toggle the whole thing with `config.OFRS_USE_GEOMETRY`; checkpoints record
which architecture they were trained with, so `predict.py` always rebuilds a
matching model.

### Honest caveats

* **Depth stream only.** GroundNet's second (surface-normal) stream and its
  geometric-consistency loss need `diffusers`/Marigold, which is not installed.
  Per the paper's own ablation (Table 4) depth+RANSAC is the stronger single
  stream (2.92° vs 6.73° on KITTI), so this captures most of the benefit — but
  it *is* a reduction from the full published method.
* **Uncalibrated photos have guessed intrinsics.** KITTI ships real calibration
  (`calib/*.txt`, `P2`); arbitrary phone photos do not, so we assume a 65°
  horizontal FOV. Absolute metric scale is then unreliable (our footage implies
  a ~7 m camera height, which is clearly wrong), but the scale error is
  *uniform*, so relative co-planarity — the thing the gate actually uses —
  survives, and the learnable `τ` values absorb the rest.
* **Unproven for this task.** The mechanism is verified to behave as designed
  (see the tests), but whether it *improves road completion* is an open
  empirical question. Train with `OFRS_USE_GEOMETRY = False` and `True` and
  compare before trusting it.

## Inference on your own images (Step 8)

`src/predict.py` runs the full deploy pipeline on any folder of images and
writes a 3-panel visualization (input / semantics / final amodal road) plus
the raw mask.

```bash
python -m src.predict --input path/to/images --out data/predictions
python -m src.predict --input data/raw/data_road/testing/image_2 --limit 20
python -m src.predict --input dashcam/ --ckpt checkpoints/ofrsnet_best.pt --no-stack
```

The final mask is **not** OFRSNet's raw output. OFRSNet's road-probability map
is bilinear-upsampled to full resolution and thresholded (never nearest-
neighbor, which double stair-steps the boundary), and then only the part of
its prediction that falls inside a nearby occluder's footprint (person/vehicle
touching the visible road) is kept:

```
final = mask2former_visible_road  OR  (ofrsnet_road AND occluder_footprint)
```

So the open-road boundary is always Mask2Former's own clean segmentation;
OFRSNet only patches the gaps under obstacles. In the visualization, that
patch contribution is highlighted in **cyan** on top of the green final mask.

Outputs: `data/predictions/<name>_viz.png` (stacked visualization) and
`data/predictions/mask/<name>.png` (binary amodal road). Accepts png/jpg/bmp/tif
and recurses into sub-directories.

---

## Annotating real-world footage (Steps 6a–6b)

KITTI (Karlsruhe, Germany) has very few motorcycles and a fixed camera
perspective, so OFRSNet trained on it alone generalizes poorly to scenes with
motorcycles or a different camera angle. To close that gap, extract frames
from your own footage and annotate a batch of them alongside KITTI:

```bash
# 6a: turn data/footage (iPhone .mov/.mp4/.heic, or any video/image folder)
#     into sampled JPG/PNG snapshots
python -m src.extract_frames --fps 0.5 --max-size 1280
#   --fps        frames sampled per second of video (default 1.0)
#   --max-size   cap the longest side in px, downscale only (0 = native)
#   --input/--out  override source/destination directories

# 6b: seed each frame's amodal mask from the model's OWN current prediction
#     (Mask2Former visible road + OFRSNet's occluded-region patch) instead of
#     a blank mask -- you correct the model's mistakes directly, which puts
#     you right on the motorcycle failure cases instead of drawing from scratch.
#     NEVER overwrites an existing data/amodal_road/<base>.png.
python -m src.s6_prepare_footage
python -m src.s6_prepare_footage --limit 20 --ckpt checkpoints/ofrsnet_best.pt

# then correct the seeded masks:
python -m src.annotator --source footage
```

Unlike KITTI, footage has no ground truth, so the annotator's "visible road"
overlay is instead Mask2Former's own `road` class (from
`data/processed/semantic_ofrs/`), and the occlusion hint is computed on the
fly from the same near-road restriction used elsewhere in the pipeline
(`config.ROAD_NEIGHBOURHOOD_PX`).

**Training automatically pools both sources** — `src/ofrs/train.py` calls
`annotated_bases()`, which is just the filename intersection of
`data/processed/semantic_ofrs/*.png` and `data/amodal_road/*.png`. Since KITTI
bases (`um_*`, `umm_*`, `uu_*`) and footage bases (`IMG_*`, `ScreenRecording_*`)
never collide, annotating footage frames folds them straight into the next
training run with zero config changes.

## Final dataset layout

```
data/
├── raw/data_road/…                     # original KITTI (RGB + GT)
├── processed/
│   ├── visible_road/  um_000000.png    # original visible road mask
│   ├── foreground/    um_000000.png    # foreground object mask
│   ├── occlusion/     um_000000.png    # occlusion hint
│   ├── semantic_ofrs/ um_000000.png    # 11-class OFRS semantic map (OFRSNet INPUT)
│   └── preview/       *_fg.png *_occ.png *_sem.png (optional QA overlays)
└── amodal_road/       um_000000.png    # ← manually corrected AMODAL GT (OFRSNet TARGET)
```

A single dataset sample is therefore:

* **RGB image** — `data/raw/data_road/training/image_2/<base>.png`
* **Visible road mask** — `data/processed/visible_road/<base>.png`
* **Foreground mask** — `data/processed/foreground/<base>.png`
* **Amodal road mask** — `data/amodal_road/<base>.png`

---

## Configuration

Everything tunable lives in `config.py`:

* `MODEL_ID` — the Mask2Former checkpoint (Mapillary Vistas semantic).
* `FOREGROUND_KEYWORDS` — class names treated as occluding foreground
  (substring-matched against the model's labels, so spelling-robust).
* `FOREGROUND_DILATE_PX` — grow object masks to cover anti-aliased edges.
* `ROAD_NEIGHBOURHOOD_PX` — how close to the road an object must be to count as
  an occluder (used by `common.occluder_blob_mask`'s whole-blob eligibility test).
* `SPLIT` — `training` (has road GT → amodal labels) or `testing`.
* `OFRS_MAX_SIDE` — cap on the longer image side (px) before feeding OFRSNet;
  downscales uniformly (aspect preserved) only if exceeded. Raise it for more
  detail at the cost of compute, lower it if inference is too slow.
* `OFRS_NET_STRIDE` — the network's total downsampling factor (8); images are
  padded up to a multiple of this so the internal down/up-sampling lines up.
  Only change this if you also change the number of DownBlocks in `ofrs/model.py`.
* `OFRS_USE_GEOMETRY` — master switch for the ground-manifold stream. `False`
  restores the original semantic-only global-context module (use it for the
  A/B ablation).
* `OFRS_ATTN_KV_STRIDE` — keys/values are pooled by this factor, so Tier 2 costs
  O(N × N/stride²) instead of O(N²). Raise it if attention is too slow/memory
  hungry, lower it for finer message passing.
* `OFRS_TAU_H_INIT` / `OFRS_TAU_D_INIT` — initial manifold-membership softness
  (metres) and ground-plane neighbourhood radius (metres²). Both are *learnable*;
  these only set the starting point.
* `GEOM_FALLBACK_HFOV_DEG` — assumed horizontal FOV for uncalibrated images.
* `GEOM_MAX_DEPTH_M`, `GEOM_RANSAC_*`, `GEOM_MIN_GROUND_PX` — plane-fitting
  robustness knobs (the 30 m depth cap follows the GroundNet paper, Sec. 5.4).

---

## Notes & caveats

* KITTI ships road GT for the **training split only** (289 images); amodal
  labels can only be made there. The testing split still gets foreground masks.
* KITTI encodes road in the **blue channel** of the GT PNG — handled in
  `common.decode_kitti_road_gt`.
* The occlusion hint is only guidance; the amodal ground truth is whatever you
  paint. Reconstruct road only where it is physically plausible (don't invent
  road behind a wall just because a car is parked in front of it).
