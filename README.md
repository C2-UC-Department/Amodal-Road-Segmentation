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
| 5  | `src/s5_export_semantics.py` | KITTI images | `data/processed/semantic_ofrs/*.png` (**OFRSNet input**) |
| 4  | `src/annotator.py` | all of the above | **`data/amodal_road/*.png`** (final GT) |
| 6  | `src/ofrs/train.py` | semantic + amodal | `checkpoints/ofrsnet_best.pt` |

All masks are single-channel **0/255 PNG**, one per image, sharing the KITTI
base name (`um_000000.png`, `umm_000012.png`, `uu_000003.png`, …).

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

## Training OFRSNet (Steps 5–6)

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
* Everything is resized to `config.OFRS_INPUT_SIZE = (384, 1248)` (paper's size).

All OFRS knobs (class scheme, Mapillary→11 mapping, loss weights, label
smoothing, split, epochs) live in `config.py`.

## Inference on your own images (Step 7)

`src/predict.py` runs the full deploy pipeline on any folder of images
(RGB → Mask2Former semantics → OFRSNet → amodal road) and writes a 3-panel
visualization (input / semantics / predicted amodal road) plus the raw mask.

```bash
python -m src.predict --input path/to/images --out data/predictions
python -m src.predict --input data/raw/data_road/testing/image_2 --limit 20
python -m src.predict --input dashcam/ --ckpt checkpoints/ofrsnet_best.pt --no-stack
```

Outputs: `data/predictions/<name>_viz.png` (stacked visualization) and
`data/predictions/mask/<name>.png` (binary amodal road). Accepts png/jpg/bmp/tif
and recurses into sub-directories.

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
  an occluder (controls the occlusion hint).
* `SPLIT` — `training` (has road GT → amodal labels) or `testing`.

---

## Notes & caveats

* KITTI ships road GT for the **training split only** (289 images); amodal
  labels can only be made there. The testing split still gets foreground masks.
* KITTI encodes road in the **blue channel** of the GT PNG — handled in
  `common.decode_kitti_road_gt`.
* The occlusion hint is only guidance; the amodal ground truth is whatever you
  paint. Reconstruct road only where it is physically plausible (don't invent
  road behind a wall just because a car is parked in front of it).
