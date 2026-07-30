"""Interactive parking-disturbance measurement.

    streamlit run app.py

Walks the workflow in three stages: segment the scene, pick which detected
vehicle is the parked one, then read off the road area it takes out of use.

All computation lives in src/disturbance.py -- this file only chooses inputs and
draws results, so the CLI (`python -m src.disturbance`) and this app can never
report different numbers for the same image.

Lives at the project root, next to run_pipeline.py, so the `import config`
bootstrap is identical to every other module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from src import bev  # noqa: E402
from src import disturbance as dist  # noqa: E402

# Click-to-select is a nicety, not a requirement: Streamlit cannot return image
# click coordinates on its own. Guarded exactly like the optional GeoCalib
# dependency in src/calibration.py -- the radio list below works either way.
try:
    from streamlit_image_coordinates import streamlit_image_coordinates as _click
    CLICK_OK = True
except Exception:  # noqa: BLE001
    CLICK_OK = False

UPLOAD_DIR = config.DATA_DIR / "uploads"
BROWSE_DIRS = [config.FRAMES_DIR, config.DATA_DIR / "RealWorld",
               config.KITTI_ROOT / config.SPLIT / "image_2"]

st.set_page_config(page_title="Parking disturbance", page_icon="🅿️", layout="wide")


# --------------------------------------------------------------------------- #
# Cached heavy work
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading models (first run downloads weights)...")
def get_models(ckpt: str, use_instance_model: bool):
    return dist.load_models(ckpt, use_instance_model=use_instance_model)


@st.cache_data(show_spinner="Segmenting, fitting the ground plane, rasterising...")
def run(_models, image_path: str, ckpt: str, use_instance_model: bool,
        camera_height: float, use_height_prior: bool, threshold: float,
        min_score: float, ppm: float, max_m2_per_px: float):
    """Cached on every parameter that can change the result, so re-picking a
    vehicle is instant -- selection is a pure read of what this returned."""
    return dist.analyze(Path(image_path), _models,
                        camera_height_m=camera_height,
                        use_height_prior=use_height_prior,
                        threshold=threshold, min_score=min_score,
                        grid=bev.BevGrid.from_config(ppm=ppm),
                        max_m2_per_px=max_m2_per_px)


def _pick_image() -> Path | None:
    mode = st.sidebar.radio("Image source", ["Browse project data", "Upload"],
                            horizontal=True)
    if mode == "Upload":
        up = st.sidebar.file_uploader("Photo", type=["png", "jpg", "jpeg", "bmp",
                                                     "tif", "tiff"])
        if up is None:
            return None
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        # Written to disk because analyze() keys the Step-7 geometry cache off the
        # file stem, and because depth/plane estimation reads the path.
        path = UPLOAD_DIR / up.name
        path.write_bytes(up.getbuffer())
        return path

    dirs = [d for d in BROWSE_DIRS if d.exists()]
    if not dirs:
        st.sidebar.warning("No image directories found under data/.")
        return None
    folder = st.sidebar.selectbox("Folder", dirs,
                                 format_func=lambda p: str(p.relative_to(config.ROOT)))
    from src import predict
    images = predict.list_images(folder)
    if not images:
        st.sidebar.warning(f"No images in {folder}")
        return None
    return st.sidebar.selectbox("Image", images, format_func=lambda p: p.name)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("Parking disturbance")
image_path = _pick_image()

st.sidebar.divider()
ckpt = st.sidebar.text_input("OFRSNet checkpoint",
                             str(config.CKPT_DIR / "ofrsnet_best.pt"))
use_instance_model = st.sidebar.checkbox(
    "Use instance segmentation", value=True,
    help="Off: occluders are split by connected components instead, which merges "
         "touching vehicles but downloads nothing.")

st.sidebar.subheader("Metric scale")
use_height_prior = st.sidebar.checkbox(
    "Correct scale with a camera-height prior", value=True,
    help="Monocular depth gets absolute scale wrong; the fitted plane exposes the "
         "camera height as 1/||n||, so rescaling to the true height fixes the "
         "whole metric world. A linear scale error is a QUADRATIC area error.")
camera_height = st.sidebar.number_input(
    "True camera height (m)", 0.3, 10.0, float(config.BEV_CAMERA_HEIGHT_M), 0.05,
    disabled=not use_height_prior)

st.sidebar.subheader("Thresholds")
threshold = st.sidebar.slider("OFRSNet road probability", 0.05, 0.95, 0.5, 0.05)
min_score = st.sidebar.slider("Instance score", 0.05, 0.95,
                              float(config.INSTANCE_MIN_SCORE), 0.05)

with st.sidebar.expander("BEV grid"):
    ppm = st.slider("Pixels per metre", 5.0, 50.0, float(config.BEV_PPM), 1.0)
    max_m2_per_px = st.select_slider(
        "Max ground m² per source pixel", [0.005, 0.01, 0.02, 0.05, 0.1, 0.5],
        value=float(config.BEV_MAX_M2_PER_PIXEL),
        help="Ground resolution decays with roughly the CUBE of range. Cells whose "
             "source pixel covers more road than this are in frame but not "
             "measurable, and are excluded from every area.")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if image_path is None:
    st.title("Parking disturbance")
    st.markdown("""
Measures the **functional road space a parked vehicle takes out of use**, by
comparing two top-down views of the same scene:

| | |
|---|---|
| **amodal** road | the road as it actually is — including what is hidden behind vehicles (OFRSNet) |
| **visible** road | the road as the camera sees it (Mask2Former's `road` class) |

Their difference, rectified onto the fitted ground plane so every cell is the same
size, is road that exists but cannot be used. Each occluded cell's source pixel
*is* a pixel of whatever hides it, so attributing that area to a specific vehicle
is exact rather than heuristic.

Pick an image in the sidebar to start.
""")
    st.stop()

models = get_models(ckpt, use_instance_model)
result = run(models, str(image_path), ckpt, use_instance_model, camera_height,
             use_height_prior, threshold, min_score, ppm, max_m2_per_px)

st.title(image_path.name)
for w in result.warnings:
    st.warning(w, icon="⚠️")

candidates = result.selectable()
if not candidates:
    st.info("No vehicles were detected on the road in this image.")

# --- stage 2: selection ---
st.subheader("1 · Select the parked vehicle")
left, right = st.columns([3, 2], gap="large")

options = [v.inst_id for v in candidates]

# Keyed widget, session_state as the single source of truth, and NO `index=`.
# Passing an explicit index to an unkeyed radio silently breaks: the index is part
# of the widget's identity, so recomputing it from the previous selection makes
# Streamlit treat it as a different widget and the reported selection lags a step
# behind the one the user clicked.
SEL_KEY = "selected_vehicle"
if st.session_state.get(SEL_KEY) not in options:
    st.session_state[SEL_KEY] = options[0] if options else None

with right:
    if options:
        def _fmt(i):
            v = next(v for v in result.vehicles if v.inst_id == i)
            score = "blob" if np.isnan(v.score) else f"{v.score:.2f}"
            return f"#{i} · {v.label} · {score} · {v.pixel_area:,} px"

        st.radio("Detected vehicles", options, key=SEL_KEY, format_func=_fmt)
    if not CLICK_OK:
        st.caption("Tip: `pip install streamlit-image-coordinates` to click a "
                   "vehicle in the image instead of using this list.")

selected = st.session_state.get(SEL_KEY)
overlay = dist.render_candidates(result, selected)

with left:
    if CLICK_OK:
        shown_w = 700
        pt = _click(overlay, key="clickimg", width=shown_w)
        st.caption("Click a vehicle to select it.")
        if pt is not None:
            # Coordinates come back in DISPLAYED pixels. Display preserves aspect,
            # so one uniform factor maps both axes back to the full-size image.
            h_img, w_img = result.inst_ids.shape
            f = w_img / float(shown_w)
            u = min(max(int(pt["x"] * f), 0), w_img - 1)
            v = min(max(int(pt["y"] * f), 0), h_img - 1)
            hit = int(result.inst_ids[v, u])
            if hit in options and hit != selected:
                # Writing the widget's own key is what makes the radio follow the
                # click on the next run.
                st.session_state[SEL_KEY] = hit
                st.rerun()
    else:
        st.image(overlay, caption="Occluders on the road, numbered "
                                  "(grey = not a selectable vehicle)",
                 width='stretch')

if not result.ok:
    st.error("No BEV could be produced for this image, so no area can be reported. "
             "See the warnings above.")
    st.stop()

# --- stage 3: measurement ---
st.subheader("2 · Measured disturbance")
sel = next((v for v in result.vehicles if v.inst_id == selected), None)
own = result.per_vehicle.get(selected, {}) if sel else {}

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"#{selected} occluded road" if sel else "Selected",
          f"{own.get('area_m2', 0):.2f} m²",
          help="Road this vehicle hides, calibrated with the camera-height prior.")
c2.metric("Share of amodal road",
          f"{100 * own.get('area_m2', 0) / max(result.total['amodal_road_m2'], 1e-9):.1f} %",
          help="Scale-immune, and therefore the figure to trust most.")
c3.metric("All occluders", f"{result.total['occluded_road_m2']:.2f} m²",
          f"{result.total['occluded_pct']:.1f}% of amodal road")
c4.metric("Uncalibrated", f"{own.get('area_m2_raw', 0):.2f} m²",
          help="Straight from monocular depth's own scale. Shown because the "
               "calibrated figure depends on a camera-height assumption; this one "
               "inherits the depth model's scale error instead.")

bev_col, tbl_col = st.columns([2, 3], gap="large")
with bev_col:
    st.image(dist.render_bev(result, selected),
             caption="BEV · green = visible road · cyan = occluded road · "
                     "magenta = selected vehicle's share",
             width='stretch')
with tbl_col:
    # Ranked by road cost, not by id: a busy scene can hold dozens of occluders
    # and only a handful of them hide any measurable road.
    st.dataframe(
        [{"id": v.inst_id, "label": v.label,
          "score": (None if np.isnan(v.score) else round(v.score, 3)),
          "source": v.source, "selectable": v.selectable,
          "occluded m²": result.per_vehicle.get(v.inst_id, {}).get("area_m2", 0.0),
          "uncalibrated m²": result.per_vehicle.get(v.inst_id, {}).get("area_m2_raw", 0.0),
          "px": v.pixel_area}
         for v in sorted(result.vehicles,
                         key=lambda v: -result.per_vehicle.get(v.inst_id, {})
                         .get("area_m2", 0.0))]
        + [{"id": None, "label": "unattributed", "source": "-", "selectable": False,
            "occluded m²": result.unattributed["area_m2"],
            "uncalibrated m²": result.unattributed["area_m2_raw"], "px": None}],
        hide_index=True, width='stretch')

    st.caption(
        f"Amodal road {result.total['amodal_road_m2']:.1f} m² · visible "
        f"{result.total['visible_road_m2']:.1f} m² · measured to "
        f"{result.resolution['measurable_range_m']:.1f} m · implied camera height "
        f"{result.scale['implied_camera_height_m']:.2f} m")

with st.expander("What exactly is being measured?"):
    st.markdown(f"""
The reported area is the **occlusion shadow**: road present in the amodal
reconstruction but absent from the visible segmentation, attributed to whichever
occluder's pixels produced each cell.

Two properties of that definition are worth keeping in mind when reading the
numbers above:

* **It is viewpoint-dependent.** A vehicle hides the road *behind* it, so a
  near-axial view down a street yields a long shadow and an oblique view a short
  one. It is not a fixed footprint.
* **It is range-truncated.** Ground resolution decays with roughly the cube of
  distance, so measurement stops at
  {result.resolution['measurable_range_m']:.1f} m here
  ({result.resolution['measurable_pct']:.0f}% of the grid). Any shadow reaching
  that boundary is a lower bound.

`occluded_pct` is the most robust figure: it is a ratio, so it survives both the
monocular depth scale error and the camera-height assumption.
""")
