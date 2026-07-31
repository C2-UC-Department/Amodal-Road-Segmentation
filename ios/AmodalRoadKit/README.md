# AmodalRoadKit

Swift Package Manager port of `src/bev.py`'s pure-math functions, for the iOS
migration (see the migration plan for full context — Section 4 covers what
does and doesn't get a native Swift port here vs. elsewhere).

## What's here

- `BevGrid.swift` — the metric top-down raster (`bev.BevGrid`).
- `Homography.swift` — the ground-plane homography (`bev.homography_bev_to_image`,
  `bev.project_points_to_bev`). **Read the header comment before editing**: `simd`
  matrices are column-major, and every matrix here is written out as rows
  (to match the Python source line-for-line) then hand-transposed into the
  column-major initializer — get this backwards and it's a bug that still
  often looks plausible.
- `PlaneGeometry.swift` — plane sanity checks and metric-scale correction
  (`bev.plane_is_usable`, `bev.implied_camera_height`, `bev.rescale_plane_to_height`).
- `Measurement.swift` — ground-resolution and area (`bev.ground_m2_per_pixel`, `bev.area_m2`).
- `Errors.swift` — `AmodalRoadKitError.unusablePlane`, matching Python's `raise ValueError`.
- `Config.swift` — the handful of `config.py` constants this package needs.
- `Warp.swift` — `warp_to_bev` (image -> BEV inverse-warp). Hand-rolled Swift loop,
  not vImage -- see the file header for why (no way to verify vImage's exact
  perspective-warp API surface without header access; correctness-verified against
  real `cv2.warpPerspective` output instead). vImage/Metal is a documented follow-up
  optimization once this baseline is profiled, not a blocker.
- `FootprintBand.swift` — `rasterize_footprint_band`. Independently-correct thick-line/
  disk rasterizer, not a `cv2.polylines`/`circle` pixel-parity port (the Python source
  itself says pixel-perfect parity isn't needed here — it's a Voronoi seed, not a
  rendered image).
- `NearestLabel.swift` — `nearest_label_bev`. Brute-force exact-Euclidean nearest-seed
  search (`O(cells * seeds)`), not a distance-transform algorithm port. Verified against
  real `cv2.distanceTransformWithLabels` output — and in the process turned up that
  OpenCV's `DIST_MASK_5` is itself only an *approximate* metric (~1% error, by its own
  docs), so the handful of golden-value disagreements found are near-exact-tie boundary
  cells where this implementation's exact Euclidean distance is arguably the more
  correct answer, not a bug. A JFA/BFS optimization is a documented follow-up if
  profiling ever calls for it on the full ~316k-cell production grid.

- `Unproject.swift` — `unproject` / `unproject_ground_points` (`src/geometry.py`).
  Pure per-pixel math, no dynamic control flow to de-risk; ports directly.
- `GroundFields.swift` — `derive_ground_fields`. Same story as `Unproject.swift`.
- `PlaneFit.swift` — `fit_plane_ls` (deterministic, golden-value tested exactly) and
  `fit_plane_ransac`/`estimate_ground_plane`. RANSAC uses `SplitMix64`, a small seedable
  PRNG, **not** an attempt to reproduce NumPy's PCG64 -- exact RNG parity with Python is
  neither achievable (different algorithm) nor useful (RANSAC's whole point is that many
  different random samples converge on the same answer). Validated instead by fitting a
  known synthetic noisy-planar point cloud and checking the recovered plane's direction
  against ground truth (cosine similarity > 0.999, across 5 independent seeds) -- the
  property that actually matters.

- `Calibration.swift` — port of `src/calibration.py`'s pure-math constructors
  (`K_from_focal`, `K_from_fov`, `scale_K`) and fallback chain (`calibrate`).
  Deliberately narrower than the Python source: drops the `kitti_calib` branch
  (dataset-only, no iOS analogue) and the `geocalib` branch (investigated in
  Phase 0 and found NO-GO for on-device use — see the migration plan, Section 3).
  The iOS chain instead leads with AVFoundation's per-shot camera-calibration
  data on live-captured photos (not available to the Python CLI, which only
  sees files on disk), then falls through EXIF -> `fov_prior` exactly as the
  Python module intends. EXIF tag *arithmetic* (`kFromExifTags`) is factored
  out from tag *extraction* (ImageIO, ties to a real image file) so the
  golden-value-testable half is isolated from the part that would need a real
  fixture image to unit-test — the same "isolate what's verifiable" pattern
  used for `Warp.swift`'s image I/O boundary. AVFoundation's
  `AVCameraCalibrationData` wiring (which needs a live capture session) is
  deferred to Phase 4's app integration for the same reason: nothing to
  meaningfully unit-test in a SwiftPM package.

- `OFRSNetModel.swift` — Swift wrapper around the actual Core ML `.mlpackage`
  produced by `tools/coreml_export_ofrsnet.py` (loads it with `MLModel`, builds
  the five input tensors, runs `model.prediction(from:)`). This is the first
  point in the migration where Swift code *runs* a converted model rather than
  reimplementing pure math. One real bug found and fixed here: an initial
  version assumed `.float32` MLMultiArrays and crashed with SIGSEGV on the
  first call — `ct.convert(..., convert_to="mlprogram")` with no explicit
  dtype on the `TensorType`s defaults every input *and* the output to
  `.float16`, not `.float32` (confirmed by inspecting the compiled model's
  `multiArrayConstraint.dataType` directly, `65552` = `MLMultiArrayDataType.float16`'s
  raw value); the crash was the output-reading code reinterpreting a
  half-width buffer as full-width and reading twice as many bytes as actually
  existed. Every buffer is `.float16` now, matching the model exactly.
  `Tests/.../OFRSNetModelTests.swift` runs the real bundled `.mlpackage`
  through Apple's on-device Core ML runtime (not just a compile check) on a
  deterministic synthetic scene, and asserts the thresholded road/not-road
  decision has IoU > 0.95 against the real PyTorch model's output on the same
  input (golden mask in `GoldenData_OFRSNetModel.swift`, generated from
  `src/ofrs/export.py::OFRSNetExport` directly) — the same road-mask-IoU bar
  the Python-side conversion tool already established for itself
  (`tools/coreml_export_ofrsnet.py` measured 0.988949 at this exact
  resolution), with headroom for Apple's runtime being a different execution
  engine than coremltools' own Python prediction path.

- `MLMultiArrayFlatten.swift` — `MLMultiArray.float16ElementsRowMajor()`. Exists
  because of a real bug found while adding `DepthModel` below: reading a Core ML
  model OUTPUT via a raw `dataPointer` assuming it's tightly packed is not safe.
  `DepthModel`'s `predicted_depth` output has logical shape `[1, 518, 518]` but
  ACTUAL strides `[281792, 544, 1]` — Core ML pads the row stride to 544 (a
  multiple of 32) — so `dataPointer[row * 518 + col]` silently reads
  garbage/zero near the row edge instead of throwing. `OFRSNetModel`'s own
  tests never caught this because its output width (96) already IS a multiple
  of 32, so no padding was ever inserted there — purely lucky, not
  correctness. Both `OFRSNetModel` and `DepthModel` read their outputs
  through this helper now, not a raw pointer. Inputs this package allocates
  itself are unaffected (confirmed via `.strides` right after allocation —
  those come back tightly packed; only Core-ML-internally-allocated outputs
  do this).
- `DepthModel.swift` — Swift wrapper around the Core ML `.mlpackage` from
  `tools/coreml_export_depth.py` (Depth-Anything-V2-Metric-Outdoor-Small; see
  that Python tool's docstring for the Phase-0 GO decision and the
  position-embedding fix already baked into the export). Same fixed-shape/
  `.float16` story as `OFRSNetModel`. `DepthModelTests.swift` runs the real
  bundled model through Apple's on-device Core ML runtime on a deterministic
  synthetic input and checks 36 sampled points against the real PyTorch
  model's output (golden values in `GoldenData_DepthModel.swift`) — max
  observed relative error 2.5%, in line with the ~2-3.5% the Python export
  tool's own docstring already documents for this exact model.
- `ImagePreprocessing.swift` — `resizeAndNormalizeImageNet`. Deliberately
  NOT a port of `DPTImageProcessor`'s aspect-ratio-preserving resize (the
  processor `tools/coreml_export_depth.py`'s model was traced with) — this
  does a plain stretch-to-fixed-size bilinear resize instead. See the file
  header for why exact parity isn't worth chasing here (depth is already
  independently scale-corrected downstream via the camera-height prior, so a
  squashed-aspect resize's distortion is small next to error already being
  corrected elsewhere) — the opposite call from `Warp.swift`'s pixel-exact
  `cv2.warpPerspective` port, and explicitly justified as such.
- `GeometryPipeline.swift` — `resolveGeometry`: wires `DepthModel` ->
  `PlaneFit.estimateGroundPlane` -> `GroundFields.derive`, on-device. Port of
  `geometry.resolve_plane`'s COMPUTE path only (not its cache-load path,
  which has no iOS analogue). Thin orchestration, no new logic of its own to
  verify beyond what each piece already covers individually — tested
  behaviorally (`GeometryPipelineTests.swift`, real model, a plausible
  synthetic road scene) rather than golden-value, since the pieces being
  composed are already independently golden-value verified.
- `MobileSemanticModel.swift` — Swift wrapper around the Core ML `.mlpackage`
  from `tools/coreml_export_mobile_semantic.py` (the Phase 2 semantic
  student, `src/mobile_semantic/model.py`). Simplest of the three model
  wrappers: one RGB input, one logits output, no geo dict or resolution
  quirks of its own. `MobileSemanticModelTests.swift` checks argmax-class
  agreement against the real PyTorch model at 36 sample points (small
  mismatch tolerance, same rationale as `NearestLabelGoldenValueTests`).
- `ImagePreprocessing.swift` gained a second entry point,
  `resizeToUnitRangeRGB` (plain bilinear resize + `[0,1]` scale, no
  normalization -- `MobileSemanticNet` normalizes internally, see that
  model's own docstring), alongside the original `resizeAndNormalizeImageNet`
  (`DepthModel`'s pre-normalized input). Both now share one internal resize
  implementation. `ImagePreprocessingTests.swift` is the first real test of
  this file's actual math (uniform-image invariance, channel-major layout,
  bounds, a hand-computed-formula check, and a downsample blend check that
  would fail under nearest-neighbour sampling) -- there wasn't one before
  because the app-level code that exercises it (`ImageBytes.swift`, real
  `CGImage`s) isn't reachable from a SwiftPM test target; what CAN be tested
  here without ImageIO now is.

## App integration (Phase 4)

`ios/Parking Disturbance` (an Xcode project, not part of this SwiftPM
package) links `AmodalRoadKit` as a local Swift Package dependency and
bundles all four converted Core ML models under
`Parking Disturbance/Models/` — Xcode auto-compiles every `.mlpackage`
there to `.mlmodelc` and copies it into the built app (confirmed by
inspecting the built `.app` directly). `ModelBundle.swift` resolves them via
`Bundle.main` and hands them to this package's typed wrappers
(`OFRSNetModel`, `DepthModel`, `MobileSemanticModel`) — the
`Bundle.main`-based counterpart to this package's own test-only
`Bundle.module`/`#filePath` resolution. `ContentView.swift`'s debug screen
loads and runs all four on synthetic input and reports pass/fail + output
shapes; verified by actually building, installing, and launching on a
booted simulator and screenshotting the result (not just a successful
build) — all four report OK. The vehicle-instance YOLO model is loaded and
run directly via `MLModel` (no typed wrapper yet): its NMS + mask-prototype
decode isn't ported to Swift yet, so today it only proves the model loads
and produces output of the expected shape, not a usable detection list.

Model files are duplicated between `ios/AmodalRoadKit/Tests/.../Resources/`
(SwiftPM test fixtures) and `ios/Parking Disturbance/.../Models/` (the
actual app bundle) — same binaries, two different consumers with different
resource-resolution mechanisms (`Bundle.module` vs `Bundle.main`). Accepted
as a reasonable tradeoff rather than engineering a shared-resource scheme;
both follow the same small-tracked/large-gitignored `.gitignore` policy
independently.

### Real-photo pipeline (first slice)

`PhotoSemanticView.swift` is the first piece of the actual "capture ->
process -> results" workflow, not another models-load smoke test:
`PhotosPicker` (or a bundled dev-only sample photo, see below) -> real JPEG
bytes -> `EXIFCalibration.swift` (ImageIO tag extraction feeding
`Intrinsics.kFromExifTags`/`calibrate`, the app-side half of
`Calibration.swift`'s documented gap) -> `ImageBytes.swift` (`CGImage` ->
raw RGB bytes) -> `ImagePreprocessing.resizeToUnitRangeRGB` ->
`MobileSemanticModel` -> an 11-class colour overlay rendered with the same
palette `s5_export_semantics.py --preview` uses.

Verification note: simulating an actual tap on the `PhotosPicker` button (or
the debug "load bundled test photo" button) needs macOS Accessibility
permission for UI scripting, which isn't granted in this environment
(`osascript ... System Events` returns error -1719, "not allowed assistive
access") — that's a sandbox limitation, not something fixable from here.
What WAS verified directly: `swift build`/`swift test` (53/53, including the
new `ImagePreprocessingTests`), and `xcodebuild ... build` + install +
launch on a booted simulator (proves the view renders and the app doesn't
crash on load). The actual photo -> EXIF -> resize -> model -> overlay path
itself is exercised end-to-end by `ImagePreprocessingTests` (the resize/
normalize math) plus the already-golden-value-tested `MobileSemanticModel`
and `Intrinsics` -- what's NOT independently verified is specifically the
`CGImage`/ImageIO decode step and the SwiftUI button-tap plumbing, both
low-risk standard-framework usage rather than this project's own logic.
Still open: a real accessibility-permission-enabled run (or an XCUITest
target) to close that specific gap.

**Depth + RANSAC ground-plane fitting is now wired into the same view.**
After the semantic overlay, the picked photo's argmax road mask (resized
nearest-neighbour onto `DepthModel`'s own fixed 518x518 grid -- independent
resolution from the semantic model's 512x512) feeds
`GeometryPipeline.resolveGeometry` end-to-end: real depth, real RANSAC plane
fit, real implied camera height, displayed alongside a grayscale depth
visualization -- the migration plan's Phase 3 "debug screen showing fitted
plane / implied camera height / depth viz" checkpoint, now on an actual
photo rather than `GeometryPipelineTests`' synthetic scene.

Verified two ways: `xcodebuild` + install + launch on a booted simulator
(renders, doesn't crash — the UI-tap-to-trigger-it gap is the same
Accessibility-permission limitation noted above), AND a full run of the
*exact* processing logic (EXIF -> resize -> semantic -> road mask -> depth
-> RANSAC) against `SampleData/SamplePhoto.jpg`, via a temporary SwiftPM
executable target (`swift run`, then removed — an ad-hoc `swift
script.swift -L ...` invocation hit dynamic-library JIT-linking errors
first, since this package builds as a static archive; a proper SwiftPM
target sidesteps that). Real, non-degenerate output on the real photo: road
29.7%/vehicle 14.0%/person 1.9%/etc. class mix, a plane fit from 78,254 real
ground points. Implied camera height came out low (~0.57m) — expected, not
a bug: this photo has no EXIF (calibration fell back to the documented
`fov_prior` blind guess) and the raw depth output has no camera-height-prior
correction applied yet (that correction is `src/instances.py`'s job, gated
on the not-yet-built vehicle-instance Swift wiring) — exactly the
"50-400% correction typically needed" scale bias `config.py` already
documents for uncorrected depth.

**OFRSNet amodal completion is wired in too.** The bundled `OFRSNetExport.mlpackage`
was re-exported at 512x512 (matching the semantic model's own resolution;
the small 64x96 one stays as the SwiftPM test fixture) since OFRSNet's input
must align pixel-for-pixel with `sem`, and depth's native 518x518 isn't
usable directly either way (not a multiple of `OFRS_NET_STRIDE=8`). Ground
fields are re-derived at 512x512 by resizing the raw DEPTH field (bilinear)
and scaling K (`Intrinsics.scaleK`), then calling `GroundFields.derive`
fresh at the target resolution -- mirroring `geometry.load_ground_fields`'s
own resize-then-rederive pattern, not a naive resize of the already-derived
G/h fields. `n` itself is resolution-independent and carries over unchanged.
The view renders visible road (green) vs. recovered occluded road (orange).

Verified the same way as the depth/geometry step: a temporary SwiftPM
executable target running the exact logic against the real sample photo.
Real, plausible numbers: 77,774 visible road px, 62,195 amodal (network)
road px, 9,622 px recovered that weren't in the visible mask (+12.4%) --
non-degenerate, and the amodal count being LESS than the visible count in
places is expected, not a bug: OFRSNet solves a different task than "always
returns visible-road", so it doesn't have to be a strict superset. Note this
view's visualization is a simplified "raw network output vs. visible mask"
diff for debugging, not `predict.py`'s `compose_amodal_mask` (which patches
the network's prediction only within near-road occluder regions, not
everywhere) -- that patching logic hasn't been ported to Swift yet.

Still missing from a full `disturbance.py`-equivalent pipeline: BEV
projection and per-vehicle attribution (including the camera-height-prior
correction mentioned above, which depends on vehicle instances), and
`compose_amodal_mask`'s occluder-patching logic (currently a simplified
diff instead).

## Test fixtures

`Tests/AmodalRoadKitTests/Resources/OFRSNetExport.mlpackage` is a real,
small (~4MB) Core ML export tracked in git — see `.gitignore`'s comment: this
is a deliberate exception, small models are committed as test fixtures.
`DepthAnythingV2Small.mlpackage` (~47MB) is the "anything larger" case that
comment already anticipated: gitignored, NOT declared as a SwiftPM
`resources:` entry (that would fail the build outright on a checkout that
doesn't have it), and located directly via `#filePath` at test time instead
— `DepthModelTests`/`GeometryPipelineTests` skip (not fail) if it's absent.
Regenerate with:

```bash
.venv/bin/python -m tools.coreml_export_depth \
    --out ios/AmodalRoadKit/Tests/AmodalRoadKitTests/Resources/DepthAnythingV2Small.mlpackage
```

## What's NOT here yet

- The ImageIO-based EXIF tag *extraction* from a real image file (only the
  arithmetic half is ported so far — see `Calibration.swift` above).
- Live `AVCameraCalibrationData` capture wiring (Phase 4, needs an app target).
- GeoCalib Swift wiring (deprioritized NO-GO, see the migration plan Section 3).
- RANSAC vectorization (`PlaneFit.fitPlaneRANSAC` is a straightforward loop
  port, not batched — a documented follow-up once there's a real device to
  profile against, same posture as `Warp.swift`/`NearestLabel.swift`'s
  deferred optimizations).
- The full `disturbance.analyze()` Swift equivalent and app-level image
  capture/display (Phase 4, needs an app target).

## Testing philosophy

`Tests/AmodalRoadKitTests/GoldenValueTests.swift` asserts against concrete
numbers computed by the REAL `src/bev.py` (see that file's header for the
Python one-liner that generates them), not just Swift-internal
self-consistency. This is deliberate and already caught one real bug: Swift's
default `Double.rounded()` uses round-half-away-from-zero, while Python's
`round()` uses round-half-to-even (banker's rounding) — they disagree exactly
on ties (`round(734.5)`: Python → 734, Swift's default → 735). Every rounding
call in this package goes through the `pythonRound` helper in `BevGrid.swift`
specifically because of this, not the bare `.rounded()`.

If you add a new ported function, add its golden-value test the same way:
compute the expected numbers from the actual Python function on some concrete
input, hardcode them, assert Swift reproduces them — not just that Swift's
own math is internally consistent.

## Building & testing

```bash
cd ios/AmodalRoadKit
swift build
swift test
```

No Xcode project needed for this package on its own (pure SwiftPM); it'll be
consumed as a dependency once the actual iOS app target exists (Phase 4 of
the migration plan).
