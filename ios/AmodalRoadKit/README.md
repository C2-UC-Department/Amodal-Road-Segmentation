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
returns visible-road", so it doesn't have to be a strict superset.

**`compose_amodal_mask`/`occluder_blob_mask` are now ported too**
(`AmodalCompose.swift`, `Dilation.swift`, `ConnectedComponents.swift`), so
the view no longer uses a `visible || rawOfrsRoad` simplification -- OFRSNet's
raw output now only PATCHES the road within person/vehicle blobs that touch
the near-road band (`OFRSClasses.roadNeighbourhoodPx = 25`), never redrawing
already-visible road, matching `predict.py`'s real behaviour. Two supporting
pieces needed porting first, neither trivial:
- `Dilation.swift` ports `common.dilate`, i.e. `cv2.dilate` with a
  `cv2.MORPH_ELLIPSE` structuring element -- verified this is NOT the same
  as a naive `dx^2+dy^2<=r^2` circle test (44 of 2005 "on" pixels differ for
  r=25) by comparing against a real `cv2.getStructuringElement` kernel
  directly; OpenCV's actual closed-form per-row half-width formula
  (`dx = round(sqrt(r^2-dy^2))`, banker's rounding) matches bit-for-bit.
  Implemented with row-prefix sums so the horizontal window-OR per `(row,
  dy)` pair is O(1), not an O(kernel width) rescan -- this runs a 51x51
  kernel over full-photo-resolution masks, so the naive approach would be
  ~1000x slower for no benefit.
- `ConnectedComponents.swift` ports `cv2.connectedComponents(...,
  connectivity=8)`'s GROUPING (not its specific label numbering, which
  nothing here depends on) via standard union-find.
- `occluderBlobMask` (whole connected blobs of person/vehicle pixels that
  touch the dilated visible-road band, admitted IN FULL -- not just the
  per-pixel slice within the band, since the whole point is not to silently
  clip a vehicle's roof/body just because its tires are the only part
  literally close to the road) and `composeAmodalMask` are both
  golden-value tested against the REAL `predict.py::compose_amodal_mask` +
  `common.occluder_blob_mask` on two synthetic scenes (40x50 and 60x60,
  including an 8-connectivity check: two vehicle blobs touching only at a
  pixel corner must be treated as ONE component, and a person blob that
  does NOT touch the road must be excluded even where OFRSNet's raw output
  is true there) -- exact match, not just close.

On the real sample photo, switching from the simplification to the real
port changed the recovered-pixel count only slightly (8,778 -> 8,743 px --
close, not identical, since the occluder-blob restriction excludes any
raw-network "recovered" pixels that aren't actually within a qualifying
occluder's footprint), confirming the simplification's visible-road
boundary handling was already close to correct on this photo and the real
gap was specifically the occluder-restriction logic, not something else
entirely.

**Vehicle-instance detection (YOLOInstanceModel) is wired in too.** NMS +
mask-prototype decode is fully implemented in `YOLOInstanceModel.swift`
(box/score/mask-coefficient parsing, greedy IoU NMS, `coeffs . protos`
mask assembly matching `ultralytics.utils.ops.process_mask(upsample=True)`
exactly -- verified by reading `Detect._inference`/`_get_decode_boxes`/
`Segment._inference` directly, not assumed, since box decode already
happens inside the traced Core ML graph and only confidence-filtering/NMS/
mask-assembly needed porting). Real photos aren't square, so unlike
`ImagePreprocessing`'s plain stretch-resize (fine for depth/semantic, whose
own downstream correction or training data tolerates it), vehicle-box
*position* needed real aspect-preserving letterboxing --
`Letterbox.swift` ports `ultralytics.data.augment.LetterBox`'s geometry
(golden-value tested against real Python parameters), with construction
(`LetterboxImage.swift`) living in the app for the same CoreGraphics/
CVPixelBuffer reasons `ImageBytes.swift` does.

Verified against real Ultralytics detections on the real sample photo via
the same temporary-executable-target technique, and found two REAL,
non-obvious CoreGraphics/CoreVideo bugs in the process, neither of which
crashed -- both silently produced wrong pixels:
1. `kCVPixelFormatType_32BGRA` is BGRA in *memory*; `CGImageAlphaInfo
   .premultipliedFirst` alone (no explicit byte order) does not produce
   that layout -- needs `.byteOrder32Little`. Without it, the first attempt
   fed the model a channel-shuffled frame and got 0 detections instead of
   the real photo's ~10-11.
2. A `CGContext` built directly over a `CVPixelBuffer`'s own memory
   (`CGContext(data: CVPixelBufferGetBaseAddress(...), ...)`) does NOT need
   a vertical flip to produce a top-left-origin image, despite Quartz's
   usual bottom-up convention -- adding one (the first attempt did)
   silently produced detections at the wrong vertical position. Confirmed
   empirically by comparing decoded box coordinates with and without the
   flip against real Ultralytics output, not assumed from documentation.

## A critical Simulator-only bug, found by actually looking at output

While wiring vehicle detection into `PhotoSemanticView`, the semantic
overlay on the SAME real photo that had worked correctly in every prior
`swift run`-based verification suddenly showed "100% road" and depth
collapsed to a solid black image -- RANSAC then had 0 usable ground points.
This reproduced consistently across relaunches (ruling out a one-off race),
and every model wrapper's own golden-value tests still passed 56/56 in the
SwiftPM suite, which _only ever runs on macOS_ (`swift test`), never on the
iOS Simulator target the app actually ships to test.

Root-caused by systematic elimination, not guessed: fixed
`ImageBytes.rgb`'s ambiguous byte order first (a real, worth-having fix,
matching `LetterboxImage`'s explicit one) -- no change. Then forced
`MLModelConfiguration.computeUnits = .cpuOnly` for just the semantic model
as a diagnostic -- the bug vanished immediately, road percentage matched
the `swift run` reference almost exactly (77,679 px vs. 77,774 px). Depth
showed the identical symptom (solid black) and the identical fix. **The iOS
Simulator's Neural Engine emulation produced silently wrong (not crashing,
not NaN, just wrong) output for these two models specifically** -- a real,
documented risk of Simulator's ANE path being an approximation of real
hardware, not a guarantee of identical numerics. `ModelBundle.swift` now
forces `.cpuOnly` for all four bundled models as the safe default until
real-device testing is possible in an environment that has one (not
available here) -- this trades inference speed for correctness on faith
that a real device's actual ANE won't reproduce this, which needs
confirming, not assuming, before it's removed.

This is the reason `swift test`/`swift run`-based verification, while a
legitimate and heavily-used technique throughout this package's development
(see every other `Tests/.../GoldenData_*.swift` file), has a real, now-
demonstrated blind spot: it never touches the iOS Simulator's own Core ML
execution path. Every model-behavior claim this README makes was re-checked
via an actual Simulator screenshot after this was found, not left resting
on the `swift test` numbers alone.

**Camera-height-prior correction is wired in too** (`VehicleHeightEstimation.swift`,
port of `src/instances.py`'s `_roofline_height_m`/
`estimate_camera_height_from_vehicles`, golden-value tested against real
Python output including the median/25th-75th-percentile spread
computation). A qualifying vehicle's roofline height above the fitted
plane, compared against a real-world prior, rescales the whole metric
world via `PlaneGeometry.rescalePlaneToHeight` -- run BEFORE amodal
completion, exactly mirroring the Python pipeline's own ordering. Known,
documented simplification: Python gates this on the instance LABEL being
exactly `"car"` (`config.VEHICLE_ROOF_HEIGHT_PRIOR_M`); the mobile instance
student (Phase 2) has no per-class labels, so this package applies the
"car" prior to every detection uniformly (`BevConfig.vehicleRoofHeightPriorM`'s
doc comment) -- a real accuracy risk specifically for trucks/buses, not
something to lose track of. On the real sample photo this took the implied
camera height from an uncorrected 0.56m (physically implausible -- nobody
holds a phone that low) to a corrected **1.57m** (a completely plausible
phone-held height), using 2 qualifying vehicles, confirmed on the actual
Simulator screenshot, not just the golden-value tests.

**BEV projection is wired in too** (`BevValidity.swift`, new: golden-value
tested `bevValidity`/`measurableMask` port of `bev.bev_validity`/
`bev.measurable_mask` -- found and fixed a real bug in the process, see
below). `PhotoSemanticView` now warps the visible and amodal road masks
into a bird's-eye-view raster via the already-tested `Warp`/`Homography`,
computes `occluded_bev = amodal_bev & ~visible_bev & bev_valid`, and reports
total amodal/visible/occluded area in m² -- using the ALREADY
height-corrected plane, so unlike Python's `occluded_road_m2_raw`, no
separate `scale_factor` is needed.

Two real findings from verifying this on the real photo, not assumed:

1. **A genuine rounding bug**, caught by `BevValidityTests`: the
   `pythonRound(x * 100) / 100` pattern used elsewhere in this package for
   2-decimal rounding is NOT equivalent to Python's `round(x, 2)` --
   multiplying by 100 first can itself introduce floating-point error that
   lands exactly on `.5`, creating an artificial tie that `.toNearestOrEven`
   then "correctly" resolves for a tie that was never really there
   (`26.774999999999998578`, definitively closer to `26.77`, rounded to the
   wrong `26.78`). Fixed with a new `pythonRound2` helper
   (`String(format: "%.2f", x)`, verified to match Python exactly on this
   value) -- `BevGrid.swift`'s header now documents why the two rounding
   helpers aren't interchangeable.
2. **`occluded_road_m2` reads 0.00 on the real sample photo** -- investigated,
   not left as a mystery, and re-investigated again after
   `compose_amodal_mask` was ported (see above), since that was the leading
   suspect and it turned out NOT to fully fix it. A temporary diagnostic
   (counting `amodalBevRaw && !visibleBevRaw` BEFORE applying `bevValid`)
   showed that count is exactly 0 on this photo: no BEV cell's
   nearest-sampled source pixel lands inside the recovered patch region at
   all. `warpToBev` is an INVERSE warp (each BEV cell samples exactly one
   nearest source pixel via the same `H`), and on this specific photo the
   real (post-fix) recovered patch sits as a thin strip right along the
   visible road's far (most-distant) edge -- exactly where ground
   resolution is already at its worst (`Measurement.groundM2PerPixel`
   degrades roughly with range^3) -- so a BEV cell there is more likely to
   nearest-sample a visible-road or off-road neighbour instead of landing
   inside the strip. This is an inherent property of nearest-neighbour
   sampling on a thin region, present in Python's own
   `cv2.warpPerspective`-based `warp_to_bev` too (`Warp`/`BevValidity`/
   `Measurement` are independently golden-value tested and match OpenCV
   exactly) -- not a Swift-side bug, and not something the
   `compose_amodal_mask` port alone resolves. Expect `occluded_road_m2` to
   read non-zero on photos where the recovered patch is wider/closer (a car
   parked squarely across a near section of visible road) rather than a
   thin far-edge sliver like this sample -- documented in
   `PhotoSemanticView.swift` directly above `runBevProjection`.

**Per-vehicle attribution is now ported too** -- the last major
`disturbance.py`-equivalent piece. Three new files, each golden-value
tested against the real Python source (not just Swift self-consistency,
except where noted):
- `BottomContour.swift` ports `instances.bottom_contour_points`: per image
  column, the bottom-most `true` pixel of a mask, excluding columns whose
  contour sits within 2 rows of the mask's own bottom edge (possibly
  frame-truncated ground contact). Pure per-pixel math, exact port.
- `OccluderInstances.swift` ports `instances.occluder_instance_ids` (plus
  a new `AmodalCompose.occluderSupport`, factored out of
  `composeAmodalMask` so both share one definition of "the occluders that
  matter"): detections are matched against the occluder support
  ascending-by-score (so the most confident detection wins any overlap),
  leftover support big enough to matter
  (`AttributionConfig.instanceMinBlobPx`) survives as its own "blob"-source
  instance via `ConnectedComponents`, and everything is renumbered
  1...N by descending pixel area. Golden-value tested on a 100x60 synthetic
  scene exercising a detected blob, an undetected-but-large leftover blob,
  a too-small sliver correctly dropped, and a person blob that must NOT
  appear at all since it doesn't touch the road.
- `Attribution.swift` ports `disturbance._ground_contact_seed_ids`
  (`groundContactSeedIds`) and `disturbance.attribute`. `attribute` is
  exact/deterministic (mask AND + area formula, no OpenCV) and is
  golden-value tested exactly, including the `unattributed` bucket (cells
  with no seed at all), not just the per-vehicle split.
  `groundContactSeedIds` composes `BottomContour` + the already-tested
  `Homography.projectPointsToBev` + `FootprintBand.rasterize` -- and
  `FootprintBand` does NOT need `cv2.polylines` pixel parity (see that
  file's header), so this is tested BEHAVIOURALLY instead (smallest-vehicle-
  first overlap resolution, low-ground-contact-coverage detection), the
  same posture `FootprintBandTests.swift` already established, not an
  exact-raster-match test that would fail for reasons unrelated to any
  real bug.

Wired into `PhotoSemanticView` as a new step after BEV projection
(`runAttribution`): the same YOLO detections used for camera-height
correction are resampled onto the semantic map's own grid (reusing
`vehicleMaskAtGrid`, generalized to target that resolution instead of
depth's), split into instances, and the already-computed `occludedBev`
raster is divided between them. `scaleFactor` (calibrated -> raw area,
mirroring `disturbance.analyze`'s `(implied / target) ** 2`) is threaded
through from the height-correction step. Verified end-to-end on the real
sample photo via Simulator screenshot: the full chain runs without error
(11 detections -> instances -> seeds -> attribution, 2382ms) and correctly
reports "no occluded area attributed to a specific vehicle" -- consistent
with, not contradicting, the already-documented finding that this
particular photo's `occludedBev` is empty (see the BEV projection section
above). Honest caveat: because of that, this run did NOT exercise the
actual non-zero per-vehicle split path on real photo data -- only the
golden-value tests (real Python fixtures, not this photo) confirm that
path's correctness directly. Worth re-checking on a photo where a vehicle
visibly occludes road once one is available.

**`width_disturbance` is ported too** (`WidthDisturbance.swift`: `rowSpanM`/
`rowCountM` port `bev.row_span_m`/`bev.row_count_m`, `measure` ports
`disturbance.width_disturbance`) -- at each along-road BEV row, what
fraction of the road's CROSS-SECTION is blocked, distinct from area on
purpose (a small occluded patch on a wide road and a full-width blockage
on a narrow one can have the same area but very different real-world
consequences). Golden-value tested against real Python output on a
synthetic scene with three road-width bands (two usable, one below
`AttributionConfig.widthMinRoadSpanM` and correctly excluded) and a vehicle
with zero occluded cells (exercises the `nil`/all-zero result path, not
just the populated one) -- including `np.argmax`'s first-occurrence
tie-breaking on the max-fraction row, matched explicitly rather than
assumed. Wired into `runAttribution`'s per-vehicle lines as a
"max width blocked N%" figure alongside area.

Still missing from a full `disturbance.py`-equivalent pipeline:
`AVCameraCalibrationData` (live-capture-only, no photo-library analogue).

## Results UI

`PhotoSemanticView` used to BE the whole screen -- every pipeline stage's
raw image/text dumped in a vertical stack, effectively a debug view wearing
the app's primary tab. That's now `ResultsView.swift`, a real product
surface built once a photo finishes processing:

- **Hero card**: occluded road area in m² (the headline number), the
  percentage of reconstructed road that figure represents, visible/total/
  measured-range metrics, and a calibration badge (green checkmark for
  EXIF/AVFoundation, amber warning for the `fov_prior` blind guess --
  every area on screen inherits that uncertainty, so it's surfaced right
  next to the numbers, not buried).
- **"Things to know"**: `PhotoSemanticView.process(data:)` now accumulates
  a `warnings: [String]` array mirroring `disturbance.DisturbanceResult
  .warnings` (single-vehicle height estimate, disagreeing height samples,
  large pre-correction scale error, no qualifying vehicle, cropped ground
  contact, truncated measurable range, large unattributed fraction) --
  surfaced as its own card, not a debug log line, per the migration plan's
  Section 9 ("port the caveat system into the UI prominently... this app
  makes quantitative real-world claims").
- **Bird's-eye view + per-vehicle list**: the attribution BEV image plus a
  list of `VehicleResult`s (colour-matched to the BEV raster via a shared
  `AttributionPalette`, not a coincidentally-similar duplicate palette),
  each showing area, max-width-blocked %, detection confidence or
  "inferred from semantic segmentation" for blob-source instances.
- **Technical details**: the old per-stage debug stack (semantic overlay,
  depth, vehicle boxes, timings) lives on behind a `DisclosureGroup`,
  collapsed by default -- nothing lost, just no longer the primary content.
- A photo with no usable ground plane has no `PipelineResult` to show (same
  posture as `disturbance.DisturbanceResult.ok`) and stays on the
  picker/status screen with a plain-language explanation instead of a
  degenerate results screen.

Verified on the real sample photo via Simulator screenshot: hero card,
warnings card (correctly showing the measurable-range caveat), BEV image,
and the empty-attribution state ("No occluded area was attributed to a
specific vehicle", consistent with that photo's already-documented 0.00 m²
occluded area) all render correctly; a temporary `ScrollViewReader`
scroll-to-bottom (removed after) confirmed the vehicle list container and
"Technical details" disclosure lay out correctly below the fold. Honest
gap: because this sample photo's occluded area is 0, the populated
per-vehicle list row layout was NOT exercised on a real screenshot, only
reasoned through -- worth a follow-up check once a photo with a real
non-zero attribution is available.

### Three real bugs found after shipping the results UI, fixed via screenshot verification

User-reported, not self-discovered -- a reminder that this file's own
"verified on the real sample photo" claims above only cover what was
actually looked at, not everything the screen renders:

1. **The "Vehicle detections" image in Technical Details was upside down.**
   `renderVehicleBoxes` (`PhotoSemanticView.swift`) draws the original photo
   into a fresh `CGContext(data: nil, ...)`. A `translateBy`/`scaleBy` flip
   had been added on the theory that such a context draws a `CGImage`
   upside down without one -- plausible-sounding, matches a well-known class
   of CoreGraphics gotcha this exact codebase has hit before (see
   `LetterboxImage.swift`'s header), and WRONG for this specific case: a
   screenshot of the enlarged image (not the tiny old debug thumbnail,
   which made this too small to notice) showed the photo itself flipped,
   proving `context.draw(image:in:)` already orients correctly in this
   context's native space and the flip was actively wrong. Fixed by
   removing the CTM flip and keeping the box rect's Y in that same native
   (bottom-left-origin) convention (`origHeight - oy2`, the function's
   original formula, paired correctly this time with no context flip) --
   verified with a second screenshot showing boxes correctly outlining the
   actual vehicles.
2. **The Bird's-eye view card looked empty.** `ResultsView`'s BEV card
   showed only `renderAttributionBev`'s output, which draws occluded cells
   ONLY -- on a photo with little/no occlusion (like the sample photo,
   0.00 m² occluded), that left the entire, much larger visible-road extent
   undrawn, rendering as a near-solid dark rectangle even though real road
   data existed. Not a data bug -- a visualization gap. Fixed by having the
   shared `BevRenderer` (see below) draw `visibleBev` as a dim-green base
   layer first, so the card now shows the whole reconstructed road, with
   occlusion layered on top -- confirmed via screenshot showing a real green
   wedge where before there was only black.
3. **No way to select a vehicle to see its own occlusion share.** The
   per-vehicle list showed each vehicle's area/width figures, but rows
   weren't interactive and the BEV card was a single static image, so
   there was no way to visually correlate "this list row" with "this region
   of the map" -- missing the interactive-selection behavior
   `disturbance.py`'s own `render_bev(selected=...)`/`--vehicle N` provides.
   Fixed: `PipelineResult` now carries the raw BEV rasters
   (`occludedBev`/`instBev`/`visibleBev`) instead of a single pre-rendered
   image, `PhotoSemanticView`'s and `ResultsView`'s renders both go through
   one shared `BevRenderer.render(..., selectedId:)` so they can't drift,
   and tapping a vehicle row toggles `@State selectedVehicleId`, live
   re-rendering the BEV image with that vehicle's cells in magenta and
   every other vehicle's dimmed to grey. Honest gap: verified this compiles
   and renders correctly for the UNSELECTED state (screenshot showed the
   new green base layer working); the sample photo's 0.00 m² occluded area
   means there is no selectable row to tap on it at all, so the actual
   tap-to-highlight interaction is unverified by screenshot -- it rests on
   the same straightforward, low-risk SwiftUI pattern (`@State` toggle +
   `onTapGesture`) as everything else in this file, not on direct
   observation. Worth confirming once a photo with real per-vehicle
   attribution is available.

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
