//
//  PhotoSemanticView.swift
//  Parking Disturbance
//
//  Phase 4: capture-or-pick-a-photo -> EXIF calibration -> mobile semantic
//  student -> 11-class overlay -> depth -> RANSAC ground plane, on a REAL
//  photo (not synthetic input, unlike the model smoke tests in
//  ContentView.swift). Matches the migration plan's Phase 3 "Demoable:
//  debug screen showing fitted plane / implied camera height / depth viz"
//  checkpoint, now on an actual photo instead of a synthetic scene
//  (GeometryPipelineTests already covered the synthetic case).
//
//  OFRSNet amodal completion is wired in too: the semantic map's road class
//  is the network's ONE input alongside G/h/gvalid, re-derived at the
//  semantic model's own 512x512 canvas (not depth's native 518x518 -- 518
//  isn't a multiple of OFRS_NET_STRIDE=8, so OFRSNet can't be exported at
//  that resolution; see resizeDepthBilinear/deriveGeometryAtSemanticRes
//  below for the resize-depth-then-rederive-fields approach, mirroring
//  geometry.load_ground_fields's own resize-then-rederive pattern rather
//  than naively resizing the already-derived G/h fields themselves). The
//  raw network output is then composed with the visible semantic road via
//  AmodalRoadKit.AmodalCompose.composeAmodalMask (predict.py's real
//  compose_amodal_mask/common.occluder_blob_mask -- OFRSNet only patches
//  the road within person/vehicle blobs that touch the near-road band, it
//  never redraws already-visible road), not a `visible || rawOfrsRoad`
//  simplification.
//
//  Vehicle-instance detection (YOLOInstanceModel's NMS + mask-prototype
//  decode, see that file) is wired in too: the photo is letterboxed to the
//  detector's own square 640x640 input (AmodalRoadKit.Letterbox +
//  LetterboxImage.swift -- NOT the plain stretch-resize
//  ImagePreprocessing's other consumers use, since detection box POSITION
//  is the whole point here, unlike depth/semantic where a squashed aspect
//  ratio is a minor, already-corrected-elsewhere distortion), detected,
//  and boxes are mapped back to the original photo via
//  `Letterbox.toOriginal` for display.
//
//  Camera-height-prior correction (src/instances.py's
//  estimate_camera_height_from_vehicles, ported as
//  AmodalRoadKit.VehicleHeightEstimation) is wired in too, now that vehicle
//  instances exist: each detection's mask gets mapped from the detector's
//  letterboxed 640x640 space onto DEPTH's own 518x518 grid (composing
//  Letterbox.toOriginal with depth's own plain-stretch mapping -- see
//  vehicleMaskAtDepthResolution below), and a qualifying car's roofline
//  height above the fitted plane, compared against a real-world prior,
//  rescales the whole metric world via PlaneGeometry.rescalePlaneToHeight
//  BEFORE amodal completion runs -- exactly mirroring
//  disturbance.py/predict.py's own ordering. Known simplification (see
//  BevConfig.vehicleRoofHeightPriorM's doc comment): the mobile instance
//  student has no per-class labels, so every detection is treated as a
//  "car" for this purpose, unlike Python's label-gated version.
//
//  BEV projection (visible/amodal/occluded road area in m^2, see
//  runBevProjection below) is wired in too.
//
//  Per-vehicle attribution is wired in too (runAttribution below): the same
//  YOLO detections used for camera-height correction are matched against
//  the semantic occluder support (AmodalRoadKit.OccluderInstances, port of
//  src/instances.py's occluder_instance_ids) to split the occluder region
//  into per-vehicle ids, each vehicle's ground-contact footprint is
//  projected to BEV and propagated by nearest-seed (AmodalRoadKit.Attribution,
//  port of disturbance.py's _ground_contact_seed_ids/attribute), and the
//  occluded BEV area is split between vehicles (+ an "unattributed"
//  remainder for occluder-support pixels no detection claimed).
//
//  Still missing from the full disturbance.py-equivalent pipeline:
//  AVCameraCalibrationData (live-capture-only, no photo-library analogue)
//  and width_disturbance (the per-row road-width-blocked metric).

import SwiftUI
import PhotosUI
import AmodalRoadKit
import simd

private let roadClassIndex = 0   // config.OFRS_CLASSES.index("road")

/// Same palette as s5_export_semantics.py::OFRS_PALETTE, so the on-device
/// overlay looks like the existing CLI preview tool's output.
private let ofrsPalette: [(UInt8, UInt8, UInt8)] = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (220, 220, 0), (107, 142, 35), (220, 20, 60), (0, 0, 142), (0, 0, 0),
]

struct PhotoSemanticView: View {
    @State private var pickerItem: PhotosPickerItem?
    @State private var sourceImage: UIImage?
    @State private var overlayImage: UIImage?
    @State private var depthImage: UIImage?
    @State private var amodalImage: UIImage?
    @State private var vehicleImage: UIImage?
    @State private var bevImage: UIImage?
    @State private var attributionImage: UIImage?
    @State private var statusText = "Pick a photo to measure how much road it occludes."
    @State private var geometryText: String?
    @State private var amodalText: String?
    @State private var vehicleText: String?
    @State private var heightCorrectionText: String?
    @State private var bevText: String?
    @State private var attributionText: String?
    @State private var isProcessing = false
    @State private var result: PipelineResult?

    var body: some View {
        Group {
            if let result {
                ResultsView(result: result, onNewPhoto: reset)
            } else {
                processingView
            }
        }
        .onChange(of: pickerItem) { _, newItem in
            Task { await handlePicked(newItem) }
        }
    }

    /// The picker + live per-stage progress screen, shown until a photo
    /// finishes processing (or fails/has no usable ground plane, in which
    /// case this stays the terminal screen -- there is no `PipelineResult`
    /// to show `ResultsView` for a photo `disturbance.py` itself would
    /// report as "no BEV produced").
    private var processingView: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Photo -> semantic overlay").font(.headline)

            PhotosPicker(selection: $pickerItem, matching: .images) {
                Label("Pick a photo", systemImage: "photo.on.rectangle")
            }
            .buttonStyle(.borderedProminent)

            // Debug-only: exercises the identical processing path
            // (process(data:)) without needing to script the system
            // PHPicker UI, which isn't reachable via xcodebuild/simctl --
            // real end-to-end verification of the code this app actually
            // owns (EXIF extraction, resize, model inference, overlay
            // rendering), not of PhotosPicker itself (standard Apple API).
            if let sampleURL = Bundle.main.url(forResource: "SamplePhoto", withExtension: "jpg") {
                Button("Load bundled test photo (debug)") {
                    Task {
                        if let data = try? Data(contentsOf: sampleURL) {
                            await process(data: data)
                        }
                    }
                }
                .buttonStyle(.bordered)
            }

            if let overlayImage {
                Image(uiImage: overlayImage)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 220)
            } else if let sourceImage {
                Image(uiImage: sourceImage)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 220)
            }

            if let depthImage {
                Text("Depth").font(.subheadline)
                Image(uiImage: depthImage)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 220)
            }

            if let amodalImage {
                Text("Amodal road (green = visible, orange = recovered occluded)").font(.subheadline)
                Image(uiImage: amodalImage)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 220)
            }

            if let vehicleImage {
                Text("Vehicle instances").font(.subheadline)
                Image(uiImage: vehicleImage)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 220)
            }

            if let bevImage {
                Text("Bird's-eye view (green = visible, orange = occluded)").font(.subheadline)
                Image(uiImage: bevImage)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 300)
            }

            if let attributionImage {
                Text("Occlusion attribution (colour per vehicle, white = unattributed)").font(.subheadline)
                Image(uiImage: attributionImage)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 300)
            }

            if isProcessing { ProgressView() }
            Text(statusText).font(.caption).foregroundStyle(.secondary)
            if let geometryText {
                Text(geometryText).font(.caption).foregroundStyle(.secondary)
            }
            if let amodalText {
                Text(amodalText).font(.caption).foregroundStyle(.secondary)
            }
            if let vehicleText {
                Text(vehicleText).font(.caption).foregroundStyle(.secondary)
            }
            if let heightCorrectionText {
                Text(heightCorrectionText).font(.caption).foregroundStyle(.secondary)
            }
            if let bevText {
                Text(bevText).font(.caption).foregroundStyle(.secondary)
            }
            if let attributionText {
                Text(attributionText).font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding()
    }

    private func handlePicked(_ item: PhotosPickerItem?) async {
        guard let item else { return }
        guard let data = try? await item.loadTransferable(type: Data.self) else {
            statusText = "Failed to load the selected photo."
            return
        }
        await process(data: data)
    }

    /// Resets every piece of per-photo state, returning to the picker --
    /// the `ResultsView`'s "New Photo" button calls this.
    private func reset() {
        result = nil
        pickerItem = nil
        sourceImage = nil
        overlayImage = nil
        depthImage = nil
        amodalImage = nil
        vehicleImage = nil
        bevImage = nil
        attributionImage = nil
        statusText = "Pick a photo to measure how much road it occludes."
        geometryText = nil
        amodalText = nil
        vehicleText = nil
        heightCorrectionText = nil
        bevText = nil
        attributionText = nil
    }

    private func process(data: Data) async {
        isProcessing = true
        overlayImage = nil
        depthImage = nil
        amodalImage = nil
        vehicleImage = nil
        bevImage = nil
        attributionImage = nil
        geometryText = nil
        amodalText = nil
        vehicleText = nil
        heightCorrectionText = nil
        bevText = nil
        attributionText = nil
        result = nil
        statusText = "Loading photo..."
        defer { isProcessing = false }

        guard let uiImage = UIImage(data: data), let cgImage = uiImage.cgImage else {
            statusText = "Failed to decode photo data."
            return
        }
        sourceImage = uiImage

        let calibration = EXIFCalibration.calibrate(imageData: data, width: cgImage.width, height: cgImage.height)
        statusText = "Calibration source: \(calibration.source.rawValue) -- running semantic model..."

        do {
            let sem = try await runSemantic(cgImage: cgImage)
            overlayImage = sem.overlay
            statusText = "Calibration source: \(calibration.source.rawValue) -- semantic \(sem.elapsedMs)ms, running depth + plane fit..."

            let geometry = try await runGeometry(cgImage: cgImage, K: calibration.K,
                                                 roadMask: sem.roadMask, roadMaskWidth: sem.width, roadMaskHeight: sem.height)
            depthImage = geometry.depthImage
            if geometry.result.valid {
                let h = PlaneGeometry.impliedCameraHeight(geometry.result.n!)
                geometryText = String(format: "Plane fit OK (%d ground points) -- implied camera height %.2f m (uncorrected)",
                                      geometry.result.pointsUsed, h)
            } else {
                geometryText = "No usable ground plane (road mask too small or RANSAC failed) -- \(geometry.result.pointsUsed) candidate points"
            }
            statusText = "Calibration source: \(calibration.source.rawValue) -- semantic \(sem.elapsedMs)ms, depth+geometry \(geometry.elapsedMs)ms, detecting vehicles..."

            let vehicles = try await runVehicleDetection(cgImage: cgImage)
            vehicleImage = vehicles.image
            vehicleText = "\(vehicles.detections.count) vehicle(s) detected in \(vehicles.elapsedMs)ms"
            statusText = "Calibration source: \(calibration.source.rawValue) -- semantic \(sem.elapsedMs)ms, "
                + "geo \(geometry.elapsedMs)ms, vehicles \(vehicles.elapsedMs)ms, correcting camera height..."

            // Plain-language caveats surfaced directly in ResultsView, not
            // just logged -- mirrors disturbance.DisturbanceResult.warnings
            // (see the migration plan's Section 9). The fov_prior/uncalibrated
            // case is deliberately NOT duplicated here -- ResultsView's hero
            // card already carries that caveat as a badge on every screen,
            // so repeating it in the warnings list would just be noise.
            var warnings: [String] = []

            var correctedN = geometry.result.n
            var heightWasCorrected = false
            // Converts calibrated area to the raw (uncalibrated) figure a
            // height-corrected BEV cell would have measured before
            // correction -- mirrors disturbance.analyze's `scale_factor =
            // (implied / target) ** 2` (a linear world-scale error is a
            // quadratic area error). Stays 1.0 (no-op) when no correction
            // was applied, since `correctedN` is then just the raw plane.
            var scaleFactor = 1.0
            if geometry.result.valid, let n = geometry.result.n {
                if let estimate = Self.estimateHeightCorrection(
                    vehicles: vehicles.detections, transform: vehicles.transform,
                    hField: geometry.result.h, hFieldWidth: geometry.depthWidth, hFieldHeight: geometry.depthHeight,
                    origWidth: cgImage.width, origHeight: cgImage.height, n: n) {
                    let (rescaledN, before) = PlaneGeometry.rescalePlaneToHeight(n, cameraHeightM: estimate.cameraHeightM)
                    correctedN = rescaledN
                    heightWasCorrected = true
                    if estimate.nSamples == 1 {
                        warnings.append("Camera height was estimated from a single parked vehicle -- "
                                       + "treat it as a rough estimate, not a precise one.")
                    } else if estimate.kSpread > 0.3 {
                        warnings.append(String(format: "The %d vehicles used to estimate camera height disagreed "
                                              + "substantially (%.0f%% spread) -- the estimate may be unreliable.",
                                              estimate.nSamples, estimate.kSpread * 100))
                    }
                    if before / estimate.cameraHeightM > 2.0 {
                        warnings.append(String(format: "Monocular depth was significantly out of scale before "
                                              + "correction (%.1fx) -- areas rely heavily on the vehicle-height calibration.",
                                              before / estimate.cameraHeightM))
                    }
                    scaleFactor = pow(before / estimate.cameraHeightM, 2)
                    heightCorrectionText = String(format: "Height corrected: %.2f m -> %.2f m (%d vehicle sample(s), k=%.3f)",
                                                 before, estimate.cameraHeightM, estimate.nSamples, estimate.kMedian)
                    geometryText = String(format: "Plane fit OK (%d ground points) -- implied camera height %.2f m (corrected from %.2f m)",
                                        geometry.result.pointsUsed, estimate.cameraHeightM, before)
                } else {
                    heightCorrectionText = "No qualifying vehicle for height correction -- using uncorrected plane"
                    warnings.append("No parked vehicle was suitable for automatic camera-height calibration -- "
                                   + "using an uncorrected estimate, which may be inaccurate.")
                }
            }
            statusText = "Calibration source: \(calibration.source.rawValue) -- semantic \(sem.elapsedMs)ms, "
                + "geo \(geometry.elapsedMs)ms, vehicles \(vehicles.elapsedMs)ms, running amodal completion..."

            let amodal = try await runAmodalCompletion(semClasses: sem.classes, semWidth: sem.width, semHeight: sem.height,
                                                       depth: geometry.result.depth, n: correctedN,
                                                       validPlane: geometry.result.valid,
                                                       depthWidth: geometry.depthWidth, depthHeight: geometry.depthHeight,
                                                       K: calibration.K)
            amodalImage = amodal.image
            amodalText = "Visible road: \(amodal.visibleRoadPx) px, recovered occluded road: \(amodal.recoveredPx) px "
                + "(+\(String(format: "%.1f", amodal.recoveredPercent))%), amodal inference \(amodal.elapsedMs)ms"
            statusText = "Calibration source: \(calibration.source.rawValue) -- sem \(sem.elapsedMs)ms, "
                + "geo \(geometry.elapsedMs)ms, vehicles \(vehicles.elapsedMs)ms, amodal \(amodal.elapsedMs)ms, "
                + "projecting to BEV..."

            if geometry.result.valid, let n = correctedN {
                let bevResult = try await runBevProjection(
                    visibleMask: amodal.visibleMask, amodalMask: amodal.amodalMask,
                    semWidth: sem.width, semHeight: sem.height,
                    n: n, K: calibration.K, depthWidth: geometry.depthWidth, depthHeight: geometry.depthHeight)
                bevImage = bevResult.image
                bevText = String(format: "Amodal road: %.2f m^2, visible: %.2f m^2, OCCLUDED: %.2f m^2 (%.1f%% of amodal) "
                                + "-- measurable to %.1f m, BEV %dms",
                                bevResult.amodalAreaM2, bevResult.visibleAreaM2, bevResult.occludedAreaM2,
                                bevResult.occludedPercent, bevResult.info.measurableRangeM, bevResult.elapsedMs)
                // An occluder hides the road behind it all the way to the
                // horizon, so once the measurable range falls meaningfully
                // short of the grid's own far edge, every area figure is a
                // lower bound, not the full picture -- mirrors
                // disturbance.analyze's identical check.
                if bevResult.info.measurableRangeM < bevResult.grid.zMax - 1.0 {
                    warnings.append(String(format: "Areas are only measured out to %.1f m (of the %.0f m grid) -- "
                                          + "beyond that, one camera pixel covers too much ground to trust.",
                                          bevResult.info.measurableRangeM, bevResult.grid.zMax))
                }
                statusText = "Calibration source: \(calibration.source.rawValue) -- sem \(sem.elapsedMs)ms, "
                    + "geo \(geometry.elapsedMs)ms, vehicles \(vehicles.elapsedMs)ms, amodal \(amodal.elapsedMs)ms, "
                    + "attributing occluded area..."

                let attribution = try await runAttribution(
                    semClasses: sem.classes, semWidth: sem.width, semHeight: sem.height,
                    detections: vehicles.detections, transform: vehicles.transform,
                    origWidth: cgImage.width, origHeight: cgImage.height,
                    occludedBev: bevResult.occludedBev, amodalBev: bevResult.amodalBev,
                    visibleBev: bevResult.visibleBev,
                    grid: bevResult.grid, H: bevResult.H, scaleFactor: scaleFactor)
                attributionImage = BevRenderer.render(
                    occludedBev: attribution.occludedBev, instBev: attribution.instBev,
                    visibleBev: attribution.visibleBev, width: bevResult.grid.width, height: bevResult.grid.height)
                attributionText = "Attribution done in \(attribution.elapsedMs)ms"

                if attribution.lowCoverageCount > 0 {
                    warnings.append("\(attribution.lowCoverageCount) vehicle(s) have their ground contact mostly "
                                   + "cropped by the frame edge, so their occluded-area figure may be incomplete.")
                }
                if bevResult.occludedAreaM2 > 0 {
                    let unattributedFrac = 100.0 * attribution.unattributedAreaM2 / bevResult.occludedAreaM2
                    if unattributedFrac > 20.0 {
                        warnings.append(String(format: "%.0f%% of the occluded area is not attributed to any "
                                              + "vehicle -- the detector and the semantic segmentation disagree "
                                              + "substantially on this photo.", unattributedFrac))
                    }
                }

                let technicalSummary = "Calibration: \(calibration.source.rawValue) -- sem \(sem.elapsedMs)ms, "
                    + "geo \(geometry.elapsedMs)ms, vehicles \(vehicles.elapsedMs)ms, amodal \(amodal.elapsedMs)ms, "
                    + "bev \(bevResult.elapsedMs)ms, attribution \(attribution.elapsedMs)ms\n"
                    + (geometryText ?? "") + "\n" + (heightCorrectionText ?? "") + "\n" + (amodalText ?? "")
                    + "\n" + (vehicleText ?? "")

                result = PipelineResult(
                    sourceImage: uiImage,
                    calibrationSource: calibration.source,
                    cameraHeightM: PlaneGeometry.impliedCameraHeight(n),
                    cameraHeightWasCorrected: heightWasCorrected,
                    amodalAreaM2: bevResult.amodalAreaM2,
                    visibleAreaM2: bevResult.visibleAreaM2,
                    occludedAreaM2: bevResult.occludedAreaM2,
                    occludedPercent: bevResult.occludedPercent,
                    measurableRangeM: bevResult.info.measurableRangeM,
                    vehicles: attribution.vehicles,
                    unattributedAreaM2: attribution.unattributedAreaM2,
                    warnings: warnings,
                    overlayImage: sem.overlay,
                    depthImage: geometry.depthImage,
                    vehicleBoxesImage: vehicles.image,
                    bevImage: bevResult.image,
                    occludedBev: attribution.occludedBev, instBev: attribution.instBev,
                    visibleBev: attribution.visibleBev,
                    bevWidth: bevResult.grid.width, bevHeight: bevResult.grid.height,
                    technicalSummary: technicalSummary)
            } else {
                bevText = "No usable ground plane -- BEV projection skipped"
                attributionText = "No usable ground plane -- attribution skipped"
                statusText = "Couldn't measure this photo: no usable ground plane was found "
                    + "(too little visible road, or the plane fit failed). Try a photo with more road visible."
            }
        } catch {
            statusText = "Pipeline failed: \(error)"
        }
    }

    /// Maps every qualifying detection's mask from the detector's
    /// letterboxed 640x640 space onto the SAME grid as `hField` (depth's
    /// own native resolution), then delegates to
    /// `VehicleHeightEstimation.estimateCameraHeight`.
    private static func estimateHeightCorrection(vehicles: [VehicleDetection], transform: LetterboxTransform,
                                                 hField: [Double], hFieldWidth: Int, hFieldHeight: Int,
                                                 origWidth: Int, origHeight: Int, n: SIMD3<Double>) -> ScaleEstimate? {
        guard !vehicles.isEmpty else { return nil }
        let forEstimation = vehicles.map { detection in
            VehicleForHeightEstimation(
                mask: vehicleMaskAtGrid(detection, transform: transform, origWidth: origWidth, origHeight: origHeight,
                                       gridWidth: hFieldWidth, gridHeight: hFieldHeight),
                score: detection.score)
        }
        let impliedBiasedHeightM = PlaneGeometry.impliedCameraHeight(n)
        return VehicleHeightEstimation.estimateCameraHeight(
            hField: hField, vehicles: forEstimation, impliedBiasedHeightM: impliedBiasedHeightM,
            width: hFieldWidth, height: hFieldHeight)
    }

    /// One detection's mask (defined on the detector's own letterboxed
    /// 640x640 grid), resampled onto an arbitrary `gridWidth x gridHeight`
    /// grid of the ORIGINAL photo (e.g. depth's own native resolution,
    /// which -- like every other model in this pipeline -- maps to the
    /// original photo with a plain stretch, not a letterbox). Composes two
    /// independent mappings (grid -> original photo -> letterboxed
    /// detector space) rather than trying to derive a single combined
    /// formula, so each half stays exactly as simple/verifiable as it is
    /// used elsewhere in this file (`nearestResizeMask`'s plain-stretch
    /// convention; `Letterbox.toOriginal`'s already golden-value-tested math).
    private static func vehicleMaskAtGrid(_ detection: VehicleDetection, transform: LetterboxTransform,
                                          origWidth: Int, origHeight: Int,
                                          gridWidth: Int, gridHeight: Int) -> [Bool] {
        var mask = [Bool](repeating: false, count: gridWidth * gridHeight)
        for gy in 0..<gridHeight {
            let yOrig = Double(gy) * Double(origHeight) / Double(gridHeight)
            let yDet = yOrig * transform.scale + transform.padY
            guard yDet >= detection.y1, yDet < detection.y2 else { continue }
            for gx in 0..<gridWidth {
                let xOrig = Double(gx) * Double(origWidth) / Double(gridWidth)
                let xDet = xOrig * transform.scale + transform.padX
                guard xDet >= detection.x1, xDet < detection.x2 else { continue }
                mask[gy * gridWidth + gx] = Self.sampleMask(detection.mask, x: xDet, y: yDet,
                                                            width: Self.detectionMaskSide)
            }
        }
        return mask
    }

    /// `VehicleDetection.mask` is always `inputWidth * inputHeight` of the
    /// bundled YOLO export -- 640 square, see `YOLOInstanceModel`.
    private static let detectionMaskSide = 640

    private static func sampleMask(_ mask: [Bool], x: Double, y: Double, width: Int) -> Bool {
        let ix = Int(x), iy = Int(y)
        guard ix >= 0, ix < width, iy >= 0, iy * width + ix < mask.count else { return false }
        return mask[iy * width + ix]
    }

    private struct SemanticRunResult {
        let overlay: UIImage
        let classes: [Int]
        let roadMask: [Bool]
        let width: Int
        let height: Int
        let elapsedMs: Int
    }

    private func runSemantic(cgImage: CGImage) async throws -> SemanticRunResult {
        try await Task.detached(priority: .userInitiated) {
            guard let rgb = ImageBytes.rgb(from: cgImage) else {
                throw ModelBundleError.resourceNotFound("could not rasterize picked photo")
            }
            let model = try ModelBundle.loadMobileSemantic()
            let px = ImagePreprocessing.resizeToUnitRangeRGB(
                rgb: rgb.bytes, srcWidth: rgb.width, srcHeight: rgb.height,
                targetWidth: model.width, targetHeight: model.height)

            let start = Date()
            let logits = try model.predict(pixelValues: px)
            let elapsedMs = Int(Date().timeIntervalSince(start) * 1000)

            let count = model.width * model.height
            var classes = [Int](repeating: 0, count: count)
            var roadMask = [Bool](repeating: false, count: count)
            for i in 0..<count {
                var bestClass = 0
                var bestVal = -Float.infinity
                for c in 0..<model.numClasses {
                    let v = logits[c * count + i]
                    if v > bestVal { bestVal = v; bestClass = c }
                }
                classes[i] = bestClass
                roadMask[i] = bestClass == roadClassIndex
            }

            let overlay = Self.renderOverlay(logits: logits, numClasses: model.numClasses,
                                            width: model.width, height: model.height,
                                            baseRGB: px)
            return SemanticRunResult(overlay: overlay, classes: classes, roadMask: roadMask,
                                    width: model.width, height: model.height, elapsedMs: elapsedMs)
        }.value
    }

    private struct GeometryRunResult {
        let result: GeometryPipeline.Result
        let depthImage: UIImage
        let depthWidth: Int
        let depthHeight: Int
        let elapsedMs: Int
    }

    /// Runs DepthModel (its own fixed 518x518 resolution, independent of the
    /// semantic model's 512x512) and RANSAC-fits a ground plane using the
    /// semantic road mask, nearest-neighbour-resized onto the depth grid.
    private func runGeometry(cgImage: CGImage, K: simd_double3x3, roadMask: [Bool],
                             roadMaskWidth: Int, roadMaskHeight: Int) async throws -> GeometryRunResult {
        try await Task.detached(priority: .userInitiated) {
            guard let rgb = ImageBytes.rgb(from: cgImage) else {
                throw ModelBundleError.resourceNotFound("could not rasterize picked photo")
            }
            let depthModel = try ModelBundle.loadDepth()
            let px = ImagePreprocessing.resizeAndNormalizeImageNet(
                rgb: rgb.bytes, srcWidth: rgb.width, srcHeight: rgb.height,
                targetWidth: depthModel.width, targetHeight: depthModel.height)

            let resizedRoadMask = Self.nearestResizeMask(
                roadMask, srcWidth: roadMaskWidth, srcHeight: roadMaskHeight,
                dstWidth: depthModel.width, dstHeight: depthModel.height)

            let start = Date()
            let result = try GeometryPipeline.resolveGeometry(
                depthModel: depthModel, pixelValues: px, roadMask: resizedRoadMask, K: K)
            let elapsedMs = Int(Date().timeIntervalSince(start) * 1000)

            let depthImage = Self.renderDepth(result.depth, width: depthModel.width, height: depthModel.height)
            return GeometryRunResult(result: result, depthImage: depthImage,
                                    depthWidth: depthModel.width, depthHeight: depthModel.height,
                                    elapsedMs: elapsedMs)
        }.value
    }

    private struct AmodalRunResult {
        let image: UIImage
        let visibleMask: [Bool]
        let amodalMask: [Bool]
        let visibleRoadPx: Int
        let recoveredPx: Int
        let recoveredPercent: Double
        let elapsedMs: Int
    }

    /// OFRSNet amodal completion. Ground fields (G/h/gvalid) must be
    /// re-derived at the SEMANTIC model's own 512x512 canvas -- OFRSNet's
    /// input is one-hot over `sem`, so it must align pixel-for-pixel with it
    /// -- not depth's native 518x518 (not a multiple of OFRS_NET_STRIDE=8
    /// anyway, so OFRSNet couldn't be exported at that resolution even if
    /// alignment weren't a concern). Mirrors src/geometry.py's
    /// `load_ground_fields`: resize DEPTH (not the already-derived G/h),
    /// scale K to match, then call `derive_ground_fields` fresh at the
    /// target resolution -- `n` itself is resolution-independent so carries
    /// over unchanged.
    ///
    /// The road mask itself now goes through `AmodalCompose.composeAmodalMask`
    /// (predict.py::compose_amodal_mask, via common.occluder_blob_mask),
    /// not the earlier `visible || rawOfrsRoad` simplification -- OFRSNet's
    /// output only patches the road within person/vehicle blobs that touch
    /// the near-road band (`OFRSClasses.roadNeighbourhoodPx`), keeping the
    /// visible road's own boundary exactly as clean as the semantic model's
    /// segmentation instead of letting OFRSNet's raw output redraw it
    /// anywhere it disagrees.
    private func runAmodalCompletion(semClasses: [Int], semWidth: Int, semHeight: Int,
                                     depth: [Float], n: SIMD3<Double>?, validPlane: Bool,
                                     depthWidth: Int, depthHeight: Int,
                                     K: simd_double3x3) async throws -> AmodalRunResult {
        try await Task.detached(priority: .userInitiated) {
            let model = try ModelBundle.loadOFRSNet()
            precondition(model.width == semWidth && model.height == semHeight,
                        "bundled OFRSNet resolution (\(model.width)x\(model.height)) must match "
                        + "the semantic model's (\(semWidth)x\(semHeight))")

            let resizedDepth = Self.resizeFloatFieldBilinear(
                depth, srcWidth: depthWidth, srcHeight: depthHeight,
                dstWidth: semWidth, dstHeight: semHeight)
            let scaledK = Intrinsics.scaleK(K, sx: Double(semWidth) / Double(depthWidth),
                                            sy: Double(semHeight) / Double(depthHeight))
            let (G, h, gvalid) = GroundFields.derive(depth: resizedDepth, n: n, K: scaledK,
                                                     width: semWidth, height: semHeight)

            let start = Date()
            let logits = try model.predict(sem: semClasses, G: G, h: h, gvalid: gvalid, validPlane: validPlane)
            let elapsedMs = Int(Date().timeIntervalSince(start) * 1000)

            let count = semWidth * semHeight
            var ofrsRoad = [Bool](repeating: false, count: count)
            for i in 0..<count {
                ofrsRoad[i] = logits[count + i] > logits[i]   // channel 1 (road) > channel 0
            }
            // predict.py::compose_amodal_mask -- OFRSNet's prediction only
            // PATCHES the road within occluder (person/vehicle) blobs that
            // touch the near-road band, it never redraws the already-visible
            // road. `patch` is the subset of `amodalMask` that came from
            // OFRSNet, used below for the recovered-pixel count/overlay.
            let (amodalMask, patch) = AmodalCompose.composeAmodalMask(
                semLabels: semClasses, ofrsRoad: ofrsRoad, width: semWidth, height: semHeight)

            var visibleRoadPx = 0, recoveredPx = 0
            var visibleMask = [Bool](repeating: false, count: count)
            var pixels = [UInt8](repeating: 255, count: count * 4)
            for i in 0..<count {
                let visible = semClasses[i] == roadClassIndex
                visibleMask[i] = visible
                if visible { visibleRoadPx += 1 }
                var r: UInt8 = 32, g: UInt8 = 32, b: UInt8 = 32
                if visible {
                    r = 0; g = 200; b = 0                       // visible road: green
                } else if patch[i] {
                    recoveredPx += 1
                    r = 255; g = 140; b = 0                     // recovered occluded road: orange
                }
                pixels[i*4+0] = r; pixels[i*4+1] = g; pixels[i*4+2] = b; pixels[i*4+3] = 255
            }
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let context = CGContext(data: &pixels, width: semWidth, height: semHeight, bitsPerComponent: 8,
                                    bytesPerRow: semWidth * 4, space: colorSpace,
                                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            let image = UIImage(cgImage: context.makeImage()!)

            let recoveredPercent = visibleRoadPx > 0 ? 100.0 * Double(recoveredPx) / Double(visibleRoadPx) : 0
            return AmodalRunResult(image: image, visibleMask: visibleMask, amodalMask: amodalMask,
                                  visibleRoadPx: visibleRoadPx, recoveredPx: recoveredPx,
                                  recoveredPercent: recoveredPercent, elapsedMs: elapsedMs)
        }.value
    }

    private struct BevRunResult {
        let image: UIImage
        let amodalAreaM2: Double
        let visibleAreaM2: Double
        let occludedAreaM2: Double
        let occludedPercent: Double
        let info: BevValidity.MeasurableInfo
        let elapsedMs: Int
        // Exposed for `runAttribution`, which needs the exact same raster
        // and homography `attribute`/`_ground_contact_seed_ids` operate on
        // in Python -- recomputing them there would risk a subtle mismatch
        // (a different K scaling, a different grid) against what this
        // function actually measured.
        let occludedBev: [Bool]
        let amodalBev: [Bool]
        let visibleBev: [Bool]
        let grid: BevGrid
        let H: simd_double3x3
    }

    /// Ports `disturbance.analyze`'s BEV half (steps 1-3, `occluded_bev`
    /// computation, and the total-area figures). Per-vehicle attribution
    /// (`attribute`/`_ground_contact_seed_ids`) is a separate step,
    /// `runAttribution` below, since it needs vehicle detections this
    /// function has no reason to depend on. `occluded_road_m2` here is
    /// computed directly from a BEV built with the ALREADY-height-corrected
    /// plane, so unlike Python's `occluded_road_m2_raw`, no separate
    /// `scale_factor` multiplication is needed -- the grid itself is
    /// already in true metres.
    ///
    /// `occluded_road_m2` still reads 0.00 on the sample photo even AFTER
    /// `runAmodalCompletion` was switched from a `visible || rawOfrsRoad`
    /// simplification to the real `compose_amodal_mask`/`occluder_blob_mask`
    /// restriction -- re-investigated (not assumed unchanged), and it's a
    /// DIFFERENT root cause than before. A temporary diagnostic (counting
    /// `amodalBevRaw && !visibleBevRaw` BEFORE applying `bevValid`) showed
    /// that count is exactly 0: no BEV cell's nearest-sampled source pixel
    /// lands inside the recovered (patch) region at all, so this isn't the
    /// resolution/`measurableRangeM` cutoff excluding an otherwise-sampled
    /// cell -- it's `warpToBev`'s inverse-nearest-neighbour sampling simply
    /// never hitting a thin recovered strip on this specific photo. On this
    /// sample, the recovered patch sits right along the visible road's far
    /// (most-distant) edge -- see the amodal road panel's overlay -- a
    /// strip only a few pixels tall in image space, at a range where ground
    /// resolution is already at its worst (`Measurement.groundM2PerPixel`
    /// degrades roughly with range^3), so a BEV cell there is more likely
    /// to nearest-sample a visible-road or off-road neighbour instead. This
    /// is an inherent property of nearest-neighbour sampling on a thin
    /// region, present in Python's `cv2.warpPerspective`-based
    /// `warp_to_bev` too (this port's `Warp`/`BevValidity`/`Measurement`
    /// are independently golden-value tested and match OpenCV exactly) --
    /// not a Swift-side bug, and not something `compose_amodal_mask`'s fix
    /// alone resolves. Expect `occluded_road_m2` to read non-zero on photos
    /// where the recovered patch is wider/closer (e.g. a car parked
    /// squarely across a near section of visible road) rather than a thin
    /// far-edge sliver like this sample.
    private func runBevProjection(visibleMask: [Bool], amodalMask: [Bool], semWidth: Int, semHeight: Int,
                                  n: SIMD3<Double>, K: simd_double3x3,
                                  depthWidth: Int, depthHeight: Int) async throws -> BevRunResult {
        try await Task.detached(priority: .userInitiated) {
            let scaledK = Intrinsics.scaleK(K, sx: Double(semWidth) / Double(depthWidth),
                                            sy: Double(semHeight) / Double(depthHeight))
            let grid = BevConfig.defaultBevGrid()
            let start = Date()
            let H = try Homography.bevToImage(K: scaledK, n: n, grid: grid)

            let (bevValid, info) = BevValidity.measurableMask(
                H: H, grid: grid, imageWidth: semWidth, imageHeight: semHeight, maxM2PerPx: BevConfig.bevMaxM2PerPixel)
            let visibleBevRaw = Warp.warpToBev(mask: visibleMask, srcWidth: semWidth, srcHeight: semHeight, H: H, grid: grid)
            let amodalBevRaw = Warp.warpToBev(mask: amodalMask, srcWidth: semWidth, srcHeight: semHeight, H: H, grid: grid)

            var visibleBev = [Bool](repeating: false, count: grid.width * grid.height)
            var amodalBev = [Bool](repeating: false, count: grid.width * grid.height)
            var occludedBev = [Bool](repeating: false, count: grid.width * grid.height)
            for i in 0..<(grid.width * grid.height) {
                visibleBev[i] = visibleBevRaw[i] && bevValid[i]
                amodalBev[i] = amodalBevRaw[i] && bevValid[i]
                occludedBev[i] = amodalBev[i] && !visibleBev[i] && bevValid[i]
            }
            let elapsedMs = Int(Date().timeIntervalSince(start) * 1000)

            let amodalArea = Measurement.areaM2(cellCount: amodalBev.filter { $0 }.count, grid: grid)
            let visibleArea = Measurement.areaM2(cellCount: visibleBev.filter { $0 }.count, grid: grid)
            let occludedArea = Measurement.areaM2(cellCount: occludedBev.filter { $0 }.count, grid: grid)
            let occludedPercent = amodalArea > 0 ? 100.0 * occludedArea / amodalArea : 0

            let image = Self.renderBev(visibleBev: visibleBev, occludedBev: occludedBev,
                                      width: grid.width, height: grid.height)
            return BevRunResult(image: image, amodalAreaM2: amodalArea, visibleAreaM2: visibleArea,
                               occludedAreaM2: occludedArea, occludedPercent: occludedPercent,
                               info: info, elapsedMs: elapsedMs,
                               occludedBev: occludedBev, amodalBev: amodalBev, visibleBev: visibleBev,
                               grid: grid, H: H)
        }.value
    }

    private struct AttributionRunResult {
        let vehicles: [VehicleResult]
        let unattributedAreaM2: Double
        let lowCoverageCount: Int
        let elapsedMs: Int
        // Raw grid data, not a pre-rendered image -- ResultsView needs to
        // re-render this reactively when the user taps a vehicle to
        // highlight it (BevRenderer.render(..., selectedId:)), which a
        // single static UIImage can't support.
        let occludedBev: [Bool]
        let instBev: [Int32]
        let visibleBev: [Bool]
    }

    /// Ports `disturbance.py`'s per-vehicle attribution (`occluder_instance_ids`
    /// + `_ground_contact_seed_ids` + `attribute` + `width_disturbance`). The
    /// same YOLO detections used for camera-height correction are resampled
    /// onto the SEMANTIC map's own grid (same `vehicleMaskAtGrid` composition
    /// used there, targeting `semWidth`x`semHeight` this time instead of
    /// depth's grid) and treated as a single collapsed "vehicle" class --
    /// matching this package's existing simplification (no per-class labels
    /// from the mobile instance student).
    private func runAttribution(semClasses: [Int], semWidth: Int, semHeight: Int,
                                detections: [VehicleDetection], transform: LetterboxTransform,
                                origWidth: Int, origHeight: Int,
                                occludedBev: [Bool], amodalBev: [Bool], visibleBev: [Bool],
                                grid: BevGrid, H: simd_double3x3,
                                scaleFactor: Double) async throws -> AttributionRunResult {
        try await Task.detached(priority: .userInitiated) {
            let start = Date()
            let occluderDetections = detections.map { d -> OccluderDetection in
                let mask = Self.vehicleMaskAtGrid(d, transform: transform, origWidth: origWidth, origHeight: origHeight,
                                                  gridWidth: semWidth, gridHeight: semHeight)
                return OccluderDetection(label: "vehicle", score: Double(d.score), mask: mask, isVehicle: true)
            }
            let (instIds, instances) = OccluderInstances.occluderInstanceIds(
                semLabels: semClasses, detections: occluderDetections, width: semWidth, height: semHeight)

            let (seedIds, lowCoverage) = Attribution.groundContactSeedIds(
                instIds: instIds, vehicles: instances, H: H, grid: grid, width: semWidth, height: semHeight)
            let instBev = NearestLabel.label(seedIds: seedIds, grid: grid,
                                             maxDistM: AttributionConfig.footprintMaxAttributionDistM)

            let (perVehicle, unattributed) = Attribution.attribute(
                occludedBev: occludedBev, instBev: instBev, vehicles: instances, grid: grid, scaleFactor: scaleFactor)
            let (widthPerVehicle, _) = WidthDisturbance.measure(
                amodalBev: amodalBev, occludedBev: occludedBev, instBev: instBev, vehicles: instances, grid: grid)
            let elapsedMs = Int(Date().timeIntervalSince(start) * 1000)

            // Ranked by area, matching `_print_report`'s ordering -- ResultsView
            // filters out zero-area occluders itself, so every instance is
            // still returned here (not just the ones with measurable area),
            // matching `disturbance.DisturbanceResult.vehicles`' own full list.
            let vehicleResults = instances
                .sorted { (perVehicle[$0.instId]?.areaM2 ?? 0) > (perVehicle[$1.instId]?.areaM2 ?? 0) }
                .map { v in
                    VehicleResult(id: v.instId, label: v.label, source: v.source, score: v.score,
                                 areaM2: perVehicle[v.instId]?.areaM2 ?? 0,
                                 widthMaxPct: widthPerVehicle[v.instId]?.widthMaxPct ?? 0)
                }

            return AttributionRunResult(vehicles: vehicleResults,
                                        unattributedAreaM2: unattributed.areaM2,
                                        lowCoverageCount: lowCoverage.count, elapsedMs: elapsedMs,
                                        occludedBev: occludedBev, instBev: instBev, visibleBev: visibleBev)
        }.value
    }

    private static func renderBev(visibleBev: [Bool], occludedBev: [Bool], width: Int, height: Int) -> UIImage {
        var pixels = [UInt8](repeating: 20, count: width * height * 4)
        for i in 0..<(width * height) {
            var r: UInt8 = 20, g: UInt8 = 20, b: UInt8 = 20
            if visibleBev[i] { r = 0; g = 200; b = 0 }
            else if occludedBev[i] { r = 255; g = 140; b = 0 }
            pixels[i*4+0] = r; pixels[i*4+1] = g; pixels[i*4+2] = b; pixels[i*4+3] = 255
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: &pixels, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        return UIImage(cgImage: context.makeImage()!)
    }

    private struct VehicleRunResult {
        let image: UIImage
        let detections: [VehicleDetection]
        let transform: LetterboxTransform
        let elapsedMs: Int
    }

    /// Letterboxes the photo to the detector's own square input (preserving
    /// aspect ratio -- see this file's header for why THIS step, unlike
    /// depth/semantic, can't tolerate a plain stretch), runs
    /// `YOLOInstanceModel.detect`, and renders box outlines mapped back to
    /// the ORIGINAL photo's own coordinates via `Letterbox.toOriginal`, so
    /// the overlay lines up with `sourceImage` regardless of the detector's
    /// fixed 640x640 working resolution.
    private func runVehicleDetection(cgImage: CGImage) async throws -> VehicleRunResult {
        try await Task.detached(priority: .userInitiated) {
            let model = try ModelBundle.loadVehicleInstance()
            guard let (pixelBuffer, transform) = LetterboxImage.makePixelBuffer(cgImage: cgImage, targetSize: model.inputWidth) else {
                throw ModelBundleError.resourceNotFound("failed to build the letterboxed detector input")
            }

            let start = Date()
            let detections = try model.detect(pixelBuffer: pixelBuffer)
            let elapsedMs = Int(Date().timeIntervalSince(start) * 1000)

            let image = Self.renderVehicleBoxes(detections, transform: transform,
                                               origWidth: cgImage.width, origHeight: cgImage.height, cgImage: cgImage)
            return VehicleRunResult(image: image, detections: detections, transform: transform, elapsedMs: elapsedMs)
        }.value
    }

    /// Draws the original photo (downscaled for display) with each
    /// detection's box outlined in the letterbox-mapped-back-to-original
    /// coordinate space.
    ///
    /// A previous version of this function flipped the CTM (`translateBy`/
    /// `scaleBy`) before `context.draw(cgImage, in:)`, on the (incorrect,
    /// empirically disproven -- verified via an actual Simulator screenshot
    /// of the enlarged Technical Details image, not assumed) theory that a
    /// fresh `data: nil` context draws a `CGImage` upside down without one.
    /// It does not: `context.draw(image:in:)` already orients the image
    /// correctly in this context's native (bottom-left origin, Y-up) space,
    /// so adding that flip was the bug -- it flipped an already-correct
    /// photo, and a paired box-formula flip made the boxes track the
    /// now-upside-down photo, so the whole composite looked consistently
    /// (but wrongly) upside down rather than obviously broken. Fixed by
    /// drawing natively (no CTM flip) and keeping the box rect's Y in the
    /// same native convention: `Letterbox.toOriginal`'s `(ox, oy)` uses a
    /// top-left-origin/Y-down photo-pixel convention, so `origHeight - oy2`
    /// converts that to this context's bottom-left-origin/Y-up space.
    private static func renderVehicleBoxes(_ detections: [VehicleDetection], transform: LetterboxTransform,
                                           origWidth: Int, origHeight: Int, cgImage: CGImage) -> UIImage {
        let displayScale = 800.0 / Double(max(origWidth, origHeight))
        let dispW = max(1, Int(Double(origWidth) * displayScale))
        let dispH = max(1, Int(Double(origHeight) * displayScale))

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: nil, width: dispW, height: dispH, bitsPerComponent: 8, bytesPerRow: 0,
                                space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: dispW, height: dispH))

        context.setStrokeColor(red: 1, green: 0.55, blue: 0, alpha: 1)
        context.setLineWidth(3)
        for d in detections {
            let (ox1, oy1) = Letterbox.toOriginal(x: d.x1, y: d.y1, transform: transform)
            let (ox2, oy2) = Letterbox.toOriginal(x: d.x2, y: d.y2, transform: transform)
            let rect = CGRect(x: ox1 * displayScale, y: (Double(origHeight) - oy2) * displayScale,
                             width: (ox2 - ox1) * displayScale, height: (oy2 - oy1) * displayScale)
            context.stroke(rect)
        }
        return UIImage(cgImage: context.makeImage()!)
    }

    /// Bilinear resize of a single-channel row-major Float field (e.g. a
    /// depth map) -- same interpolation convention as
    /// ImagePreprocessing's RGB resize, just one channel instead of three.
    private static func resizeFloatFieldBilinear(_ field: [Float], srcWidth: Int, srcHeight: Int,
                                                  dstWidth: Int, dstHeight: Int) -> [Float] {
        precondition(field.count == srcWidth * srcHeight)
        var out = [Float](repeating: 0, count: dstWidth * dstHeight)
        let sx = Double(srcWidth) / Double(dstWidth)
        let sy = Double(srcHeight) / Double(dstHeight)
        for ty in 0..<dstHeight {
            let fy = (Double(ty) + 0.5) * sy - 0.5
            let y0 = max(0, min(srcHeight - 1, Int(floor(fy))))
            let y1 = max(0, min(srcHeight - 1, y0 + 1))
            let wy = max(0.0, min(1.0, fy - Double(y0)))
            for tx in 0..<dstWidth {
                let fx = (Double(tx) + 0.5) * sx - 0.5
                let x0 = max(0, min(srcWidth - 1, Int(floor(fx))))
                let x1 = max(0, min(srcWidth - 1, x0 + 1))
                let wx = max(0.0, min(1.0, fx - Double(x0)))
                let p00 = Double(field[y0 * srcWidth + x0]), p01 = Double(field[y0 * srcWidth + x1])
                let p10 = Double(field[y1 * srcWidth + x0]), p11 = Double(field[y1 * srcWidth + x1])
                let top = p00 * (1 - wx) + p01 * wx
                let bot = p10 * (1 - wx) + p11 * wx
                out[ty * dstWidth + tx] = Float(top * (1 - wy) + bot * wy)
            }
        }
        return out
    }

    private static func nearestResizeMask(_ mask: [Bool], srcWidth: Int, srcHeight: Int,
                                          dstWidth: Int, dstHeight: Int) -> [Bool] {
        var out = [Bool](repeating: false, count: dstWidth * dstHeight)
        for y in 0..<dstHeight {
            let sy = min(srcHeight - 1, y * srcHeight / dstHeight)
            for x in 0..<dstWidth {
                let sx = min(srcWidth - 1, x * srcWidth / dstWidth)
                out[y * dstWidth + x] = mask[sy * srcWidth + sx]
            }
        }
        return out
    }

    /// Grayscale depth visualization, near = dark, far = light, normalized
    /// to this image's own min/max (a debug view, not a calibrated colour scale).
    private static func renderDepth(_ depth: [Float], width: Int, height: Int) -> UIImage {
        let lo = depth.min() ?? 0
        let hi = depth.max() ?? 1
        let range = max(hi - lo, 1e-6)
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            let v = UInt8(clamping: Int((depth[i] - lo) / range * 255))
            pixels[i * 4 + 0] = v; pixels[i * 4 + 1] = v; pixels[i * 4 + 2] = v; pixels[i * 4 + 3] = 255
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: &pixels, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        return UIImage(cgImage: context.makeImage()!)
    }

    /// argmax per pixel -> OFRS palette colour, alpha-blended 50/50 over the
    /// (already resized, [0,1]-scaled) input the model itself saw -- same
    /// blend factor as s5_export_semantics.py's `--preview` output.
    private static func renderOverlay(logits: [Float], numClasses: Int, width: Int, height: Int,
                                      baseRGB: [Float]) -> UIImage {
        let count = width * height
        var pixels = [UInt8](repeating: 255, count: count * 4)
        for i in 0..<count {
            var bestClass = 0
            var bestVal = -Float.infinity
            for c in 0..<numClasses {
                let v = logits[c * count + i]
                if v > bestVal { bestVal = v; bestClass = c }
            }
            let color = ofrsPalette[bestClass]
            let baseR = baseRGB[0 * count + i] * 255
            let baseG = baseRGB[1 * count + i] * 255
            let baseB = baseRGB[2 * count + i] * 255
            pixels[i * 4 + 0] = UInt8(clamping: Int(0.5 * baseR + 0.5 * Float(color.0)))
            pixels[i * 4 + 1] = UInt8(clamping: Int(0.5 * baseG + 0.5 * Float(color.1)))
            pixels[i * 4 + 2] = UInt8(clamping: Int(0.5 * baseB + 0.5 * Float(color.2)))
            pixels[i * 4 + 3] = 255
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: &pixels, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        let cgImage = context.makeImage()!
        return UIImage(cgImage: cgImage)
    }
}

#Preview {
    PhotoSemanticView()
}
