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
//  than naively resizing the already-derived G/h fields themselves).
//
//  Still missing from the full disturbance.py-equivalent pipeline: BEV
//  projection and per-vehicle attribution -- both need the vehicle-instance
//  YOLO model's Swift-side NMS/mask decode first.

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
    @State private var statusText = "Pick a photo to run the mobile semantic student on-device."
    @State private var geometryText: String?
    @State private var amodalText: String?
    @State private var isProcessing = false

    var body: some View {
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

            if isProcessing { ProgressView() }
            Text(statusText).font(.caption).foregroundStyle(.secondary)
            if let geometryText {
                Text(geometryText).font(.caption).foregroundStyle(.secondary)
            }
            if let amodalText {
                Text(amodalText).font(.caption).foregroundStyle(.secondary)
            }
        }
        .onChange(of: pickerItem) { _, newItem in
            Task { await handlePicked(newItem) }
        }
    }

    private func handlePicked(_ item: PhotosPickerItem?) async {
        guard let item else { return }
        guard let data = try? await item.loadTransferable(type: Data.self) else {
            statusText = "Failed to load the selected photo."
            return
        }
        await process(data: data)
    }

    private func process(data: Data) async {
        isProcessing = true
        overlayImage = nil
        depthImage = nil
        amodalImage = nil
        geometryText = nil
        amodalText = nil
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
                geometryText = String(format: "Plane fit OK (%d ground points) -- implied camera height %.2f m",
                                      geometry.result.pointsUsed, h)
            } else {
                geometryText = "No usable ground plane (road mask too small or RANSAC failed) -- \(geometry.result.pointsUsed) candidate points"
            }
            statusText = "Calibration source: \(calibration.source.rawValue) -- semantic \(sem.elapsedMs)ms, depth+geometry \(geometry.elapsedMs)ms, running amodal completion..."

            let amodal = try await runAmodalCompletion(semClasses: sem.classes, semWidth: sem.width, semHeight: sem.height,
                                                       geometry: geometry.result, depthWidth: geometry.depthWidth,
                                                       depthHeight: geometry.depthHeight, K: calibration.K)
            amodalImage = amodal.image
            amodalText = "Visible road: \(amodal.visibleRoadPx) px, recovered occluded road: \(amodal.recoveredPx) px "
                + "(+\(String(format: "%.1f", amodal.recoveredPercent))%), amodal inference \(amodal.elapsedMs)ms"
            statusText = "Calibration source: \(calibration.source.rawValue) -- full pipeline OK "
                + "(sem \(sem.elapsedMs)ms, geo \(geometry.elapsedMs)ms, amodal \(amodal.elapsedMs)ms)"
        } catch {
            statusText = "Pipeline failed: \(error)"
        }
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
    private func runAmodalCompletion(semClasses: [Int], semWidth: Int, semHeight: Int,
                                     geometry: GeometryPipeline.Result, depthWidth: Int, depthHeight: Int,
                                     K: simd_double3x3) async throws -> AmodalRunResult {
        try await Task.detached(priority: .userInitiated) {
            let model = try ModelBundle.loadOFRSNet()
            precondition(model.width == semWidth && model.height == semHeight,
                        "bundled OFRSNet resolution (\(model.width)x\(model.height)) must match "
                        + "the semantic model's (\(semWidth)x\(semHeight))")

            let resizedDepth = Self.resizeFloatFieldBilinear(
                geometry.depth, srcWidth: depthWidth, srcHeight: depthHeight,
                dstWidth: semWidth, dstHeight: semHeight)
            let scaledK = Intrinsics.scaleK(K, sx: Double(semWidth) / Double(depthWidth),
                                            sy: Double(semHeight) / Double(depthHeight))
            let (G, h, gvalid) = GroundFields.derive(depth: resizedDepth, n: geometry.n, K: scaledK,
                                                     width: semWidth, height: semHeight)

            let start = Date()
            let logits = try model.predict(sem: semClasses, G: G, h: h, gvalid: gvalid, validPlane: geometry.valid)
            let elapsedMs = Int(Date().timeIntervalSince(start) * 1000)

            let count = semWidth * semHeight
            var visibleRoadPx = 0, recoveredPx = 0
            var pixels = [UInt8](repeating: 255, count: count * 4)
            for i in 0..<count {
                let amodalRoad = logits[count + i] > logits[i]   // channel 1 (road) > channel 0
                let visible = semClasses[i] == roadClassIndex
                if visible { visibleRoadPx += 1 }
                var r: UInt8 = 32, g: UInt8 = 32, b: UInt8 = 32
                if visible {
                    r = 0; g = 200; b = 0                       // visible road: green
                } else if amodalRoad {
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
            return AmodalRunResult(image: image, visibleRoadPx: visibleRoadPx, recoveredPx: recoveredPx,
                                  recoveredPercent: recoveredPercent, elapsedMs: elapsedMs)
        }.value
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
