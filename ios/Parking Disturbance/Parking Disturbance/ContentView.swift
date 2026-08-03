//
//  ContentView.swift
//  Parking Disturbance
//
//  Created by Gilang Banyu Biru Erassunu on 31/07/26.
//

import SwiftUI
import AmodalRoadKit
import CoreML
import simd

/// Root view: the real-photo workflow (PhotoSemanticView) is the actual
/// product surface; DebugView below is the Phase-4 verification screen from
/// earlier in this session (proves the package links and the bundled models
/// run at all) -- kept as a second tab rather than deleted, since it's still
/// the fastest way to tell "model problem" from "photo/preprocessing
/// problem" if something ever breaks.
struct ContentView: View {
    var body: some View {
        TabView {
            PhotoSemanticView()
                .tabItem { Label("Photo", systemImage: "photo") }
            DebugView()
                .tabItem { Label("Debug", systemImage: "wrench.and.screwdriver") }
        }
    }
}

/// Phase 4 checkpoint: proves AmodalRoadKit links and runs inside a real
/// app target (Section "Geometry math" below), AND that all four converted
/// Core ML models bundled into this app target (see Models/ and
/// ModelBundle.swift) actually load and run through Bundle.main resolution
/// -- not just the SPM test-bundle resolution ios/AmodalRoadKit's own tests
/// use. Synthetic (not real-photo) inputs throughout -- PhotoSemanticView is
/// where a real picked photo is actually exercised.
struct DebugView: View {
    private let k = Homography.pinholeK(fx: 1000, fy: 1000, cx: 960, cy: 540)
    private let n = SIMD3<Double>(0.0, 0.6667, -0.05)
    private let grid = BevGrid(ppm: 20, xMin: -10, xMax: 10, zMin: 2, zMax: 40)

    private var impliedHeightM: Double { PlaneGeometry.impliedCameraHeight(n) }
    private var planeUsable: Bool { PlaneGeometry.planeIsUsable(n) }

    @State private var modelStatus: [String] = ["Loading bundled Core ML models..."]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("AmodalRoadKit geometry math")
                    .font(.headline)
                Text("Plane usable: \(planeUsable ? "yes" : "no")")
                Text(String(format: "Implied camera height: %.2f m", impliedHeightM))
                Text("BEV grid: \(grid.width) x \(grid.height) px @ \(Int(grid.ppm)) ppm")
                if let H = try? Homography.bevToImage(K: k, n: n, grid: grid) {
                    let m2 = Measurement.groundM2PerPixel(u: Double(grid.width) / 2, v: Double(grid.height) / 2,
                                                          H: H, grid: grid)
                    Text(String(format: "Ground resolution at grid centre: %.4f m^2/px", m2))
                }

                Divider().padding(.vertical, 4)

                Text("Bundled Core ML models")
                    .font(.headline)
                ForEach(modelStatus, id: \.self) { line in
                    Text(line).font(.system(.body, design: .monospaced)).font(.caption)
                }
            }
            .padding()
        }
        .task { modelStatus = await runModelSmokeTests() }
    }

    /// Loads and runs all four bundled models on synthetic input, reporting
    /// pass/fail + a shape/sample-value summary per model. Not a
    /// correctness check (that's ios/AmodalRoadKit's golden-value test
    /// suite, run against the identical model files) -- this is "does
    /// Bundle.main resolution + on-device execution work from inside a real
    /// app," which a SwiftPM test target cannot exercise.
    private func runModelSmokeTests() async -> [String] {
        await Task.detached(priority: .userInitiated) {
            var lines: [String] = []

            do {
                let model = try ModelBundle.loadMobileSemantic()
                let px = [Float](repeating: 0.4, count: 3 * model.height * model.width)
                let logits = try model.predict(pixelValues: px)
                lines.append("MobileSemanticNet: OK, \(model.height)x\(model.width), logits.count=\(logits.count)")
            } catch {
                lines.append("MobileSemanticNet: FAILED (\(error))")
            }

            do {
                let model = try ModelBundle.loadOFRSNet()
                let count = model.height * model.width
                let sem = [Int](repeating: 0, count: count)
                let g = [SIMD3<Double>](repeating: .zero, count: count)
                let h = [Double](repeating: 0, count: count)
                let gvalid = [Bool](repeating: false, count: count)
                let logits = try model.predict(sem: sem, G: g, h: h, gvalid: gvalid, validPlane: false)
                lines.append("OFRSNetExport: OK, \(model.height)x\(model.width), logits.count=\(logits.count)")
            } catch {
                lines.append("OFRSNetExport: FAILED (\(error))")
            }

            do {
                let model = try ModelBundle.loadDepth()
                let px = [Float](repeating: 0.0, count: 3 * model.height * model.width)
                let depth = try model.predict(pixelValues: px)
                let mean = depth.reduce(0, +) / Float(depth.count)
                lines.append("DepthAnythingV2Small: OK, \(model.height)x\(model.width), mean depth=\(String(format: "%.2f", mean))m")
            } catch {
                lines.append("DepthAnythingV2Small: FAILED (\(error))")
            }

            do {
                let model = try ModelBundle.loadVehicleInstance()
                let pixelBuffer = try Self.makePixelBuffer(width: model.inputWidth, height: model.inputHeight)
                let detections = try model.detect(pixelBuffer: pixelBuffer)
                lines.append("VehicleInstanceYOLO: OK, \(model.inputWidth)x\(model.inputHeight), "
                    + "\(detections.count) detections on a flat grey synthetic frame (expect 0 -- nothing to detect)")
            } catch {
                lines.append("VehicleInstanceYOLO: FAILED (\(error))")
            }

            return lines
        }.value
    }

    private static func makePixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32BGRA, nil, &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw ModelBundleError.resourceNotFound("CVPixelBuffer allocation failed (status \(status))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            memset(base, 128, CVPixelBufferGetDataSize(buffer))
        }
        return buffer
    }
}

#Preview {
    ContentView()
}

#Preview {
    DebugView()
}
