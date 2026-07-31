import XCTest
import simd
import Foundation
@testable import AmodalRoadKit

/// Runs the REAL Core ML `.mlpackage` (bundled as a test resource, exported
/// by `tools/coreml_export_ofrsnet.py` from the actual trained checkpoint --
/// see GoldenData_OFRSNetModel.swift's header for the exact commands) through
/// Apple's on-device Core ML runtime, not just checking that `OFRSNetModel`
/// compiles against the CoreML API. This is the first point in the migration
/// where Swift code actually executes a converted model rather than
/// reimplementing pure math -- the acceptance bar is different accordingly:
/// bit-exact output isn't expected (this runs through Apple's CoreML runtime,
/// not coremltools' Python prediction path the export tool's own parity
/// numbers were measured against), but the THRESHOLDED road/not-road
/// decision should closely track the real PyTorch model's, the same
/// road_iou bar the Python-side tool already established
/// (tools/coreml_export_ofrsnet.py measured 0.988949 at this exact
/// resolution, Python CoreML vs PyTorch -- see that file's docstring).
final class OFRSNetModelTests: XCTestCase {
    let height = GoldenData.ofrsRoadMaskHeight
    let width = GoldenData.ofrsRoadMaskWidth

    func loadModel() throws -> OFRSNetModel {
        guard let url = Bundle.module.url(forResource: "OFRSNetExport", withExtension: "mlpackage") else {
            XCTFail("OFRSNetExport.mlpackage not found in test bundle resources")
            throw XCTSkip("missing test fixture")
        }
        return try OFRSNetModel(contentsOf: url, numClasses: 11)
    }

    /// Exact port of GoldenData_OFRSNetModel.swift's generation script --
    /// same class layout, same G formula, same deterministic (non-RNG) h
    /// formula, same gvalid cutoff. Must reproduce the identical scene the
    /// golden road mask was computed from.
    func makeScene() -> (sem: [Int], G: [SIMD3<Double>], h: [Double], gvalid: [Bool]) {
        let road = 0, vehicle = 9, building = 2   // config.OFRS_CLASSES indices
        var sem = [Int](repeating: building, count: height * width)
        for v in (height / 3)..<height {
            for u in 0..<width { sem[v * width + u] = road }
        }
        let vy0 = height / 2, vy1 = height / 2 + max(1, height / 12)
        let vx0 = width / 3, vx1 = width / 3 + max(1, width / 6)
        for v in vy0..<vy1 {
            for u in vx0..<vx1 { sem[v * width + u] = vehicle }
        }

        var G = [SIMD3<Double>](repeating: .zero, count: height * width)
        var h = [Double](repeating: 0, count: height * width)
        var gvalid = [Bool](repeating: false, count: height * width)
        for v in 0..<height {
            let gz = 2.0 + Double(v) * (28.0 / Double(height - 1))          // linspace(2, 30, height)
            let hFactor = (Double(v % 7) - 3.0) / 3.0
            let valid = v >= height / 3
            for u in 0..<width {
                let gx = -5.0 + Double(u) * (10.0 / Double(width - 1))      // linspace(-5, 5, width)
                G[v * width + u] = SIMD3(gx, 0, gz)
                let wFactor = (Double(u % 5) - 2.0) / 2.0
                h[v * width + u] = 0.1 * hFactor * wFactor
                gvalid[v * width + u] = valid
            }
        }
        return (sem, G, h, gvalid)
    }

    func unpackGoldenRoadMask() -> [Bool] {
        let data = Data(base64Encoded: GoldenData.ofrsRoadMaskPackedBase64)!
        let count = height * width
        var mask = [Bool](repeating: false, count: count)
        for i in 0..<count {
            let byte = data[i / 8]
            let bitIndexFromMSB = 7 - (i % 8)     // np.packbits is MSB-first
            mask[i] = (byte >> bitIndexFromMSB) & 1 == 1
        }
        return mask
    }

    func testModelReportsExpectedResolution() throws {
        let model = try loadModel()
        XCTAssertEqual(model.height, height)
        XCTAssertEqual(model.width, width)
    }

    func testPredictReturnsExpectedShape() throws {
        let model = try loadModel()
        let scene = makeScene()
        let logits = try model.predict(sem: scene.sem, G: scene.G, h: scene.h,
                                       gvalid: scene.gvalid, validPlane: true)
        XCTAssertEqual(logits.count, GoldenData.ofrsExpectedLogitsCount)
    }

    /// The real acceptance test: thresholded road/not-road agreement between
    /// Apple's on-device Core ML execution and the real PyTorch model, on
    /// the same input.
    func testPredictRoadMaskMatchesPyTorchWithinTolerance() throws {
        let model = try loadModel()
        let scene = makeScene()
        let logits = try model.predict(sem: scene.sem, G: scene.G, h: scene.h,
                                       gvalid: scene.gvalid, validPlane: true)
        let count = height * width
        var predictedRoad = [Bool](repeating: false, count: count)
        for i in 0..<count { predictedRoad[i] = logits[count + i] > logits[i] }  // channel 1 > channel 0

        let golden = unpackGoldenRoadMask()
        var intersection = 0, union = 0
        for i in 0..<count {
            if predictedRoad[i] && golden[i] { intersection += 1 }
            if predictedRoad[i] || golden[i] { union += 1 }
        }
        let iou = union > 0 ? Double(intersection) / Double(union) : 1.0
        // Python-side tool measured 0.988949 (Python CoreML vs PyTorch) at
        // this exact resolution -- 0.95 leaves headroom for Apple's runtime
        // being a different execution engine on the same mlprogram, without
        // being so loose it'd pass a badly broken conversion.
        XCTAssertGreaterThan(iou, 0.95, "road mask IoU vs PyTorch reference too low: \(iou)")
    }
}
