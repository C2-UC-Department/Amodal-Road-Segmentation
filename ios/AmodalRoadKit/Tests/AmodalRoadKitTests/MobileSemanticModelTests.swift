import XCTest
@testable import AmodalRoadKit

/// Same posture as OFRSNetModelTests/DepthModelTests: runs the REAL bundled
/// `.mlpackage` through Apple's on-device Core ML runtime and checks against
/// real PyTorch output at a set of sample points, not bit-exact equality.
final class MobileSemanticModelTests: XCTestCase {
    let size = GoldenData.mobileSemanticSize

    func loadModel() throws -> MobileSemanticModel {
        guard let url = Bundle.module.url(forResource: "MobileSemanticNet", withExtension: "mlpackage") else {
            XCTFail("MobileSemanticNet.mlpackage not found in test bundle resources")
            throw XCTSkip("missing test fixture")
        }
        return try MobileSemanticModel(contentsOf: url)
    }

    /// Exact port of GoldenData_MobileSemanticModel.swift's generation
    /// script's synthetic-input formula.
    func makePixelValues() -> [Float] {
        var out = [Float](repeating: 0, count: 3 * size * size)
        let plane = size * size
        for v in 0..<size {
            for u in 0..<size {
                for c in 0..<3 {
                    let cc = Double(c)
                    let raw = sin(Double(v) * 0.01 + cc) * 0.5 + cos(Double(u) * 0.013 - cc * 0.7) * 0.5
                    out[c * plane + v * size + u] = Float(raw * 0.5 + 0.5)
                }
            }
        }
        return out
    }

    func testModelReportsExpectedShape() throws {
        let model = try loadModel()
        XCTAssertEqual(model.height, size)
        XCTAssertEqual(model.width, size)
        XCTAssertEqual(model.numClasses, 11)
    }

    /// Argmax class agreement at 36 sample points against the real PyTorch
    /// model -- allows a small number of mismatches (near class-boundary
    /// pixels can legitimately flip between adjacent logits under fp16
    /// Core ML rounding, same rationale as NearestLabelGoldenValueTests'
    /// tolerance), same spirit as OFRSNetModelTests' IoU threshold.
    func testPredictMatchesPyTorchClassesAtSamplePoints() throws {
        let model = try loadModel()
        let px = makePixelValues()
        let logits = try model.predict(pixelValues: px)

        var mismatches = 0
        for (row, col, expectedClass) in GoldenData.mobileSemanticSamples {
            var bestClass = 0
            var bestVal = -Float.infinity
            for c in 0..<model.numClasses {
                let v = logits[c * size * size + row * size + col]
                if v > bestVal { bestVal = v; bestClass = c }
            }
            if bestClass != expectedClass { mismatches += 1 }
        }
        XCTAssertLessThanOrEqual(mismatches, 2,
                                "too many class mismatches vs PyTorch: \(mismatches)/\(GoldenData.mobileSemanticSamples.count)")
    }
}
