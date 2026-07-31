import XCTest
import Foundation
@testable import AmodalRoadKit

/// Runs the REAL Depth-Anything-V2 Core ML `.mlpackage` (see
/// GoldenData_DepthModel.swift's header for the exact export command)
/// through Apple's on-device Core ML runtime, same posture as
/// OFRSNetModelTests: bit-exact match isn't the bar (different execution
/// engine than coremltools' own Python prediction path), tracking the
/// real PyTorch model's output within a documented tolerance is.
///
/// UNLIKE OFRSNetModelTests, this fixture is NOT git-tracked (~47MB, over
/// the "small models get committed" line -- see .gitignore and
/// ios/AmodalRoadKit/README.md's Test Fixtures section). If it hasn't been
/// regenerated locally, these tests skip rather than fail -- a missing
/// optional fixture is not a code defect.
final class DepthModelTests: XCTestCase {
    static var fixtureURL: URL {
        // Resources/ is a sibling directory of this file.
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Resources/DepthAnythingV2Small.mlpackage")
    }

    func loadModel() throws -> DepthModel {
        let url = Self.fixtureURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("""
                DepthAnythingV2Small.mlpackage not present (large model, gitignored by \
                design). Regenerate with:
                  .venv/bin/python -m tools.coreml_export_depth --out \(url.path)
                """)
        }
        return try DepthModel(contentsOf: url)
    }

    /// Exact port of GoldenData_DepthModel.swift's generation script's
    /// synthetic-input formula -- must reproduce the identical tensor the
    /// golden depth samples were computed from.
    func makePixelValues(size: Int) -> [Float] {
        var out = [Float](repeating: 0, count: 3 * size * size)
        let plane = size * size
        for v in 0..<size {
            for u in 0..<size {
                for c in 0..<3 {
                    let cc = Double(c)
                    let val = sin(Double(v) * 0.01 + cc) * 0.5 + cos(Double(u) * 0.013 - cc * 0.7) * 0.5
                    out[c * plane + v * size + u] = Float(val)
                }
            }
        }
        return out
    }

    func testModelReportsExpectedResolution() throws {
        let model = try loadModel()
        XCTAssertEqual(model.height, GoldenData.depthModelSize)
        XCTAssertEqual(model.width, GoldenData.depthModelSize)
    }

    func testPredictReturnsExpectedShape() throws {
        let model = try loadModel()
        let px = makePixelValues(size: model.width)
        let depth = try model.predict(pixelValues: px)
        XCTAssertEqual(depth.count, model.height * model.width)
    }

    /// The real acceptance test: sampled depth values from Apple's on-device
    /// Core ML execution vs the real PyTorch model, on the same input.
    /// Tolerance matches what tools/coreml_export_depth.py's own docstring
    /// already documents for PyTorch-vs-Core-ML drift on this exact model
    /// (~2-3.5% relative, fp16 Neural-Engine-oriented graph) -- 8% leaves
    /// headroom for Apple's runtime being a different execution engine than
    /// coremltools' own Python prediction path (the same gap
    /// OFRSNetModelTests' 0.95 IoU bar, vs the Python tool's own 0.99, is
    /// there for) without being loose enough to pass a broken conversion.
    func testPredictMatchesPyTorchWithinTolerance() throws {
        let model = try loadModel()
        let px = makePixelValues(size: model.width)
        let depth = try model.predict(pixelValues: px)

        var maxRelErr = 0.0
        for (row, col, expected) in GoldenData.depthModelSamples {
            let actual = Double(depth[row * model.width + col])
            let relErr = abs(actual - expected) / abs(expected)
            maxRelErr = max(maxRelErr, relErr)
            XCTAssertEqual(actual, expected, accuracy: abs(expected) * 0.08,
                          "row \(row) col \(col): got \(actual), expected \(expected)")
        }
        print("[DepthModelTests] max relative error across \(GoldenData.depthModelSamples.count) samples: \(maxRelErr)")
    }
}
