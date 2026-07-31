import XCTest
import Foundation
import simd
@testable import AmodalRoadKit

/// Behavioral, not golden-value: `DepthModel`, `PlaneFit`, and `GroundFields`
/// are each already independently verified against real Python output
/// (DepthModelTests, PlaneFitGoldenValueTests, GeometryGoldenValueTests) --
/// what's worth checking here is only that composing them through
/// `GeometryPipeline.resolveGeometry` actually wires together correctly on a
/// REAL model (not a hand-built fixture) and produces a sane, non-degenerate
/// result, not that the composition reproduces a specific number.
///
/// Uses the same DepthAnythingV2Small.mlpackage fixture as DepthModelTests
/// (gitignored, large -- skips rather than fails if not regenerated locally).
final class GeometryPipelineTests: XCTestCase {
    func loadModel() throws -> DepthModel {
        let url = DepthModelTests.fixtureURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("DepthAnythingV2Small.mlpackage not present -- see DepthModelTests.")
        }
        return try DepthModel(contentsOf: url)
    }

    func testResolveGeometryProducesValidPlane() throws {
        let model = try loadModel()
        let size = model.width

        // A plausible scene: sky/building texture in the top 2/3, a flat
        // road-textured band in the bottom 1/3 -- same "road below h/3"
        // convention GoldenData_OFRSNetModel.swift's synthetic scene uses.
        var pixelValues = [Float](repeating: 0, count: 3 * size * size)
        var roadMask = [Bool](repeating: false, count: size * size)
        for v in 0..<size {
            for u in 0..<size {
                let isRoad = v >= (2 * size) / 3
                roadMask[v * size + u] = isRoad
                let base: Float = isRoad ? -0.3 : 0.2
                for c in 0..<3 {
                    pixelValues[c * size * size + v * size + u] = base
                }
            }
        }

        // A plausible pinhole K for a 518x518 frame (moderate FOV).
        let K = Homography.pinholeK(fx: 500, fy: 500, cx: Double(size) / 2, cy: Double(size) / 2)

        let result = try GeometryPipeline.resolveGeometry(depthModel: model, pixelValues: pixelValues,
                                                           roadMask: roadMask, K: K)

        XCTAssertEqual(result.depth.count, size * size)
        XCTAssertTrue(result.depth.allSatisfy { $0.isFinite && $0 > 0 },
                     "depth should be finite and positive everywhere for a metric depth model")

        // A flat, textured lower-third road band should give RANSAC enough
        // of a planar point cloud to fit SOMETHING -- not asserting a
        // specific plane, just that this real scene doesn't degrade to nil.
        XCTAssertTrue(result.valid, "expected a fitted plane on a plausible road scene")
        XCTAssertGreaterThan(result.pointsUsed, 100)

        XCTAssertEqual(result.G.count, size * size)
        XCTAssertEqual(result.h.count, size * size)
        XCTAssertEqual(result.gvalid.count, size * size)
        XCTAssertTrue(result.gvalid.contains(true), "at least some rays should hit the fitted plane")
    }

    func testResolveGeometryHandlesEmptyRoadMask() throws {
        let model = try loadModel()
        let size = model.width
        let pixelValues = [Float](repeating: 0, count: 3 * size * size)
        let roadMask = [Bool](repeating: false, count: size * size)
        let K = Homography.pinholeK(fx: 500, fy: 500, cx: Double(size) / 2, cy: Double(size) / 2)

        let result = try GeometryPipeline.resolveGeometry(depthModel: model, pixelValues: pixelValues,
                                                           roadMask: roadMask, K: K)
        XCTAssertFalse(result.valid, "no road pixels -- RANSAC has nothing to fit, n must be nil")
        XCTAssertTrue(result.gvalid.allSatisfy { !$0 })
    }
}
