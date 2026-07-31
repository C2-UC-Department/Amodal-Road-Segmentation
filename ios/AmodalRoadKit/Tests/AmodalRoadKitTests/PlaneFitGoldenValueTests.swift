import XCTest
import simd
@testable import AmodalRoadKit

final class PlaneFitGoldenValueTests: XCTestCase {
    func makePoints() -> [SIMD3<Double>] {
        let flat = GoldenData.planeFitPointsFlat
        precondition(flat.count % 3 == 0)
        var pts: [SIMD3<Double>] = []
        pts.reserveCapacity(flat.count / 3)
        for i in stride(from: 0, to: flat.count, by: 3) {
            pts.append(SIMD3(flat[i], flat[i + 1], flat[i + 2]))
        }
        return pts
    }

    /// Deterministic: `fitPlaneLS` has no RNG, so this asserts against the
    /// exact output of the real `geometry.fit_plane_ls`.
    func testFitPlaneLSMatchesPython() {
        let pts = makePoints()
        guard let n = PlaneFit.fitPlaneLS(pts) else {
            return XCTFail("fitPlaneLS returned nil")
        }
        let expected = GoldenData.planeFitExpectedLS
        XCTAssertEqual(n.x, expected[0], accuracy: 1e-6)
        XCTAssertEqual(n.y, expected[1], accuracy: 1e-6)
        XCTAssertEqual(n.z, expected[2], accuracy: 1e-6)
    }

    func testFitPlaneLSRejectsTooFewPoints() {
        XCTAssertNil(PlaneFit.fitPlaneLS([SIMD3(0, 0, 0), SIMD3(1, 1, 1)]))
        XCTAssertNil(PlaneFit.fitPlaneLS([]))
    }

    /// NOT a golden-value test -- `fitPlaneRANSAC` uses a different RNG than
    /// NumPy on purpose (see PlaneFit.swift's doc comment). This asserts the
    /// property that actually matters: on the same noisy synthetic point
    /// cloud used to validate the Python implementation, the recovered
    /// normal's DIRECTION agrees closely with the known true plane.
    func testRansacRecoversKnownPlane() {
        let pts = makePoints()
        let trueN = GoldenData.planeFitTrueN
        let trueNVec = SIMD3(trueN[0], trueN[1], trueN[2])

        guard let n = PlaneFit.fitPlaneRANSAC(pts, distThreshM: 0.05, iters: 500, seed: 0) else {
            return XCTFail("fitPlaneRANSAC returned nil")
        }
        let cosSim = simd_dot(n, trueNVec) / (simd_length(n) * simd_length(trueNVec))
        XCTAssertGreaterThan(cosSim, 0.999, "recovered plane direction diverges from ground truth")

        // Implied height (1/||n||) should also be close to the true plane's.
        let impliedHeight = PlaneGeometry.impliedCameraHeight(n)
        let trueHeight = PlaneGeometry.impliedCameraHeight(trueNVec)
        XCTAssertEqual(impliedHeight, trueHeight, accuracy: trueHeight * 0.02)
    }

    /// A second seed must independently converge to the same plane -- if
    /// this were sensitive to the specific PRNG draws rather than robustly
    /// finding the dominant inlier structure, different seeds could disagree.
    func testRansacIsRobustAcrossSeeds() {
        let pts = makePoints()
        let trueN = GoldenData.planeFitTrueN
        let trueNVec = SIMD3(trueN[0], trueN[1], trueN[2])

        for seed: UInt64 in [1, 2, 3, 100] {
            guard let n = PlaneFit.fitPlaneRANSAC(pts, distThreshM: 0.05, iters: 500, seed: seed) else {
                XCTFail("seed \(seed): fitPlaneRANSAC returned nil")
                continue
            }
            let cosSim = simd_dot(n, trueNVec) / (simd_length(n) * simd_length(trueNVec))
            XCTAssertGreaterThan(cosSim, 0.999, "seed \(seed) diverged")
        }
    }

    func testRansacRejectsTooFewPoints() {
        XCTAssertNil(PlaneFit.fitPlaneRANSAC([SIMD3(0, 0, 0), SIMD3(1, 1, 1)]))
    }

    func testRansacHandlesAnImpossiblyTightThreshold() {
        // With distThreshM this small, essentially only each iteration's own
        // 3 sampled points are ever inliers (any 3 points are trivially their
        // own exact-fit plane) -- must still return a valid, non-crashing
        // result rather than propagating a degenerate empty-inlier state.
        let pts = makePoints()
        let n = PlaneFit.fitPlaneRANSAC(pts, distThreshM: 1e-12, iters: 10, seed: 0)
        XCTAssertNotNil(n)
    }
}
