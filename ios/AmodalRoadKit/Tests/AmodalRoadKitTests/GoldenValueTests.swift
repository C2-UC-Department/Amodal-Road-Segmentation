import XCTest
import simd
@testable import AmodalRoadKit

/// Cross-language parity tests: every expected value here was computed by
/// the REAL src/bev.py (not re-derived independently in Swift), using the
/// exact fixture from tests/test_bev.py's `_K()`/`_plane()` helpers. This
/// proves AmodalRoadKit reproduces the Python implementation's actual
/// numbers, not just that the Swift code is internally self-consistent --
/// the failure mode a self-consistency-only test suite cannot catch (e.g. a
/// transposed matrix that is still invertible and still produces
/// plausible-looking output).
///
/// To regenerate these values after a genuine algorithm change (not a
/// regression), see the golden-value generation script referenced in this
/// package's history -- it runs the equivalent computation through
/// src/bev.py and prints exactly the numbers below.
final class GoldenValueTests: XCTestCase {
    /// n = normalize([0.06, 0.99, -0.12]) / 1.65 -- implied camera height
    /// 1.65m, tilted plane. Matches test_bev.py's `_plane()`.
    let n = SIMD3<Double>(0.03639823112352753, 0.6005708135382043, -0.07279646224705506)
    let K = Homography.pinholeK(fx: 800, fy: 800, cx: 640, cy: 360)
    let grid = BevGrid(ppm: 20.0, xMin: -10.0, xMax: 10.0, zMin: 0.5, zMax: 40.0)

    func testGridShape() {
        XCTAssertEqual(grid.width, 400)
        XCTAssertEqual(grid.height, 790)
        XCTAssertEqual(grid.cellAreaM2, 0.0025, accuracy: 1e-12)
    }

    func testWorldToPixel() {
        XCTAssertEqual(grid.worldToPixel(x: 0.0, z: 10.0).u, 200)
        XCTAssertEqual(grid.worldToPixel(x: 0.0, z: 10.0).v, 600)
        XCTAssertEqual(grid.worldToPixel(x: -7.5, z: 3.25).u, 50)
        XCTAssertEqual(grid.worldToPixel(x: -7.5, z: 3.25).v, 734)
        XCTAssertEqual(grid.worldToPixel(x: 4.0, z: 39.0).u, 280)
        XCTAssertEqual(grid.worldToPixel(x: 4.0, z: 39.0).v, 20)
    }

    func testImpliedCameraHeight() {
        XCTAssertEqual(PlaneGeometry.impliedCameraHeight(n), 1.65, accuracy: 1e-9)
    }

    func testPlaneIsUsable() {
        XCTAssertTrue(PlaneGeometry.planeIsUsable(n))
        XCTAssertFalse(PlaneGeometry.planeIsUsable(SIMD3<Double>(0, 0, 0)))
        XCTAssertFalse(PlaneGeometry.planeIsUsable(SIMD3<Double>(0, -0.6, 0)))   // n.y <= 0
        XCTAssertFalse(PlaneGeometry.planeIsUsable(SIMD3<Double>(0, 5.0, 0)))   // height 0.2m, too short
        XCTAssertFalse(PlaneGeometry.planeIsUsable(SIMD3<Double>(0, 0.02, 0)))  // height 50m, too tall
        XCTAssertFalse(PlaneGeometry.planeIsUsable(nil))
    }

    func testRescalePlaneToHeight() {
        let (n2, implied) = PlaneGeometry.rescalePlaneToHeight(n, cameraHeightM: 1.65)
        XCTAssertEqual(implied, 1.65, accuracy: 1e-9)
        XCTAssertEqual(n2.x, 0.03639823112352753, accuracy: 1e-12)
        XCTAssertEqual(n2.y, 0.6005708135382043, accuracy: 1e-12)
        XCTAssertEqual(n2.z, -0.07279646224705506, accuracy: 1e-12)

        // Rescaling to a DIFFERENT target height must still report the same
        // "implied height BEFORE rescale" (1.65 again, since n itself is
        // fixed) -- this caught a real copy-paste risk when writing this test.
        let (_, implied3) = PlaneGeometry.rescalePlaneToHeight(n, cameraHeightM: 6.92)
        XCTAssertEqual(implied3, 1.65, accuracy: 1e-9)
    }

    /// The homography matrix itself, element-for-element against Python's
    /// `bev.homography_bev_to_image(K, n, grid)`. This is the test that
    /// would catch a row/column transposition in the simd port -- see
    /// Homography.swift's header comment.
    func testHomographyMatrixElements() throws {
        let H = try Homography.bevToImage(K: K, n: n, grid: grid)
        let expected: [[Double]] = [
            [40.0, -32.0, 17604.0],
            [-2.424242424242424, -22.848484848484848, 20083.066064427738],
            [0.0, -0.05, 39.975],
        ]
        for row in 0..<3 {
            for col in 0..<3 {
                // simd subscript is [column][row] -- see Measurement.swift's
                // header comment for the same indexing note.
                XCTAssertEqual(H[col][row], expected[row][col], accuracy: 1e-9,
                               "H[\(row),\(col)]")
            }
        }
    }

    func testProjectPointsToBevRoundTrip() throws {
        let H = try Homography.bevToImage(K: K, n: n, grid: grid)
        let imgPts = [SIMD2<Double>(400.0, 600.0), SIMD2<Double>(700.0, 550.0),
                     SIMD2<Double>(900.0, 650.0)]
        let bevPts = Homography.projectPointsToBev(imgPts, H: H)
        let expected = [SIMD2<Double>(137.29502812342128, 592.1500937447388),
                        SIMD2<Double>(220.16999065491285, 523.9001246011572),
                        SIMD2<Double>(240.96998125105213, 671.9000576890699)]
        for i in 0..<3 {
            XCTAssertEqual(bevPts[i].x, expected[i].x, accuracy: 1e-6, "point \(i) x")
            XCTAssertEqual(bevPts[i].y, expected[i].y, accuracy: 1e-6, "point \(i) y")
        }
    }

    /// Forward through H (BEV->image) then back through its inverse must
    /// recover the original BEV point -- the same invariant
    /// test_bev.py::test_project_points_to_bev_round_trips_through_H checks,
    /// but exercised purely in Swift so it also catches a Swift-side
    /// `.inverse` bug that happened to still agree with Python by accident.
    func testHomographyRoundTripIsSelfConsistent() throws {
        let H = try Homography.bevToImage(K: K, n: n, grid: grid)
        let bevPt = SIMD3<Double>(150.0, 300.0, 1.0)
        let imgHom = H * bevPt
        let imgPt = SIMD2(imgHom.x / imgHom.z, imgHom.y / imgHom.z)
        let back = Homography.projectPointsToBev([imgPt], H: H)[0]
        XCTAssertEqual(back.x, bevPt.x, accuracy: 1e-6)
        XCTAssertEqual(back.y, bevPt.y, accuracy: 1e-6)
    }

    func testGroundM2PerPixelGrowsWithRange() throws {
        let H = try Homography.bevToImage(K: K, n: n, grid: grid)
        let cases: [(z: Double, expected: Double)] = [
            (5.0, 0.00011554828497554563),
            (10.0, 0.0009313715371179705),
            (20.0, 0.007479018587376662),
            (35.0, 0.040147399359425524),
        ]
        var previous = 0.0
        for c in cases {
            let (u, v) = grid.worldToPixel(x: 0.0, z: c.z)
            let got = Measurement.groundM2PerPixel(u: Double(u), v: Double(v), H: H, grid: grid)
            XCTAssertEqual(got, c.expected, accuracy: abs(c.expected) * 1e-6, "z=\(c.z)")
            XCTAssertGreaterThan(got, previous, "resolution must coarsen monotonically with range")
            previous = got
        }
    }

    func testAreaM2() {
        XCTAssertEqual(Measurement.areaM2(cellCount: 1000, grid: grid), 2.5, accuracy: 1e-9)
    }

    /// Parity with test_bev.py::test_homography_raises_on_unusable_plane --
    /// a degenerate plane must fail loudly, not silently produce garbage.
    func testHomographyThrowsOnUnusablePlane() {
        let badN = SIMD3<Double>(0.0, -1.0, 0.0)
        XCTAssertThrowsError(try Homography.bevToImage(K: K, n: badN, grid: grid)) { error in
            guard case AmodalRoadKitError.unusablePlane = error else {
                XCTFail("expected .unusablePlane, got \(error)")
                return
            }
        }
    }
}
