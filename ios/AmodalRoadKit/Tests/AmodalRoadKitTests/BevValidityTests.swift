import XCTest
import simd
@testable import AmodalRoadKit

/// Golden values from the REAL `bev.bev_validity` / `bev.measurable_mask`
/// (src/bev.py), via:
///   .venv/bin/python -c "
///   import numpy as np, sys; sys.path.insert(0,'.')
///   from src import bev
///   K = np.array([[800.0,0,640],[0,800.0,360],[0,0,1]])
///   n = np.array([0.0, 0.6667, -0.05])
///   grid = bev.BevGrid(ppm=20, x_min=-10, x_max=10, z_min=2, z_max=40)
///   H = bev.homography_bev_to_image(K, n, grid)
///   validity = bev.bev_validity(H, grid, (720, 1280))
///   mask, info = bev.measurable_mask(H, grid, (720,1280), max_m2_per_px=0.02)
///   "
final class BevValidityTests: XCTestCase {
    let K = Homography.pinholeK(fx: 800, fy: 800, cx: 640, cy: 360)
    let n = SIMD3<Double>(0.0, 0.6667, -0.05)
    let grid = BevGrid(ppm: 20, xMin: -10, xMax: 10, zMin: 2, zMax: 40)
    let imageWidth = 1280
    let imageHeight = 720

    func testBevValidityMatchesPython() throws {
        let H = try Homography.bevToImage(K: K, n: n, grid: grid)
        let validity = BevValidity.bevValidity(H: H, grid: grid, imageWidth: imageWidth, imageHeight: imageHeight)

        XCTAssertEqual(validity.count, grid.width * grid.height)
        let sum = validity.reduce(0) { $0 + ($1 ? 1 : 0) }
        XCTAssertEqual(sum, 264880)

        let samples: [(v: Int, u: Int, expected: Bool)] = [
            (0, 0, true), (380, 200, true), (759, 399, false), (300, 200, true), (100, 150, true),
        ]
        for s in samples {
            XCTAssertEqual(validity[s.v * grid.width + s.u], s.expected, "validity[\(s.v),\(s.u)]")
        }
    }

    func testMeasurableMaskMatchesPython() throws {
        let H = try Homography.bevToImage(K: K, n: n, grid: grid)
        let (mask, info) = BevValidity.measurableMask(H: H, grid: grid, imageWidth: imageWidth,
                                                       imageHeight: imageHeight, maxM2PerPx: 0.02)

        let sum = mask.reduce(0) { $0 + ($1 ? 1 : 0) }
        XCTAssertEqual(sum, 159280)

        let samples: [(v: Int, u: Int, expected: Bool)] = [
            (0, 0, false), (380, 200, true), (759, 399, false), (300, 200, true), (100, 150, false),
        ]
        for s in samples {
            XCTAssertEqual(mask[s.v * grid.width + s.u], s.expected, "mask[\(s.v),\(s.u)]")
        }

        XCTAssertEqual(info.measurableRangeM, 26.77, accuracy: 0.01)
        XCTAssertEqual(info.inFramePct, 87.13, accuracy: 0.01)
        XCTAssertEqual(info.measurablePct, 52.39, accuracy: 0.01)
    }
}
