import XCTest
import simd
@testable import AmodalRoadKit

/// Behavioural tests, not cross-language golden values: `rasterize` does not
/// need to match `cv2.polylines`/`cv2.circle` pixel-for-pixel (see
/// FootprintBand.swift's header), so these mirror the same properties
/// tests/test_bev.py asserts for the Python version --
/// `test_rasterize_footprint_band_has_expected_width`,
/// `..._single_point_is_a_disk`, `..._empty_points`.
final class FootprintBandTests: XCTestCase {
    let grid = BevGrid(ppm: 20.0, xMin: -10.0, xMax: 10.0, zMin: 0.5, zMax: 40.0)

    func testEmptyPointsGivesEmptyCanvas() {
        let out = FootprintBand.rasterize(points: [], grid: grid, bandWidthM: 0.5)
        XCTAssertEqual(out.count, grid.width * grid.height)
        XCTAssertFalse(out.contains(true))
    }

    func testSinglePointIsADisk() {
        let out = FootprintBand.rasterize(points: [SIMD2(100.0, 300.0)], grid: grid,
                                          bandWidthM: 0.5)
        let expectedRadius = Double(max(1, Int((0.5 * grid.ppm).rounded()) / 2))
        let expectedArea = Double.pi * expectedRadius * expectedRadius
        let actualArea = Double(out.filter { $0 }.count)
        XCTAssertTrue(out.contains(true))
        XCTAssertEqual(actualArea, expectedArea, accuracy: expectedArea * 0.3)
    }

    func testHorizontalLineHasExpectedWidth() {
        let pts = [SIMD2(50.0, 300.0), SIMD2(150.0, 300.0), SIMD2(250.0, 300.0)]
        let out = FootprintBand.rasterize(points: pts, grid: grid, bandWidthM: 0.5)
        let expectedPx = Int((0.5 * grid.ppm).rounded())

        // Column 150 is squarely inside the drawn segment -- measure the
        // band's thickness there.
        var thickness = 0
        for y in 0..<grid.height where out[y * grid.width + 150] {
            thickness += 1
        }
        XCTAssertGreaterThan(thickness, 0)
        XCTAssertLessThanOrEqual(abs(thickness - expectedPx), 2)

        // The band must not bleed into a column far from the drawn segment.
        var columnZeroHits = 0
        for y in 0..<grid.height where out[y * grid.width + 0] {
            columnZeroHits += 1
        }
        XCTAssertEqual(columnZeroHits, 0)
    }

    func testAllFilledCellsAreWithinBounds() {
        // Points near/outside the grid edge must not crash or corrupt memory
        // -- the bounding-box clamping is the thing under test here.
        let pts = [SIMD2(-50.0, -50.0), SIMD2(1000.0, 1000.0)]
        let out = FootprintBand.rasterize(points: pts, grid: grid, bandWidthM: 0.5)
        XCTAssertEqual(out.count, grid.width * grid.height)
    }

    func testWiderBandCoversMoreArea() {
        let pts = [SIMD2(50.0, 300.0), SIMD2(250.0, 300.0)]
        let narrow = FootprintBand.rasterize(points: pts, grid: grid, bandWidthM: 0.2)
        let wide = FootprintBand.rasterize(points: pts, grid: grid, bandWidthM: 1.0)
        XCTAssertLessThan(narrow.filter { $0 }.count, wide.filter { $0 }.count)
    }
}
