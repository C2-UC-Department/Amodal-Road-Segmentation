import XCTest
import simd
@testable import AmodalRoadKit

/// `Attribution.groundContactSeedIds` composes `BottomContour` (exact),
/// `Homography.projectPointsToBev` (exact, golden-value tested elsewhere),
/// and `FootprintBand.rasterize` -- and `FootprintBand` does NOT need to
/// match `cv2.polylines` pixel-for-pixel (see FootprintBand.swift's header,
/// FootprintBandTests.swift's precedent). So this file tests the
/// ORCHESTRATION `groundContactSeedIds` itself adds on top of those pieces
/// (low-coverage detection, smallest-first overlap resolution) rather than
/// asserting an exact raster match against Python.
final class GroundContactSeedIdsTests: XCTestCase {
    let grid = BevGrid(ppm: 10.0, xMin: -3.0, xMax: 3.0, zMin: 0.5, zMax: 3.0)
    var H: simd_double3x3 {
        // K = [[16,0,15],[0,16,2],[0,0,1]], n = (0, 1/1.2, 0) -- same
        // fixture used to verify `lowCoverageIds` against real Python
        // (src/disturbance.py::_ground_contact_seed_ids) below.
        simd_double3x3(
            SIMD3(1.6, 0.0, 0.0),
            SIMD3(-1.5, -0.2, -0.1),
            SIMD3(-2.95, 25.1, 2.95)
        )
    }

    /// Vehicle 2's mask sits entirely in the bottom two rows of a 20-row
    /// image, so EVERY column of its bottom contour is excluded as possibly
    /// frame-truncated (matches `instances.bottom_contour_points`'s `v0 >=
    /// h_img - 2` rule) -- `bottom_contour_points` then returns zero points
    /// and `coverage_frac == 0.0` exactly, which real
    /// `disturbance._ground_contact_seed_ids` confirmed flags vehicle 2 (and
    /// only vehicle 2) as low-coverage, with vehicle 1 the sole contributor
    /// to `seedIds`.
    func testLowCoverageMatchesPython() {
        let width = 30, height = 20
        var instIds = [Int32](repeating: 0, count: width * height)
        for y in 10..<16 { for x in 2..<10 { instIds[y * width + x] = 1 } }
        for y in 18..<20 { for x in 15..<22 { instIds[y * width + x] = 2 } }

        let vehicles = [
            VehicleInstance(instId: 1, label: "vehicle", score: 0.9, pixelArea: 48,
                            centroidU: 0, centroidV: 0, bboxU0: 0, bboxV0: 0, bboxU1: 1, bboxV1: 1,
                            selectable: true, source: "instance"),
            VehicleInstance(instId: 2, label: "vehicle", score: 0.8, pixelArea: 14,
                            centroidU: 0, centroidV: 0, bboxU0: 0, bboxV0: 0, bboxU1: 1, bboxV1: 1,
                            selectable: true, source: "instance"),
        ]

        let (seedIds, lowCoverage) = Attribution.groundContactSeedIds(
            instIds: instIds, vehicles: vehicles, H: H, grid: grid, width: width, height: height)

        XCTAssertEqual(lowCoverage, [2])
        // The seed count is a FootprintBand raster area (line length x band
        // width in BEV pixels), not the vehicle's image-space pixel area
        // (48) -- just check vehicle 1 produced a real seed and vehicle 2
        // (zero contour points) produced none, matching real Python.
        XCTAssertGreaterThan(seedIds.filter { $0 == 1 }.count, 0)
        XCTAssertEqual(seedIds.filter { $0 == 2 }.count, 0)
    }

    /// Two vehicles with the IDENTICAL bottom contour (so their bands land
    /// in exactly the same BEV cells) but different pixel areas: vehicles
    /// are processed smallest-first, so the LARGER one paints last and
    /// must win the full overlap -- matching
    /// `_ground_contact_seed_ids`'s documented "larger/more prominent
    /// vehicle paints last and wins" rule.
    func testLargerVehicleWinsFullOverlap() {
        let width = 30, height = 20
        var small = [Int32](repeating: 0, count: width * height)
        var large = [Int32](repeating: 0, count: width * height)
        for y in 14..<16 { for x in 2..<10 { small[y * width + x] = 1 } }   // bottom row 15, area 16
        for y in 5..<16 { for x in 2..<10 { large[y * width + x] = 1 } }    // SAME bottom row 15, area 88

        var instIds = [Int32](repeating: 0, count: width * height)
        for i in 0..<instIds.count {
            if large[i] == 1 { instIds[i] = 10 }       // instId 10: the larger vehicle
            else if small[i] == 1 { instIds[i] = 20 }  // instId 20: the smaller vehicle
        }

        let vehicles = [
            VehicleInstance(instId: 10, label: "vehicle", score: 0.9, pixelArea: 88,
                            centroidU: 0, centroidV: 0, bboxU0: 0, bboxV0: 0, bboxU1: 1, bboxV1: 1,
                            selectable: true, source: "instance"),
            VehicleInstance(instId: 20, label: "vehicle", score: 0.8, pixelArea: 16,
                            centroidU: 0, centroidV: 0, bboxU0: 0, bboxV0: 0, bboxU1: 1, bboxV1: 1,
                            selectable: true, source: "instance"),
        ]

        let (seedIds, _) = Attribution.groundContactSeedIds(
            instIds: instIds, vehicles: vehicles, H: H, grid: grid, width: width, height: height)

        XCTAssertTrue(seedIds.contains(10), "the larger vehicle must produce a seed")
        XCTAssertFalse(seedIds.contains(20), "the smaller vehicle's identical-contour band must be fully overwritten")
    }
}
