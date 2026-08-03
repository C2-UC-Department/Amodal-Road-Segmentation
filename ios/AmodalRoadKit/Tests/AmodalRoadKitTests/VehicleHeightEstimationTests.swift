import XCTest
@testable import AmodalRoadKit

/// Golden values from the REAL `src/instances.py::_roofline_height_m` /
/// `estimate_camera_height_from_vehicles`, on deterministic synthetic
/// scenes (a smooth h-field formula + rectangular vehicle masks with a
/// forced roofline value) -- generation commands documented per test.
final class VehicleHeightEstimationTests: XCTestCase {
    let width = 192
    let height = 128

    func makeBaseHField() -> [Double] {
        var h = [Double](repeating: 0, count: width * height)
        for v in 0..<height {
            for u in 0..<width {
                h[v * width + u] = 0.1 * (Double(v % 7) - 3) + 0.05 * (Double(u % 5) - 2)
            }
        }
        return h
    }

    func makeMask(v0: Int, v1: Int, u0: Int, u1: Int) -> [Bool] {
        var mask = [Bool](repeating: false, count: width * height)
        for v in v0..<v1 { for u in u0..<u1 { mask[v * width + u] = true } }
        return mask
    }

    /// .venv/bin/python -c "
    /// import numpy as np, sys; sys.path.insert(0,'.')
    /// from src.instances import _roofline_height_m
    /// h,w = 128,192
    /// h_field = np.zeros((h,w))
    /// for v in range(h):
    ///     for u in range(w): h_field[v,u] = 0.1*((v%7)-3) + 0.05*((u%5)-2)
    /// mask = np.zeros((h,w), dtype=bool); mask[40:100, 40:130] = True
    /// h_field[40:45, 40:130] = -1.4
    /// print(_roofline_height_m(h_field, mask))
    /// "
    /// -> (1.4, True)
    func testRoofHeightMMatchesPython() {
        var h = makeBaseHField()
        let mask = makeMask(v0: 40, v1: 100, u0: 40, u1: 130)
        for v in 40..<45 { for u in 40..<130 { h[v * width + u] = -1.4 } }

        let (roofHeight, ok) = VehicleHeightEstimation.roofHeightM(hField: h, mask: mask, width: width, height: height)
        XCTAssertTrue(ok)
        XCTAssertEqual(roofHeight!, 1.4, accuracy: 1e-9)
    }

    func testRoofHeightMRejectsMaskTouchingTopEdge() {
        let h = makeBaseHField()
        let mask = makeMask(v0: 0, v1: 20, u0: 10, u1: 50)
        let (_, ok) = VehicleHeightEstimation.roofHeightM(hField: h, mask: mask, width: width, height: height)
        XCTAssertFalse(ok, "roofline touching the frame's top edge must be rejected, not silently biased")
    }

    /// Single qualifying vehicle. Python (see testRoofHeightMMatchesPython's
    /// scene, same mask/h_field, implied_biased_height=0.57):
    ///   n_samples=1, k_median=1.0714285714285714, k_spread=0.0,
    ///   camera_height_m=0.6107142857142857
    func testEstimateCameraHeightSingleVehicleMatchesPython() {
        var h = makeBaseHField()
        let mask = makeMask(v0: 40, v1: 100, u0: 40, u1: 130)
        for v in 40..<45 { for u in 40..<130 { h[v * width + u] = -1.4 } }

        let vehicle = VehicleForHeightEstimation(mask: mask, score: 0.95)
        let result = VehicleHeightEstimation.estimateCameraHeight(
            hField: h, vehicles: [vehicle], impliedBiasedHeightM: 0.57, width: width, height: height)

        XCTAssertNotNil(result)
        XCTAssertEqual(result!.nSamples, 1)
        XCTAssertEqual(result!.kMedian, 1.0714285714285714, accuracy: 1e-9)
        XCTAssertEqual(result!.kSpread, 0.0, accuracy: 1e-9)
        XCTAssertEqual(result!.cameraHeightM, 0.6107142857142857, accuracy: 1e-9)
        XCTAssertEqual(result!.perVehicle.count, 1)
        XCTAssertEqual(result!.perVehicle[0].pixelArea, 5400)
        XCTAssertEqual(result!.perVehicle[0].roofHeightBiasedM, 1.4, accuracy: 1e-9)
    }

    /// A second, too-small mask (750px < SCALE_EST_MIN_PIXELS=3000) must be
    /// silently excluded, not crash or corrupt the single qualifying result.
    func testEstimateCameraHeightRejectsTooSmallVehicle() {
        var h = makeBaseHField()
        let goodMask = makeMask(v0: 40, v1: 100, u0: 40, u1: 130)
        for v in 40..<45 { for u in 40..<130 { h[v * width + u] = -1.4 } }
        let tinyMask = makeMask(v0: 10, v1: 15, u0: 5, u1: 10)

        let result = VehicleHeightEstimation.estimateCameraHeight(
            hField: h,
            vehicles: [VehicleForHeightEstimation(mask: goodMask, score: 0.95),
                      VehicleForHeightEstimation(mask: tinyMask, score: 0.9)],
            impliedBiasedHeightM: 0.57, width: width, height: height)

        XCTAssertNotNil(result)
        XCTAssertEqual(result!.nSamples, 1, "the tiny mask must be excluded, not counted")
    }

    /// Three qualifying vehicles -- exercises the median/25th-75th-percentile
    /// spread computation against real Python output:
    ///   n_samples=3, k_median=0.9677419354838709,
    ///   k_spread=0.06919642857142845, camera_height_m=0.5516129032258064
    func testEstimateCameraHeightThreeVehiclesMatchesPython() {
        var h = makeBaseHField()
        let mask1 = makeMask(v0: 40, v1: 100, u0: 40, u1: 130)
        for v in 40..<45 { for u in 40..<130 { h[v * width + u] = -1.4 } }

        let mask2 = makeMask(v0: 20, v1: 90, u0: 10, u1: 100)
        for v in 20..<26 { for u in 10..<100 { h[v * width + u] = -1.6 } }

        let mask3 = makeMask(v0: 30, v1: 110, u0: 100, u1: 190)
        for v in 30..<36 { for u in 100..<190 { h[v * width + u] = -1.55 } }

        let vehicles = [
            VehicleForHeightEstimation(mask: mask1, score: 0.95),
            VehicleForHeightEstimation(mask: mask2, score: 0.90),
            VehicleForHeightEstimation(mask: mask3, score: 0.88),
        ]
        let result = VehicleHeightEstimation.estimateCameraHeight(
            hField: h, vehicles: vehicles, impliedBiasedHeightM: 0.57, width: width, height: height)

        XCTAssertNotNil(result)
        XCTAssertEqual(result!.nSamples, 3)
        XCTAssertEqual(result!.kMedian, 0.9677419354838709, accuracy: 1e-9)
        XCTAssertEqual(result!.kSpread, 0.06919642857142845, accuracy: 1e-9)
        XCTAssertEqual(result!.cameraHeightM, 0.5516129032258064, accuracy: 1e-9)
    }

    func testEstimateCameraHeightReturnsNilWithNoQualifyingVehicles() {
        let h = makeBaseHField()
        let result = VehicleHeightEstimation.estimateCameraHeight(
            hField: h, vehicles: [], impliedBiasedHeightM: 0.57, width: width, height: height)
        XCTAssertNil(result)
    }
}
