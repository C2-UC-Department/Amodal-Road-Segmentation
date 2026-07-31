import XCTest
import simd
@testable import AmodalRoadKit

final class CalibrationGoldenValueTests: XCTestCase {
    /// `simd_double3x3` is column-major; golden fixtures are row-major flat
    /// arrays (numpy's default), matching the convention already established
    /// in GeometryGoldenValueTests/GoldenValueTests.
    func assertMatches(_ K: simd_double3x3, _ flatRowMajor: [Double],
                        accuracy: Double = 1e-9, file: StaticString = #filePath, line: UInt = #line) {
        let rows = [[K[0].x, K[1].x, K[2].x],
                    [K[0].y, K[1].y, K[2].y],
                    [K[0].z, K[1].z, K[2].z]]
        let flat = rows.flatMap { $0 }
        XCTAssertEqual(flat.count, flatRowMajor.count, file: file, line: line)
        for i in 0..<flat.count {
            XCTAssertEqual(flat[i], flatRowMajor[i], accuracy: accuracy, "index \(i)", file: file, line: line)
        }
    }

    func testKFromFocalMatchesPython() {
        let K = Intrinsics.kFromFocal(fPx: 1000.0, w: 640, h: 480)
        assertMatches(K, GoldenData.calibKFromFocal)
    }

    func testKFromFovMatchesPython() {
        let K = Intrinsics.kFromFOV(w: 640, h: 480, hfovDeg: 65.0)
        assertMatches(K, GoldenData.calibKFromFov)
    }

    func testScaleKMatchesPython() {
        let K = Intrinsics.kFromFocal(fPx: 1000.0, w: 640, h: 480)
        let scaled = Intrinsics.scaleK(K, sx: 0.5, sy: 0.75)
        assertMatches(scaled, GoldenData.calibScaleK)
    }

    /// `FocalLengthIn35mmFilm` path -- src/calibration.py:174-178.
    func testKFromExifTagsF35PathMatchesPython() {
        let K = Intrinsics.kFromExifTags(focalLengthIn35mmFilm: 26.0,
                                          focalLengthMM: nil,
                                          focalPlaneXResolution: nil,
                                          focalPlaneResolutionUnit: nil,
                                          w: 640, h: 480)
        XCTAssertNotNil(K)
        assertMatches(K!, GoldenData.calibExifF35Expected)
    }

    /// `FocalLength` + `FocalPlaneXResolution` fallback path --
    /// src/calibration.py:180-189. f35 absent, so this must fall through to
    /// the focal-plane arithmetic exactly the way Python does.
    func testKFromExifTagsFocalPlanePathMatchesPython() {
        let K = Intrinsics.kFromExifTags(focalLengthIn35mmFilm: nil,
                                          focalLengthMM: 4.25,
                                          focalPlaneXResolution: 8000.0,
                                          focalPlaneResolutionUnit: 2,
                                          w: 640, h: 480)
        XCTAssertNotNil(K)
        assertMatches(K!, GoldenData.calibExifFocalPlaneExpected)
    }

    func testKFromExifTagsReturnsNilWithNoUsableTags() {
        XCTAssertNil(Intrinsics.kFromExifTags(focalLengthIn35mmFilm: nil, focalLengthMM: nil,
                                               focalPlaneXResolution: nil, focalPlaneResolutionUnit: nil,
                                               w: 640, h: 480))
        // FocalLength without FocalPlaneXResolution isn't enough either.
        XCTAssertNil(Intrinsics.kFromExifTags(focalLengthIn35mmFilm: nil, focalLengthMM: 4.25,
                                               focalPlaneXResolution: nil, focalPlaneResolutionUnit: nil,
                                               w: 640, h: 480))
    }

    func testCalibratePicksHighestPriorityAvailableSource() {
        let avK = Intrinsics.kFromFocal(fPx: 900, w: 640, h: 480)
        let exifK = Intrinsics.kFromFocal(fPx: 800, w: 640, h: 480)

        let av = Intrinsics.calibrate(avFoundationK: avK, exifK: exifK, w: 640, h: 480)
        XCTAssertEqual(av.source, .avfoundation)
        XCTAssertTrue(av.calibrated)

        let exif = Intrinsics.calibrate(avFoundationK: nil, exifK: exifK, w: 640, h: 480)
        XCTAssertEqual(exif.source, .exif)
        XCTAssertTrue(exif.calibrated)

        let fallback = Intrinsics.calibrate(avFoundationK: nil, exifK: nil, w: 640, h: 480)
        XCTAssertEqual(fallback.source, .fovPrior)
        XCTAssertFalse(fallback.calibrated)
        assertMatches(fallback.K, GoldenData.calibKFromFov)
    }
}
