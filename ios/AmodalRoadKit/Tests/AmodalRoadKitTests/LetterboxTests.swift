import XCTest
@testable import AmodalRoadKit

/// Golden values from the REAL `ultralytics.data.augment.LetterBox.get_params`
/// on the actual 3024x4032 sample photo this session's other verification
/// steps used:
///   r = min(640/4032, 640/3024) = 0.15873015873015872
///   new_unpad = (round(3024*r), round(4032*r)) = (480, 640)
///   dw, dh = 640-480, 640-640 = 160, 0  ->  padX=80, padY=0 (centered)
final class LetterboxTests: XCTestCase {
    func testComputeTransformMatchesPythonOnRealPhotoDimensions() {
        let t = Letterbox.computeTransform(srcWidth: 3024, srcHeight: 4032, targetSize: 640)
        XCTAssertEqual(t.scale, 0.15873015873015872, accuracy: 1e-9)
        XCTAssertEqual(t.newUnpadWidth, 480)
        XCTAssertEqual(t.newUnpadHeight, 640)
        XCTAssertEqual(t.padX, 80.0, accuracy: 1e-9)
        XCTAssertEqual(t.padY, 0.0, accuracy: 1e-9)
    }

    func testComputeTransformSquareImageHasNoPadding() {
        let t = Letterbox.computeTransform(srcWidth: 640, srcHeight: 640, targetSize: 640)
        XCTAssertEqual(t.scale, 1.0, accuracy: 1e-9)
        XCTAssertEqual(t.padX, 0.0, accuracy: 1e-9)
        XCTAssertEqual(t.padY, 0.0, accuracy: 1e-9)
    }

    func testToOriginalIsTheInverseMapping() {
        let t = Letterbox.computeTransform(srcWidth: 3024, srcHeight: 4032, targetSize: 640)
        // A point at the letterboxed frame's padded-content top-left corner
        // (padX, padY) should map back to the original image's (0, 0).
        let origin = Letterbox.toOriginal(x: t.padX, y: t.padY, transform: t)
        XCTAssertEqual(origin.x, 0.0, accuracy: 1e-6)
        XCTAssertEqual(origin.y, 0.0, accuracy: 1e-6)

        // The far corner of the unpadded content maps back to (srcWidth, srcHeight).
        let farCorner = Letterbox.toOriginal(x: t.padX + Double(t.newUnpadWidth),
                                             y: t.padY + Double(t.newUnpadHeight), transform: t)
        XCTAssertEqual(farCorner.x, 3024.0, accuracy: 1e-6)
        XCTAssertEqual(farCorner.y, 4032.0, accuracy: 1e-6)
    }
}
