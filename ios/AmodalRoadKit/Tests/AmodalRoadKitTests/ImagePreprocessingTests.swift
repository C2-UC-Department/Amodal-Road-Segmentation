import XCTest
@testable import AmodalRoadKit

/// Behavioral tests for `ImagePreprocessing` -- not golden-value (the resize
/// half is an intentional simplification, not a port; see the file's
/// header), but real numeric checks that the actual bilinear-resize and
/// normalize arithmetic behaves correctly, not just that it compiles. Uses
/// plain `[UInt8]` arrays throughout, no CGImage/ImageIO -- this logic is
/// exercised from the app via `ImageBytes.rgb(from:)` on a real photo, which
/// isn't reachable from a SwiftPM test target, so what CAN be unit tested
/// here (the resize/normalize math itself) should be, precisely because the
/// image-decoding half can't be.
final class ImagePreprocessingTests: XCTestCase {
    func testResizeToUnitRangeUniformImageStaysUniform() {
        let src = [UInt8](repeating: 200, count: 4 * 4 * 3)
        let out = ImagePreprocessing.resizeToUnitRangeRGB(rgb: src, srcWidth: 4, srcHeight: 4,
                                                          targetWidth: 8, targetHeight: 8)
        let expected: Float = 200.0 / 255.0
        for v in out {
            XCTAssertEqual(v, expected, accuracy: 1e-5)
        }
    }

    func testResizeToUnitRangeIsWithinBounds() {
        var rng = SplitMix64(seed: 1)
        let src = (0..<(6 * 6 * 3)).map { _ in UInt8(rng.next() % 256) }
        let out = ImagePreprocessing.resizeToUnitRangeRGB(rgb: src, srcWidth: 6, srcHeight: 6,
                                                          targetWidth: 10, targetHeight: 3)
        XCTAssertEqual(out.count, 3 * 3 * 10)
        for v in out {
            XCTAssertGreaterThanOrEqual(v, 0.0)
            XCTAssertLessThanOrEqual(v, 1.0)
        }
    }

    func testResizeToUnitRangeChannelLayoutIsChannelMajor() {
        // Pure red image: R=255, G=0, B=0 everywhere.
        var src = [UInt8](repeating: 0, count: 4 * 4 * 3)
        for i in 0..<(4 * 4) { src[i * 3 + 0] = 255 }
        let out = ImagePreprocessing.resizeToUnitRangeRGB(rgb: src, srcWidth: 4, srcHeight: 4,
                                                          targetWidth: 4, targetHeight: 4)
        let plane = 4 * 4
        XCTAssertTrue(out[0..<plane].allSatisfy { $0 > 0.99 }, "R channel plane should be ~1.0")
        XCTAssertTrue(out[plane..<(2 * plane)].allSatisfy { $0 == 0 }, "G channel plane should be 0")
        XCTAssertTrue(out[(2 * plane)..<(3 * plane)].allSatisfy { $0 == 0 }, "B channel plane should be 0")
    }

    func testResizeAndNormalizeImageNetMatchesHandComputedFormula() {
        // A single uniform-value image: normalize(v) = (v/255 - mean[c]) / std[c].
        let src = [UInt8](repeating: 128, count: 2 * 2 * 3)
        let out = ImagePreprocessing.resizeAndNormalizeImageNet(rgb: src, srcWidth: 2, srcHeight: 2,
                                                                targetWidth: 2, targetHeight: 2)
        let v = 128.0 / 255.0
        let plane = 2 * 2
        for c in 0..<3 {
            let expected = Float((v - ImagePreprocessing.imagenetMean[c]) / ImagePreprocessing.imagenetStd[c])
            for i in 0..<plane {
                XCTAssertEqual(out[c * plane + i], expected, accuracy: 1e-5)
            }
        }
    }

    func testResizeDownsampleAveragesNeighboringPixels() {
        // Two-pixel-wide image, left half 0, right half 255 -- downsampling
        // to 1 pixel wide should land near the midpoint, not snap to either
        // extreme (proves bilinear blending is actually happening, not
        // nearest-neighbour).
        var src = [UInt8](repeating: 0, count: 4 * 2 * 3)
        for y in 0..<2 {
            for c in 0..<3 { src[(y * 4 + 2) * 3 + c] = 255; src[(y * 4 + 3) * 3 + c] = 255 }
        }
        let out = ImagePreprocessing.resizeToUnitRangeRGB(rgb: src, srcWidth: 4, srcHeight: 2,
                                                          targetWidth: 1, targetHeight: 1)
        XCTAssertGreaterThan(out[0], 0.1)
        XCTAssertLessThan(out[0], 0.9)
    }
}
