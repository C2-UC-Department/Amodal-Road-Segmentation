//
//  ImageBytes.swift
//  Parking Disturbance
//
//  CGImage -> row-major interleaved RGB UInt8 bytes, the input format
//  AmodalRoadKit.ImagePreprocessing.resizeAndNormalizeImageNet expects.
//  App-level (CoreGraphics/UIKit-tied), not part of the SwiftPM package for
//  the same reason ImageIO/EXIF extraction isn't -- see EXIFCalibration.swift.
//
//  Explicit `.byteOrder32Big` below (rather than leaving `bitmapInfo` as
//  just `.premultipliedLast`, host-default byte order): defensive, matching
//  `LetterboxImage.swift`'s explicit `.byteOrder32Little`. NOTE: this did
//  NOT turn out to be the explanation for a real bug found in the same
//  investigation -- see the "100% road" note in PhotoSemanticView.swift /
//  the migration plan for what that actually was (a macOS-vs-iOS-Simulator
//  Core ML execution difference on the semantic/depth models, not an image
//  decode issue). Verify any future change here ON THE SIMULATOR, not via
//  `swift run` on macOS -- the two are not guaranteed to agree.
import CoreGraphics
import Foundation

enum ImageBytes {
    struct RGB {
        let bytes: [UInt8]
        let width: Int
        let height: Int
    }

    static func rgb(from cgImage: CGImage) -> RGB? {
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0, height > 0 else { return nil }

        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let context = CGContext(data: &bytes, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: colorSpace, bitmapInfo: bitmapInfo) else {
            return nil
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        var rgbBytes = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            rgbBytes[i * 3 + 0] = bytes[i * 4 + 0]
            rgbBytes[i * 3 + 1] = bytes[i * 4 + 1]
            rgbBytes[i * 3 + 2] = bytes[i * 4 + 2]
        }
        return RGB(bytes: rgbBytes, width: width, height: height)
    }
}
