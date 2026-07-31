//
//  ImageBytes.swift
//  Parking Disturbance
//
//  CGImage -> row-major interleaved RGB UInt8 bytes, the input format
//  AmodalRoadKit.ImagePreprocessing.resizeAndNormalizeImageNet expects.
//  App-level (CoreGraphics/UIKit-tied), not part of the SwiftPM package for
//  the same reason ImageIO/EXIF extraction isn't -- see EXIFCalibration.swift.

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
        guard let context = CGContext(data: &bytes, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
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
