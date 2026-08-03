//
//  LetterboxImage.swift
//  Parking Disturbance
//
//  Builds the letterboxed 640x640 BGRA CVPixelBuffer YOLOInstanceModel's
//  Core ML `ImageType` input expects, per AmodalRoadKit.Letterbox's
//  geometry (aspect-preserving resize + centered grey pad, matching
//  Ultralytics' own LetterBox preprocessing closely enough for real-world
//  detection quality -- see that file's header for the one deliberate
//  simplification, sub-pixel pad-rounding).
//
//  Two things were NOT obvious and got verified empirically (against real
//  Ultralytics detections on the same real photo, via a temporary SwiftPM
//  executable target, same technique used for the depth/geometry/amodal
//  steps) rather than assumed from CoreGraphics/CoreVideo documentation:
//    1. Byte order: kCVPixelFormatType_32BGRA is BGRA in MEMORY. CGContext's
//       `.premultipliedFirst` alone defaults to a byte order that does NOT
//       match this (silently produces a channel-shuffled image -- the first
//       attempt here got 0 detections instead of the real photo's ~10-11).
//       `.byteOrder32Little` is required to actually get BGRA in memory.
//    2. Vertical flip: despite Quartz's usual bottom-up coordinate
//       convention, a CGContext created directly over a CVPixelBuffer's own
//       memory (`CGContext(data: CVPixelBufferGetBaseAddress(...), ...)`)
//       does NOT need a flip to produce a top-left-origin image matching
//       what CVPixelBuffer/Core ML expect -- adding one (the first attempt
//       here did) produces detections at the wrong vertical position
//       entirely. Confirmed by comparing decoded box coordinates against
//       real Ultralytics output with and without the flip.

import CoreGraphics
import CoreVideo
import AmodalRoadKit

enum LetterboxImage {
    static func makePixelBuffer(cgImage: CGImage, targetSize: Int) -> (CVPixelBuffer, LetterboxTransform)? {
        let transform = Letterbox.computeTransform(srcWidth: cgImage.width, srcHeight: cgImage.height,
                                                    targetSize: targetSize)

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, targetSize, targetSize,
                                         kCVPixelFormatType_32BGRA, nil, &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        guard let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                      width: targetSize, height: targetSize, bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                      space: colorSpace, bitmapInfo: bitmapInfo) else { return nil }

        context.setFillColor(red: 114.0 / 255, green: 114.0 / 255, blue: 114.0 / 255, alpha: 1.0)
        context.fill(CGRect(x: 0, y: 0, width: targetSize, height: targetSize))

        let drawRect = CGRect(x: transform.padX, y: transform.padY,
                              width: Double(transform.newUnpadWidth), height: Double(transform.newUnpadHeight))
        context.draw(cgImage, in: drawRect)

        return (buffer, transform)
    }
}
