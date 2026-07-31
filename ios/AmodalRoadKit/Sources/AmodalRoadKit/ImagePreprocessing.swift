import Foundation

/// RGB image -> model-ready tensors, for `DepthModel` and `MobileSemanticModel`.
///
/// The resize half is deliberately NOT a port of
/// `transformers.models.dpt.image_processing_dpt` (`DPTImageProcessor`),
/// which `tools/coreml_export_depth.py`'s model was traced with: that
/// processor preserves aspect ratio (scale-to-fit, then round to a multiple
/// of 14 via `get_resize_output_image_size`, effectively letterboxing).
/// This does a plain stretch-to-`(targetWidth, targetHeight)` bilinear
/// resize instead. Rationale for not chasing exact parity here (the
/// opposite call from `Warp.swift`'s pixel-exact `cv2.warpPerspective`
/// port): depth is already uncorrected/uncalibrated at this stage and gets
/// its own independent metric-scale correction from the camera-height prior
/// (config.py documents 50-400% correction magnitude on real footage) -- the
/// geometric distortion from a squashed-aspect resize is small next to error
/// that is already being corrected downstream. `MobileSemanticNet` has no
/// such downstream correction, but its own training data (`src/mobile_semantic
/// /dataset.py`) is likewise never letterboxed -- `fit_within_max_side`
/// preserves aspect only up to a cap, never pads -- so a plain stretch here
/// is consistent with what the model actually saw during training, not a
/// new source of train/inference mismatch.
///
/// The NORMALIZE half (ImageNet mean/std, for `DepthModel` only --
/// `MobileSemanticNet` normalizes internally, see `MobileSemanticModel.swift`)
/// IS exact -- that's a fixed arithmetic formula, not an approximated
/// algorithm, so there's no reason for it to be anything but precise.
public enum ImagePreprocessing {
    public static let imagenetMean: [Double] = [0.485, 0.456, 0.406]
    public static let imagenetStd: [Double] = [0.229, 0.224, 0.225]

    /// `rgb`: row-major, interleaved R,G,B `UInt8`, `srcWidth x srcHeight`.
    /// Returns a channel-major `Float` array, `3 * targetHeight * targetWidth`,
    /// each channel independently resampled via `sample`.
    private static func resizeChannelMajor(rgb: [UInt8], srcWidth: Int, srcHeight: Int,
                                           targetWidth: Int, targetHeight: Int,
                                           sample: (Double, Int) -> Float) -> [Float] {
        precondition(rgb.count == srcWidth * srcHeight * 3)
        var out = [Float](repeating: 0, count: 3 * targetHeight * targetWidth)
        let planeSize = targetHeight * targetWidth

        let sx = Double(srcWidth) / Double(targetWidth)
        let sy = Double(srcHeight) / Double(targetHeight)

        for ty in 0..<targetHeight {
            let fy = (Double(ty) + 0.5) * sy - 0.5
            let y0 = max(0, min(srcHeight - 1, Int(floor(fy))))
            let y1 = max(0, min(srcHeight - 1, y0 + 1))
            let wy = max(0.0, min(1.0, fy - Double(y0)))

            for tx in 0..<targetWidth {
                let fx = (Double(tx) + 0.5) * sx - 0.5
                let x0 = max(0, min(srcWidth - 1, Int(floor(fx))))
                let x1 = max(0, min(srcWidth - 1, x0 + 1))
                let wx = max(0.0, min(1.0, fx - Double(x0)))

                for c in 0..<3 {
                    let p00 = Double(rgb[(y0 * srcWidth + x0) * 3 + c])
                    let p01 = Double(rgb[(y0 * srcWidth + x1) * 3 + c])
                    let p10 = Double(rgb[(y1 * srcWidth + x0) * 3 + c])
                    let p11 = Double(rgb[(y1 * srcWidth + x1) * 3 + c])
                    let top = p00 * (1 - wx) + p01 * wx
                    let bot = p10 * (1 - wx) + p11 * wx
                    let v = (top * (1 - wy) + bot * wy) / 255.0
                    out[c * planeSize + ty * targetWidth + tx] = sample(v, c)
                }
            }
        }
        return out
    }

    /// For `DepthModel.predict(pixelValues:)`: bilinear resize + ImageNet normalize.
    public static func resizeAndNormalizeImageNet(rgb: [UInt8], srcWidth: Int, srcHeight: Int,
                                                  targetWidth: Int, targetHeight: Int) -> [Float] {
        resizeChannelMajor(rgb: rgb, srcWidth: srcWidth, srcHeight: srcHeight,
                          targetWidth: targetWidth, targetHeight: targetHeight) { v, c in
            Float((v - imagenetMean[c]) / imagenetStd[c])
        }
    }

    /// For `MobileSemanticModel.predict(pixelValues:)`: bilinear resize only,
    /// scaled to `[0, 1]` -- the model normalizes internally (see
    /// `MobileSemanticModel.swift`), so no mean/std subtraction here.
    public static func resizeToUnitRangeRGB(rgb: [UInt8], srcWidth: Int, srcHeight: Int,
                                            targetWidth: Int, targetHeight: Int) -> [Float] {
        resizeChannelMajor(rgb: rgb, srcWidth: srcWidth, srcHeight: srcHeight,
                          targetWidth: targetWidth, targetHeight: targetHeight) { v, _ in Float(v) }
    }
}
