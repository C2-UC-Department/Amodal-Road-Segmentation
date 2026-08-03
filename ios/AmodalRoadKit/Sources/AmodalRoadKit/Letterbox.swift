import Foundation

/// Port of `ultralytics.data.augment.LetterBox`'s geometry (the
/// aspect-preserving resize + centered pad Ultralytics' own training/
/// inference pipeline uses) -- needed because `YOLOInstanceModel`'s bundled
/// export is FIXED at 640x640 square, and this project's real photos are
/// very much not square (3024x4032 test photo, aspect 0.75); feeding a
/// naively stretched frame would systematically skew every detected box's
/// position, unlike the small, already-corrected-downstream distortion
/// `ImagePreprocessing`'s plain stretch-resize accepts for `DepthModel`/
/// `MobileSemanticModel` -- see that file's header for why THAT
/// simplification is fine and this one is not.
///
/// Deliberately does NOT replicate `LetterBox.get_params`'s exact
/// `round(dh - 0.1)`/`round(dh + 0.1)` asymmetric-rounding trick for
/// top/bottom padding (a tie-breaking detail that only matters at the
/// sub-pixel level); this uses plain symmetric rounding instead. Fine for
/// this package's actual use (drawing detection overlays on a photo), not
/// fine if pixel-exact parity with Ultralytics' own preprocessing were ever
/// required -- revisit if that changes.
public struct LetterboxTransform {
    public let scale: Double
    public let padX: Double
    public let padY: Double
    public let newUnpadWidth: Int
    public let newUnpadHeight: Int

    public init(scale: Double, padX: Double, padY: Double, newUnpadWidth: Int, newUnpadHeight: Int) {
        self.scale = scale
        self.padX = padX
        self.padY = padY
        self.newUnpadWidth = newUnpadWidth
        self.newUnpadHeight = newUnpadHeight
    }
}

public enum Letterbox {
    /// `r = min(target/srcH, target/srcW)` (Ultralytics: `scaleup=True`, no
    /// downscale-only cap), `new_unpad = round(src * r)`, padding is the
    /// remainder split evenly on both sides.
    public static func computeTransform(srcWidth: Int, srcHeight: Int, targetSize: Int) -> LetterboxTransform {
        let r = min(Double(targetSize) / Double(srcHeight), Double(targetSize) / Double(srcWidth))
        let newUnpadWidth = Int((Double(srcWidth) * r).rounded())
        let newUnpadHeight = Int((Double(srcHeight) * r).rounded())
        let padX = Double(targetSize - newUnpadWidth) / 2.0
        let padY = Double(targetSize - newUnpadHeight) / 2.0
        return LetterboxTransform(scale: r, padX: padX, padY: padY,
                                  newUnpadWidth: newUnpadWidth, newUnpadHeight: newUnpadHeight)
    }

    /// Letterboxed-frame `(x, y)` -> original-photo `(x, y)`.
    public static func toOriginal(x: Double, y: Double, transform: LetterboxTransform) -> (x: Double, y: Double) {
        ((x - transform.padX) / transform.scale, (y - transform.padY) / transform.scale)
    }
}
