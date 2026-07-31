import simd

/// Port of `bev.warp_to_bev` (src/bev.py) -- inverse-warps an image-space
/// array into the BEV grid via the homography `H` (which already maps
/// BEV pixel -> image pixel, exactly `cv2.WARP_INVERSE_MAP`'s convention,
/// so no extra inversion happens here).
///
/// Implemented as a plain Swift loop, not vImage: this repo has no way to
/// verify Apple's exact vImage perspective-warp API surface without a full
/// Xcode project and header access, and getting a numeric API detail wrong
/// silently (rather than a compile error) is exactly the failure mode this
/// migration's golden-value testing approach exists to avoid. This
/// implementation IS verified, against real `cv2.warpPerspective` output
/// (see WarpGoldenValueTests.swift) -- correct first, matching the plan's
/// "optimize with vImage/Metal later if profiling requires it" framing
/// (src/bev.py's own docstring for `nearest_label_bev` makes the same
/// correctness-first-then-optimize call for the Voronoi labeling).
public enum Interpolation {
    case nearest
    case bilinear
}

public enum Warp {
    /// `src` is a row-major `[Float]` of shape `(srcHeight, srcWidth)`.
    /// Returns a row-major `[Float]` of shape `(grid.height, grid.width)`.
    /// Out-of-bounds samples are 0 (matches `cv2.BORDER_CONSTANT,
    /// borderValue=0`); `.bilinear` blends smoothly with that 0 border
    /// rather than hard-clipping at the source edge, matching cv2.
    public static func warpToBev(src: [Float], srcWidth: Int, srcHeight: Int,
                                 H: simd_double3x3, grid: BevGrid,
                                 interpolation: Interpolation) -> [Float] {
        var dst = [Float](repeating: 0, count: grid.width * grid.height)
        for v in 0..<grid.height {
            for u in 0..<grid.width {
                let hom = H * SIMD3<Double>(Double(u), Double(v), 1.0)
                guard hom.z != 0 else { continue }
                let sx = hom.x / hom.z
                let sy = hom.y / hom.z
                let value: Float
                switch interpolation {
                case .nearest:
                    value = sampleNearest(src, srcWidth, srcHeight, sx, sy)
                case .bilinear:
                    value = sampleBilinear(src, srcWidth, srcHeight, sx, sy)
                }
                dst[v * grid.width + u] = value
            }
        }
        return dst
    }

    /// Convenience overload for boolean masks (visible/amodal road masks,
    /// bev_validity's solid-image warp) -- always nearest, matching
    /// `warp_to_bev`'s dtype-dispatch in Python (bool forces `nearest=True`).
    public static func warpToBev(mask: [Bool], srcWidth: Int, srcHeight: Int,
                                 H: simd_double3x3, grid: BevGrid) -> [Bool] {
        let src = mask.map { $0 ? Float(1) : Float(0) }
        let out = warpToBev(src: src, srcWidth: srcWidth, srcHeight: srcHeight,
                            H: H, grid: grid, interpolation: .nearest)
        return out.map { $0 > 0.5 }
    }

    private static func sampleNearest(_ src: [Float], _ w: Int, _ h: Int,
                                      _ x: Double, _ y: Double) -> Float {
        let xi = Int(pythonRound(x))
        let yi = Int(pythonRound(y))
        guard xi >= 0, xi < w, yi >= 0, yi < h else { return 0 }
        return src[yi * w + xi]
    }

    private static func sampleBilinear(_ src: [Float], _ w: Int, _ h: Int,
                                       _ x: Double, _ y: Double) -> Float {
        let x0 = Int(floor(x)), y0 = Int(floor(y))
        let fx = x - Double(x0), fy = y - Double(y0)

        func at(_ xx: Int, _ yy: Int) -> Double {
            guard xx >= 0, xx < w, yy >= 0, yy < h else { return 0 }
            return Double(src[yy * w + xx])
        }

        let top = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx
        let bottom = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx
        return Float(top * (1 - fy) + bottom * fy)
    }
}
