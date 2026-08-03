import Foundation

/// Port of `common.dilate` (src/common.py), i.e. `cv2.dilate` with a
/// `cv2.MORPH_ELLIPSE` structuring element of size `(2*radiusPx+1)^2`.
public enum Dilation {
    /// Per-row half-width of OpenCV's square MORPH_ELLIPSE kernel. OpenCV
    /// builds the kernel from a closed-form formula
    /// (`dx = round(r * sqrt((r^2-dy^2)/r^2))`, banker's-rounding
    /// `saturate_cast<int>`), not a plain `dx^2+dy^2<=r^2` circle test --
    /// verified directly against a real `cv2.getStructuringElement` kernel
    /// for r=25: 44 of 2005 "on" pixels differ from the naive circle test,
    /// so this formula is load-bearing, not a simplification. `dx` reduces
    /// to `round(sqrt(r^2-dy^2))` exactly when the kernel is square (r==c,
    /// always true here), which is the only case this package needs.
    static func ellipseHalfWidths(radiusPx r: Int) -> [Int] {
        (-r...r).map { dy in
            let v = Double(r * r - dy * dy)
            return Int(pythonRound(v.squareRoot()))
        }
    }

    /// `radiusPx <= 0` returns `mask` unchanged, matching `common.dilate`'s
    /// own early return (`cv2.getStructuringElement` isn't even valid for a
    /// zero/negative kernel size).
    public static func dilate(mask: [Bool], width: Int, height: Int, radiusPx r: Int) -> [Bool] {
        guard r > 0 else { return mask }
        let halfWidths = ellipseHalfWidths(radiusPx: r)

        // Row-prefix sums of `mask`, computed once, so each (output row, dy)
        // pair's horizontal window-OR is an O(1) lookup instead of an
        // O(width) rescan -- this runs at ROAD_NEIGHBOURHOOD_PX=25 (a 51x51
        // kernel) over full-photo-resolution masks, so the naive
        // O(H*W*kernelArea) approach would be ~1000x slower for no benefit.
        var rowPrefix = [Int32](repeating: 0, count: height * (width + 1))
        for y in 0..<height {
            let base = y * (width + 1)
            var running: Int32 = 0
            for x in 0..<width {
                if mask[y * width + x] { running += 1 }
                rowPrefix[base + x + 1] = running
            }
        }

        var out = [Bool](repeating: false, count: width * height)
        for y in 0..<height {
            for dy in -r...r {
                let sy = y + dy
                guard sy >= 0, sy < height else { continue }
                let d = halfWidths[dy + r]
                let base = sy * (width + 1)
                let rowOut = y * width
                for x in 0..<width {
                    if out[rowOut + x] { continue }
                    let lo = max(0, x - d)
                    let hi = min(width - 1, x + d)
                    if rowPrefix[base + hi + 1] - rowPrefix[base + lo] > 0 {
                        out[rowOut + x] = true
                    }
                }
            }
        }
        return out
    }
}
