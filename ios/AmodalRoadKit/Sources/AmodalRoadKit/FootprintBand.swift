import Foundation
import simd

/// Port of `bev.rasterize_footprint_band` (src/bev.py). Draws a vehicle's
/// projected ground-contact contour as a thin BEV band -- the attribution
/// seed region for `nearest_label_bev`/Voronoi propagation.
///
/// Unlike `Warp.swift`, this does NOT need to match `cv2.polylines`/`cv2.circle`
/// pixel-for-pixel: the Python source's own comment on this function says so
/// explicitly ("pixel-perfect line-width parity isn't critical -- it only
/// serves as a Voronoi seed"), and the migration plan repeats that framing.
/// So this is a straightforward, independently-correct rasterizer (thick
/// line = fill every cell within `bandWidth/2` of the segment; a single
/// point = a filled disk), not a port requiring golden-value parity against
/// OpenCV's specific antialiasing/thickness algorithm.
public enum FootprintBand {
    /// `points` are BEV pixel coordinates, already ordered (as
    /// `bottom_contour_points` produces them: left-to-right by image
    /// column). Returns a row-major `[Bool]` of shape `(grid.height, grid.width)`.
    public static func rasterize(points: [SIMD2<Double>], grid: BevGrid,
                                 bandWidthM: Double) -> [Bool] {
        var canvas = [Bool](repeating: false, count: grid.width * grid.height)
        guard !points.isEmpty else { return canvas }

        let bandPx = max(1, Int(pythonRound(bandWidthM * grid.ppm)))
        let radius = Double(max(1, bandPx / 2))

        if points.count == 1 {
            fillDisk(&canvas, grid: grid, center: points[0], radius: radius)
            return canvas
        }
        for i in 0..<(points.count - 1) {
            fillSegment(&canvas, grid: grid, p0: points[i], p1: points[i + 1], radius: radius)
        }
        return canvas
    }

    private static func fillDisk(_ canvas: inout [Bool], grid: BevGrid,
                                 center: SIMD2<Double>, radius: Double) {
        let x0 = max(0, Int(floor(center.x - radius)))
        let x1 = min(grid.width - 1, Int(ceil(center.x + radius)))
        let y0 = max(0, Int(floor(center.y - radius)))
        let y1 = min(grid.height - 1, Int(ceil(center.y + radius)))
        guard x0 <= x1, y0 <= y1 else { return }
        let r2 = radius * radius
        for y in y0...y1 {
            for x in x0...x1 {
                let dx = Double(x) - center.x, dy = Double(y) - center.y
                if dx * dx + dy * dy <= r2 {
                    canvas[y * grid.width + x] = true
                }
            }
        }
    }

    private static func fillSegment(_ canvas: inout [Bool], grid: BevGrid,
                                    p0: SIMD2<Double>, p1: SIMD2<Double>, radius: Double) {
        let x0 = max(0, Int(floor(min(p0.x, p1.x) - radius)))
        let x1 = min(grid.width - 1, Int(ceil(max(p0.x, p1.x) + radius)))
        let y0 = max(0, Int(floor(min(p0.y, p1.y) - radius)))
        let y1 = min(grid.height - 1, Int(ceil(max(p0.y, p1.y) + radius)))
        guard x0 <= x1, y0 <= y1 else { return }

        let d = p1 - p0
        let lenSq = simd_dot(d, d)
        let r2 = radius * radius

        for y in y0...y1 {
            for x in x0...x1 {
                let p = SIMD2<Double>(Double(x), Double(y))
                let distSq: Double
                if lenSq < 1e-12 {
                    let diff = p - p0
                    distSq = simd_dot(diff, diff)
                } else {
                    // Project p onto the segment, clamped to [0, 1].
                    let t = max(0, min(1, simd_dot(p - p0, d) / lenSq))
                    let closest = p0 + d * t
                    let diff = p - closest
                    distSq = simd_dot(diff, diff)
                }
                if distSq <= r2 {
                    canvas[y * grid.width + x] = true
                }
            }
        }
    }
}
