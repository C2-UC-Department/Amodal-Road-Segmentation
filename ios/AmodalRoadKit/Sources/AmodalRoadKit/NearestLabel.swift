import Foundation

/// Port of `bev.nearest_label_bev` (src/bev.py) -- labels every BEV cell
/// with its nearest seed's id (a Voronoi diagram of the seed cells),
/// optionally cut off beyond `maxDistM`.
///
/// Python computes this via `cv2.distanceTransformWithLabels`. This
/// implementation is a plain brute-force "nearest seed" search instead:
/// for each grid cell, scan every seed cell and keep the closest. That's
/// `O(cells * seeds)`, correct by construction (no distance-transform
/// approximation to get subtly wrong), and fine for Phase 1's
/// correctness-first goal -- the migration plan explicitly calls for
/// optimizing (a proper Jump-Flooding-Algorithm or multi-source BFS pass)
/// "only if profiling requires it," which needs a working, verified baseline
/// to profile against in the first place. Verified against real
/// `cv2.distanceTransformWithLabels` output in NearestLabelGoldenValueTests.swift.
public enum NearestLabel {
    /// `seedIds` is a row-major `[Int32]` of shape `(grid.height, grid.width)`,
    /// `0` meaning "no seed". Returns a same-shape array where every cell
    /// holds its nearest seed's id, or `0` if farther than `maxDistM` (in
    /// grid pixels, i.e. `maxDistM * grid.ppm`) from every seed.
    public static func label(seedIds: [Int32], grid: BevGrid, maxDistM: Double? = nil) -> [Int32] {
        let w = grid.width, h = grid.height
        precondition(seedIds.count == w * h, "seedIds must be grid.width * grid.height")

        var seedX: [Double] = [], seedY: [Double] = [], seedLabel: [Int32] = []
        for y in 0..<h {
            for x in 0..<w {
                let id = seedIds[y * w + x]
                if id != 0 {
                    seedX.append(Double(x)); seedY.append(Double(y)); seedLabel.append(id)
                }
            }
        }
        guard !seedLabel.isEmpty else { return [Int32](repeating: 0, count: w * h) }

        let maxDistSq = maxDistM.map { pow($0 * grid.ppm, 2) } ?? .infinity
        var out = [Int32](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                var bestDistSq = Double.infinity
                var bestLabel: Int32 = 0
                let fx = Double(x), fy = Double(y)
                for i in 0..<seedLabel.count {
                    let dx = fx - seedX[i], dy = fy - seedY[i]
                    let d2 = dx * dx + dy * dy
                    if d2 < bestDistSq {
                        bestDistSq = d2
                        bestLabel = seedLabel[i]
                    }
                }
                out[y * w + x] = bestDistSq <= maxDistSq ? bestLabel : 0
            }
        }
        return out
    }
}
