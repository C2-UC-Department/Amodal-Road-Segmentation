import simd

/// Port of `instances.bottom_contour_points` (src/instances.py).
public enum BottomContour {
    /// Per image column, the bottom-most (max-row) `true` pixel of `mask`.
    /// Only this contour is actually near the ground plane -- everything
    /// above it (bonnet, roof) is not, and would smear if projected through
    /// a ground-plane homography (see `Homography.projectPointsToBev`).
    /// Points come out already ordered left-to-right by column, ready for
    /// `FootprintBand.rasterize`.
    ///
    /// Columns whose bottom pixel sits on the mask's last row (or the one
    /// above it) are excluded: the vehicle's true ground contact may extend
    /// below the visible frame, so that "contour" point is not trustworthy.
    /// `coverageFrac` is the fraction of spanned columns that were NOT
    /// excluded this way; low coverage means the footprint is likely
    /// incomplete.
    ///
    /// Returns `(points, coverageFrac)`; `points` may be empty for a
    /// degenerate mask.
    public static func points(mask: [Bool], width: Int, height: Int) -> (points: [SIMD2<Double>], coverageFrac: Double) {
        precondition(mask.count == width * height)
        var bottomRow = [Int](repeating: -1, count: width)
        for y in 0..<height {
            for x in 0..<width where mask[y * width + x] {
                bottomRow[x] = y
            }
        }
        let cols = (0..<width).filter { bottomRow[$0] >= 0 }
        guard !cols.isEmpty else { return ([], 0.0) }

        let truncated = cols.map { bottomRow[$0] >= height - 2 }
        let coverageFrac = 1.0 - Double(truncated.filter { $0 }.count) / Double(truncated.count)
        let usable = cols.enumerated().filter { !truncated[$0.offset] }.map { $0.element }
        guard !usable.isEmpty else { return ([], coverageFrac) }

        let pts = usable.map { SIMD2<Double>(Double($0), Double(bottomRow[$0])) }
        return (pts, coverageFrac)
    }
}
