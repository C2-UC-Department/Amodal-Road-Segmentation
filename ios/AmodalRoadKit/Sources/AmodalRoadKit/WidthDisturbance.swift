/// Port of `bev.row_span_m`/`bev.row_count_m` and `disturbance.width_disturbance`.
///
/// Area alone conflates a small occluded patch on a wide road with a
/// full-width blockage on a narrow one -- only the second one actually
/// stops other traffic passing. A BEV row is a fixed-distance
/// cross-section of the road, so reducing `amodal_bev`/`occluded_bev`
/// row-by-row gives "what fraction of the road's WIDTH is blocked here"
/// for free, no new segmentation or geometry needed.
public enum WidthDisturbance {
    /// Port of `bev.row_span_m`. Per BEV row, the OUTER span (rightmost -
    /// leftmost `true` column + 1) in metres -- 0 for an empty row.
    /// Tolerant of a small internal gap (noise, a sliver misclassified as
    /// non-road), since the reconstructed road is expected to be one
    /// contiguous strip and its OUTER extent is the more honest width.
    /// Contrast `rowCountM`, which must NOT be gap-tolerant.
    public static func rowSpanM(mask: [Bool], grid: BevGrid) -> [Double] {
        precondition(mask.count == grid.width * grid.height)
        var out = [Double](repeating: 0, count: grid.height)
        for y in 0..<grid.height {
            var rightmost = -1
            var leftmost = grid.width
            let row = y * grid.width
            for x in 0..<grid.width where mask[row + x] {
                if x > rightmost { rightmost = x }
                if x < leftmost { leftmost = x }
            }
            out[y] = rightmost >= 0 ? Double(rightmost - leftmost + 1) / grid.ppm : 0
        }
        return out
    }

    /// Port of `bev.row_count_m`. Per BEV row, the actually-occupied width
    /// in metres -- a plain count. Unlike `rowSpanM`, a visible gap between
    /// two occluders' shadows within the same row is real, passable road,
    /// and must not be counted as blocked just because it sits between two
    /// blocked cells.
    public static func rowCountM(mask: [Bool], grid: BevGrid) -> [Double] {
        precondition(mask.count == grid.width * grid.height)
        var out = [Double](repeating: 0, count: grid.height)
        for y in 0..<grid.height {
            var count = 0
            let row = y * grid.width
            for x in 0..<grid.width where mask[row + x] { count += 1 }
            out[y] = Double(count) / grid.ppm
        }
        return out
    }

    public struct Measure {
        public let widthMaxPct: Double
        public let widthMeanPct: Double
        public let widthMaxM: Double
        public let widthRoadMAtMax: Double
        public let widthMaxAtZM: Double?
    }

    /// Port of `disturbance.width_disturbance`. At each along-road position
    /// (a BEV row), what fraction of the road's CROSS-SECTION is blocked --
    /// distinct from area on purpose (see the module doc). Rows narrower
    /// than `AttributionConfig.widthMinRoadSpanM` are excluded -- the BEV
    /// wedge narrows near the camera, so a very thin span there is
    /// measurement boundary noise, not a real usable road width.
    public static func measure(amodalBev: [Bool], occludedBev: [Bool], instBev: [Int32],
                               vehicles: [VehicleInstance], grid: BevGrid) -> (perVehicle: [Int32: Measure], total: Measure) {
        let roadSpan = rowSpanM(mask: amodalBev, grid: grid)
        let usable = roadSpan.map { $0 >= AttributionConfig.widthMinRoadSpanM }
        let zOfRow = (0..<grid.height).map { grid.zMax - (Double($0) + 0.5) / grid.ppm }

        func measureMask(_ mask: [Bool]) -> Measure {
            let blocked = rowCountM(mask: mask, grid: grid)
            let rows = (0..<grid.height).filter { usable[$0] && blocked[$0] > 0 }
            guard !rows.isEmpty else {
                return Measure(widthMaxPct: 0, widthMeanPct: 0, widthMaxM: 0, widthRoadMAtMax: 0, widthMaxAtZM: nil)
            }
            let frac = rows.map { blocked[$0] / roadSpan[$0] }
            // np.argmax returns the FIRST occurrence of the max value on ties.
            var bestIdx = 0
            for i in 1..<frac.count where frac[i] > frac[bestIdx] { bestIdx = i }
            let row = rows[bestIdx]
            let meanFrac = frac.reduce(0, +) / Double(frac.count)
            return Measure(widthMaxPct: pythonRound2(100.0 * frac[bestIdx]),
                          widthMeanPct: pythonRound2(100.0 * meanFrac),
                          widthMaxM: pythonRound3(blocked[row]),
                          widthRoadMAtMax: pythonRound3(roadSpan[row]),
                          widthMaxAtZM: pythonRound2(zOfRow[row]))
        }

        var perVehicle: [Int32: Measure] = [:]
        for v in vehicles {
            let mask = (0..<occludedBev.count).map { occludedBev[$0] && instBev[$0] == v.instId }
            perVehicle[v.instId] = measureMask(mask)
        }
        return (perVehicle, measureMask(occludedBev))
    }
}
