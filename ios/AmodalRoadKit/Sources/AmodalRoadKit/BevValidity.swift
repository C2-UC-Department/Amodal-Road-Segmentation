import simd

/// Port of `bev.bev_validity` / `bev.measurable_mask` (src/bev.py).
public enum BevValidity {
    /// Which BEV cells are actually observed by the image. Python computes
    /// this by warping a solid (all-255) image through `H` with nearest
    /// sampling and thresholding `> 127` -- which reduces exactly to "does
    /// this BEV cell's nearest-rounded source coordinate fall inside the
    /// image", so that's what this computes directly rather than
    /// allocating and warping a fake solid array. Same nearest-sampling
    /// convention (`pythonRound`) as `Warp.warpToBev`, so this can never
    /// disagree with it -- mirroring the Python docstring's own claim.
    public static func bevValidity(H: simd_double3x3, grid: BevGrid,
                                   imageWidth: Int, imageHeight: Int) -> [Bool] {
        var validity = [Bool](repeating: false, count: grid.width * grid.height)
        for v in 0..<grid.height {
            for u in 0..<grid.width {
                let hom = H * SIMD3<Double>(Double(u), Double(v), 1.0)
                guard hom.z != 0 else { continue }
                let xi = Int(pythonRound(hom.x / hom.z))
                let yi = Int(pythonRound(hom.y / hom.z))
                validity[v * grid.width + u] = xi >= 0 && xi < imageWidth && yi >= 0 && yi < imageHeight
            }
        }
        return validity
    }

    public struct MeasurableInfo {
        public let maxM2PerPx: Double
        public let measurableRangeM: Double
        public let inFramePct: Double
        public let measurablePct: Double
    }

    /// BEV cells that are both observed AND resolved well enough to
    /// measure (ground resolution decays with roughly the cube of range --
    /// see `Measurement.groundM2PerPixel`'s docstring -- so without this
    /// cap, near-horizon boundary disagreement would dominate any area
    /// figure). Returns the mask plus `MeasurableInfo` so callers can
    /// report what was actually measured rather than implying the whole
    /// grid was.
    public static func measurableMask(H: simd_double3x3, grid: BevGrid,
                                      imageWidth: Int, imageHeight: Int,
                                      maxM2PerPx: Double) -> (mask: [Bool], info: MeasurableInfo) {
        let inFrame = bevValidity(H: H, grid: grid, imageWidth: imageWidth, imageHeight: imageHeight)
        var mask = [Bool](repeating: false, count: grid.width * grid.height)
        var minValidRow: Int?
        for v in 0..<grid.height {
            for u in 0..<grid.width {
                let idx = v * grid.width + u
                let resolved = Measurement.groundM2PerPixel(u: Double(u), v: Double(v), H: H, grid: grid) <= maxM2PerPx
                mask[idx] = inFrame[idx] && resolved
                if mask[idx], minValidRow == nil { minValidRow = v }
                else if mask[idx], let cur = minValidRow, v < cur { minValidRow = v }
            }
        }
        // Row 0 is the far edge, so the smallest surviving row index is the range cap.
        let zMaxUsed = minValidRow.map { grid.zMax - (Double($0) + 0.5) / grid.ppm } ?? 0.0

        let inFrameCount = inFrame.reduce(0) { $0 + ($1 ? 1 : 0) }
        let maskCount = mask.reduce(0) { $0 + ($1 ? 1 : 0) }
        let total = Double(grid.width * grid.height)
        let info = MeasurableInfo(
            maxM2PerPx: maxM2PerPx,
            measurableRangeM: pythonRound2(zMaxUsed),
            inFramePct: pythonRound2(100.0 * Double(inFrameCount) / total),
            measurablePct: pythonRound2(100.0 * Double(maskCount) / total))
        return (mask, info)
    }
}
