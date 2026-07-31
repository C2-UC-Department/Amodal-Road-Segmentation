import simd

/// Ground-resolution and area measurement. Port of the corresponding section
/// of src/bev.py (`ground_m2_per_pixel`, `area_m2`).
public enum Measurement {
    /// Ground area a single BEV cell's SOURCE image pixel covers, at BEV
    /// pixel `(u, v)`.
    ///
    /// Ground resolution decays with roughly the CUBE of range (see
    /// src/bev.py's docstring for the derivation), so a couple of pixels of
    /// mask disagreement near the horizon can be worth tens of square
    /// metres. For a homography the Jacobian determinant has a closed form,
    /// `det J = det(H) / (h3 . q)^3` at `q = [u, v, 1]`.
    public static func groundM2PerPixel(u: Double, v: Double, H: simd_double3x3, grid: BevGrid) -> Double {
        let denom = H[0][2] * u + H[1][2] * v + H[2][2]
        // NOTE: simd_double3x3 subscript is [column][row], so H[0][2] is
        // row 2, column 0 of the mathematical matrix -- i.e. H's THIRD ROW,
        // first two entries, matching Python's H[2,0]*u + H[2,1]*v + H[2,2].
        let detH = simd_determinant(H)
        let imgPxPerCell = abs(detH) / max(pow(abs(denom), 3), 1e-30)
        return grid.cellAreaM2 / max(imgPxPerCell, 1e-30)
    }

    /// Ground area of a BEV mask: every cell is the same size, so area is
    /// just a cell count times cell area.
    public static func areaM2(cellCount: Int, grid: BevGrid) -> Double {
        Double(cellCount) * grid.cellAreaM2
    }
}
