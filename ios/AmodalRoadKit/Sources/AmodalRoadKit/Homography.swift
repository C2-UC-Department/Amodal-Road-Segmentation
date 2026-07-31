import simd

/// Ground-plane homography: image <-> BEV, direct port of the matrix
/// construction in src/bev.py (`_S`, `_M`, `homography_bev_to_image`) and
/// `project_points_to_bev`.
///
/// IMPORTANT convention note for anyone editing this file: `simd_double3x3`
/// is COLUMN-MAJOR -- its initializer takes three SIMD3<Double> COLUMNS, not
/// rows. Every matrix below is built by writing out the mathematical matrix
/// as rows (matching the Python/numpy source directly, so the two stay easy
/// to diff against each other) and then transposing by hand into the
/// column-major initializer. Get this backwards and every downstream
/// computation is silently wrong in a way that can look almost-plausible
/// (a homography built from a transposed sub-block is still often invertible
/// and still often produces IN-FRAME-looking output) -- this is exactly the
/// kind of bug the golden-value tests in AmodalRoadKitTests exist to catch:
/// they assert against concrete numbers computed by the real src/bev.py, not
/// just internal Swift self-consistency.
public enum Homography {
    /// Pinhole intrinsics matrix [[fx,0,cx],[0,fy,cy],[0,0,1]].
    public static func pinholeK(fx: Double, fy: Double, cx: Double, cy: Double) -> simd_double3x3 {
        // Row0=(fx,0,cx) Row1=(0,fy,cy) Row2=(0,0,1) -> columns:
        simd_double3x3(
            SIMD3(fx, 0, 0),
            SIMD3(0, fy, 0),
            SIMD3(cx, cy, 1)
        )
    }

    /// BEV pixel [u, v, 1] -> ground [X, Z, 1], metres. Port of `bev._S`.
    /// Pixel (u, v) samples the CENTRE of its cell (the +-0.5 offsets), same
    /// as the Python source -- without them a mask's area is biased by half
    /// a cell along each axis.
    static func sMatrix(grid: BevGrid) -> simd_double3x3 {
        let inv = 1.0 / grid.ppm
        // Row0=(inv,0,xMin+0.5*inv) Row1=(0,-inv,zMax-0.5*inv) Row2=(0,0,1)
        return simd_double3x3(
            SIMD3(inv, 0, 0),
            SIMD3(0, -inv, 0),
            SIMD3(grid.xMin + 0.5 * inv, grid.zMax - 0.5 * inv, 1)
        )
    }

    /// Ground [X, Z, 1] -> camera-frame [Xc, Yc, Zc], via Y on the plane
    /// `n . X = 1`. Port of `bev._M`.
    static func mMatrix(n: SIMD3<Double>) -> simd_double3x3 {
        let nx = n.x, ny = n.y, nz = n.z
        // Row0=(1,0,0) Row1=(-nx/ny,-nz/ny,1/ny) Row2=(0,1,0)
        return simd_double3x3(
            SIMD3(1, -nx / ny, 0),
            SIMD3(0, -nz / ny, 1),
            SIMD3(0, 1 / ny, 0)
        )
    }

    /// 3x3 mapping a BEV pixel to the image pixel that observes that ground
    /// cell: `H = K . M(n) . S`. Port of `bev.homography_bev_to_image`.
    ///
    /// Throws `AmodalRoadKitError.unusablePlane` for a degenerate `n`,
    /// matching Python's explicit `raise ValueError` there (see
    /// `PlaneGeometry.planeIsUsable`) rather than silently producing NaN/inf
    /// garbage -- `bev.py`'s own test suite specifically asserts this raises
    /// (`test_homography_raises_on_unusable_plane`), so a silent Swift
    /// equivalent would be a real behavioural gap, not just a style choice.
    public static func bevToImage(K: simd_double3x3, n: SIMD3<Double>, grid: BevGrid) throws -> simd_double3x3 {
        guard PlaneGeometry.planeIsUsable(n) else {
            throw AmodalRoadKitError.unusablePlane(n: n)
        }
        return K * mMatrix(n: n) * sMatrix(grid: grid)
    }

    /// `(N, 2)` image `(u, v)` -> `(N, 2)` continuous BEV pixel coordinates.
    /// The point-wise inverse of `bevToImage`: H maps BEV -> image, so points
    /// go through H^-1. Port of `bev.project_points_to_bev`.
    public static func projectPointsToBev(_ points: [SIMD2<Double>], H: simd_double3x3) -> [SIMD2<Double>] {
        let hInv = H.inverse
        return points.map { p in
            let hom = hInv * SIMD3(p.x, p.y, 1)
            return SIMD2(hom.x / hom.z, hom.y / hom.z)
        }
    }
}
