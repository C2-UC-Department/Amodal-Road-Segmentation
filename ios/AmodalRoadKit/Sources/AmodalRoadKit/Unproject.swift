import simd

/// Port of `geometry.unproject` / `geometry.unproject_ground_points`
/// (src/geometry.py) -- pinhole unprojection, Eq. 1-2 in the paper this
/// module follows (see geometry.py's header). Pure per-pixel math, no
/// Python-level control flow beyond the depth-value clamp, so this ports
/// directly with no dynamic-shape or RNG concerns (unlike PlaneFit.swift's
/// RANSAC half).
public enum Unproject {
    /// Lift every pixel to 3D camera coordinates. `depth` is row-major
    /// `(height, width)`. Returns row-major `(height, width)` of `(x, y, z)`.
    ///
    /// `K` here is a `simd_double3x3` built the same way `Homography.pinholeK`
    /// builds it -- `fx = K[0][0]`, `fy = K[1][1]`, `cx = K[2][0]`, `cy = K[2][1]`
    /// (see that function's column layout).
    public static func unproject(depth: [Float], width: Int, height: Int,
                                 K: simd_double3x3) -> [SIMD3<Double>] {
        let fx = K[0][0], fy = K[1][1], cx = K[2][0], cy = K[2][1]
        var out = [SIMD3<Double>](repeating: .zero, count: width * height)
        for v in 0..<height {
            for u in 0..<width {
                var z = Double(depth[v * width + u])
                if !z.isFinite { z = 0 }
                z = max(0.0, min(z, 1e4))
                let x = (Double(u) - cx) * z / fx
                let y = (Double(v) - cy) * z / fy
                out[v * width + u] = SIMD3(x, y, z)
            }
        }
        return out
    }

    /// Ground-masked 3D point cloud. `groundMask` is row-major `(height, width)`
    /// booleans. Port of `unproject_ground_points`, including its depth
    /// sanity filter (`0.1 < z < maxDepth`).
    public static func unprojectGroundPoints(depth: [Float], groundMask: [Bool],
                                             width: Int, height: Int, K: simd_double3x3,
                                             maxDepthM: Double) -> [SIMD3<Double>] {
        let fx = K[0][0], fy = K[1][1], cx = K[2][0], cy = K[2][1]
        var out: [SIMD3<Double>] = []
        out.reserveCapacity(groundMask.filter { $0 }.count)
        for v in 0..<height {
            for u in 0..<width {
                guard groundMask[v * width + u] else { continue }
                let z = Double(depth[v * width + u])
                guard z.isFinite, z > 0.1, z < maxDepthM else { continue }
                let x = (Double(u) - cx) * z / fx
                let y = (Double(v) - cy) * z / fy
                let p = SIMD3<Double>(x, y, z)
                if p.x.isFinite, p.y.isFinite, p.z.isFinite {
                    out.append(p)
                }
            }
        }
        return out
    }
}
