import simd

/// Port of `geometry.derive_ground_fields` (src/geometry.py) -- the dense
/// per-pixel fields OFRSNet's `MultimodalContextModule` consumes. See the
/// Python source's docstring for the full derivation (`G` = ray-plane
/// intersection, depth-independent; `h` = signed height above the plane of
/// the pixel's own 3D point).
///
/// Straight per-pixel loop, not a numpy-style vectorized formulation --
/// there is no dynamic-shape or Python-level control flow here to work
/// around (only the single `n` validity check at the top, ported directly),
/// so there is nothing to de-risk the way OFRSNet's export needed.
public enum GroundFields {
    public static func derive(depth: [Float], n: SIMD3<Double>?, K: simd_double3x3,
                              width: Int, height: Int,
                              maxGroundDistM: Double = 60.0)
    -> (G: [SIMD3<Double>], h: [Double], gvalid: [Bool]) {
        let count = width * height
        guard let n, n.x.isFinite, n.y.isFinite, n.z.isFinite, simd_length(n) >= 1e-9 else {
            return ([SIMD3<Double>](repeating: .zero, count: count),
                    [Double](repeating: 0, count: count),
                    [Bool](repeating: false, count: count))
        }
        let nNorm = simd_length(n)
        let kInv = K.inverse

        var G = [SIMD3<Double>](repeating: .zero, count: count)
        var gvalid = [Bool](repeating: false, count: count)
        for v in 0..<height {
            for u in 0..<width {
                let ray = kInv * SIMD3<Double>(Double(u), Double(v), 1.0)
                let denom = simd_dot(ray, n)
                let valid = denom > 1e-6
                gvalid[v * width + u] = valid
                guard valid else { continue }

                var g = ray * (1.0 / denom)
                let dist = simd_length(g)
                if dist > maxGroundDistM, dist > 1e-9 {
                    g *= maxGroundDistM / dist
                }
                G[v * width + u] = g
            }
        }

        let Q = Unproject.unproject(depth: depth, width: width, height: height, K: K)
        var h = [Double](repeating: 0, count: count)
        for i in 0..<count {
            let hv = (simd_dot(Q[i], n) - 1.0) / nNorm
            h[i] = hv.isFinite ? hv : 0
        }

        return (G, h, gvalid)
    }
}
