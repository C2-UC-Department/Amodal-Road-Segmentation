import simd

/// A small seedable PRNG (SplitMix64) for RANSAC's random sampling.
/// `RandomNumberGenerator` conformance is needed for `Int.random(in:using:)`.
/// Deliberately NOT an attempt to reproduce NumPy's PCG64 bit-for-bit --
/// see `PlaneFit.fitPlaneRANSAC`'s doc comment for why that's neither
/// achievable nor useful here.
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Port of `geometry.fit_plane_ls` / `geometry.fit_plane_ransac` (src/geometry.py).
public enum PlaneFit {
    /// Solve `C^T C n = C^T 1` in the least-squares sense (Eq. 4-5 in the
    /// paper geometry.py follows). `n` is deliberately NOT normalized --
    /// `||n||` is `1/distance to the plane`, needed downstream. Deterministic;
    /// verified against real `geometry.fit_plane_ls` output
    /// (PlaneFitGoldenValueTests.swift), unlike the RANSAC half below.
    public static func fitPlaneLS(_ points: [SIMD3<Double>]) -> SIMD3<Double>? {
        guard points.count >= 3 else { return nil }

        var gram = [[Double]](repeating: [0, 0, 0], count: 3)
        var rhs = [Double](repeating: 0, count: 3)
        for p in points {
            let v = [p.x, p.y, p.z]
            for i in 0..<3 {
                rhs[i] += v[i]
                for j in 0..<3 { gram[i][j] += v[i] * v[j] }
            }
        }
        // Symmetric, so row/column order doesn't matter, but built the same
        // "rows-as-written, hand-transposed into columns" way as Homography.swift
        // for consistency.
        let G = simd_double3x3(
            SIMD3(gram[0][0], gram[1][0], gram[2][0]),
            SIMD3(gram[0][1], gram[1][1], gram[2][1]),
            SIMD3(gram[0][2], gram[1][2], gram[2][2])
        )
        guard abs(simd_determinant(G)) > 1e-18 else { return nil }   // singular system
        let n = G.inverse * SIMD3(rhs[0], rhs[1], rhs[2])
        guard n.x.isFinite, n.y.isFinite, n.z.isFinite, simd_length(n) > 1e-9 else { return nil }
        return n
    }

    /// RANSAC plane fit, then least-squares refit on inliers.
    ///
    /// Uses `SplitMix64`, NOT NumPy's PCG64 -- exact RNG-seed parity with
    /// Python is neither achievable (different algorithm entirely) nor
    /// useful (RANSAC's whole point is that many different random samples
    /// converge on the same answer; matching Python's specific draws would
    /// only prove this implementation copies numbers, not that it fits
    /// planes correctly). Validated instead by fitting a KNOWN synthetic
    /// noisy-planar point cloud and checking the recovered normal's
    /// direction against the true generating plane
    /// (PlaneFitGoldenValueTests.swift's `testRansacRecoversKnownPlane`) --
    /// this is the same fixture and same acceptance bar
    /// (cosine similarity, not bit-exact match) used to sanity-check the
    /// Python implementation when this Swift port's golden values were
    /// generated.
    public static func fitPlaneRANSAC(_ points: [SIMD3<Double>],
                                      distThreshM: Double = 0.05,
                                      iters: Int = 500,
                                      seed: UInt64 = 0) -> SIMD3<Double>? {
        let pts = points.filter { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }
        guard pts.count >= 3 else { return nil }

        var rng = SplitMix64(seed: seed)
        let n = pts.count
        var bestInliers: [Bool]?
        var bestCount = -1

        for _ in 0..<iters {
            var idxSet = Set<Int>()
            while idxSet.count < 3 { idxSet.insert(Int.random(in: 0..<n, using: &rng)) }
            let idx = Array(idxSet)
            let p0 = pts[idx[0]], p1 = pts[idx[1]], p2 = pts[idx[2]]

            let normalRaw = simd_cross(p1 - p0, p2 - p0)
            let nn = simd_length(normalRaw)
            guard nn.isFinite, nn >= 1e-9 else { continue }
            let normal = normalRaw / nn

            var inliers = [Bool](repeating: false, count: n)
            var count = 0
            for i in 0..<n {
                let d = abs(simd_dot(pts[i] - p0, normal))
                if d.isFinite, d < distThreshM {
                    inliers[i] = true
                    count += 1
                }
            }
            if count > bestCount {
                bestCount = count
                bestInliers = inliers
            }
        }

        guard let inliers = bestInliers, bestCount >= 3 else {
            return fitPlaneLS(pts)
        }
        var inlierPoints: [SIMD3<Double>] = []
        inlierPoints.reserveCapacity(bestCount)
        for i in 0..<n where inliers[i] { inlierPoints.append(pts[i]) }
        return fitPlaneLS(inlierPoints)
    }

    /// Port of `geometry.estimate_ground_plane`: unproject the ground-masked
    /// pixels, then RANSAC a plane. Returns `(n, pointsUsed)`, `n == nil` on
    /// failure -- thin orchestration, no new logic of its own to verify
    /// beyond what `Unproject`/`fitPlaneRANSAC` already cover.
    public static func estimateGroundPlane(depth: [Float], groundMask: [Bool],
                                           width: Int, height: Int, K: simd_double3x3,
                                           maxDepthM: Double, distThreshM: Double = 0.05,
                                           iters: Int = 500, seed: UInt64 = 0)
    -> (n: SIMD3<Double>?, pointsUsed: Int) {
        let pts = Unproject.unprojectGroundPoints(depth: depth, groundMask: groundMask,
                                                   width: width, height: height, K: K,
                                                   maxDepthM: maxDepthM)
        guard pts.count >= 3 else { return (nil, 0) }
        return (fitPlaneRANSAC(pts, distThreshM: distThreshM, iters: iters, seed: seed), pts.count)
    }
}
