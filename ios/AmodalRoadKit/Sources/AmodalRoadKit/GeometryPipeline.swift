import simd

/// Wires depth -> RANSAC -> the dense geometry fields, on-device. Port of
/// `geometry.resolve_plane`'s COMPUTE path (src/geometry.py) -- not its
/// cache-load path, which has no iOS analogue: there is no precomputed-
/// geometry cache for a photo the user just took, only the Step-7-style
/// on-the-fly computation `resolve_plane` falls back to for images that
/// were never precomputed.
///
/// This is thin orchestration, not new logic -- `DepthModel`, `PlaneFit`,
/// and `GroundFields` are each already independently golden-value verified
/// against the real Python pipeline; what's asserted here is only that
/// composing them produces a valid, non-crashing result on a real model.
public enum GeometryPipeline {
    public struct Result {
        public let depth: [Float]
        public let n: SIMD3<Double>?
        public let pointsUsed: Int
        public let G: [SIMD3<Double>]
        public let h: [Double]
        public let gvalid: [Bool]

        public var valid: Bool { n != nil }
    }

    /// `roadMask`: the OFRS semantic map's road class, at `depthModel`'s own
    /// (fixed, traced) resolution -- callers resize/derive that mask
    /// themselves, same division of responsibility as `Calibration.calibrate`
    /// taking already-resolved K candidates rather than doing I/O itself.
    public static func resolveGeometry(depthModel: DepthModel, pixelValues: [Float],
                                       roadMask: [Bool], K: simd_double3x3,
                                       maxDepthM: Double = BevConfig.geomMaxDepthM,
                                       distThreshM: Double = BevConfig.geomRansacThreshM,
                                       iters: Int = BevConfig.geomRansacIters,
                                       maxGroundDistM: Double = BevConfig.geomMaxGroundDistM,
                                       seed: UInt64 = 0) throws -> Result {
        let depth = try depthModel.predict(pixelValues: pixelValues)
        let (n, pointsUsed) = PlaneFit.estimateGroundPlane(
            depth: depth, groundMask: roadMask, width: depthModel.width, height: depthModel.height,
            K: K, maxDepthM: maxDepthM, distThreshM: distThreshM, iters: iters, seed: seed)
        let (G, h, gvalid) = GroundFields.derive(
            depth: depth, n: n, K: K, width: depthModel.width, height: depthModel.height,
            maxGroundDistM: maxGroundDistM)
        return Result(depth: depth, n: n, pointsUsed: pointsUsed, G: G, h: h, gvalid: gvalid)
    }
}
