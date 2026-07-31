import simd

/// Plane sanity & metric scale. Direct port of the corresponding section of
/// src/bev.py (`implied_camera_height`, `plane_is_usable`,
/// `rescale_plane_to_height`).
public enum PlaneGeometry {
    /// Distance from the camera centre to the plane, in metres.
    ///
    /// With `n . X = 1`, the origin (camera centre) is at signed distance
    /// `-1/||n||` from the plane, so the plane sits `1/||n||` away. For a
    /// ground plane that IS the camera height.
    public static func impliedCameraHeight(_ n: SIMD3<Double>) -> Double {
        let norm = simd_length(n)
        guard norm >= 1e-9 else { return .infinity }
        return 1.0 / norm
    }

    /// Reject plane fits that cannot describe a ground plane below the
    /// camera. `n.y > 0` is a real constraint, not just a divide-by-zero
    /// guard: the plane's nearest point to the camera is `n/||n||^2`, so for
    /// the plane to be BELOW the camera (y is DOWN, matching OpenCV camera
    /// convention) its normal must have a positive y component.
    public static func planeIsUsable(_ n: SIMD3<Double>?) -> Bool {
        guard let n else { return false }
        guard n.x.isFinite, n.y.isFinite, n.z.isFinite else { return false }
        guard simd_length(n) >= 1e-9, n.y > 1e-9 else { return false }
        let h = impliedCameraHeight(n)
        return h >= BevConfig.minCameraHeightM && h <= BevConfig.maxCameraHeightM
    }

    /// Rescale the metric world so the camera sits at a known height.
    ///
    /// Monocular metric depth carries a scale error; because the plane fit
    /// exposes the camera height directly as `1/||n||`, a uniform rescale of
    /// `n` corrects the whole metric world. Returns `(scaledN, impliedHeightBefore)`.
    public static func rescalePlaneToHeight(_ n: SIMD3<Double>, cameraHeightM: Double) -> (SIMD3<Double>, Double) {
        let implied = impliedCameraHeight(n)
        guard implied.isFinite, cameraHeightM > 0 else { return (n, implied) }
        return (n * (implied / cameraHeightM), implied)
    }
}
