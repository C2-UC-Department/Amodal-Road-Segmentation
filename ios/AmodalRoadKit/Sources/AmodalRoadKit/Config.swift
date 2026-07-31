/// Constants mirrored from config.py. Kept as a single small file (not
/// scattered inline) so it's obvious what to re-check if the Python side's
/// values ever change -- these two files should be diffed together, not
/// independently tuned.
public enum BevConfig {
    /// config.py: BEV_MIN_CAMERA_HEIGHT_M / BEV_MAX_CAMERA_HEIGHT_M --
    /// reject plainly broken plane fits rather than rasterising garbage.
    public static let minCameraHeightM = 0.3
    public static let maxCameraHeightM = 10.0

    /// config.py: GEOM_FALLBACK_HFOV_DEG -- the last-resort intrinsics guess
    /// when no calibration source is available. Documented in
    /// src/calibration.py as "a blind guess and measurably poor -- on our
    /// own footage it underestimates the focal length by 35-45%," i.e. use
    /// it only as a final fallback, not a default.
    public static let geomFallbackHFOVDeg = 65.0

    /// config.py: GEOM_MAX_DEPTH_M / GEOM_RANSAC_THRESH_M / GEOM_RANSAC_ITERS /
    /// GEOM_MAX_GROUND_DIST_M -- GeometryPipeline's defaults.
    public static let geomMaxDepthM = 30.0
    public static let geomRansacThreshM = 0.05
    public static let geomRansacIters = 500
    public static let geomMaxGroundDistM = 60.0
}
