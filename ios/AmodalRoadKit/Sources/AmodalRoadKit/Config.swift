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

    /// config.py: VEHICLE_ROOF_HEIGHT_PRIOR_M["car"] / SCALE_EST_* --
    /// `VehicleHeightEstimation`'s defaults. Python keys this prior by
    /// instance LABEL ("car" only, deliberately not extended to truck/bus/
    /// motorcycle -- see config.py's comment on why). The mobile vehicle-
    /// instance student (Phase 2) is a single collapsed "vehicle" class with
    /// no label breakdown, so this package applies the "car" prior
    /// uniformly to every detection -- a real, documented simplification:
    /// a detected truck or bus will be height-corrected as if it were a
    /// sedan, which is wrong for that vehicle specifically. Revisit if/when
    /// the mobile instance student gains per-class labels.
    public static let vehicleRoofHeightPriorM = 1.5
    public static let scaleEstMinScore: Float = 0.85
    public static let scaleEstMinPixels = 3000
    public static let scaleEstMinMaskRows = 20
    public static let scaleEstTopFrac = 0.08
    public static let scaleEstMinTopRows = 6
    public static let scaleEstMinRoofHeightM = 0.3

    /// config.py: BEV_PPM / BEV_RANGE_X_M / BEV_RANGE_Z_M / BEV_MAX_M2_PER_PIXEL
    /// -- the default measurement grid and resolution cap for
    /// `BevValidity.measurableMask`.
    public static let bevPpm = 20.0
    public static let bevRangeXM: (Double, Double) = (-10.0, 10.0)
    public static let bevRangeZM: (Double, Double) = (0.5, 40.0)
    public static let bevMaxM2PerPixel = 0.02

    /// A default `BevGrid` matching `bev.BevGrid.from_config()`'s defaults.
    public static func defaultBevGrid() -> BevGrid {
        BevGrid(ppm: bevPpm, xMin: bevRangeXM.0, xMax: bevRangeXM.1, zMin: bevRangeZM.0, zMax: bevRangeZM.1)
    }
}

/// config.py: `OFRS_CLASSES` (index == channel) / `ROAD_NEIGHBOURHOOD_PX` --
/// the class indices `predict.py::compose_amodal_mask` keys off of.
public enum OFRSClasses {
    public static let roadIdx = 0
    public static let personIdx = 8
    public static let vehicleIdx = 9
    public static let roadNeighbourhoodPx = 25
}

/// config.py: the "Ground-contact footprint attribution" / "Road-width
/// disturbance" constants `OccluderInstances`/`Attribution` key off of.
public enum AttributionConfig {
    public static let footprintBandWidthM = 0.4
    public static let footprintMaxAttributionDistM = 35.0
    public static let footprintMinCoverage = 0.5
    public static let instanceMinBlobPx = 200
    public static let widthMinRoadSpanM = 1.0
}
