import simd

/// Port of `src/calibration.py`'s pure-math constructors and fallback chain.
///
/// Scope note (deliberately narrower than the Python source): the Python
/// chain is `kitti_calib -> geocalib -> exif -> fov_prior`. `kitti_calib`
/// only ever applies to the KITTI dataset (never a real user photo, so it
/// has no iOS analogue), and GeoCalib was investigated in Phase 0 and found
/// NO-GO for on-device use (see the migration plan, Section 3: trace
/// failures past the position-embedding-style fix, plus a 1.57s/image eager
/// CPU cost before any other model runs) -- both branches are dropped
/// rather than ported. The iOS chain instead leads with AVFoundation's
/// free per-shot camera-calibration data on live-captured photos (not
/// available to the Python CLI, which only ever sees files on disk), then
/// falls through EXIF -> fov_prior exactly as the Python module intends.
///
/// Every 3D quantity downstream -- the unprojected point cloud, the fitted
/// ground plane, `h`, `G` -- is computed through K, so getting this wrong
/// shears the whole reconstruction; see the Python module's docstring for
/// the full rationale, which applies unchanged here.
public enum CalibrationSource: String {
    case avfoundation
    case exif
    case fovPrior = "fov_prior"
}

public struct Calibration {
    public let K: simd_double3x3
    public let source: CalibrationSource

    public init(K: simd_double3x3, source: CalibrationSource) {
        self.K = K
        self.source = source
    }

    /// True when K came from a real calibration source rather than the
    /// blind `fov_prior` guess. Port of `Calibration.calibrated`.
    public var calibrated: Bool { source != .fovPrior }
}

public enum Intrinsics {
    /// Pinhole K with a principal point at the image centre. Port of
    /// `calibration.K_from_focal`. Delegates to `Homography.pinholeK`
    /// (same matrix, `cx = w/2, cy = h/2`) rather than re-deriving the
    /// column-major layout a second time.
    public static func kFromFocal(fPx: Double, w: Double, h: Double) -> simd_double3x3 {
        Homography.pinholeK(fx: fPx, fy: fPx, cx: w / 2.0, cy: h / 2.0)
    }

    /// Port of `calibration.K_from_fov`.
    public static func kFromFOV(w: Double, h: Double, hfovDeg: Double) -> simd_double3x3 {
        let fPx = (w / 2.0) / tan(hfovDeg * .pi / 180.0 / 2.0)
        return kFromFocal(fPx: fPx, w: w, h: h)
    }

    /// Port of `calibration.scale_K`: scales row 0 (fx, 0, cx) by `sx` and
    /// row 1 (0, fy, cy) by `sy`. `simd_double3x3` is column-major, so "row
    /// i" here means the `.x`/`.y` component of every column -- see
    /// `Homography.swift`'s header for the same convention note.
    public static func scaleK(_ K: simd_double3x3, sx: Double, sy: Double) -> simd_double3x3 {
        var m = K
        m[0].x *= sx; m[1].x *= sx; m[2].x *= sx
        m[0].y *= sy; m[1].y *= sy; m[2].y *= sy
        return m
    }

    /// Pure arithmetic half of `calibration.K_from_exif`, factored out from
    /// the EXIF tag *extraction* (see `EXIFIntrinsics.swift`) so the part
    /// that's actually golden-value-testable (this) is isolated from the
    /// part that isn't easily testable in a SwiftPM package without a real
    /// image fixture (ImageIO tag reading). Preferred source is
    /// `FocalLengthIn35mmFilm`; falls back to `FocalLength` +
    /// `FocalPlaneXResolution`. Returns `nil` exactly where the Python
    /// function would fall through to the next chain link.
    public static func kFromExifTags(focalLengthIn35mmFilm: Double?,
                                      focalLengthMM: Double?,
                                      focalPlaneXResolution: Double?,
                                      focalPlaneResolutionUnit: Int?,
                                      w: Double, h: Double) -> simd_double3x3? {
        if let f35 = focalLengthIn35mmFilm, f35 > 0 {
            return kFromFocal(fPx: f35 / 36.0 * max(w, h), w: w, h: h)
        }
        if let fMM = focalLengthMM, let fpRes = focalPlaneXResolution, fMM > 0, fpRes > 0 {
            let unit = focalPlaneResolutionUnit ?? 2   // 2 = inch, 3 = cm
            let perMM = fpRes / (unit == 2 ? 25.4 : 10.0)
            let sensorWidthMM = w / perMM
            if sensorWidthMM > 0 {
                return kFromFocal(fPx: fMM / sensorWidthMM * w, w: w, h: h)
            }
        }
        return nil
    }

    /// The fallback chain, ported from `calibration.calibrate` minus the
    /// KITTI/GeoCalib branches (see this file's header). Callers resolve
    /// `avFoundationK`/`exifK` themselves (camera-calibration-data lookup
    /// and EXIF tag extraction are both I/O, not pure math -- kept out of
    /// this package the same way `Warp.swift` keeps image loading out of
    /// itself) and pass already-resolved, already-scaled-to-(w,h) candidates
    /// in priority order.
    public static func calibrate(avFoundationK: simd_double3x3?,
                                  exifK: simd_double3x3?,
                                  w: Double, h: Double,
                                  fallbackHFOVDeg: Double = BevConfig.geomFallbackHFOVDeg) -> Calibration {
        if let k = avFoundationK { return Calibration(K: k, source: .avfoundation) }
        if let k = exifK { return Calibration(K: k, source: .exif) }
        return Calibration(K: kFromFOV(w: w, h: h, hfovDeg: fallbackHFOVDeg), source: .fovPrior)
    }
}
