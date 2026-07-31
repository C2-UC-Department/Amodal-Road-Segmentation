// Auto-generated from src/calibration.py's K_from_focal / K_from_fov / scale_K,
// via:
//   .venv/bin/python -c "import src.calibration as c; ...K_from_focal(1000.0,640,480)... etc"
// (see CalibrationGoldenValueTests.swift for the exact calls). scale_K and
// K_from_fov are exercised via the same fixture; the EXIF cases assert the
// arithmetic in K_from_exif directly (f35 path and focal-plane path), not
// through a real image file -- see Intrinsics.kFromExifTags's doc comment
// for why that split exists.
extension GoldenData {
    static let calibKFromFocal: [Double] = [1000.0, 0.0, 320.0, 0.0, 1000.0, 240.0, 0.0, 0.0, 1.0]
    static let calibKFromFov: [Double] = [502.2993846775969, 0.0, 320.0, 0.0, 502.2993846775969, 240.0, 0.0, 0.0, 1.0]
    static let calibScaleK: [Double] = [500.0, 0.0, 160.0, 0.0, 750.0, 180.0, 0.0, 0.0, 1.0]
    static let calibExifF35Expected: [Double] = [462.22222222222223, 0.0, 320.0, 0.0, 462.22222222222223, 240.0, 0.0, 0.0, 1.0]
    static let calibExifFocalPlaneExpected: [Double] = [1338.5826771653544, 0.0, 320.0, 0.0, 1338.5826771653544, 240.0, 0.0, 0.0, 1.0]
}
