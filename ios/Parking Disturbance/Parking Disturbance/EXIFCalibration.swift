//
//  EXIFCalibration.swift
//  Parking Disturbance
//
//  ImageIO-based EXIF tag *extraction* -- the app-side half AmodalRoadKit's
//  Calibration.swift deliberately left out (see that file's header: the
//  arithmetic is golden-value tested in the package, tag extraction from a
//  real image file needs a real fixture and ties to ImageIO/UIKit, so it
//  lives here instead). Mirrors src/calibration.py::K_from_exif's own tag
//  preference order exactly: FocalLengthIn35mmFilm first, then
//  FocalLength + FocalPlaneXResolution as a fallback -- see
//  Intrinsics.kFromExifTags (AmodalRoadKit) for the arithmetic itself.

import Foundation
import ImageIO
import simd
import AmodalRoadKit

enum EXIFCalibration {
    /// Reads the same four EXIF tags src/calibration.py::K_from_exif reads,
    /// then applies the identical, already golden-value-tested arithmetic.
    /// Returns nil wherever the Python function would also return nil (no
    /// usable tags present -- most photos that have been re-encoded/stripped
    /// of metadata, e.g. anything that has been through a messaging app).
    static func resolveK(imageData: Data, width: Int, height: Int) -> simd_double3x3? {
        guard let source = CGImageSourceCreateWithData(imageData as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let exif = properties[kCGImagePropertyExifDictionary] as? [CFString: Any] else {
            return nil
        }

        let f35 = (exif[kCGImagePropertyExifFocalLenIn35mmFilm] as? NSNumber)?.doubleValue
        let fMM = (exif[kCGImagePropertyExifFocalLength] as? NSNumber)?.doubleValue
        let fpRes = (exif[kCGImagePropertyExifFocalPlaneXResolution] as? NSNumber)?.doubleValue
        let fpUnit = (exif[kCGImagePropertyExifFocalPlaneResolutionUnit] as? NSNumber)?.intValue

        return Intrinsics.kFromExifTags(focalLengthIn35mmFilm: f35, focalLengthMM: fMM,
                                        focalPlaneXResolution: fpRes, focalPlaneResolutionUnit: fpUnit,
                                        w: Double(width), h: Double(height))
    }

    /// The fallback chain a picked/captured photo actually uses -- EXIF ->
    /// fov_prior (no AVFoundation calibration data for a photo picked from
    /// the library, only for a live capture session -- see
    /// Calibration.swift's header for why that's a separate, not-yet-built
    /// path).
    static func calibrate(imageData: Data, width: Int, height: Int) -> Calibration {
        let exifK = resolveK(imageData: imageData, width: width, height: height)
        return Intrinsics.calibrate(avFoundationK: nil, exifK: exifK, w: Double(width), h: Double(height))
    }
}
