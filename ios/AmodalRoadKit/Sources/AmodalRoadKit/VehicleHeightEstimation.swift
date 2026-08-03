import Foundation

/// Port of `src/instances.py`'s `_roofline_height_m` / `estimate_camera_height_from_vehicles`.
///
/// Every 3D quantity in this pipeline is scaled by whatever bias the
/// monocular depth model has on a given photo -- the implied camera height
/// from the raw RANSAC plane fit is typically off by 50-400% (config.py).
/// A detected vehicle's ROOF height above the fitted plane, compared
/// against a fixed real-world prior (an actual car's roof height), gives a
/// correction factor `k` that fixes the whole metric world -- because both
/// the roofline measurement and the plane's implied height came from the
/// SAME biased depth, they share the same bias.
public struct VehicleHeightSample {
    public let label: String
    public let score: Float
    public let pixelArea: Int
    public let roofHeightBiasedM: Double
    public let k: Double

    public init(label: String, score: Float, pixelArea: Int, roofHeightBiasedM: Double, k: Double) {
        self.label = label
        self.score = score
        self.pixelArea = pixelArea
        self.roofHeightBiasedM = roofHeightBiasedM
        self.k = k
    }
}

public struct ScaleEstimate {
    public let cameraHeightM: Double
    public let nSamples: Int
    public let kMedian: Double
    public let kSpread: Double
    public let perVehicle: [VehicleHeightSample]
}

/// One detected vehicle, as `estimateCameraHeight` needs it: a mask already
/// resized/mapped onto the SAME grid as `hField` (row-major, `width x
/// height`), plus the detection's own score and (collapsed, single-class --
/// see `BevConfig.vehicleRoofHeightPriorM`'s doc comment) label.
public struct VehicleForHeightEstimation {
    public let mask: [Bool]
    public let score: Float
    public let label: String

    public init(mask: [Bool], score: Float, label: String = "car") {
        self.mask = mask
        self.score = score
        self.label = label
    }
}

public enum VehicleHeightEstimation {
    /// Median height-above-plane over a vehicle mask's own top slice (its
    /// roofline) -- NOT the whole body, which mixes hood/roof pixels and is
    /// unreliable for a close or partially-framed vehicle. `ok` is false
    /// when the mask touches row 0/1 (the roofline may be cut off by the
    /// frame, which would silently bias the reading low) or too few pixels
    /// qualify.
    static func roofHeightM(hField: [Double], mask: [Bool], width: Int, height: Int) -> (height: Double?, ok: Bool) {
        var v0 = Int.max, v1 = -1
        for v in 0..<height {
            for u in 0..<width where mask[v * width + u] {
                v0 = min(v0, v); v1 = max(v1, v)
            }
        }
        guard v1 >= 0 else { return (nil, false) }
        guard v0 > 1 else { return (nil, false) }   // roofline possibly truncated by the frame edge

        let span = max(1, v1 - v0)
        let cutoff = v0 + max(BevConfig.scaleEstMinTopRows,
                             Int((Double(span) * BevConfig.scaleEstTopFrac).rounded()))

        var topValues: [Double] = []
        for v in v0...min(cutoff, height - 1) {
            for u in 0..<width where mask[v * width + u] { topValues.append(hField[v * width + u]) }
        }
        guard topValues.count >= 3 else { return (nil, false) }
        // h is negative above the plane (see GroundFields.derive); height
        // above ground is therefore -h.
        return (-median(topValues), true)
    }

    /// Returns nil when no detection meets the quality bar (`BevConfig
    /// .scaleEst*`) -- callers fall back to the uncorrected plane in that
    /// case. `nSamples == 1` is a real estimate, just lower-confidence than
    /// `>= 2` (no cross-vehicle agreement to report).
    public static func estimateCameraHeight(hField: [Double], vehicles: [VehicleForHeightEstimation],
                                            impliedBiasedHeightM: Double, width: Int, height: Int) -> ScaleEstimate? {
        var samples: [VehicleHeightSample] = []
        var ks: [Double] = []

        for vehicle in vehicles {
            guard vehicle.score >= BevConfig.scaleEstMinScore else { continue }
            let pixelArea = vehicle.mask.reduce(0) { $0 + ($1 ? 1 : 0) }
            guard pixelArea >= BevConfig.scaleEstMinPixels else { continue }

            var v0 = Int.max, v1 = -1
            for v in 0..<height {
                for u in 0..<width where vehicle.mask[v * width + u] { v0 = min(v0, v); v1 = max(v1, v) }
            }
            guard v1 >= 0, (v1 - v0) >= BevConfig.scaleEstMinMaskRows else { continue }

            let (roofBiased, ok) = roofHeightM(hField: hField, mask: vehicle.mask, width: width, height: height)
            guard ok, let roofBiased, roofBiased >= BevConfig.scaleEstMinRoofHeightM else { continue }

            let k = BevConfig.vehicleRoofHeightPriorM / roofBiased
            ks.append(k)
            samples.append(VehicleHeightSample(label: vehicle.label, score: vehicle.score, pixelArea: pixelArea,
                                              roofHeightBiasedM: roofBiased, k: k))
        }
        guard !ks.isEmpty else { return nil }

        let kMed = median(ks)
        let spread: Double
        if ks.count > 1, kMed > 0 {
            spread = (percentile(ks, 75) - percentile(ks, 25)) / kMed
        } else {
            spread = 0.0
        }
        return ScaleEstimate(cameraHeightM: impliedBiasedHeightM * kMed, nSamples: ks.count,
                            kMedian: kMed, kSpread: spread, perVehicle: samples)
    }
}

/// numpy-compatible median (average the two middle elements on an even count).
func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    let n = sorted.count
    guard n > 0 else { return .nan }
    if n % 2 == 1 { return sorted[n / 2] }
    return (sorted[n / 2 - 1] + sorted[n / 2]) / 2
}

/// numpy-compatible `percentile` with its default ("linear") interpolation:
/// the rank `p/100 * (n-1)` is generally fractional, linearly interpolated
/// between its two neighbouring sorted values.
func percentile(_ values: [Double], _ p: Double) -> Double {
    let sorted = values.sorted()
    let n = sorted.count
    guard n > 0 else { return .nan }
    guard n > 1 else { return sorted[0] }
    let rank = p / 100.0 * Double(n - 1)
    let lo = Int(floor(rank)), hi = Int(ceil(rank))
    if lo == hi { return sorted[lo] }
    let frac = rank - Double(lo)
    return sorted[lo] * (1 - frac) + sorted[hi] * frac
}
