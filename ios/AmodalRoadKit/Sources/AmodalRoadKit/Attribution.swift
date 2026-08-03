import simd

/// Port of `disturbance.py`'s "Ground-contact footprint attribution" section
/// (`_ground_contact_seed_ids`, `attribute`).
///
/// `Warp.warpToBev`/the homography assume every pixel warped lies ON the
/// plane -- true for road pixels, false for a vehicle's roof/hood/windshield
/// (up to ~1.5 m above it). Warping those elevated pixels finds where their
/// camera ray would meet the ground PAST the vehicle, smearing its claimed
/// BEV region away from the camera relative to where it actually sits. Only
/// a vehicle's BOTTOM CONTOUR (`BottomContour`) is near the plane and
/// projects reliably, so attribution seeds come from that contour alone:
/// projected to BEV (`Homography.projectPointsToBev`), rasterised into a
/// small band (`FootprintBand.rasterize`), then propagated to the rest of
/// the shadow by nearest seed (`NearestLabel.label`).
public enum Attribution {
    /// Port of `disturbance._ground_contact_seed_ids`. Builds the
    /// attribution seed image: each vehicle's ground-contact band, stamped
    /// with its id. Vehicles are processed smallest-first, so on any direct
    /// band overlap the larger/more prominent vehicle paints last and wins
    /// -- mirroring `OccluderInstances`' own ascending-score overlap rule.
    ///
    /// Returns `(seedIds, lowCoverageIds)` -- the latter lists the ids of
    /// vehicles whose ground contact was mostly cropped by the frame (low
    /// `coverageFrac`), for a caller warning.
    public static func groundContactSeedIds(instIds: [Int32], vehicles: [VehicleInstance],
                                            H: simd_double3x3, grid: BevGrid,
                                            width: Int, height: Int) -> (seedIds: [Int32], lowCoverageIds: [Int32]) {
        var seedIds = [Int32](repeating: 0, count: grid.width * grid.height)
        var lowCoverage: [Int32] = []
        for v in vehicles.sorted(by: { $0.pixelArea < $1.pixelArea }) {
            let mask = instIds.map { $0 == v.instId }
            let (points, coverage) = BottomContour.points(mask: mask, width: width, height: height)
            if coverage < AttributionConfig.footprintMinCoverage {
                lowCoverage.append(v.instId)
            }
            guard !points.isEmpty else { continue }
            let ptsBev = Homography.projectPointsToBev(points, H: H)
            let band = FootprintBand.rasterize(points: ptsBev, grid: grid, bandWidthM: AttributionConfig.footprintBandWidthM)
            for i in 0..<seedIds.count where band[i] {
                seedIds[i] = v.instId
            }
        }
        return (seedIds, lowCoverage)
    }

    public struct Measure {
        public let areaM2: Double
        public let areaM2Raw: Double
        public let cells: Int
    }

    /// Port of `disturbance.attribute`. Splits the occluded BEV area
    /// between the instances that hide it. `scaleFactor` converts calibrated
    /// area to the raw (uncalibrated) figure; both are reported because the
    /// calibrated one depends on a camera-height prior while the raw one
    /// inherits monocular depth's scale error.
    public static func attribute(occludedBev: [Bool], instBev: [Int32], vehicles: [VehicleInstance],
                                 grid: BevGrid, scaleFactor: Double) -> (perVehicle: [Int32: Measure], unattributed: Measure) {
        func measure(_ mask: [Bool]) -> Measure {
            let cells = mask.reduce(0) { $0 + ($1 ? 1 : 0) }
            let area = Measurement.areaM2(cellCount: cells, grid: grid)
            return Measure(areaM2: pythonRound3(area), areaM2Raw: pythonRound3(area * scaleFactor), cells: cells)
        }
        var perVehicle: [Int32: Measure] = [:]
        for v in vehicles {
            let mask = (0..<occludedBev.count).map { occludedBev[$0] && instBev[$0] == v.instId }
            perVehicle[v.instId] = measure(mask)
        }
        let unattributedMask = (0..<occludedBev.count).map { occludedBev[$0] && instBev[$0] == 0 }
        return (perVehicle, measure(unattributedMask))
    }
}
