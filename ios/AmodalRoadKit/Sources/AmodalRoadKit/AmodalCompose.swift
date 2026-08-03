/// Port of `common.occluder_blob_mask` / `predict.py::compose_amodal_mask`.
public enum AmodalCompose {
    /// Port of `common.occluder_blob_mask` (src/common.py). Whole connected
    /// components of `occluders` that touch the near-road band -- see that
    /// function's docstring for why per-object (not per-pixel) eligibility
    /// matters: a per-pixel version (`occluders & dilate(visible, nearPx)`)
    /// only admits the slice of an occluder within `nearPx` of the visible
    /// road -- for a vehicle, roughly the tire/undercarriage band, since the
    /// roof can be 100+ px further away -- silently discarding the rest of
    /// its footprint no matter how well OFRSNet reconstructed it. Here any
    /// connected blob that touches the near-road band at all is admitted IN
    /// FULL.
    public static func occluderBlobMask(occluders: [Bool], visible: [Bool], nearPx: Int,
                                        width: Int, height: Int) -> [Bool] {
        let roadNeigh = Dilation.dilate(mask: visible, width: width, height: height, radiusPx: nearPx)
        let (labels, count) = ConnectedComponents.label8(mask: occluders, width: width, height: height)
        guard count > 0 else { return [Bool](repeating: false, count: width * height) }

        var touchesRoad = [Bool](repeating: false, count: count)
        for i in 0..<(width * height) where occluders[i] && roadNeigh[i] {
            touchesRoad[labels[i]] = true
        }

        var result = [Bool](repeating: false, count: width * height)
        for i in 0..<(width * height) where occluders[i] && touchesRoad[labels[i]] {
            result[i] = true
        }
        return result
    }

    /// Port of `instances.occluder_support`. The exact region
    /// `composeAmodalMask` is allowed to patch -- kept as its own function
    /// (mirroring the Python source) so `OccluderInstances.occluderInstanceIds`
    /// can share one definition of "the occluders that matter" with
    /// `composeAmodalMask` instead of quietly recomputing it.
    public static func occluderSupport(semLabels: [Int], width: Int, height: Int) -> [Bool] {
        let visible = semLabels.map { $0 == OFRSClasses.roadIdx }
        let occluders = semLabels.map { $0 == OFRSClasses.vehicleIdx || $0 == OFRSClasses.personIdx }
        return occluderBlobMask(occluders: occluders, visible: visible,
                                nearPx: OFRSClasses.roadNeighbourhoodPx, width: width, height: height)
    }

    /// Port of `predict.py::compose_amodal_mask`. Merges the visible road
    /// with ONLY the occluded-region patch OFRSNet predicted -- OFRSNet's
    /// output never replaces or redraws already-visible road, it only fills
    /// the footprint of nearby occluders (`OFRSClasses.roadNeighbourhoodPx`).
    /// Returns `(final, patch)`; `patch` is the subset of `final` that came
    /// from OFRSNet, useful for QA/debug visualization, matching
    /// `compose_amodal_mask`'s own return shape.
    public static func composeAmodalMask(semLabels: [Int], ofrsRoad: [Bool],
                                         width: Int, height: Int) -> (final: [Bool], patch: [Bool]) {
        let visible = semLabels.map { $0 == OFRSClasses.roadIdx }
        let occluders = semLabels.map { $0 == OFRSClasses.personIdx || $0 == OFRSClasses.vehicleIdx }
        let occlusionRegion = occluderBlobMask(occluders: occluders, visible: visible,
                                               nearPx: OFRSClasses.roadNeighbourhoodPx,
                                               width: width, height: height)
        var patch = [Bool](repeating: false, count: width * height)
        var final = [Bool](repeating: false, count: width * height)
        for i in 0..<(width * height) {
            patch[i] = ofrsRoad[i] && occlusionRegion[i]
            final[i] = visible[i] || patch[i]
        }
        return (final, patch)
    }
}
