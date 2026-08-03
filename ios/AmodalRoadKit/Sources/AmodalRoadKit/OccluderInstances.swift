/// Port of `instances.py`'s "Instances on the occluder support" section
/// (`VehicleInstance`, `_describe`, `_dominant_label`, `occluder_instance_ids`).
///
/// The load-bearing design decision (see src/instances.py's module
/// docstring) is what the ids are defined ON: `AmodalCompose.composeAmodalMask`
/// only ever patches inside `AmodalCompose.occluderSupport` -- whole
/// connected blobs of the vehicle/person classes that touch the near-road
/// band. Ids are therefore built by taking that support as given and using
/// detections only to SPLIT it, not the other way around, so attribution is
/// complete by construction and disagreement between the detector and the
/// semantic support becomes a diagnostic (the "unattributed" figure) rather
/// than a mystery gap.
public struct VehicleInstance {
    public let instId: Int32
    public let label: String
    public let score: Double          // .nan for a "blob" (undetected) instance
    public let pixelArea: Int
    public let centroidU: Int
    public let centroidV: Int
    public let bboxU0: Int
    public let bboxV0: Int
    public let bboxU1: Int            // exclusive
    public let bboxV1: Int            // exclusive
    public let selectable: Bool
    public let source: String         // "instance" | "blob"
}

/// One raw detection, already resampled onto the semantic map's own grid
/// (callers own that resampling -- see `PhotoSemanticView`'s
/// `vehicleMaskAtGrid`, since it composes app-specific letterbox geometry
/// this package has no dependency on). Mirrors `instances.Detection`, minus
/// the fields nothing here reads.
public struct OccluderDetection {
    public let label: String
    public let score: Double
    public let mask: [Bool]           // same width*height as the semantic map
    public let isVehicle: Bool

    public init(label: String, score: Double, mask: [Bool], isVehicle: Bool) {
        self.label = label
        self.score = score
        self.mask = mask
        self.isVehicle = isVehicle
    }
}

public enum OccluderInstances {
    /// Port of `instances.occluder_instance_ids`. Returns
    /// `(instIds, instances)`: `instIds` is a row-major `[Int32]` of shape
    /// `(height, width)`, `0` meaning "not an occluder"; `instances` are
    /// renumbered `1...N` by DESCENDING pixel area (so `#1` is the most
    /// prominent occluder). Support pixels no detection claims are kept as
    /// their own `source: "blob"` instance when big enough
    /// (`AttributionConfig.instanceMinBlobPx`) to matter, and left at `0`
    /// otherwise.
    public static func occluderInstanceIds(semLabels: [Int], detections: [OccluderDetection],
                                           width: Int, height: Int) -> (instIds: [Int32], instances: [VehicleInstance]) {
        let count = width * height
        precondition(semLabels.count == count)
        let support = AmodalCompose.occluderSupport(semLabels: semLabels, width: width, height: height)

        struct Meta { let label: String; let score: Double; let selectable: Bool; let source: String }
        var ids = [Int32](repeating: 0, count: count)
        var meta: [Int32: Meta] = [:]
        var nextId: Int32 = 1

        // Ascending score, so the most confident detection paints last and
        // wins any overlap -- matches Python's `sorted(detections, key=score)`.
        for det in detections.sorted(by: { $0.score < $1.score }) {
            var any = false
            for i in 0..<count where det.mask[i] && support[i] {
                ids[i] = nextId
                any = true
            }
            if any {
                meta[nextId] = Meta(label: det.label, score: det.score, selectable: det.isVehicle, source: "instance")
                nextId += 1
            }
        }

        var leftover = [Bool](repeating: false, count: count)
        var hasLeftover = false
        for i in 0..<count where support[i] && ids[i] == 0 {
            leftover[i] = true
            hasLeftover = true
        }
        if hasLeftover {
            let (labels, compCount) = ConnectedComponents.label8(mask: leftover, width: width, height: height)
            if compCount > 0 {
                var compPixels = [[Int]](repeating: [], count: compCount)
                for i in 0..<count where leftover[i] {
                    compPixels[labels[i]].append(i)
                }
                for comp in compPixels {
                    guard comp.count >= AttributionConfig.instanceMinBlobPx else { continue }
                    var nVehicle = 0, nPerson = 0
                    for i in comp {
                        if semLabels[i] == OFRSClasses.vehicleIdx { nVehicle += 1 }
                        else if semLabels[i] == OFRSClasses.personIdx { nPerson += 1 }
                    }
                    let (label, selectable) = nVehicle >= nPerson ? ("vehicle", true) : ("person", false)
                    for i in comp { ids[i] = nextId }
                    meta[nextId] = Meta(label: label, score: .nan, selectable: selectable, source: "blob")
                    nextId += 1
                }
            }
        }

        // Renumber by descending pixel area. Swift's `sorted` is stable, so
        // ties preserve ascending-old-id order -- matching Python's
        // `sorted(meta)` (ascending keys) followed by a stable descending
        // area sort.
        var areaByOld: [Int32: Int] = [:]
        for i in 0..<count where ids[i] != 0 {
            areaByOld[ids[i], default: 0] += 1
        }
        let present = (Int32(1)..<nextId).compactMap { old -> (old: Int32, area: Int)? in
            guard let a = areaByOld[old], a > 0 else { return nil }
            return (old, a)
        }.sorted { $0.area > $1.area }

        var lut = [Int32](repeating: 0, count: Int(nextId))
        for (newIdx, entry) in present.enumerated() {
            lut[Int(entry.old)] = Int32(newIdx + 1)
        }
        for i in 0..<count where ids[i] != 0 {
            ids[i] = lut[Int(ids[i])]
        }

        // One pass over the renumbered ids to describe every instance
        // (count/centroid/bbox) -- O(N) total rather than O(N * instances).
        let n = present.count
        var pxCount = [Int](repeating: 0, count: n + 1)
        var sumU = [Int](repeating: 0, count: n + 1)
        var sumV = [Int](repeating: 0, count: n + 1)
        var minU = [Int](repeating: .max, count: n + 1)
        var minV = [Int](repeating: .max, count: n + 1)
        var maxU = [Int](repeating: .min, count: n + 1)
        var maxV = [Int](repeating: .min, count: n + 1)
        for y in 0..<height {
            for x in 0..<width {
                let id = Int(ids[y * width + x])
                guard id != 0 else { continue }
                pxCount[id] += 1
                sumU[id] += x; sumV[id] += y
                minU[id] = min(minU[id], x); maxU[id] = max(maxU[id], x)
                minV[id] = min(minV[id], y); maxV[id] = max(maxV[id], y)
            }
        }

        var instances: [VehicleInstance] = []
        instances.reserveCapacity(n)
        for (newIdx, entry) in present.enumerated() {
            let new = newIdx + 1
            let m = meta[entry.old]!
            instances.append(VehicleInstance(
                instId: Int32(new), label: m.label, score: m.score, pixelArea: pxCount[new],
                centroidU: sumU[new] / pxCount[new], centroidV: sumV[new] / pxCount[new],
                bboxU0: minU[new], bboxV0: minV[new], bboxU1: maxU[new] + 1, bboxV1: maxV[new] + 1,
                selectable: m.selectable, source: m.source))
        }
        return (ids, instances)
    }
}
