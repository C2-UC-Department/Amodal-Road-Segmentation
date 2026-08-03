/// 8-connectivity connected-components labelling, matching
/// `cv2.connectedComponents(mask, connectivity=8)`'s GROUPING -- not its
/// specific label values/order, which no caller in this package depends on
/// (`AmodalCompose.occluderBlobMask` only needs "which pixels belong to the
/// same blob").
public enum ConnectedComponents {
    /// Returns per-pixel labels (`-1` where `mask` is false) and the number
    /// of distinct components. Labels are dense `0..<count`, in first-seen
    /// raster-scan order (an implementation detail, not a contract).
    public static func label8(mask: [Bool], width: Int, height: Int) -> (labels: [Int], count: Int) {
        var parent = [Int](repeating: -1, count: width * height)
        func find(_ x: Int) -> Int {
            var x = x
            while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x] }
            return x
        }
        func union(_ a: Int, _ b: Int) {
            let ra = find(a), rb = find(b)
            if ra != rb { parent[ra] = rb }
        }
        for i in 0..<(width * height) where mask[i] { parent[i] = i }

        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                guard mask[idx] else { continue }
                // 8-connectivity: only look "backward" (left, and the three
                // upper neighbours) in raster order -- forward neighbours
                // will look back at this pixel on their own turn, so every
                // adjacent pair still gets unioned exactly once.
                if x > 0, mask[idx - 1] { union(idx, idx - 1) }
                if y > 0 {
                    let up = idx - width
                    if mask[up] { union(idx, up) }
                    if x > 0, mask[up - 1] { union(idx, up - 1) }
                    if x < width - 1, mask[up + 1] { union(idx, up + 1) }
                }
            }
        }

        var labels = [Int](repeating: -1, count: width * height)
        var nextLabel = 0
        var rootToLabel: [Int: Int] = [:]
        for i in 0..<(width * height) where mask[i] {
            let root = find(i)
            if let l = rootToLabel[root] {
                labels[i] = l
            } else {
                rootToLabel[root] = nextLabel
                labels[i] = nextLabel
                nextLabel += 1
            }
        }
        return (labels, nextLabel)
    }
}
