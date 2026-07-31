import CoreML

/// Reading a Core ML model OUTPUT via a raw pointer is not safe to treat as
/// tightly packed. Found the hard way: `DepthModel`'s `predicted_depth`
/// output has logical shape `[1, 518, 518]`, but its ACTUAL `strides` are
/// `[281792, 544, 1]` -- Core ML pads the row stride to 544 (a
/// hardware-alignment-friendly multiple of 32), not the logical width 518.
/// Reading it with `dataPointer[row * 518 + col]` silently walks into the
/// padding and reads garbage/zero once `col` gets close to 518, rather than
/// throwing -- it happened to go unnoticed in `OFRSNetModel`'s own tests
/// because its output width (96) already IS a multiple of 32, so no padding
/// was ever inserted there. Inputs this package allocates ITSELF (via
/// `MLMultiArray(shape:dataType:)`) are unaffected -- confirmed by
/// inspecting `.strides` right after allocation, they come back tightly
/// packed -- so this is specifically an output-reading concern.
extension MLMultiArray {
    /// Every element in row-major logical order, respecting this array's
    /// OWN strides. Only `.float16` is implemented -- the only dtype this
    /// package's Core ML exports ever use (see DepthModel.swift/
    /// OFRSNetModel.swift's header notes on why `ct.convert(...,
    /// convert_to="mlprogram")` defaults to it).
    func float16ElementsRowMajor() -> [Float] {
        precondition(dataType == .float16)
        let dims = shape.map { $0.intValue }
        let strd = strides.map { $0.intValue }
        let total = dims.reduce(1, *)
        let bufferElements = zip(dims, strd).map { ($0 - 1) * $1 }.reduce(1, +)
        let ptr = dataPointer.bindMemory(to: Float16.self, capacity: bufferElements)

        var result = [Float](repeating: 0, count: total)
        var idx = [Int](repeating: 0, count: dims.count)
        for flat in 0..<total {
            var offset = 0
            for d in 0..<dims.count { offset += idx[d] * strd[d] }
            result[flat] = Float(ptr[offset])

            var d = dims.count - 1
            while d >= 0 {
                idx[d] += 1
                if idx[d] < dims[d] { break }
                idx[d] = 0
                d -= 1
            }
        }
        return result
    }
}
