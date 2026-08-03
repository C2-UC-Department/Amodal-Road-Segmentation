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
    private func elementsRowMajor<T>(as type: T.Type, load: (UnsafePointer<T>, Int) -> Float) -> [Float] {
        let dims = shape.map { $0.intValue }
        let strd = strides.map { $0.intValue }
        let total = dims.reduce(1, *)
        let bufferElements = zip(dims, strd).map { ($0 - 1) * $1 }.reduce(1, +)
        let ptr = dataPointer.bindMemory(to: T.self, capacity: bufferElements)

        var result = [Float](repeating: 0, count: total)
        var idx = [Int](repeating: 0, count: dims.count)
        for flat in 0..<total {
            var offset = 0
            for d in 0..<dims.count { offset += idx[d] * strd[d] }
            result[flat] = load(ptr, offset)

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

    /// Every element in row-major logical order, respecting this array's
    /// OWN strides. `.float16` is what `OFRSNetModel`/`DepthModel`/
    /// `MobileSemanticModel`'s exports use (`ct.convert(..., convert_to=
    /// "mlprogram")` defaults to it with no dtype on the `TensorType`s --
    /// see those files' header notes); `.float32` is what the Ultralytics
    /// YOLO export uses instead (confirmed by inspecting the compiled
    /// model's `multiArrayConstraint.dataType` directly, `65568`, not
    /// assumed) -- so both are real, not a guess either way.
    func float16ElementsRowMajor() -> [Float] {
        precondition(dataType == .float16)
        return elementsRowMajor(as: Float16.self) { ptr, offset in Float(ptr[offset]) }
    }

    func float32ElementsRowMajor() -> [Float] {
        precondition(dataType == .float32)
        return elementsRowMajor(as: Float32.self) { ptr, offset in ptr[offset] }
    }
}
