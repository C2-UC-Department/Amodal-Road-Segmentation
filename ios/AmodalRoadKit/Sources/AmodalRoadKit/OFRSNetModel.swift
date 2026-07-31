import CoreML
import simd

/// Swift wrapper around the Core ML `.mlpackage` produced by
/// `tools/coreml_export_ofrsnet.py` (`OFRSNetExport`, see `src/ofrs/export.py`).
///
/// The exported graph is FIXED-SHAPE: `torch.jit.trace` bakes in H/W at
/// conversion time (the tool's `TracingWrapper`), so one `OFRSNetModel`
/// instance is only valid for the resolution it was exported at. `height`/
/// `width` are read back from the loaded model's own input description
/// rather than assumed by the caller, so a resolution mismatch is caught by
/// `predict`'s precondition instead of silently misreading a flat buffer.
///
/// Input tensor names/shapes (`x`, `G`, `h`, `gvalid_f`, `valid_f`) and the
/// output name (`logits`) mirror the tool's `convert()` exactly -- see that
/// file if these ever need to change together.
public final class OFRSNetModel {
    private let model: MLModel
    public let height: Int
    public let width: Int
    public let numClasses: Int

    /// `modelURL` may point at either a `.mlpackage` (compiled on first
    /// load -- slow, do this once and hold the instance) or an
    /// already-compiled `.mlmodelc`.
    public init(contentsOf modelURL: URL, numClasses: Int,
               configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        let compiledURL = modelURL.pathExtension == "mlmodelc"
            ? modelURL : try MLModel.compileModel(at: modelURL)
        self.model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        self.numClasses = numClasses

        guard let xDesc = model.modelDescription.inputDescriptionsByName["x"],
              let shape = xDesc.multiArrayConstraint?.shape, shape.count == 4 else {
            throw OFRSNetModelError.unexpectedInputShape
        }
        self.height = shape[2].intValue
        self.width = shape[3].intValue
    }

    /// `ct.convert(..., convert_to="mlprogram")` with no explicit dtype on
    /// any `ct.TensorType` (the export tool doesn't set one) defaults every
    /// input AND the output to `.float16`, not `.float32` -- confirmed by
    /// inspecting the compiled model's `multiArrayConstraint.dataType`
    /// directly (`65552`, `MLMultiArrayDataType.float16`'s raw value), after
    /// an initial `.float32` version of this file crashed with SIGSEGV: the
    /// output-reading code was reinterpreting a half-width Float16 buffer as
    /// Float32 and reading twice as many bytes as actually existed. Every
    /// buffer below is therefore `.float16`, matching the model exactly
    /// rather than assuming a "natural" default.
    private func makeFloat16Array(shape: [Int],
                                  fill: (UnsafeMutablePointer<Float16>, Int) -> Void) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        let count = shape.reduce(1, *)
        let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: count)
        fill(ptr, count)
        return array
    }

    /// One forward pass. `sem` is the OFRS-11 semantic map (class indices,
    /// row-major `height x width`, matching `predict.py`'s `sem`); this
    /// wrapper one-hot-encodes it into `x` itself, mirroring the Python
    /// export tool's `_plausible_example_inputs`/training-time encoding, so
    /// callers pass the same plain class-index array `GroundFields`/`bev`
    /// deal in elsewhere in this package. `G`/`h`/`gvalid` are
    /// `GroundFields.derive`'s dense outputs; pass all-zero `G`/`h`,
    /// all-false `gvalid`, `validPlane: false` when no plane was resolved --
    /// that's the exact zeroed-dict input the Python-side regression test
    /// (`tests/test_ofrsnet_coreml_readiness.py`) proves is bit-identical to
    /// `geo=None`.
    ///
    /// Returns the raw `[background, road]` logits (widened from the
    /// model's native float16 to Float), flattened `2 x height x width`,
    /// channel-major (index `c * height*width + i`) -- same layout
    /// `tools/coreml_export_ofrsnet.py` reads `logits` in.
    public func predict(sem: [Int], G: [SIMD3<Double>], h: [Double],
                        gvalid: [Bool], validPlane: Bool) throws -> [Float] {
        let count = height * width
        precondition(sem.count == count && G.count == count && h.count == count && gvalid.count == count,
                    "OFRSNetModel.predict: input size \(sem.count) doesn't match the exported resolution \(height)x\(width)")

        let xArray = try makeFloat16Array(shape: [1, numClasses, height, width]) { ptr, cnt in
            ptr.update(repeating: 0, count: cnt)
            for i in 0..<count {
                let c = sem[i]
                if c >= 0, c < numClasses { ptr[c * count + i] = 1 }
            }
        }
        let gArray = try makeFloat16Array(shape: [1, 3, height, width]) { ptr, _ in
            for i in 0..<count {
                ptr[0 * count + i] = Float16(G[i].x)
                ptr[1 * count + i] = Float16(G[i].y)
                ptr[2 * count + i] = Float16(G[i].z)
            }
        }
        let hArray = try makeFloat16Array(shape: [1, 1, height, width]) { ptr, _ in
            for i in 0..<count { ptr[i] = Float16(h[i]) }
        }
        let gvalidArray = try makeFloat16Array(shape: [1, 1, height, width]) { ptr, _ in
            for i in 0..<count { ptr[i] = gvalid[i] ? 1 : 0 }
        }
        let validArray = try makeFloat16Array(shape: [1]) { ptr, _ in
            ptr[0] = validPlane ? 1 : 0
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: xArray),
            "G": MLFeatureValue(multiArray: gArray),
            "h": MLFeatureValue(multiArray: hArray),
            "gvalid_f": MLFeatureValue(multiArray: gvalidArray),
            "valid_f": MLFeatureValue(multiArray: validArray),
        ])

        let output = try model.prediction(from: input)
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw OFRSNetModelError.missingOutput
        }
        // NOT a raw dataPointer walk -- Core ML output buffers can pad their
        // row stride to a hardware-alignment-friendly multiple (found via
        // DepthModel's 518-wide output actually being stored at stride 544);
        // this test fixture's width (96) happens to already be a multiple of
        // 32 so it never triggered here, but reading through
        // float16ElementsRowMajor() rather than assuming packed strides is
        // correct regardless of what width a future export uses. See
        // MLMultiArrayFlatten.swift.
        return logits.float16ElementsRowMajor()
    }
}

public enum OFRSNetModelError: Error {
    case unexpectedInputShape
    case missingOutput
}
