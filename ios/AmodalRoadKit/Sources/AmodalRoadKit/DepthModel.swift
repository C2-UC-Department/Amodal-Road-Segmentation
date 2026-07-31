import CoreML

/// Swift wrapper around the Core ML `.mlpackage` produced by
/// `tools/coreml_export_depth.py` (Depth-Anything-V2-Metric-Outdoor-Small,
/// see that file's docstring for the Phase-0 conversion spike this came
/// from -- GO, with the DINOv2 position-embedding fix already baked into
/// the exported graph).
///
/// Same fixed-shape / `.float16` story as `OFRSNetModel`: the HF
/// `DPTImageProcessor` this model was traced with always targets 518x518
/// (`ensure_multiple_of=14`, DINOv2's patch size), regardless of the source
/// photo's own resolution or aspect ratio -- confirmed by inspecting the
/// compiled model's `multiArrayConstraint` directly (`pixel_values`
/// `[1,3,518,518]`, `predicted_depth` `[1,518,518]`, both dataType `65552` =
/// `.float16`), the same lesson `OFRSNetModel.swift` already learned the
/// hard way (that file's header has the SIGSEGV post-mortem).
///
/// This wrapper takes an ALREADY preprocessed `pixel_values` tensor, not a
/// raw image -- see `ImagePreprocessing.swift` for the resize+normalize
/// step, which is a DELIBERATE simplification of `DPTImageProcessor`'s own
/// aspect-ratio-preserving resize (documented there, not silently dropped).
public final class DepthModel {
    private let model: MLModel
    public let height: Int
    public let width: Int

    public init(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        let compiledURL = modelURL.pathExtension == "mlmodelc"
            ? modelURL : try MLModel.compileModel(at: modelURL)
        self.model = try MLModel(contentsOf: compiledURL, configuration: configuration)

        guard let desc = model.modelDescription.inputDescriptionsByName["pixel_values"],
              let shape = desc.multiArrayConstraint?.shape, shape.count == 4 else {
            throw DepthModelError.unexpectedInputShape
        }
        self.height = shape[2].intValue
        self.width = shape[3].intValue
    }

    /// `pixelValues`: normalized RGB, channel-major, `3 * height * width`
    /// (see `ImagePreprocessing.resizeAndNormalizeImageNet`). Returns the
    /// metric depth map in metres, row-major `height * width`.
    public func predict(pixelValues: [Float]) throws -> [Float] {
        let count = height * width
        precondition(pixelValues.count == 3 * count,
                    "DepthModel.predict: input size \(pixelValues.count) doesn't match 3x\(height)x\(width)")

        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)],
                                    dataType: .float16)
        let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: 3 * count)
        for i in 0..<(3 * count) { ptr[i] = Float16(pixelValues[i]) }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: array),
        ])
        let output = try model.prediction(from: input)
        guard let depth = output.featureValue(for: "predicted_depth")?.multiArrayValue else {
            throw DepthModelError.missingOutput
        }
        // NOT a raw dataPointer walk -- see MLMultiArrayFlatten.swift's
        // header for why (Core ML pads this output's row stride to 544, not
        // the logical width 518).
        return depth.float16ElementsRowMajor()
    }
}

public enum DepthModelError: Error {
    case unexpectedInputShape
    case missingOutput
}
