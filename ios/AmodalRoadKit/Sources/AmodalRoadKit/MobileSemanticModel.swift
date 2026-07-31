import CoreML

/// Swift wrapper around the Core ML `.mlpackage` produced by
/// `tools/coreml_export_mobile_semantic.py` (`MobileSemanticNet`, see
/// `src/mobile_semantic/model.py`) -- the Phase 2 semantic student
/// (LR-ASPP/MobileNetV3-Large) that replaces the Mask2Former/Mapillary-Vistas
/// teacher on-device.
///
/// Simplest of this package's three Core ML wrappers: a single RGB input,
/// a single logits output, no geo dict (`OFRSNetModel`) or preprocessing
/// quirks of its own (`DepthModel`'s 518x518 DPT resolution). Same fixed-shape
/// `.float16` story as the other two, and outputs are read through
/// `float16ElementsRowMajor()` on principle (see `MLMultiArrayFlatten.swift`)
/// even though this model's width (512) is already a multiple of 32 and so
/// wouldn't itself trigger the padded-stride bug found via `DepthModel`.
public final class MobileSemanticModel {
    private let model: MLModel
    public let height: Int
    public let width: Int
    public let numClasses: Int

    public init(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        let compiledURL = modelURL.pathExtension == "mlmodelc"
            ? modelURL : try MLModel.compileModel(at: modelURL)
        self.model = try MLModel(contentsOf: compiledURL, configuration: configuration)

        guard let xDesc = model.modelDescription.inputDescriptionsByName["rgb"],
              let shape = xDesc.multiArrayConstraint?.shape, shape.count == 4,
              let yDesc = model.modelDescription.outputDescriptionsByName["logits"],
              let outShape = yDesc.multiArrayConstraint?.shape, outShape.count == 4 else {
            throw MobileSemanticModelError.unexpectedShape
        }
        self.height = shape[2].intValue
        self.width = shape[3].intValue
        self.numClasses = outShape[1].intValue
    }

    /// `pixelValues`: normalized RGB, channel-major, `3 * height * width`
    /// (e.g. `ImagePreprocessing.resizeAndNormalizeImageNet`). Returns raw
    /// per-class logits, channel-major, `numClasses * height * width`.
    public func predict(pixelValues: [Float]) throws -> [Float] {
        let count = height * width
        precondition(pixelValues.count == 3 * count,
                    "MobileSemanticModel.predict: input size \(pixelValues.count) doesn't match 3x\(height)x\(width)")

        let array = try MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)],
                                    dataType: .float16)
        let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: 3 * count)
        for i in 0..<(3 * count) { ptr[i] = Float16(pixelValues[i]) }

        let input = try MLDictionaryFeatureProvider(dictionary: ["rgb": MLFeatureValue(multiArray: array)])
        let output = try model.prediction(from: input)
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw MobileSemanticModelError.missingOutput
        }
        return logits.float16ElementsRowMajor()
    }
}

public enum MobileSemanticModelError: Error {
    case unexpectedShape
    case missingOutput
}
