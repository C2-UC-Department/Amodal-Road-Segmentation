import CoreML

/// One decoded, post-NMS vehicle detection. Box coordinates and mask are
/// both in the model's own fixed input resolution (`YOLOInstanceModel.
/// inputWidth/inputHeight`, 640x640 for the bundled export) -- callers
/// scale to the original photo themselves, same division of responsibility
/// `Calibration`/`GeometryPipeline` already use elsewhere in this package.
public struct VehicleDetection {
    public let x1: Double
    public let y1: Double
    public let x2: Double
    public let y2: Double
    public let score: Float
    /// Row-major bool mask, `maskWidth * maskHeight` (== the model's own
    /// input resolution -- upsampled from the prototype grid, see
    /// `YOLOInstanceModel.decode`'s header), already cropped to this
    /// detection's own box.
    public let mask: [Bool]
}

/// Swift wrapper + postprocessing for the vehicle-instance YOLOv8n-seg
/// Core ML export (`tools/train_yolo_instance.py`, `checkpoints/
/// yolo_instance/weights/best.mlpackage`). Exported with `nms=False`
/// (Ultralytics' standard mobile-deployment pattern, see that model's
/// export invocation): the Core ML graph returns raw box/score/mask-
/// coefficient predictions AND already-decoded pixel-space boxes (box
/// regression -> anchor decode -> stride scaling all happen inside the
/// traced graph, confirmed by reading `ultralytics.nn.modules.head.
/// Detect._inference`/`_get_decode_boxes` directly, not assumed) -- what's
/// NOT done by the graph, and IS done here, is confidence filtering, NMS,
/// and turning the 32 mask coefficients + prototype maps into actual
/// per-detection binary masks.
///
/// Output layout (confirmed against `Segment._inference`/`Segment.forward`,
/// with `nc=1` since this is a single-class "vehicle" detector -- see
/// `src/mobile_instance/labels.py`):
///   box-scores-coeffs tensor: (1, 4 + nc + nm, numAnchors) = (1, 37, 8400)
///     channels [0:4]   = box, CENTER-x, center-y, width, height, in pixels
///                        of the model's own input resolution (NOT xyxy --
///                        `Detect.decode_bboxes` returns xywh when not
///                        end2end/xyxy-flagged, which this export is not)
///     channel  [4]     = sigmoid'd class confidence (nc=1, so just one channel)
///     channels [5:37]  = 32 raw (not sigmoid'd) mask coefficients
///   prototypes tensor: (1, nm, protoH, protoW) = (1, 32, 160, 160)
/// Mask assembly matches `ultralytics.utils.ops.process_mask(upsample=True)`
/// exactly: per-detection mask = coefficients . prototypes (plain dot
/// product, NOT sigmoid -- threshold at > 0 is the sigmoid-0.5 equivalent),
/// bilinear-upsampled to the model's own input resolution BEFORE cropping
/// to the box (Ultralytics' own comment: cropping first would smear the
/// bilinear edge outside the bbox).
public final class YOLOInstanceModel {
    private let model: MLModel
    public let inputWidth: Int
    public let inputHeight: Int

    public init(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        let compiledURL = modelURL.pathExtension == "mlmodelc"
            ? modelURL : try MLModel.compileModel(at: modelURL)
        self.model = try MLModel(contentsOf: compiledURL, configuration: configuration)

        guard let imageInput = model.modelDescription.inputDescriptionsByName.values.first(where: { $0.imageConstraint != nil }),
              let constraint = imageInput.imageConstraint else {
            throw YOLOInstanceModelError.unexpectedInputType
        }
        self.inputWidth = constraint.pixelsWide
        self.inputHeight = constraint.pixelsHigh
    }

    public func detect(pixelBuffer: CVPixelBuffer, confThreshold: Float = 0.25,
                       iouThreshold: Float = 0.7) throws -> [VehicleDetection] {
        let inputName = model.modelDescription.inputDescriptionsByName.first(where: { $0.value.imageConstraint != nil })!.key
        let input = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: pixelBuffer)])
        let output = try model.prediction(from: input)

        var boxScoresCoeffs: MLMultiArray?
        var protos: MLMultiArray?
        for name in output.featureNames {
            guard let arr = output.featureValue(for: name)?.multiArrayValue else { continue }
            if arr.shape.count == 3 { boxScoresCoeffs = arr }
            else if arr.shape.count == 4 { protos = arr }
        }
        guard let boxScoresCoeffs, let protos else {
            throw YOLOInstanceModelError.unexpectedOutputShape
        }

        return Self.decode(boxScoresCoeffs: boxScoresCoeffs, protos: protos,
                          inputWidth: inputWidth, inputHeight: inputHeight,
                          confThreshold: confThreshold, iouThreshold: iouThreshold)
    }

    static func decode(boxScoresCoeffs: MLMultiArray, protos: MLMultiArray,
                       inputWidth: Int, inputHeight: Int,
                       confThreshold: Float, iouThreshold: Float) -> [VehicleDetection] {
        let bscDims = boxScoresCoeffs.shape.map { $0.intValue }        // [1, channels, numAnchors]
        let channels = bscDims[1], numAnchors = bscDims[2]
        let numCoeffs = channels - 5                                   // 4 box + 1 score + numCoeffs mask coeffs
        let bsc = boxScoresCoeffs.float32ElementsRowMajor()            // channel-major: c*numAnchors + a

        // --- confidence filter + box format conversion (cxcywh -> xyxy) ---
        var candidates: [Candidate] = []
        for a in 0..<numAnchors {
            let score = bsc[4 * numAnchors + a]
            guard score >= confThreshold else { continue }
            let cx = Double(bsc[0 * numAnchors + a])
            let cy = Double(bsc[1 * numAnchors + a])
            let w = Double(bsc[2 * numAnchors + a])
            let h = Double(bsc[3 * numAnchors + a])
            var coeffs = [Float](repeating: 0, count: numCoeffs)
            for c in 0..<numCoeffs { coeffs[c] = bsc[(5 + c) * numAnchors + a] }
            candidates.append(Candidate(x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2,
                                       score: score, coeffs: coeffs))
        }

        // --- greedy NMS, highest score first ---
        candidates.sort { $0.score > $1.score }
        var keep: [Candidate] = []
        for c in candidates {
            let overlaps = keep.contains { iou($0, c) > Double(iouThreshold) }
            if !overlaps { keep.append(c) }
        }
        guard !keep.isEmpty else { return [] }

        // --- mask assembly: coeffs . protos, upsample, threshold, crop ---
        let protoDims = protos.shape.map { $0.intValue }               // [1, nm, protoH, protoW]
        let protoChannels = protoDims[1], protoH = protoDims[2], protoW = protoDims[3]
        let protoFlat = protos.float32ElementsRowMajor()               // channel-major: c*protoH*protoW + y*protoW + x

        return keep.map { c in
            var protoMask = [Float](repeating: 0, count: protoH * protoW)
            for py in 0..<protoH {
                for px in 0..<protoW {
                    var acc: Float = 0
                    for ch in 0..<protoChannels {
                        acc += c.coeffs[ch] * protoFlat[ch * protoH * protoW + py * protoW + px]
                    }
                    protoMask[py * protoW + px] = acc
                }
            }
            let upsampled = resizeBilinear(protoMask, srcWidth: protoW, srcHeight: protoH,
                                          dstWidth: inputWidth, dstHeight: inputHeight)
            var mask = [Bool](repeating: false, count: inputWidth * inputHeight)
            for y in 0..<inputHeight {
                let inBoxY = Double(y) >= c.y1 && Double(y) < c.y2
                guard inBoxY else { continue }
                for x in 0..<inputWidth {
                    guard Double(x) >= c.x1, Double(x) < c.x2 else { continue }
                    mask[y * inputWidth + x] = upsampled[y * inputWidth + x] > 0
                }
            }
            return VehicleDetection(x1: c.x1, y1: c.y1, x2: c.x2, y2: c.y2, score: c.score, mask: mask)
        }
    }

    private static func resizeBilinear(_ field: [Float], srcWidth: Int, srcHeight: Int,
                                       dstWidth: Int, dstHeight: Int) -> [Float] {
        var out = [Float](repeating: 0, count: dstWidth * dstHeight)
        let sx = Double(srcWidth) / Double(dstWidth)
        let sy = Double(srcHeight) / Double(dstHeight)
        for ty in 0..<dstHeight {
            let fy = (Double(ty) + 0.5) * sy - 0.5
            let y0 = max(0, min(srcHeight - 1, Int(floor(fy))))
            let y1 = max(0, min(srcHeight - 1, y0 + 1))
            let wy = max(0.0, min(1.0, fy - Double(y0)))
            for tx in 0..<dstWidth {
                let fx = (Double(tx) + 0.5) * sx - 0.5
                let x0 = max(0, min(srcWidth - 1, Int(floor(fx))))
                let x1 = max(0, min(srcWidth - 1, x0 + 1))
                let wx = max(0.0, min(1.0, fx - Double(x0)))
                let p00 = Double(field[y0 * srcWidth + x0]), p01 = Double(field[y0 * srcWidth + x1])
                let p10 = Double(field[y1 * srcWidth + x0]), p11 = Double(field[y1 * srcWidth + x1])
                let top = p00 * (1 - wx) + p01 * wx
                let bot = p10 * (1 - wx) + p11 * wx
                out[ty * dstWidth + tx] = Float(top * (1 - wy) + bot * wy)
            }
        }
        return out
    }
}

public enum YOLOInstanceModelError: Error {
    case unexpectedInputType
    case unexpectedOutputShape
}

private struct Candidate {
    let x1: Double, y1: Double, x2: Double, y2: Double
    let score: Float
    let coeffs: [Float]
}

private func iou(_ a: Candidate, _ b: Candidate) -> Double {
    let ix1 = max(a.x1, b.x1), iy1 = max(a.y1, b.y1)
    let ix2 = min(a.x2, b.x2), iy2 = min(a.y2, b.y2)
    let interW = max(0, ix2 - ix1), interH = max(0, iy2 - iy1)
    let inter = interW * interH
    guard inter > 0 else { return 0 }
    let areaA = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
    let areaB = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
    let union = areaA + areaB - inter
    return union > 0 ? inter / union : 0
}
