import SwiftUI
import AmodalRoadKit

/// The user-facing output of one full `PhotoSemanticView.process(data:)` run
/// -- everything `ResultsView` needs to render, decoupled from the pile of
/// `@State` vars `process(data:)` itself uses to drive live per-stage
/// progress text while it's still running. Only constructed on a full
/// success (valid ground plane -> BEV -> attribution); a photo with no
/// usable ground plane has nothing here to show and stays on the
/// picker/status screen instead (mirrors `disturbance.DisturbanceResult.ok`
/// -- `occluded_bev is not None` -- which the Python CLI/Streamlit app also
/// treat as "nothing to report" rather than a degenerate all-zero result).
struct PipelineResult {
    let sourceImage: UIImage

    let calibrationSource: CalibrationSource
    let cameraHeightM: Double
    let cameraHeightWasCorrected: Bool

    let amodalAreaM2: Double
    let visibleAreaM2: Double
    let occludedAreaM2: Double
    let occludedPercent: Double
    let measurableRangeM: Double

    let vehicles: [VehicleResult]
    let unattributedAreaM2: Double

    /// Plain-language caveats, surfaced directly in the results UI rather
    /// than buried in a debug log -- mirrors `disturbance.DisturbanceResult
    /// .warnings`, which the migration plan's Section 9 specifically calls
    /// out as needing to stay prominent: this app makes a quantitative
    /// real-world claim ("occluded road area in m²"), and an unqualified
    /// number next to an uncalibrated-scale photo is misleading, not just
    /// incomplete.
    let warnings: [String]

    // Debug/technical imagery -- shown behind a "Technical details"
    // disclosure in ResultsView, not deleted, since it's still the fastest
    // way to tell "the model is wrong" from "the photo is hard."
    let overlayImage: UIImage
    let depthImage: UIImage?
    let vehicleBoxesImage: UIImage?
    let bevImage: UIImage

    // Raw BEV rasters, not a single pre-rendered image -- ResultsView
    // re-renders these via `BevRenderer.render` every time the user taps a
    // vehicle to highlight it, which a static UIImage can't support.
    let occludedBev: [Bool]
    let instBev: [Int32]
    let visibleBev: [Bool]
    let bevWidth: Int
    let bevHeight: Int

    let technicalSummary: String
}

/// One occluder's share of the occluded road, ready for display.
struct VehicleResult: Identifiable {
    let id: Int32
    let label: String
    let source: String        // "instance" | "blob"
    let score: Double         // .nan for a "blob" (undetected) instance
    let areaM2: Double
    let widthMaxPct: Double

    var color: Color { AttributionPalette.color(forInstId: id) }
    var isDetected: Bool { source == "instance" }
}

/// The palette `PhotoSemanticView.renderAttributionBev` colours the BEV
/// raster with -- shared here (not duplicated) so a vehicle's swatch in
/// `ResultsView`'s list is the EXACT colour it appears as in the BEV image,
/// not a coincidentally-similar one.
enum AttributionPalette {
    static let colors: [(UInt8, UInt8, UInt8)] = [
        (255, 0, 200), (0, 200, 255), (255, 215, 0), (0, 255, 120), (255, 100, 100), (150, 120, 255),
    ]

    static func color(forInstId id: Int32) -> Color {
        let c = colors[Int(id - 1) % colors.count]
        return Color(red: Double(c.0) / 255, green: Double(c.1) / 255, blue: Double(c.2) / 255)
    }
}

/// Renders the BEV raster shown in `ResultsView`'s "Bird's-eye view" card.
/// Shared (not duplicated) between `PhotoSemanticView.process(data:)` (the
/// initial, unselected render) and `ResultsView` (re-rendered live when the
/// user taps a vehicle to highlight it), so the two can never drift apart.
enum BevRenderer {
    /// `visibleBev` is drawn as dim green FIRST so the BEV card shows the
    /// whole reconstructed road extent, not just the occluded slice -- a
    /// photo with little/no occlusion used to render as an almost-entirely
    /// empty dark rectangle (nothing drawn for the, usually much larger,
    /// unoccluded portion of the road) even though real road data existed;
    /// this is why. Occluded cells draw on top, coloured per vehicle
    /// (`AttributionPalette`) or white for `attribute`'s `unattributed`
    /// bucket. When `selectedId` is set, mirrors `disturbance.py`'s
    /// `render_bev(selected=...)`: that vehicle's occluded cells highlight
    /// in magenta and every other vehicle's cells dim to grey, so the
    /// selected vehicle's share is unambiguous even when its own palette
    /// colour would otherwise be easy to miss in a small thumbnail.
    static func render(occludedBev: [Bool], instBev: [Int32], visibleBev: [Bool],
                       width: Int, height: Int, selectedId: Int32? = nil) -> UIImage {
        precondition(occludedBev.count == width * height && instBev.count == width * height
                    && visibleBev.count == width * height)
        var pixels = [UInt8](repeating: 20, count: width * height * 4)
        for i in 0..<(width * height) {
            var r: UInt8 = 20, g: UInt8 = 20, b: UInt8 = 20
            if occludedBev[i] {
                let id = instBev[i]
                if id == 0 {
                    r = 255; g = 255; b = 255
                } else if let selectedId, id == selectedId {
                    r = 255; g = 0; b = 200
                } else if selectedId != nil {
                    r = 90; g = 90; b = 90
                } else {
                    let c = AttributionPalette.colors[Int(id - 1) % AttributionPalette.colors.count]
                    r = c.0; g = c.1; b = c.2
                }
            } else if visibleBev[i] {
                r = 0; g = 110; b = 40
            }
            pixels[i*4+0] = r; pixels[i*4+1] = g; pixels[i*4+2] = b; pixels[i*4+3] = 255
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: &pixels, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        return UIImage(cgImage: context.makeImage()!)
    }
}
