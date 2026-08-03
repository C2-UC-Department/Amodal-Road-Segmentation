import SwiftUI
import AmodalRoadKit

/// The actual product surface: "how much road does this occlude, and how
/// sure should I be." `PhotoSemanticView` builds a `PipelineResult` once a
/// photo finishes processing and swaps to this view -- the per-stage debug
/// stack (semantic overlay, depth viz, timings) that used to be the whole
/// screen is still here, just behind a "Technical details" disclosure
/// instead of being the primary content.
struct ResultsView: View {
    let result: PipelineResult
    let onNewPhoto: () -> Void

    /// Tapping a vehicle row selects/deselects it -- `bevCard` re-renders
    /// the BEV image live (via `BevRenderer`, shared with the initial
    /// render so the two can't drift) highlighting that vehicle's occluded
    /// cells in magenta and dimming every other vehicle's, mirroring
    /// `disturbance.py`'s own `render_bev(selected=...)`/`--vehicle N`
    /// interactive-selection feature.
    @State private var selectedVehicleId: Int32?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                heroCard
                if !result.warnings.isEmpty {
                    warningsCard
                }
                bevCard
                vehiclesSection
                technicalDetails
            }
            .padding()
        }
        .safeAreaInset(edge: .bottom) {
            Button(action: onNewPhoto) {
                Label("New Photo", systemImage: "arrow.counterclockwise")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .padding()
            .background(.bar)
        }
    }

    // MARK: - Header

    private var header: some View {
        Image(uiImage: result.sourceImage)
            .resizable()
            .scaledToFit()
            .frame(maxHeight: 200)
            .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Hero

    private var heroCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Occluded road area").font(.subheadline).foregroundStyle(.secondary)
            Text(String(format: "%.2f m²", result.occludedAreaM2))
                .font(.system(size: 44, weight: .bold, design: .rounded))
            Text(String(format: "%.1f%% of the reconstructed road here is out of use", result.occludedPercent))
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Divider().padding(.vertical, 4)

            HStack(spacing: 16) {
                metric("Visible", String(format: "%.2f m²", result.visibleAreaM2))
                metric("Total (amodal)", String(format: "%.2f m²", result.amodalAreaM2))
                metric("Measured to", String(format: "%.0f m", result.measurableRangeM))
            }

            calibrationBadge
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 16).fill(.thinMaterial))
    }

    private func metric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value).font(.callout).fontWeight(.semibold)
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
    }

    /// Mirrors `disturbance.analyze`'s own distinction between a real
    /// calibration source and the "blind guess" `fov_prior` fallback --
    /// surfaced as a badge, not buried, since every area figure on this
    /// screen inherits that uncertainty.
    private var calibrationBadge: some View {
        let calibrated = result.calibrationSource != .fovPrior
        return HStack(spacing: 6) {
            Image(systemName: calibrated ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(calibrated ? .green : .orange)
            Text(calibratedText)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.top, 2)
    }

    private var calibratedText: String {
        switch result.calibrationSource {
        case .avfoundation:
            return "Calibrated from camera metadata"
        case .exif:
            return "Calibrated from photo EXIF data"
        case .fovPrior:
            return "Estimated scale (no camera metadata found) — areas may be off by 35-45%"
        }
    }

    // MARK: - Warnings

    private var warningsCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Things to know", systemImage: "exclamationmark.triangle")
                .font(.subheadline).fontWeight(.semibold)
                .foregroundStyle(.orange)
            ForEach(result.warnings, id: \.self) { warning in
                Text("•  \(warning)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 16).fill(Color.orange.opacity(0.12)))
    }

    // MARK: - BEV map

    /// Recomputed whenever `selectedVehicleId` changes -- cheap (one pass
    /// over the BEV grid, no ML), so doing it inline in `body` rather than
    /// caching is simpler and fine at this grid size.
    private var bevDisplayImage: UIImage {
        BevRenderer.render(occludedBev: result.occludedBev, instBev: result.instBev,
                          visibleBev: result.visibleBev, width: result.bevWidth, height: result.bevHeight,
                          selectedId: selectedVehicleId)
    }

    private var bevCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Bird's-eye view").font(.subheadline).fontWeight(.semibold)
            Image(uiImage: bevDisplayImage)
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 280)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            Text(selectedVehicleId == nil
                 ? "Green = road, unoccluded. Colour = occluded road attributed to that vehicle. "
                   + "White = occluded road no detection claimed. Tap a vehicle below to highlight its share."
                 : "Magenta = the selected vehicle's occluded share. Grey = other vehicles'.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Vehicles

    private var vehiclesSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Occluded by").font(.subheadline).fontWeight(.semibold)

            let attributed = result.vehicles.filter { $0.areaM2 > 0 }
            if attributed.isEmpty {
                Text("No occluded area was attributed to a specific vehicle.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                VStack(spacing: 0) {
                    ForEach(attributed.sorted { $0.areaM2 > $1.areaM2 }) { vehicle in
                        vehicleRow(vehicle)
                        if vehicle.id != attributed.last?.id {
                            Divider()
                        }
                    }
                }
                .background(RoundedRectangle(cornerRadius: 16).fill(.thinMaterial))
            }

            if result.unattributedAreaM2 > 0 {
                Text(String(format: "+ %.2f m² not attributed to a specific vehicle", result.unattributedAreaM2))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func vehicleRow(_ vehicle: VehicleResult) -> some View {
        let isSelected = selectedVehicleId == vehicle.id
        return HStack(spacing: 12) {
            Circle().fill(vehicle.color).frame(width: 12, height: 12)
            VStack(alignment: .leading, spacing: 2) {
                Text(vehicle.label.capitalized).font(.callout).fontWeight(.medium)
                Text(vehicle.isDetected
                     ? String(format: "detected, %.0f%% confidence", vehicle.score * 100)
                     : "inferred from semantic segmentation")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Text(String(format: "%.2f m²", vehicle.areaM2)).font(.callout).fontWeight(.semibold)
                if vehicle.widthMaxPct > 0 {
                    Text(String(format: "%.0f%% width blocked", vehicle.widthMaxPct))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            Image(systemName: isSelected ? "checkmark.circle.fill" : "chevron.right")
                .foregroundStyle(isSelected ? Color.accentColor : Color.secondary.opacity(0.4))
                .imageScale(.small)
        }
        .padding()
        .background(isSelected ? Color.accentColor.opacity(0.12) : Color.clear)
        .contentShape(Rectangle())
        .onTapGesture {
            selectedVehicleId = isSelected ? nil : vehicle.id
        }
    }

    // MARK: - Technical details

    private var technicalDetails: some View {
        DisclosureGroup("Technical details") {
            VStack(alignment: .leading, spacing: 12) {
                Text("Semantic segmentation").font(.caption).foregroundStyle(.secondary)
                Image(uiImage: result.overlayImage)
                    .resizable().scaledToFit().frame(maxHeight: 180)
                    .clipShape(RoundedRectangle(cornerRadius: 8))

                if let depthImage = result.depthImage {
                    Text("Depth").font(.caption).foregroundStyle(.secondary)
                    Image(uiImage: depthImage)
                        .resizable().scaledToFit().frame(maxHeight: 180)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }

                if let vehicleBoxesImage = result.vehicleBoxesImage {
                    Text("Vehicle detections").font(.caption).foregroundStyle(.secondary)
                    Image(uiImage: vehicleBoxesImage)
                        .resizable().scaledToFit().frame(maxHeight: 180)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }

                Text(result.technicalSummary)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            .padding(.top, 8)
        }
        .font(.subheadline)
    }
}
