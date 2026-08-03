//
//  ModelBundle.swift
//  Parking Disturbance
//
//  Resolves the four Core ML models bundled into this app target (see
//  Models/ -- Xcode auto-compiles every .mlpackage placed there to
//  .mlmodelc and copies it into the built app's Bundle.main root; confirmed
//  by inspecting the built .app directly, not assumed) into the AmodalRoadKit
//  wrapper types built and golden-value tested in ios/AmodalRoadKit.
//
//  This is the Bundle.main-based counterpart to those wrappers' test-only
//  #filePath/Bundle.module lookups -- same models, same wrapper classes,
//  different resolution strategy because a running app and an SPM test
//  target ship resources differently.

import CoreML
import Foundation
import AmodalRoadKit

enum ModelBundleError: Error {
    case resourceNotFound(String)
}

enum ModelBundle {
    /// `.cpuOnly` for every model, not just the one caught in the act: a
    /// REAL bug, found by actually looking at output rather than trusting a
    /// successful build --  `MobileSemanticNet` under default (ANE/GPU-
    /// eligible) compute-unit dispatch on the iOS SIMULATOR silently
    /// classified 100% of a real photo as "road" (`DepthAnythingV2Small`
    /// showed the same symptom, solid-black/degenerate depth), while the
    /// identical `.mlpackage` gave correct, varied, real output both via
    /// `swift run` on macOS AND via `.cpuOnly` right here on the Simulator.
    /// This point at Simulator-specific ANE emulation, not a bug in this
    /// app's own code (`ImageBytes`/`ImagePreprocessing` were audited and
    /// are not the cause -- confirmed by this exact fix resolving it with
    /// no other change). `.cpuOnly` everywhere is the safe default until
    /// this can be re-tested on a REAL device (not available in this
    /// environment) -- Simulator's Neural Engine path is documented by
    /// Apple as an approximation of real ANE hardware, not a guarantee of
    /// identical numerics, and this is exactly the kind of divergence that
    /// warning exists for. Trades inference speed for correctness; revisit
    /// (per-model, not blanket) once real-device testing is possible.
    private static func cpuOnlyConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        return config
    }

    static func url(forCompiledModel name: String) throws -> URL {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw ModelBundleError.resourceNotFound("\(name).mlmodelc")
        }
        return url
    }

    static func loadOFRSNet() throws -> OFRSNetModel {
        try OFRSNetModel(contentsOf: url(forCompiledModel: "OFRSNetExport"), numClasses: 11,
                        configuration: cpuOnlyConfiguration())
    }

    static func loadDepth() throws -> DepthModel {
        try DepthModel(contentsOf: url(forCompiledModel: "DepthAnythingV2Small"),
                      configuration: cpuOnlyConfiguration())
    }

    static func loadMobileSemantic() throws -> MobileSemanticModel {
        try MobileSemanticModel(contentsOf: url(forCompiledModel: "MobileSemanticNet"),
                               configuration: cpuOnlyConfiguration())
    }

    static func loadVehicleInstance() throws -> YOLOInstanceModel {
        try YOLOInstanceModel(contentsOf: url(forCompiledModel: "VehicleInstanceYOLO"),
                             configuration: cpuOnlyConfiguration())
    }
}
