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
    static func url(forCompiledModel name: String) throws -> URL {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw ModelBundleError.resourceNotFound("\(name).mlmodelc")
        }
        return url
    }

    static func loadOFRSNet() throws -> OFRSNetModel {
        try OFRSNetModel(contentsOf: url(forCompiledModel: "OFRSNetExport"), numClasses: 11)
    }

    static func loadDepth() throws -> DepthModel {
        try DepthModel(contentsOf: url(forCompiledModel: "DepthAnythingV2Small"))
    }

    static func loadMobileSemantic() throws -> MobileSemanticModel {
        try MobileSemanticModel(contentsOf: url(forCompiledModel: "MobileSemanticNet"))
    }

    /// Vehicle-instance YOLOv8n-seg. No AmodalRoadKit wrapper exists yet --
    /// NMS + mask-prototype decode aren't ported to Swift yet (see the
    /// migration plan's Phase 4 open items) -- so this returns the raw
    /// `MLModel` rather than a typed wrapper. Callers that just want to
    /// prove the model loads/runs (this app's current debug screen) can use
    /// it directly; a real typed wrapper should replace this once the
    /// postprocessing is written.
    static func loadVehicleInstanceRawModel() throws -> MLModel {
        let compiledURL = try url(forCompiledModel: "VehicleInstanceYOLO")
        return try MLModel(contentsOf: compiledURL)
    }
}
