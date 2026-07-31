// swift-tools-version: 5.9
import PackageDescription

/// AmodalRoadKit: the iOS/Core ML migration's Swift-side geometry port.
///
/// Phase 1 of the migration plan (see /Users/biru/.claude/plans -- or ask for
/// the plan doc) calls for the pure-computation pieces (BEV homography,
/// warping, RANSAC, ground-manifold math) to be ported as self-contained
/// packages, independently of the mobile segmentation models (Phase 2) and
/// the Core ML geometry stack (Phase 3). This package is that BEV/geometry
/// port, mirroring src/bev.py and (later) src/geometry.py's pure-math
/// functions -- not the cv2-only rasterization primitives, which get their
/// own native implementations as noted in the plan.
let package = Package(
    name: "AmodalRoadKit",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "AmodalRoadKit", targets: ["AmodalRoadKit"]),
    ],
    targets: [
        .target(name: "AmodalRoadKit"),
        .testTarget(name: "AmodalRoadKitTests", dependencies: ["AmodalRoadKit"],
                   // DepthAnythingV2Small.mlpackage is NOT declared as a
                   // resource: at ~47MB it follows the "large models stay
                   // gitignored, regenerate locally" policy (see .gitignore),
                   // so it won't exist on a fresh checkout -- an SwiftPM
                   // `resources:` entry would fail the build outright when
                   // absent. DepthModelTests locates it directly via
                   // `#filePath` instead and skips (not fails) if missing.
                   // `exclude:` silences the "unhandled file" warning when it
                   // IS present locally; excluding a path that doesn't exist
                   // is a no-op, so this is safe either way.
                   exclude: ["Resources/DepthAnythingV2Small.mlpackage"],
                   resources: [.copy("Resources/OFRSNetExport.mlpackage"),
                              .copy("Resources/MobileSemanticNet.mlpackage")]),
    ]
)
