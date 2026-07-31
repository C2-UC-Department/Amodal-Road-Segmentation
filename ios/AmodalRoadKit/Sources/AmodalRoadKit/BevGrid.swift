import Foundation

/// Python's built-in `round()` uses round-half-to-even (banker's rounding);
/// Swift's `Double.rounded()` defaults to round-half-away-from-zero. These
/// disagree exactly on ties (e.g. `round(734.5)`: Python -> 734, Swift's
/// default -> 735) -- caught by GoldenValueTests.testWorldToPixel, which is
/// why every rounding call in this file goes through this helper instead of
/// the bare `.rounded()`, rather than fixing the one call site the test
/// happened to exercise.
func pythonRound(_ x: Double) -> Double {
    x.rounded(.toNearestOrEven)
}

/// A metric top-down raster: `ppm` pixels per metre over `[x, z]` extents.
///
/// Direct port of `bev.BevGrid` in src/bev.py -- see that file for the full
/// derivation. Convention (unchanged from Python): X is lateral (camera at
/// X=0, +X right), Z is forward. Row v=0 is the FAR edge (z_max), so the
/// raster reads like a map with the camera at the bottom.
public struct BevGrid: Equatable {
    public let ppm: Double
    public let xMin: Double
    public let xMax: Double
    public let zMin: Double
    public let zMax: Double

    public init(ppm: Double, xMin: Double, xMax: Double, zMin: Double, zMax: Double) {
        self.ppm = ppm
        self.xMin = xMin
        self.xMax = xMax
        self.zMin = zMin
        self.zMax = zMax
    }

    public var width: Int {
        max(1, Int(pythonRound((xMax - xMin) * ppm)))
    }

    public var height: Int {
        max(1, Int(pythonRound((zMax - zMin) * ppm)))
    }

    public var cellAreaM2: Double {
        1.0 / (ppm * ppm)
    }

    /// Ground `(x, z)` metres -> `(u, v)` BEV pixel. Inverse of `sMatrix`.
    public func worldToPixel(x: Double, z: Double) -> (u: Int, v: Int) {
        let u = (x - xMin) * ppm - 0.5
        let v = (zMax - z) * ppm - 0.5
        return (Int(pythonRound(u)), Int(pythonRound(v)))
    }
}
