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

/// `round(x, 2)` -- Python's real per-decimal-place rounding, which
/// correctly-rounds the actual binary value directly. Found the hard way
/// (`BevValidityTests`) that the seemingly-equivalent `pythonRound(x * 100)
/// / 100` is NOT the same thing: multiplying by 100 first can itself
/// introduce floating-point error that lands EXACTLY on `.5`, creating an
/// artificial tie that `.toNearestOrEven` then resolves "correctly" for a
/// tie that was never really there -- concretely, `26.774999999999998578`
/// (definitively closer to `26.77`) times 100 rounds to the exact float
/// `2677.5`, which then ties-to-even up to `2678`, giving the wrong `26.78`.
/// `String(format: "%.2f", x)` performs the decimal rounding directly on
/// the binary value (the same class of algorithm CPython's `round()` uses),
/// without that intermediate multiplication, and was verified to give the
/// same `26.77` Python does on this exact value.
func pythonRound2(_ x: Double) -> Double {
    Double(String(format: "%.2f", x)) ?? x
}

/// `round(x, 3)` -- same rationale as `pythonRound2`, one more decimal place
/// (used by `Attribution.attribute`, which mirrors `disturbance.attribute`'s
/// `round(a, 3)`).
func pythonRound3(_ x: Double) -> Double {
    Double(String(format: "%.3f", x)) ?? x
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
