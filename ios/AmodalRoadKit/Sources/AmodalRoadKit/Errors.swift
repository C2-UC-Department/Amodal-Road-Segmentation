import simd

public enum AmodalRoadKitError: Error, CustomStringConvertible {
    /// Mirrors the `ValueError` raised by `bev.homography_bev_to_image` in
    /// src/bev.py for a plane that fails `PlaneGeometry.planeIsUsable`.
    case unusablePlane(n: SIMD3<Double>)

    public var description: String {
        switch self {
        case .unusablePlane(let n):
            return "plane normal \(n) is not a usable ground plane " +
                   "(see PlaneGeometry.planeIsUsable)"
        }
    }
}
