import XCTest
import simd
@testable import AmodalRoadKit

/// Golden values from the REAL `geometry.unproject` / `geometry.derive_ground_fields`
/// (src/geometry.py), on a small 8x6 synthetic depth map (including a NaN and
/// an Inf value, to exercise the clamp) with K sized to match the small image
/// (so some rays genuinely hit the ground plane -- an arbitrary K far from the
/// image content would give an all-invalid, uninformative test, as an earlier
/// attempt at this fixture found).
final class GeometryGoldenValueTests: XCTestCase {
    let width = 8, height = 6
    let K = Homography.pinholeK(fx: 20.0, fy: 20.0, cx: 4.0, cy: 2.0)
    let n = SIMD3<Double>(0.03639823112352753, 0.6005708135382043, -0.07279646224705506)

    func makeDepth() -> [Float] {
        var depth = [Float](repeating: 0, count: width * height)
        for i in 0..<depth.count {
            depth[i] = 2.0 + Float(i) * (18.0 / Float(depth.count - 1))
        }
        depth[2 * width + 3] = .nan
        depth[4 * width + 5] = .infinity
        return depth
    }

    func testUnprojectMatchesPython() {
        let depth = makeDepth()
        let Q = Unproject.unproject(depth: depth, width: width, height: height, K: K)
        let expected = GoldenData.unprojectFlat
        XCTAssertEqual(Q.count * 3, expected.count)
        for i in 0..<Q.count {
            XCTAssertEqual(Q[i].x, expected[i * 3 + 0], accuracy: 1e-4, "Q[\(i)].x")
            XCTAssertEqual(Q[i].y, expected[i * 3 + 1], accuracy: 1e-4, "Q[\(i)].y")
            XCTAssertEqual(Q[i].z, expected[i * 3 + 2], accuracy: 1e-4, "Q[\(i)].z")
        }
    }

    func testDeriveGroundFieldsMatchesPython() {
        let depth = makeDepth()
        let (G, h, gvalid) = GroundFields.derive(depth: depth, n: n, K: K,
                                                 width: width, height: height,
                                                 maxGroundDistM: 60.0)
        let expectedG = GoldenData.groundFieldsGFlat
        let expectedH = GoldenData.groundFieldsHFlat
        let expectedValid = GoldenData.groundFieldsGvalidFlat

        XCTAssertEqual(G.count * 3, expectedG.count)
        XCTAssertEqual(h.count, expectedH.count)
        XCTAssertEqual(gvalid.count, expectedValid.count)

        for i in 0..<G.count {
            XCTAssertEqual(gvalid[i], expectedValid[i] == 1, "gvalid[\(i)]")
            XCTAssertEqual(G[i].x, expectedG[i * 3 + 0], accuracy: 1e-3, "G[\(i)].x")
            XCTAssertEqual(G[i].y, expectedG[i * 3 + 1], accuracy: 1e-3, "G[\(i)].y")
            XCTAssertEqual(G[i].z, expectedG[i * 3 + 2], accuracy: 1e-3, "G[\(i)].z")
            XCTAssertEqual(h[i], expectedH[i], accuracy: 1e-3, "h[\(i)]")
        }

        // At least one cell must actually be valid, or this fixture would
        // silently be testing nothing meaningful (see the file header).
        XCTAssertTrue(gvalid.contains(true))
    }

    func testDeriveGroundFieldsHandlesNilPlane() {
        let depth = makeDepth()
        let (G, h, gvalid) = GroundFields.derive(depth: depth, n: nil, K: K,
                                                  width: width, height: height)
        XCTAssertTrue(G.allSatisfy { $0 == .zero })
        XCTAssertTrue(h.allSatisfy { $0 == 0 })
        XCTAssertTrue(gvalid.allSatisfy { $0 == false })
    }
}
