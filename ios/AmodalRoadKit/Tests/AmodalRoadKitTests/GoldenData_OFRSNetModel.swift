// Auto-generated from the REAL src/ofrs/export.py::OFRSNetExport (loaded from
// checkpoints/ofrsnet_best.pt, NOT the Core ML conversion) on a deterministic
// 64x96 synthetic scene: building everywhere, road below h/3, a small vehicle
// blob, G a linear ground-footprint gradient, h a deterministic (non-RNG,
// Swift-reproducible) per-pixel formula, gvalid true below h/3 -- see
// OFRSNetModelTests.swift for the exact Swift-side reconstruction of this
// same scene, which must match bit-for-bit. Generation script kept inline
// here for anyone who needs to regenerate this fixture:
//
//   .venv/bin/python -c "
//   import numpy as np, torch, sys; sys.path.insert(0,'.')
//   import config
//   from src.ofrs.export import OFRSNetExport
//   from src.ofrs.model import OFRSNet
//   ckpt = torch.load('checkpoints/ofrsnet_best.pt', map_location='cpu')
//   model = OFRSNet(in_channels=config.OFRS_NUM_CLASSES, num_classes=2, use_geometry=True)
//   model.load_state_dict(ckpt['model']); model.eval()
//   export_model = OFRSNetExport.from_trained(model)
//   h, w = 64, 96
//   ROAD, VEHICLE, BUILDING = (config.OFRS_CLASSES.index(c) for c in ('road','vehicle','building'))
//   sem = np.full((h, w), BUILDING, np.int64)
//   sem[h//3:, :] = ROAD
//   sem[h//2:h//2+max(1,h//12), w//3:w//3+max(1,w//6)] = VEHICLE
//   x = torch.from_numpy(np.eye(config.OFRS_NUM_CLASSES, dtype=np.float32)[sem].transpose(2,0,1))[None]
//   G = torch.zeros(1,3,h,w)
//   G[0,2] = torch.linspace(2,30,h).view(h,1).expand(h,w)
//   G[0,0] = torch.linspace(-5,5,w).view(1,w).expand(h,w)
//   ii = torch.arange(h).view(h,1).expand(h,w).float()
//   jj = torch.arange(w).view(1,w).expand(h,w).float()
//   hh = (0.1*(((ii.long()%7).float()-3.0)/3.0)) * (((jj.long()%5).float()-2.0)/2.0)
//   hh = hh.view(1,1,h,w)
//   gvalid = torch.zeros(1,1,h,w); gvalid[0,0,h//3:,:] = 1.0
//   geo = {'G': G, 'h': hh, 'gvalid': gvalid > 0.5, 'valid': torch.ones(1) > 0.5}
//   with torch.no_grad(): out = export_model(x, geo).numpy()
//   road = out[0,1] > out[0,0]
//   packed = np.packbits(road.astype(np.uint8).flatten())
//   import base64; print(base64.b64encode(packed.tobytes()).decode())
//   "
//
// The .mlpackage bundled as a test resource (Resources/OFRSNetExport.mlpackage)
// was exported at this exact 64x96 resolution via:
//   .venv/bin/python -m tools.coreml_export_ofrsnet --height 64 --width 96 \
//       --out ios/AmodalRoadKit/Tests/AmodalRoadKitTests/Resources/OFRSNetExport.mlpackage --no-real-geo
extension GoldenData {
    static let ofrsRoadMaskHeight = 64
    static let ofrsRoadMaskWidth = 96

    /// packbits-packed (row-major, MSB-first, same as numpy default),
    /// base64-encoded road/not-road mask from the real PyTorch model.
    static let ofrsRoadMaskPackedBase64 =
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/////////////4AP/////////////+AH/////////////+" +
        "AD/////////////+AD/////////////+AB/////////////+AA/////////////+AAf////////////+AAP////////////+" +
        "AAD////////////+AAAP///////////+AAAA///////////+AAAAP//////////+AAAAD//////////+AAAABD/////////+" +
        "AAAAAA/////////+AAAAAB/////////+AAAADv/////////+AAAAD//////////+AAAAD//////////+AAAAD//////////+" +
        "AAAAH//////////+AAAAP//////////+AAAAf//////////+QAAAf///////////QAAA///////////+YAAA///////////+" +
        "8AAA///////////++AAB///////////+/gAP///////////+/8P////////////+///////////////+///////////////+" +
        "///////////////+///////////////+////////////////////////////////////////////////////////////////" +
        "///////////////+///////////////+///////////////+f//////////////+"

    /// logits.count at the exported resolution (2 * height * width), so
    /// OFRSNetModelTests can assert the Core ML output shape before ever
    /// touching the mask comparison.
    static let ofrsExpectedLogitsCount = 2 * ofrsRoadMaskHeight * ofrsRoadMaskWidth
}
