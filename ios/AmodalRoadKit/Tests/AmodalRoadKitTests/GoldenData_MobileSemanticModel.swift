// Auto-generated from the REAL src/mobile_semantic/model.py::MobileSemanticNet
// (loaded from checkpoints/mobile_semantic_best.pt, NOT the Core ML
// conversion) on a deterministic synthetic 512x512 RGB image -- a smooth
// sin/cos field in [0,1] (this model does its own ImageNet normalization
// internally, see MobileSemanticNet.forward, so the input here is plain
// RGB, unlike DepthModel's already-normalized pixel_values). See
// MobileSemanticModelTests.swift for the exact Swift-side reconstruction.
// Generation script:
//
//   .venv/bin/python -c "
//   import numpy as np, torch, sys; sys.path.insert(0,'.')
//   import config
//   from src.mobile_semantic.model import MobileSemanticNet
//   ckpt = torch.load(config.CKPT_DIR / 'mobile_semantic_best.pt', map_location='cpu')
//   model = MobileSemanticNet(pretrained_backbone=False)
//   model.load_state_dict(ckpt['model']); model.eval()
//   size = 512
//   ii = torch.arange(size).view(size,1,1).expand(size,size,3).float()
//   jj = torch.arange(size).view(1,size,1).expand(size,size,3).float()
//   cc = torch.arange(3).view(1,1,3).expand(size,size,3).float()
//   px = (torch.sin(ii*0.01 + cc) * 0.5 + torch.cos(jj*0.013 - cc*0.7) * 0.5) * 0.5 + 0.5
//   px = px.permute(2,0,1).unsqueeze(0).contiguous()
//   with torch.no_grad(): out = model(px).numpy()
//   cls = out[0].argmax(0)
//   "
//
// The .mlpackage this test runs against (Resources/MobileSemanticNet.mlpackage,
// git-tracked -- small enough to follow OFRSNetExport.mlpackage's precedent,
// not DepthAnythingV2Small's) was exported at this same 512x512 resolution via:
//   .venv/bin/python -m tools.coreml_export_mobile_semantic \
//       --out ios/AmodalRoadKit/Tests/AmodalRoadKitTests/Resources/MobileSemanticNet.mlpackage
extension GoldenData {
    static let mobileSemanticSize = 512

    /// (row, col, expected argmax OFRS-11 class) from the real PyTorch model.
    static let mobileSemanticSamples: [(Int, Int, Int)] = [
        (20, 20, 10), (20, 114, 10), (20, 208, 10), (20, 303, 10), (20, 397, 10), (20, 492, 10),
        (114, 20, 10), (114, 114, 10), (114, 208, 10), (114, 303, 8), (114, 397, 7), (114, 492, 10),
        (208, 20, 8), (208, 114, 8), (208, 208, 8), (208, 303, 8), (208, 397, 8), (208, 492, 10),
        (303, 20, 8), (303, 114, 8), (303, 208, 8), (303, 303, 8), (303, 397, 8), (303, 492, 8),
        (397, 20, 10), (397, 114, 10), (397, 208, 10), (397, 303, 10), (397, 397, 8), (397, 492, 10),
        (492, 20, 10), (492, 114, 10), (492, 208, 10), (492, 303, 0), (492, 397, 8), (492, 492, 10),
    ]
}
