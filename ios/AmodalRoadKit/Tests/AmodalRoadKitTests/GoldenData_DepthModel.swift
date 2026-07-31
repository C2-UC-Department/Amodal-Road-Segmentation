// Auto-generated from the REAL depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf
// model (via transformers.AutoModelForDepthEstimation, with the same
// position-embedding-interpolation patch tools/coreml_export_depth.py
// applies -- NOT going through the real DPTImageProcessor's resize, since
// ImagePreprocessing.swift deliberately doesn't replicate that algorithm;
// see its header). Deterministic synthetic pixel_values -- a smooth
// sin/cos field, Swift-reproducible without RNG or a real image file -- at
// the model's fixed 518x518 trace resolution. See DepthModelTests.swift for
// the exact Swift-side reconstruction, which must match this formula
// exactly. Generation script:
//
//   .venv/bin/python -c "
//   import torch, sys; sys.path.insert(0,'.')
//   import config
//   from transformers import AutoModelForDepthEstimation
//   from tools.coreml_export_depth import patch_position_embedding_interpolation, DepthWrapper
//   h, w = 518, 518
//   model = AutoModelForDepthEstimation.from_pretrained(config.DEPTH_MODEL_ID).eval()
//   patch_position_embedding_interpolation(model, h, w)
//   wrapper = DepthWrapper(model).eval()
//   ii = torch.arange(h).view(h,1,1).expand(h,w,3).float()
//   jj = torch.arange(w).view(1,w,1).expand(h,w,3).float()
//   cc = torch.arange(3).view(1,1,3).expand(h,w,3).float()
//   px = (torch.sin(ii*0.01 + cc) * 0.5 + torch.cos(jj*0.013 - cc*0.7) * 0.5)
//   px = px.permute(2,0,1).unsqueeze(0).contiguous()
//   with torch.no_grad(): out = wrapper(px).numpy()
//   "
//
// The .mlpackage this test runs against (Resources/DepthAnythingV2Small.mlpackage,
// gitignored -- see Package.swift) was exported at this same 518x518
// resolution via:
//   .venv/bin/python -m tools.coreml_export_depth --out ios/AmodalRoadKit/Tests/AmodalRoadKitTests/Resources/DepthAnythingV2Small.mlpackage
extension GoldenData {
    static let depthModelSize = 518

    /// (row, col, expected depth in metres from the real PyTorch model), a
    /// 6x6 grid spaced away from the image edges.
    static let depthModelSamples: [(Int, Int, Double)] = [
        (20, 20, 3.30794620513916), (20, 115, 3.3057198524475098), (20, 211, 3.1577601432800293),
        (20, 306, 3.1402370929718018), (20, 402, 3.0533390045166016), (20, 498, 3.103423595428467),
        (115, 20, 3.277454376220703), (115, 115, 3.2548980712890625), (115, 211, 3.167475700378418),
        (115, 306, 3.0648765563964844), (115, 402, 3.005528450012207), (115, 498, 3.0793585777282715),
        (211, 20, 3.285311460494995), (211, 115, 3.1653218269348145), (211, 211, 3.1294240951538086),
        (211, 306, 3.0320262908935547), (211, 402, 2.960723638534546), (211, 498, 3.0412375926971436),
        (306, 20, 3.178112506866455), (306, 115, 3.097836494445801), (306, 211, 3.0549099445343018),
        (306, 306, 2.9859275817871094), (306, 402, 2.8748669624328613), (306, 498, 2.9329447746276855),
        (402, 20, 3.171858310699463), (402, 115, 3.1065850257873535), (402, 211, 3.0129761695861816),
        (402, 306, 2.9309325218200684), (402, 402, 2.890305280685425), (402, 498, 2.9863502979278564),
        (498, 20, 3.262080669403076), (498, 115, 3.1501832008361816), (498, 211, 3.013002872467041),
        (498, 306, 2.960934638977051), (498, 402, 2.996786117553711), (498, 498, 3.0817108154296875),
    ]
}
