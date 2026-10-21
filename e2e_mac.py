# e2e_mac.py: run our Core ML encoder and Core ML decoder together on a photo, compare with Meta's PyTorch pipeline at each step,
# and draw both sets of 2D keypoints on the photo
# run: python -u e2e_mac.py <photo> <path to sam3d_backbone.mlpackage>   (writes e2e_keypoints.png)
import sys, os, types, inspect
import numpy as np
import torch
import coremltools as ct
from PIL import Image, ImageDraw

# --- Meta's inference code calls .cuda(): point it at the CPU (same shims as test_decoder.py), and fake pyrender ---
def _cpu(x):
    if isinstance(x, str) and x.startswith("cuda"): return "cpu"
    if isinstance(x, torch.device) and x.type == "cuda": return torch.device("cpu")
    return x

_t_to, _m_to = torch.Tensor.to, torch.nn.Module.to
torch.Tensor.to = lambda self, *a, **k: _t_to(self, *[_cpu(x) for x in a], **{n: _cpu(v) for n, v in k.items()})
torch.nn.Module.to = lambda self, *a, **k: _m_to(self, *[_cpu(x) for x in a], **{n: _cpu(v) for n, v in k.items()})
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self

class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})

sys.modules["pyrender"] = _Stub("pyrender")

import sam_3d_body
import sam_3d_body.sam_3d_body_estimator as est_mod
if hasattr(est_mod, "recursive_to"):
    _orig_rt = est_mod.recursive_to
    est_mod.recursive_to = lambda x, dev: _orig_rt(x, "cpu")

photo, enc_path = sys.argv[1], sys.argv[2]
base = os.environ.get("SAM3D_CKPT_DIR", "checkpoints/sam-3d-body-dinov3")
model, cfg = sam_3d_body.load_sam_3d_body(checkpoint_path=f"{base}/model.ckpt", device="cpu", mhr_path=f"{base}/assets/mhr_model.pt")
model = model.float().eval()
model.backbone_dtype = torch.float32  # same as test_decoder.py, so the reference runs in float32 like the conversion

# --- reference: Meta's PyTorch pipeline, body decoder only; catch the encoder's input/output and the decoder's inputs/output ---
caught = {}
def _enc_hook(mod, args, kwargs, out):
    if "encoder" not in caught:
        caught["encoder"] = (args[0].detach().clone(), (out[-1] if isinstance(out, tuple) else out).detach().clone())
model.backbone.register_forward_hook(_enc_hook, with_kwargs=True)
_orig_fd = model.forward_decoder
def _catch(*a, **kw):
    out = _orig_fd(*a, **kw)
    if "decoder" not in caught:
        caught["decoder"] = (inspect.signature(_orig_fd).bind(*a, **kw).arguments, out)
    return out
model.forward_decoder = _catch

estimator = sam_3d_body.SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
with torch.no_grad():
    estimator.process_one_image(photo, inference_type="body")

enc_in, enc_out_pt = caught["encoder"]
args, (_, pose_outs) = caught["decoder"]
batch = args["batch"]
ref3 = pose_outs[-1]["pred_keypoints_3d"].numpy()  # PyTorch's final keypoints: 3D, and 2D in photo pixels
ref2 = pose_outs[-1]["pred_keypoints_2d"].numpy()
print("E2E encoder input", tuple(enc_in.shape), "| crop", tuple(batch["img"].shape), "max value", float(batch["img"].max()))

# --- step 1: our Core ML encoder on the same crop (as an RGB image; the model normalizes it itself) ---
crop = batch["img"][0, 0].permute(1, 2, 0).numpy()  # (512, 512, 3), not yet normalized
crop = crop * 255.0 if crop.max() <= 1.0 else crop
crop_img = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8), "RGB")
enc = ct.models.MLModel(enc_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
feats = enc.predict({"image": crop_img})["features"].astype(np.float32)
d = np.abs(feats - enc_out_pt.numpy())
print(f"E2E encoder    Core ML vs PyTorch: max diff {d.max():.4f}, mean diff {d.mean():.4f}, PyTorch max abs {np.abs(enc_out_pt.numpy()).max():.4f}")

# --- step 2: our Core ML decoder, once with PyTorch's features (decoder alone) and once with our encoder's (the full chain) ---
dec = ct.models.MLModel("sam3d_decoder.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
names = [i.name for i in dec.get_spec().description.input]
def feed(features):
    f = {}
    for n in names:
        v = features if n == "image_embeddings" else (args[n] if n in args else batch[n])
        f[n] = np.asarray(v.numpy() if torch.is_tensor(v) else v, dtype=np.float32)
    return f
for tag, features in (("decoder only ", enc_out_pt.numpy()), ("full chain   ", feats)):
    res = dec.predict(feed(features))
    print(f"E2E {tag} Core ML vs PyTorch: 3D max diff {np.abs(res['keypoints3d'] - ref3).max():.4f}   2D max diff {np.abs(res['keypoints2d'] - ref2).max():.1f} px")
chain2 = res["keypoints2d"]  # the full chain's 2D keypoints, from the last loop pass

# --- draw: PyTorch keypoints in green, our Core ML chain in orange ---
img = Image.open(photo).convert("RGB")
draw = ImageDraw.Draw(img)
for pts, color in ((ref2[0], (40, 200, 90)), (chain2[0], (235, 104, 52))):
    for x, y in pts:
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=color, width=2)
img.save("e2e_keypoints.png")
print("E2E drew e2e_keypoints.png (green: PyTorch, orange: Core ML encoder + decoder)")
