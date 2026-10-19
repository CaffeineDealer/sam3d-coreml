import sys, types, os, time
import numpy as np
import torch
import cv2

def _cpu(x):  # turn any "cuda" device into "cpu"
    if isinstance(x, str) and x.startswith("cuda"): return "cpu"
    if isinstance(x, torch.device) and x.type == "cuda": return torch.device("cpu")
    return x

_t_to, _m_to = torch.Tensor.to, torch.nn.Module.to
torch.Tensor.to = lambda self, *a, **k: _t_to(self, *[_cpu(x) for x in a], **{n: _cpu(v) for n, v in k.items()})
torch.nn.Module.to = lambda self, *a, **k: _m_to(self, *[_cpu(x) for x in a], **{n: _cpu(v) for n, v in k.items()})
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
for _n in ("zeros", "ones", "tensor", "arange", "full", "empty", "linspace", "eye", "rand", "randn", "as_tensor", "zeros_like", "ones_like"):
    _f = getattr(torch, _n)
    setattr(torch, _n, (lambda f: lambda *a, **k: f(*a, **{n: (_cpu(v) if n == "device" else v) for n, v in k.items()}))(_f))

class _Stub(types.ModuleType):  # fake pyrender, same as before
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})

sys.modules["pyrender"] = _Stub("pyrender")

import sam_3d_body
import sam_3d_body.sam_3d_body_estimator as est_mod

if hasattr(est_mod, "recursive_to"):  # the estimator may move data to "cuda"; point it at the CPU
    _orig = est_mod.recursive_to
    est_mod.recursive_to = lambda x, dev: _orig(x, "cpu")

base = os.environ.get("SAM3D_CKPT_DIR", "checkpoints/sam-3d-body-dinov3")
model, cfg = sam_3d_body.load_sam_3d_body(
    checkpoint_path=f"{base}/model.ckpt", device="cpu", mhr_path=f"{base}/assets/mhr_model.pt")
model.eval()
estimator = sam_3d_body.SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)

path = sys.argv[1]
t0 = time.time()
with torch.no_grad():
    people = estimator.process_one_image(path)
print(f"people found: {len(people)}, time: {time.time() - t0:.1f} s")

img = cv2.imread(path)
h, w = img.shape[:2]
r = max(3, min(h, w) // 250)  # dot size scales with the image

for p in people:
    kp = p["pred_keypoints_2d"].cpu().numpy()  # (70, 2)
    print("keypoints x range:", kp[:, 0].min(), kp[:, 0].max(), "| y range:", kp[:, 1].min(), kp[:, 1].max(), "| image:", w, "x", h)
    x0, y0, x1, y1 = [int(v) for v in np.asarray(p["bbox"]).reshape(-1)[:4]]
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 0), r)  # person box
    for i, (x, y) in enumerate(kp):
        cv2.circle(img, (int(x), int(y)), r, (0, 255, 0), -1)  # green dot
        cv2.putText(img, str(i), (int(x) + r, int(y) - r), cv2.FONT_HERSHEY_SIMPLEX, r / 6, (0, 0, 255), max(1, r // 3))  # index

scale = 1600 / max(h, w)
cv2.imwrite("sam3d_out.jpg", cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img)
print("saved sam3d_out.jpg")