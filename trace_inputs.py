import sys, types, os, inspect, traceback
import numpy as np
import torch

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
    _f = getattr(torch, _n)  # tensor factories that take device="cuda"
    setattr(torch, _n, (lambda f: lambda *a, **k: f(*a, **{n: (_cpu(v) if n == "device" else v) for n, v in k.items()}))(_f))

class _Stub(types.ModuleType):  # fake pyrender, same as before
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})

sys.modules["pyrender"] = _Stub("pyrender")

import sam_3d_body
import sam_3d_body.sam_3d_body_estimator as est_mod

def d(x):  # short description of a value: shapes for arrays, keys for dicts
    if torch.is_tensor(x) or isinstance(x, np.ndarray):
        return f"{type(x).__name__}{tuple(x.shape)} {x.dtype}"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {d(v)}" for k, v in x.items()) + "}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[" + ", ".join(d(v) for v in list(x)[:4]) + (", ..." if len(x) > 4 else "") + "]"
    if isinstance(x, (int, float, bool, str)) or x is None:
        return repr(x)
    return type(x).__name__

log = []
def note(label, args, kwargs, out):  # one entry per call, in the order calls finish
    log.append(f"{label}\n   in : {d(list(args))} {d(kwargs) if kwargs else ''}\n   out: {d(out)}"[:1500])

base = os.environ.get("SAM3D_CKPT_DIR", "checkpoints/sam-3d-body-dinov3")
model, cfg = sam_3d_body.load_sam_3d_body(
    checkpoint_path=f"{base}/model.ckpt", device="cpu", mhr_path=f"{base}/assets/mhr_model.pt")
model.eval()

# log every call into the main parts
for name in ["backbone", "prompt_encoder", "decoder", "decoder_hand",
             "head_pose", "head_pose_hand", "head_camera", "head_camera_hand"]:
    m = getattr(model, name, None)
    if m is not None:
        m.register_forward_hook(lambda mod, a, kw, o, n=name: note(f"module {n}", a, kw, o), with_kwargs=True)

# log every public method of the model itself, to see how the parts are chained
for name, fn in list(type(model).__dict__.items()):
    if callable(fn) and not name.startswith("_") and name != "forward":
        def wrapper(*a, _o=getattr(model, name), _n=name, **kw):
            r = _o(*a, **kw)
            note(f"method {_n}", a, kw, r)
            return r
        setattr(model, name, wrapper)

if hasattr(est_mod, "recursive_to"):  # the estimator may move data to "cuda"; point it at the CPU
    _orig = est_mod.recursive_to
    est_mod.recursive_to = lambda x, dev: _orig(x, "cpu")

print("estimator args:", inspect.signature(sam_3d_body.SAM3DBodyEstimator.__init__))
try:
    estimator = sam_3d_body.SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    with torch.no_grad():
        result = estimator.process_one_image(sys.argv[1])
    print("RESULT:", d(result))
except Exception:
    traceback.print_exc()
print("\n".join(log))