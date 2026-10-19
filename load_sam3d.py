import sys, types, inspect
import sam_3d_body  # the model package itself, without the notebook helpers
import inspect, torch
import os


class _Stub(types.ModuleType):  # fake pyrender: any attribute becomes an empty dummy class
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})

sys.modules["pyrender"] = _Stub("pyrender")  # we don't render, and its EGL library doesn't exist on macOS

names = [n for n in dir(sam_3d_body) if not n.startswith("_")]
print(names)  # what the package exposes
for n in names:
    if n.startswith("load"):
        print(n, inspect.signature(getattr(sam_3d_body, n)))  # loader arguments


base = os.environ.get("SAM3D_CKPT_DIR", "checkpoints/sam-3d-body-dinov3")  # set SAM3D_CKPT_DIR, or put the checkpoint in ./checkpoints

out = sam_3d_body.load_sam_3d_body(
    checkpoint_path=f"{base}/model.ckpt",
    device="cpu",
    mhr_path=f"{base}/assets/mhr_model.pt",
)
model = out[0] if isinstance(out, tuple) else out  # the loader may return (model, config)
model.eval()

print(type(model))
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
print([n for n, _ in model.named_children()])  # top-level parts
print(inspect.signature(model.forward))  # what the model takes as input

cfg = out[1] if isinstance(out, tuple) else None  # the loader may return (model, config)
bb = model.backbone
print(type(bb), inspect.signature(bb.forward))  # what the image encoder takes
keys = ("IMAGE_SIZE", "IMG_SIZE", "RESOLUTION", "PATCH", "BACKBONE", "INPUT")
print("\n".join(l for l in str(cfg).splitlines() if any(k in l.upper() for k in keys)))  # only size-related config lines

import coremltools as ct  # converts the traced PyTorch model to Core ML
import coremltools.converters.mil.frontend.torch.ops as _tops  # coremltools' op converters

def _cast_fixed(context, node, dtype, dtype_name):
    x = _tops._get_inputs(context, node, expected=1)[0]
    if x.can_be_folded_to_const():
        v = x.val
        v = v.item() if hasattr(v, "item") else v  # unwrap 1-element arrays before int()/float()/bool()
        res = _tops.mb.const(val=dtype(v), name=node.name)
    elif len(x.shape) > 0:
        x = _tops.mb.squeeze(x=x, name=node.name + "_item")
        res = _tops.mb.cast(x=x, dtype=dtype_name, name=node.name)
    else:
        res = _tops.mb.cast(x=x, dtype=dtype_name, name=node.name)
    context.add(res, node.name)

_tops._cast = _cast_fixed

class BackboneWrapper(torch.nn.Module):  # tracing needs a plain tensor out
    def __init__(self, b):
        super().__init__()
        self.b = b
    def forward(self, x):
        y = self.b(x)  # may return a tensor, tuple, list or dict
        if isinstance(y, dict):
            y = next(iter(y.values()))
        if isinstance(y, (tuple, list)):
            y = y[0]
        return y

wrapped = BackboneWrapper(bb).float().eval()  # weights are bfloat16; float32 matches the dummy input
dummy = torch.randn(1, 3, 512, 512)  # one 512x512 RGB image, as the config says
with torch.no_grad():
    print("output shape:", wrapped(dummy).shape)
    traced = torch.jit.trace(wrapped, dummy, strict=False)  # freeze into a graph

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(
    name="image",
    shape=(1, 3, 512, 512),
    color_layout=ct.colorlayout.RGB,
    scale=1 / (255 * 0.226),                                # ImageNet std, averaged
    bias=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],  # ImageNet mean / std
)],
    outputs=[ct.TensorType(name="features")],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS18,
)
mlmodel.save("sam3d_backbone.mlpackage")
print("saved")