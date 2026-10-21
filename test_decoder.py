# test_decoder.py: convert SAM 3D Body's whole body decoder pass to Core ML: 6 transformer layers, with head_pose,
# head_camera, the camera projection and the keypoint-token updates between layers. Reuses every patch from test_head_pose.py.
# run: python test_decoder.py <photo>   (writes sam3d_decoder.mlpackage)
import sys, os, inspect
import numpy as np
import torch

here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(here, "test_head_pose.py")).read().split("# --- load the model ---")[0])  # imports and patches 1-33

# --- decoder patches (head_pose's patches 1-33 come from test_head_pose.py above) ---
# --- patch 34: x.mT, the transpose of the last two dims (used in _full_to_crop on the crop transform) ---
@register_torch_op
def mt(context, node):
    x = _tops._get_inputs(context, node, expected=1)[0]
    perm = list(range(x.rank))
    perm[-2], perm[-1] = perm[-1], perm[-2]  # swap the last two dims
    context.add(mb.transpose(x=x, perm=perm, name=node.name))

# --- patch 35: a | b on bool masks (the invalid-keypoint mask in the keypoint-token update) ---
def _or_op(context, node):
    a, b = _tops._get_inputs(context, node, expected=2)
    if isinstance(a, Var) and isinstance(b, Var) and _tops.types.is_bool(a.dtype) and _tops.types.is_bool(b.dtype):
        context.add(mb.logical_or(x=a, y=b, name=node.name))  # element-wise or of two masks
        return
    if not isinstance(a, Var) and not isinstance(b, Var):  # two Python values known now
        context.add(mb.const(val=bool(a) or bool(b), name=node.name))
        return
    raise NotImplementedError(f"__or__ on {getattr(a, 'dtype', type(a))} and {getattr(b, 'dtype', type(b))}")

from coremltools.converters.mil.frontend.torch.utils import sanitize_op_kind, unify_inplace_and_functional

def _register(fn, raw_kind):  # register under the exact keys coremltools looks up: for "aten::__or__" (scan) and "__or__" (converter)
    for raw in (raw_kind, raw_kind.split("::")[-1]):
        _TORCH_OPS_REGISTRY.set_func_by_name(fn, unify_inplace_and_functional(sanitize_op_kind(raw)))

_register(_or_op, "aten::__or__")  # keys "__or_" and "or"

# --- Meta's inference code calls .cuda() and moves data to "cuda": point all of it at the CPU (same shims as trace_inputs.py) ---
def _cpu(x):
    if isinstance(x, str) and x.startswith("cuda"): return "cpu"
    if isinstance(x, torch.device) and x.type == "cuda": return torch.device("cpu")
    return x

_t_to, _m_to = torch.Tensor.to, torch.nn.Module.to
torch.Tensor.to = lambda self, *a, **k: _t_to(self, *[_cpu(x) for x in a], **{n: _cpu(v) for n, v in k.items()})
torch.nn.Module.to = lambda self, *a, **k: _m_to(self, *[_cpu(x) for x in a], **{n: _cpu(v) for n, v in k.items()})
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self

import sam_3d_body
import sam_3d_body.sam_3d_body_estimator as est_mod
if hasattr(est_mod, "recursive_to"):
    _orig_rt = est_mod.recursive_to
    est_mod.recursive_to = lambda x, dev: _orig_rt(x, "cpu")

# --- load the model ---
base = os.environ.get("SAM3D_CKPT_DIR", "checkpoints/sam-3d-body-dinov3")
model, cfg = sam_3d_body.load_sam_3d_body(checkpoint_path=f"{base}/model.ckpt", device="cpu", mhr_path=f"{base}/assets/mhr_model.pt")
model = model.float().eval()  # patches 17, 25, 26 look up attributes on this global
model.backbone_dtype = torch.float32  # Meta feeds the encoder x.type(backbone_dtype), which was bfloat16; weights are float32 now

# --- are the 6 decoder layers the same apart from their weights? ---
reprs = [repr(l) for l in model.decoder.layers]
print("DECODER layers", len(reprs), "| layers 1-5 identical structure:", len(set(reprs[1:])) == 1,
      "| layer 0 same as layer 1:", reprs[0] == reprs[1],
      "| skip_first_pe per layer:", [getattr(l, "skip_first_pe", None) for l in model.decoder.layers])

# --- run Meta's estimator once on the photo, body decoder only, and catch the first forward_decoder call's arguments ---
caught = {}
_orig_fd = model.forward_decoder
def _catch(*a, **kw):
    if not caught:  # the first call is the body pass
        caught.update(inspect.signature(_orig_fd).bind(*a, **kw).arguments)
    return _orig_fd(*a, **kw)
model.forward_decoder = _catch

estimator = sam_3d_body.SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
kw = {"inference_type": "body"} if "inference_type" in inspect.signature(estimator.process_one_image).parameters else {}
print("ESTIMATOR kwargs", kw)
with torch.no_grad():
    estimator.process_one_image(sys.argv[1], **kw)
model.forward_decoder = _orig_fd

batch = caught.pop("batch")
describe = lambda v: f"{tuple(v.shape)} {v.dtype}" if torch.is_tensor(v) else repr(v)[:60]
print("CAUGHT args:", {k: describe(v) for k, v in caught.items()})
print("CAUGHT batch:", {k: describe(v) for k, v in batch.items()})

# per-frame values become model inputs; everything else stays a constant
PER_FRAME = ["ray_cond", "cam_int", "bbox_center", "bbox_scale", "ori_img_size", "affine_trans", "img_size"]
arg_names = [k for k, v in caught.items() if torch.is_tensor(v)]  # image_embeddings, keypoints, condition_info, ...
batch_names = [k for k in PER_FRAME if k in batch and torch.is_tensor(batch[k])]
in_names = arg_names + batch_names
fixed_args = {k: v for k, v in caught.items() if not torch.is_tensor(v)}  # None or plain values
fixed_batch = {k: v for k, v in batch.items() if k not in batch_names}  # e.g. img (only its shape is used)

class DecoderPass(torch.nn.Module):  # one body decoder pass, final keypoints out
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, *vals):
        args = dict(fixed_args, **{k: v.float() for k, v in zip(arg_names, vals[:len(arg_names)])})
        b = dict(fixed_batch, **dict(zip(batch_names, vals[len(arg_names):])))
        _, outs = self.m.forward_decoder(batch=b, **args)
        last = outs[-1] if isinstance(outs, (list, tuple)) else outs  # the pose output after layer 6
        return last["pred_keypoints_3d"], last["pred_keypoints_2d"]

inputs_a = [caught[k] for k in arg_names] + [batch[k] for k in batch_names]
inputs_a = [v.float() if v.is_floating_point() else v for v in inputs_a]
wrapped = DecoderPass(model).eval()
with torch.no_grad():
    ref_a = [r.numpy() for r in wrapped(*inputs_a)]

# a second, different input to test on (not the trace input): features scaled and noised, box moved by 4 pixels
g = torch.Generator().manual_seed(0)
inputs_b = [v.clone() for v in inputs_a]
inputs_b[in_names.index("image_embeddings")] = inputs_a[in_names.index("image_embeddings")] * 1.05 + 0.05 * torch.randn(inputs_a[in_names.index("image_embeddings")].shape, generator=g)
if "bbox_center" in in_names:
    inputs_b[in_names.index("bbox_center")] = inputs_a[in_names.index("bbox_center")] + 4.0
with torch.no_grad():
    ref_b = [r.numpy() for r in wrapped(*inputs_b)]

# --- trace, the same graph rewrites as for head_pose (patches 27, 31, 32), convert ---
model._jit_is_scripting = True  # Lightning's own export flag: its `trainer` attribute then returns None instead of raising while tracing
traced = torch.jit.trace(wrapped, tuple(inputs_a), strict=False, check_trace=False)
torch._C._jit_pass_inline(traced.forward.graph)
_split_inplace_arith(traced.forward.graph)  # patch 31
_drop_noop_to(traced.forward.graph)  # patch 32
_reroute_alias_writes(traced.forward.graph)  # patch 27

# --- scan: list every op in the graph that has no converter, so all missing ops show in one run ---
def _kinds(block):
    ks = []
    for n in block.nodes():
        ks.append(n.kind())
        for b in n.blocks():
            ks += _kinds(b)
    return ks

missing = sorted({k for k in _kinds(traced.forward.graph) if _TORCH_OPS_REGISTRY.get_func(k) is None
                  and k not in ("prim::GetAttr", "prim::SetAttr", "prim::CallMethod")})
print("DECODER unsupported ops:", missing)

mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name=n, shape=tuple(v.shape)) for n, v in zip(in_names, inputs_a)],
    outputs=[ct.TensorType(name="keypoints3d"), ct.TensorType(name="keypoints2d")],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS18,
    skip_model_load=True,  # build and save only; loading (compiling) happens below, so a crash there is separate
)
mlmodel.save("sam3d_decoder.mlpackage")
print("saved")
print("DECODER package size MB:", round(sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk("sam3d_decoder.mlpackage") for f in fs) / 1e6))

# --- save the test inputs and PyTorch's results; compare_decoder.py runs the model on them, one compute setting per process ---
np.savez("decoder_test.npz", names=np.array(in_names),
         **{f"a_{n}": v.numpy() for n, v in zip(in_names, inputs_a)},  # a: the trace input
         **{f"b_{n}": v.numpy() for n, v in zip(in_names, inputs_b)},  # b: a different input
         ref_a3=ref_a[0], ref_a2=ref_a[1], ref_b3=ref_b[0], ref_b2=ref_b[1])
print("DECODER test inputs saved", flush=True)
