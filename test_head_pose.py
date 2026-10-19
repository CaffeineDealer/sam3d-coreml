import sys, types, os
import numpy as np
import torch

class _Stub(types.ModuleType):  # fake pyrender, same as before
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})

sys.modules["pyrender"] = _Stub("pyrender")

import coremltools as ct
import coremltools.converters.mil.frontend.torch.ops as _tops  # coremltools' op converters
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
from coremltools.converters.mil.mil import Var

mb = _tops.mb

# --- patch 1: int/float/bool casts of 1-element arrays (newer numpy rejects them) ---
def _cast_fixed(context, node, dtype, dtype_name):
    x = _tops._get_inputs(context, node, expected=1)[0]
    if x.can_be_folded_to_const():
        v = x.val
        v = v.item() if hasattr(v, "item") else v
        res = mb.const(val=dtype(v), name=node.name)
    elif len(x.shape) > 0:
        x = mb.squeeze(x=x, name=node.name + "_item")
        res = mb.cast(x=x, dtype=dtype_name, name=node.name)
    else:
        res = mb.cast(x=x, dtype=dtype_name, name=node.name)
    context.add(res, node.name)

_tops._cast = _cast_fixed

# --- patch 2: skip the optional cleanup pass that crashes on the rig ---
torch._C._jit_pass_canonicalize_graph_fuser_ops = lambda graph: None

# --- patch 3: linalg_cross reuses coremltools' own cross product ---
_TORCH_OPS_REGISTRY.set_func_by_name(_tops.cross, "linalg_cross")

# --- patch 4: general torch.diagonal (any offset, any two dims) ---
def _int(a):  # a constant index argument, as a Python int
    return int(a.val) if isinstance(a, Var) else int(a)

@register_torch_op(torch_alias=["diagonal_copy"], override=True)
def diagonal(context, node):
    inputs = _tops._get_inputs(context, node, min_expected=1)
    x = inputs[0]
    offset = _int(inputs[1]) if len(inputs) > 1 else 0
    dim1 = _int(inputs[2]) if len(inputs) > 2 else 0
    dim2 = _int(inputs[3]) if len(inputs) > 3 else 1
    rank = x.rank
    d1, d2 = dim1 % rank, dim2 % rank
    rest = [i for i in range(rank) if i not in (d1, d2)]
    xt = mb.transpose(x=x, perm=rest + [d1, d2])  # the two dims go last: [..., n1, n2]
    n1, n2 = int(x.shape[d1]), int(x.shape[d2])
    mask = np.zeros((n1, n2), dtype=np.float32)
    for i in range(n1):
        if 0 <= i + offset < n2:
            mask[i, i + offset] = 1.0
    d = mb.reduce_sum(x=mb.mul(x=xt, y=mask), axes=[-1], keep_dims=False)  # [..., n1]
    start, stop = max(0, -offset), min(n1, n2 - offset)
    nb = rank - 2
    res = mb.slice_by_index(x=d, begin=[0] * nb + [start], end=[0] * nb + [stop],
                            begin_mask=[True] * nb + [False], end_mask=[True] * nb + [False], name=node.name)
    context.add(res)

# --- patch 5: view() to a scalar, without the size assert that fails on unknown-size inputs ---
_orig_view = _tops.view

@register_torch_op(torch_alias=["view_copy", "_unsafe_view", "reshape"], override=True)
def view(context, node):
    inputs = _tops._get_inputs(context, node, expected=2)
    x, shape = inputs[0], inputs[1]
    if isinstance(shape, Var) and int(np.prod(shape.shape)) == 0:  # reshape to a scalar
        print("VIEW-SCALAR input shape:", x.shape)
        flat = mb.reshape(x=x, shape=[1])  # assumes it holds exactly one element
        context.add(mb.squeeze(x=flat, axes=[0], name=node.name))
        return
    return _orig_view(context, node)

# --- patch 6: hypot, index_add, index_copy ---
@register_torch_op
def hypot(context, node):
    a, b = _tops._get_inputs(context, node, expected=2)
    sq = mb.add(x=mb.mul(x=a, y=a), y=mb.mul(x=b, y=b))
    context.add(mb.sqrt(x=sq, name=node.name))

def _index_scatter(context, node, mode):  # index_add / index_copy with a 1-D index along dim
    inputs = _tops._get_inputs(context, node, min_expected=4)
    x = inputs[0]
    dim = _int(inputs[1]) % x.rank
    index = mb.cast(x=inputs[2], dtype="int32")
    src = inputs[3]
    if mode == "add" and len(inputs) > 4:  # optional alpha multiplier
        alpha = float(inputs[4].val if isinstance(inputs[4], Var) else inputs[4])
        if alpha != 1.0:
            src = mb.mul(x=src, y=alpha)
    context.add(mb.scatter(data=x, indices=index, updates=src, axis=dim, mode=mode, name=node.name))

@register_torch_op
def index_add(context, node):
    _index_scatter(context, node, "add")

@register_torch_op
def index_copy(context, node):  # also serves index_copy_, coremltools treats in-place ops as functional
    _index_scatter(context, node, "update")

# --- load the model ---
import sam_3d_body

base = os.environ.get("SAM3D_CKPT_DIR", "checkpoints/sam-3d-body-dinov3")
out = sam_3d_body.load_sam_3d_body(checkpoint_path=f"{base}/model.ckpt", device="cpu", mhr_path=f"{base}/assets/mhr_model.pt")
model = out[0] if isinstance(out, tuple) else out
model.eval()

class HeadWrapper(torch.nn.Module):  # tracing needs one plain tensor out
    def __init__(self, h):
        super().__init__()
        self.h = h
    def forward(self, token, init):
        return self.h(token, init)["pred_keypoints_3d"]  # (1, 70, 3)

wrapped = HeadWrapper(model.head_pose.float()).eval()
token = torch.randn(1, 1024)  # shapes taken from the trace
init = torch.randn(1, 519)
with torch.no_grad():
    ref = wrapped(token, init)
    print("keypoints shape:", ref.shape)

traced = torch.jit.trace(wrapped, (token, init), strict=False, check_trace=False)

# --- scan: list ops in the live graph that have no converter ---
g = traced.graph
torch._C._jit_pass_inline(g)  # flatten the rig's code so every op is visible
torch._C._jit_pass_dce(g)  # drop unused branches, so the list shows only ops the output needs

def _kinds(block):  # every op in the graph, including inside nested blocks
    ks = []
    for n in block.nodes():
        ks.append(n.kind())
        for b in n.blocks():
            ks += _kinds(b)
    return ks

missing = sorted({k for k in _kinds(g) if _TORCH_OPS_REGISTRY.get_func(k) is None
                  and k not in ("prim::GetAttr", "prim::SetAttr", "prim::CallMethod")})
print("unsupported ops:", missing)

# --- convert ---
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="token", shape=(1, 1024)), ct.TensorType(name="init", shape=(1, 519))],
    outputs=[ct.TensorType(name="keypoints3d")],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS18,
)
mlmodel.save("sam3d_head_pose.mlpackage")
print("saved")

try:  # compare Core ML against PyTorch on the same input
    res = mlmodel.predict({"token": token.numpy(), "init": init.numpy()})
    print("max abs diff:", np.abs(ref.numpy() - list(res.values())[0]).max())
except Exception as e:
    print("predict failed:", e)