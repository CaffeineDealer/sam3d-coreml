import sys, types, os
import torch

class _Stub(types.ModuleType):  # fake pyrender, same as before
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})

sys.modules["pyrender"] = _Stub("pyrender")

import sam_3d_body

base = os.environ.get("SAM3D_CKPT_DIR", "checkpoints/sam-3d-body-dinov3")
out = sam_3d_body.load_sam_3d_body(checkpoint_path=f"{base}/model.ckpt", device="cpu", mhr_path=f"{base}/assets/mhr_model.pt")
model = out[0] if isinstance(out, tuple) else out
model.eval()

class HeadWrapper(torch.nn.Module):  # same wrapper as in test_head_pose.py
    def __init__(self, h):
        super().__init__()
        self.h = h
    def forward(self, token, init):
        return self.h(token, init)["pred_keypoints_3d"]

wrapped = HeadWrapper(model.head_pose.float()).eval()
token, init = torch.randn(1, 1024), torch.randn(1, 519)

scripts = [n for n, m in wrapped.named_modules() if isinstance(m, torch.jit.ScriptModule)]
print("## script modules:", len(scripts), scripts[:5])  # is the rig a saved TorchScript file?

with torch.no_grad():
    traced = torch.jit.trace(wrapped, (token, init), strict=False, check_trace=False)

g = traced.inlined_graph
bad = [n for n in g.findAllNodes("aten::mul", True) if [str(i.type()) for i in n.inputs()][:2] != ["Tensor", "Tensor"]]
print("## mul nodes with a non-tensor input:", len(bad))
for n in bad[:8]:
    print("##", [str(i.type()) for i in n.inputs()])
    print(str(n)[:400])  # includes the file and line it came from