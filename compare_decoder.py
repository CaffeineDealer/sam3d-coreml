# compare_decoder.py: run sam3d_decoder.mlpackage with one compute setting on the saved test inputs and compare with PyTorch
# run: python -u compare_decoder.py CPU_ONLY | CPU_AND_NE | ALL
import sys, time
import numpy as np
import coremltools as ct

cu = sys.argv[1]
z = np.load("decoder_test.npz")
names = [str(n) for n in z["names"]]
t0 = time.perf_counter()
m = ct.models.MLModel("sam3d_decoder.mlpackage", compute_units=getattr(ct.ComputeUnit, cu))
print(f"COMPARE {cu:10s} loaded in {time.perf_counter() - t0:.1f} s", flush=True)
for tag in ("a", "b"):  # a: trace input, b: different input
    feed = {n: z[f"{tag}_{n}"] for n in names}
    try:
        t0 = time.perf_counter()
        res = m.predict(feed)
        ms = (time.perf_counter() - t0) * 1000
        print(f"COMPARE {cu:10s} input {tag}: keypoints3d max diff {np.abs(res['keypoints3d'] - z[f'ref_{tag}3']).max():.4f}  "
              f"keypoints2d max diff {np.abs(res['keypoints2d'] - z[f'ref_{tag}2']).max():.2f} px  ({ms:.0f} ms)", flush=True)
    except Exception as e:
        print(f"COMPARE {cu:10s} input {tag}: FAILED {str(e)[:300]}", flush=True)
