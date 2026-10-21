# sam3d-coreml

Converting Meta's [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) to Core ML, so it can run on the iPhone.

This repo holds only my conversion and test scripts. Meta's code and weights are not included and keep their own licenses. The iPhone app that runs the converted models is a separate project (PoseCam).

## Status

| Part | Status |
|---|---|
| Image encoder (DINOv3, 512 x 512 in, `[1, 1280, 32, 32]` features out) | converted, runs on the iPhone |
| Rig and pose heads (`head_pose`: pose token in, 70 3D keypoints out) | converted with 33 patches; matches PyTorch exactly in float32, within about 3-6 mm in float16; about 8 ms on the iPhone CPU |
| Body decoder (6 layers, with `head_pose`, the camera head, projection and keypoint-token updates inside) | converted (patches 34, 35); final keypoints within about 3-5 mm (3D) and about 3 px (2D) of PyTorch on the Mac and the iPhone; about 0.3-0.9 s per pass on the iPhone |
| Encoder + decoder chained on a photo | matches PyTorch within about 1.2 cm (3D) and 3.3 px (2D) |
| Hand branch | not converted (body-only tracking uses one body pass, Meta's `inference_type="body"`) |

On the iPhone: the encoder must run with compute units `.cpuAndGPU` (with the Neural Engine allowed, its Neural Engine compile fails, loading takes 4-5 minutes and keypoints come out shifted; on the CPU alone the keypoints collapse), and the decoder with `.all` (without the Neural Engine it crashes inside Core ML). Root causes not known yet.

## Setup

1. Clone Meta's code next to this folder:

        git clone https://github.com/facebookresearch/sam-3d-body.git

2. Create the environment (Python 3.11):

        uv venv --python 3.11 .venv
        source .venv/bin/activate
        uv pip install -r requirements.txt

3. Download the weights (needs approved access on Hugging Face and an HF_TOKEN):

        hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3

4. Create a `.env` file in this folder (it is git-ignored) with the two paths:

        PYTHONPATH=../sam-3d-body
        SAM3D_CKPT_DIR=checkpoints/sam-3d-body-dinov3

Versions: torch 2.14.1, coremltools 9.0 (officially tested only up to torch 2.7), roma 1.6.1. The patches depend on coremltools' and roma's internals; other versions may need them adjusted.

## Run

In every new terminal, from this folder, activate the environment and load `.env`:

    source .venv/bin/activate && set -a && source .env && set +a

Convert `head_pose` (writes `sam3d_head_pose.mlpackage`, compares with PyTorch):

    python test_head_pose.py > head_out.txt 2>&1

Convert the body decoder from a photo (writes `sam3d_decoder.mlpackage` and `decoder_test.npz`), then check it with each compute setting, one process per setting:

    python -u test_decoder.py ../sam-3d-body/notebook/images/dancing.jpg > decoder_out.txt 2>&1
    python -u compare_decoder.py CPU_AND_NE
    python -u compare_decoder.py ALL

Run the converted encoder and decoder together on a photo and compare with Meta's PyTorch pipeline (writes `e2e_keypoints.png`):

    python -u e2e_mac.py <photo> <path to sam3d_backbone.mlpackage>

## Scripts

| Script | What it does |
|---|---|
| `load_sam3d.py` | converts the image encoder to Core ML (`sam3d_backbone.mlpackage`) |
| `test_head_pose.py` | converts the rig and pose heads; holds patches 1 to 33 (converter fixes and graph rewrites), reused by the decoder |
| `test_decoder.py` | converts one whole body decoder pass from a real photo; adds patches 34 and 35; saves test inputs and PyTorch results |
| `compare_decoder.py` | runs the saved decoder with one compute setting on the saved inputs and compares with PyTorch |
| `e2e_mac.py` | Core ML encoder + Core ML decoder on a photo vs Meta's PyTorch pipeline, step by step, and draws both sets of keypoints |
| `trace_inputs.py` | logs the inputs and outputs of each model part during Meta's pipeline |
| `draw_sam3d.py` | draws the 70 keypoints on a photo |
| `find_bool_mul.py` | debugging helper |

## What the patches fix

Most problems fall into two groups:

1. Conversion errors: ops coremltools does not support, or receives with unexpected argument types (Python lists, strings, module attributes arriving as their names, dtypes as strings), and control flow (static `if`s, loops unrolled).
2. Silent wrong results: in-place writes (`x[:, idx] = v`, `x -= 1`, `index_copy_` in a loop, writes after `.to()`), which PyTorch applies through shared memory but Core ML, having no shared memory, loses unless every later read is pointed at the new value; and code whose traced result depends on the example input (masked writes, `nonzero()`), which only works for inputs like the traced one.

Each patch in `test_head_pose.py` and `test_decoder.py` has a comment explaining the problem it fixes.

Lessons that cost the most time:

1. A conversion that succeeds is not a correct conversion: compare outputs, and intermediate results, with PyTorch.
2. Always test on an input other than the one used for tracing.
3. On the device, try each model with each compute-unit setting; here the setting decided between wrong, collapsed, crashing and correct results.

## Third-party code

Patches 29 and 33 in `test_head_pose.py` are adapted from [RoMa](https://github.com/naver/roma), Copyright (c) 2020 NAVER Corp., BSD-3-Clause license (RoMa adapts these conversions from SciPy, BSD-3-Clause). Patch 7 re-implements the math of Meta's `batchXYZfrom6D` (sam-3d-body) without in-place writes.

## License

My code in this repo is under the MIT License (see `LICENSE`). Meta's SAM 3D Body code and weights, and the RoMa-derived parts (BSD-3-Clause), keep their own licenses.
