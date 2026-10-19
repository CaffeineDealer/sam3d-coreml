# sam3d-coreml
Work in progress: converting parts of Meta's SAM 3D Body to Core ML for iPhone.

This repo holds only my scripts. Meta's code and weights are not included and have their own licenses.

## Setup

1. Clone Meta's code next to this folder:

        git clone https://github.com/facebookresearch/sam-3d-body.git

2. Create the environment (Python 3.11):

        uv venv --python 3.11 .venv
        source .venv/bin/activate
        uv pip install -r requirements.txt

3. Download the weights (needs approved access on Hugging Face and an HF_TOKEN):

        hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3

4. Run a script:

        PYTHONPATH=../sam-3d-body SAM3D_CKPT_DIR=checkpoints/sam-3d-body-dinov3 python test_head_pose.py

## Scripts

- load_sam3d.py: converts the image encoder to Core ML
- test_head_pose.py: tests converting the rig and pose heads
- trace_inputs.py: logs the inputs and outputs of each model part
- draw_sam3d.py: draws the 70 keypoints on a photo
- find_bool_mul.py: debugging helper
