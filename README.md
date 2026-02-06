# UniST

A Unified Computational Framework for 3D Spatial Transcriptomics Reconstruction.

[Tutorials](https://unist-tutorial.readthedocs.io/en/latest/)

<p align="center">
  <img src="fig.png" width="1000">
</p>

## Installation

**From GitHub:**

```
pip install git+https://github.com/lanshui98/UniST.git
```

**Using requirements.txt:**
```
pip install -r requirements.txt
pip install -e .
```

#### Build CUDA extensions (only for upsampling)

```
# For A100, RTX 3090, etc.
export TORCH_CUDA_ARCH_LIST="8.0"

# For V100
# export TORCH_CUDA_ARCH_LIST="7.0"

# For RTX 2080, Titan RTX
# export TORCH_CUDA_ARCH_LIST="7.5"

# For RTX 3090, A6000
# export TORCH_CUDA_ARCH_LIST="8.6"
```

Build the CUDA extensions:

```
python setup_cuda_extensions.py
```

Or build manually:

```
# Build Chamfer3D
cd external/RepKPU_ops/models/Chamfer3D
python setup.py install
cd ../../../../

# Build pointops
cd external/RepKPU_ops/models/pointops
python setup.py install
cd ../../../
```

## Quick Start

### Upsampling

Test upsampling on point cloud files:

```bash
python -m upsampling.test_upsampling \
    --dataset pugan \
    --input_dir /path/to/input/pointclouds \
    --ckpt /path/to/pretrain/ckpt-best.pth \
    --r 2 \
    --save_dir /path/to/output \
    --flexible \
    --no_gt
```

**Arguments:**
- `--dataset`: Dataset type
- `--input_dir`: Directory containing input `.xyz` point cloud files
- `--ckpt`: Path to model checkpoint file (e.g., `pretrain/ckpt-best.pth`)
- `--r`: Upsampling rate
- `--save_dir`: Directory to save upsampled point clouds
- `--flexible`: Enable flexible upsampling rate (recommended)
- `--no_gt`: Skip evaluation (use when you don't have ground truth)

**Pre-trained weights**
Download the weights from google drive [link](https://drive.google.com/file/d/1af_de8YnG5eOAJaSDy1C5CmZRqaCCYb-/view?usp=sharing) and put under ./external/RepKPU_ops/pretrain/

### Interpolation

Run slice interpolation using InterpolAI:

```bash
python -m interpolation.main \
    --mode auto \
    --tile_size 1024 1024 \
    --pth /path/to/image/folder
```

**Modes:**
- `auto`: Automatically detect and interpolate missing slices
- `no_skip`: Interpolate without skipping slices (requires `--skip` argument)
- `skip`: Interpolate with specified skip values (requires `--skip` argument)

**Pre-trained weights**
Download the weights from google drive [link](https://drive.google.com/drive/folders/1zw6kgpnxat_CEFoDWaVIHndxuKqk5vmD?usp=sharing) and put under ./external/InterpolAI/interpolation/

### Gene imputation

**Step 1: Train Embedder (Graph Autoencoder)**

Train a graph autoencoder to learn embeddings from spatial transcriptomics data:

```
python -m external.SUICA_pro.train \
    --mode embedder \
    --conf external/SUICA_pro/configs/ST/embedder_gae.yaml
```

**Step 2: Train INR (Implicit Neural Representation)**

Train an INR model to learn continuous representations:

```
python -m external.SUICA_pro.train \
    --mode inr \
    --conf external/SUICA_pro/configs/ST/inr_embd.yaml
```

**Step 3:  Prediction/Imputation**

After training, use the trained model to predict/impute gene expression at custom coordinates:

```
python -m external.SUICA_pro.predict \
    --mode inr \
    --conf external/SUICA_pro/configs/ST/inr_pred.yaml
```
