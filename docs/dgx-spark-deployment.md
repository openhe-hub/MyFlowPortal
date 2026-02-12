# FlowPortal DGX Spark Deployment Guide

## Overview

FlowPortal has been deployed on the AIR DGX Spark cluster (30 x NVIDIA DGX GB10 nodes). This document covers the hardware specifics, environment setup, file transfer, job submission, and known issues.

## Hardware

| Spec | Detail |
|------|--------|
| Nodes | 30 x NVIDIA DGX GB10 |
| Architecture | ARM64 (aarch64) |
| GPU | NVIDIA GB10 (Grace Blackwell), compute capability sm_121 |
| CUDA Driver | 580.95.05 |
| CUDA Version | 13.0 |
| CPU | ARM Neoverse (per node) |
| Local Disk | ~3.7 TB NVMe per node (`/home/cvpr/`) |
| Shared Storage | `/CVPR` — 2 TB SMB/CIFS mount |

**Important**: The login node is x86_64, while compute nodes are aarch64. All Python environments and binaries must be ARM64-compatible.

## SSH Access

```bash
# Direct access from local machine (SSH config alias "dgx-login")
ssh dgx-login

# Run a command
ssh dgx-login "<command>"
```

## Project Location

```
/CVPR/zhewen/FlowPortal/
```

- Synced from local machine at `/home/nyuair/zhewen/FlowPortalRelease/`
- `/CVPR` is an SMB mount shared across all nodes (login + compute)

## Python Environment

### Location

```
/CVPR/python_envs/flowportal/
```

- Python 3.10.19
- PyTorch 2.11.0.dev20260210+cu130 (CUDA 13.0 — required for GB10 sm_121)
- All FlowPortal dependencies installed

### How the Environment Was Created

The SMB mount (`/CVPR`) does not support `chmod`, `symlink`, or certain filesystem operations, so conda/venv cannot create environments directly on `/CVPR`. The workaround:

1. Submit a SLURM job to a compute node (aarch64)
2. Use the existing conda at `/home/cvpr/miniconda3/bin/conda` on the compute node
3. Create the env on the compute node's local disk (`/home/cvpr/zhewen/python_envs/flowportal`)
4. Install all dependencies via pip
5. Copy the completed env to `/CVPR/python_envs/flowportal/`

```bash
# Step 1: On compute node (via SLURM)
/home/cvpr/miniconda3/bin/conda create -p /home/cvpr/zhewen/python_envs/flowportal python=3.10 -y

# Step 2: Activate and install
export PATH=/home/cvpr/zhewen/python_envs/flowportal/bin:$PATH

# PyTorch cu130 nightly (required for GB10 sm_121 support)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# FlowPortal dependencies
pip install transformers diffusers accelerate safetensors einops omegaconf \
    opencv-python-headless imageio imageio-ffmpeg kornia controlnet_aux \
    huggingface_hub sentencepiece protobuf scipy pillow

# Step 3: Copy to shared storage
cp -r /home/cvpr/zhewen/python_envs/flowportal /CVPR/python_envs/flowportal
```

### Why cu130?

The GB10 GPU has compute capability sm_121 (Blackwell architecture). Only CUDA 13.0 includes native NVRTC JIT support for sm_121. Earlier CUDA versions fail:

| CUDA | PyTorch | Result |
|------|---------|--------|
| cu126 | 2.10.0 | `cudaErrorNoKernelImageForDevice` — no sm_121 kernel |
| cu128 | 2.11.0.dev | NVRTC JIT error — `invalid value for --gpu-architecture` |
| **cu130** | **2.11.0.dev** | **Works** — native sm_121 support |

### Key Packages

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.11.0.dev20260210+cu130 | CUDA 13.0, nightly build |
| transformers | 4.49.0 | |
| diffusers | 0.32.2 | |
| accelerate | 1.3.0 | |
| controlnet_aux | 0.0.10 | HED/Midas detectors |
| kornia | 0.8.2 | Required by MatAnyone |

### Code Modifications for aarch64

`decord` has no ARM64 wheel, so conditional imports were added:

```python
# videox_fun/data/dataset_image_video.py
# videox_fun/data/dataset_video.py
try:
    from decord import VideoReader
except ImportError:
    VideoReader = None
```

This is safe because the demo pipeline does not use `decord` (it uses OpenCV for frame extraction).

## Pretrained Models

All models are stored under `/CVPR/zhewen/FlowPortal/pretrained_models/`:

| Model | Path | Size | Source |
|-------|------|------|--------|
| Wan2.1-Fun-V1.1-1.3B-Control | `models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control/` | 19GB | `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control` |
| IC-Light FC | `IC-Light/models/iclight_sd15_fc.safetensors` | 1.7GB | `lllyasviel/ic-light` |
| IC-Light FBC | `IC-Light/models/iclight_sd15_fbc.safetensors` | 1.7GB | `lllyasviel/ic-light` |
| MatAnyone | `MatAnyone/pretrained_models/matanyone.pth` | 135MB | GitHub releases |
| Annotators (HED/Midas) | `Annotators/` | 470MB | Transferred from local |

Note: The SD1.5 base model (`stablediffusionapi/realistic-vision-v51`) used by IC-Light will be auto-downloaded on first run and cached to `${LOCAL_MODELS}/.huggingface_cache` on NVMe (persistent across jobs on the same node).

## File Transfer (Local -> DGX)

### Standard Files

```bash
rsync -avh --inplace --no-perms --no-owner --no-group \
  /home/nyuair/zhewen/FlowPortalRelease/ \
  dgx-login:/CVPR/zhewen/FlowPortal/
```

The flags `--inplace --no-perms --no-owner --no-group` are required because the SMB mount does not support `mkstemp`, `chmod`, or ownership changes.

### Large Files (>6 GB)

SMB has issues transferring very large single files (transfers stall around 6.5 GB). Use the split-transfer-reassemble approach:

```bash
# 1. Split locally into 2GB chunks
split -b 2G /path/to/large_file.pth /tmp/chunks/chunk_

# 2. Transfer each chunk
for f in /tmp/chunks/chunk_*; do
  rsync -avh --inplace --no-perms --no-owner --no-group \
    "$f" dgx-login:/CVPR/zhewen/t5_chunks/
done

# 3. Reassemble on DGX
ssh dgx-login "cat /CVPR/zhewen/t5_chunks/chunk_* > /CVPR/zhewen/FlowPortal/path/to/large_file.pth"

# 4. Verify file size matches
ssh dgx-login "ls -l /CVPR/zhewen/FlowPortal/path/to/large_file.pth"

# 5. Clean up
ssh dgx-login "rm -rf /CVPR/zhewen/t5_chunks"
rm -rf /tmp/chunks
```

This was needed for `models_t5_umt5-xxl-enc-bf16.pth` (10.6 GB).

## SLURM Job Submission

### Cluster Info

- **Scheduler**: SLURM
- **GPU partition**: `spark`
- **Available GPU**: NVIDIA GB10 (Grace Blackwell)
- **Nodes**: 30 (node names: ADUAED21xxx)

### SLURM Script Template

```bash
#!/bin/bash
#SBATCH --job-name=FlowPortal
#SBATCH --output=/CVPR/zhewen/FlowPortal/logs/flowportal_%j.out
#SBATCH --error=/CVPR/zhewen/FlowPortal/logs/flowportal_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00

set -e

PYTHON=/CVPR/python_envs/flowportal/bin/python
PROJECT=/CVPR/zhewen/FlowPortal

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/huggingface_cache
export TORCH_CUDA_ARCH_LIST="12.0"

cd $PROJECT
mkdir -p logs

# ... your FlowPortal commands here ...
```

### Key Environment Variables

| Variable | Value | Reason |
|----------|-------|--------|
| `HF_HOME` | `${LOCAL_MODELS}/.huggingface_cache` | Persistent HF cache on NVMe; SMB mount doesn't support `chmod` |
| `TORCH_CUDA_ARCH_LIST` | `"12.0"` | Target GB10 GPU architecture |
| `TORCHINDUCTOR_CACHE_DIR` | `${LOCAL_MODELS}/.torchinductor_cache` | Persistent torch.compile cache on NVMe |
| `PYTHONUNBUFFERED` | `1` | Ensure real-time log output |

### Submitting Jobs

```bash
# Submit
ssh dgx-login "sbatch /CVPR/zhewen/FlowPortal/run-flow-portal-cat-dgx.sh"

# Check status
ssh dgx-login "squeue -u \$(whoami)"

# Cancel job
ssh dgx-login "scancel <job_id>"

# View output
ssh dgx-login "tail -50 /CVPR/zhewen/FlowPortal/logs/flowportal_cat_<job_id>.out"

# View progress (stderr has tqdm bars)
ssh dgx-login "tail -5 /CVPR/zhewen/FlowPortal/logs/flowportal_cat_<job_id>.err"
```

### Cat Demo Script

The cat demo script is `run-flow-portal-cat-dgx.sh`. It runs the full 5-stage pipeline:

```
Stage 1: Preprocessing (frame extraction, masks, Canny edge)
Stage 2: MatAnyone matting (alpha mask refinement)
Stage 3: Re-preprocessing with refined masks
Stage 4: IC-Light relighting (SD1.5-based reference image)
Stage 5: FlowPortal generation (Wan2.1 video diffusion)
```

Key parameters for the cat demo:
- Resolution: 480x720
- Frames: 49 (6.1s at 8fps)
- `partial_edit=true` (preserve foreground cat, generate new background)
- `transfer_blurring="mid"` (0.2)
- `gpu_memory_mode="no_offload"` (GB10 has unified CPU/GPU memory, no offload needed)

## NVMe Model Caching

The SLURM script caches all pretrained models from the SMB mount (`/CVPR`) to the compute node's local NVMe (`/home/cvpr/zhewen/flowportal_models/`). This is persistent across jobs on the same node and avoids repeated slow reads from SMB.

Cached models:
- Wan2.1-Fun-V1.1-1.3B-Control (~19GB)
- IC-Light models (~3.4GB)
- MatAnyone (~135MB)
- Annotators / HED / Midas weights (~470MB)
- HuggingFace model cache (SD1.5 for IC-Light, auto-populated on first run)

The cache sentinel check uses `models_t5_umt5-xxl-enc-bf16.pth` — if present, skips the main copy. Annotators have a separate check since they were added later.

## Performance

| GPU | Cat Demo (49 frames, 480x720) | Notes |
|-----|-------------------------------|-------|
| A6000 (48GB) | ~15-20 min | Local machine, `no_offload` |
| A100-40GB | ~7.5 min | Jubail cluster, `no_offload` |
| **GB10 (DGX Spark)** | **~7 min** | `no_offload`, optimized pipeline |

### Per-Stage Breakdown (GB10)

| Stage | Baseline | Optimized | Speedup | Key Change |
|-------|----------|-----------|---------|------------|
| 1. Preprocessing | 339s | 95s | 3.6x | Canny-only edge detection (skip HED+Midas) |
| 2. MatAnyone | 154s | 52s | 3.0x | NVMe model cache |
| 3. Re-preprocessing | 87s | 8s | 10.9x | Lazy torch import + skip redundant work |
| 4. IC-Light | 164s | 114s | 1.4x | 15 steps, skip highres pass |
| 5. FlowPortal | 294s | 150s | 2.0x | 20 steps, DPM++, TeaCache, cfg_skip |
| **Total** | **1038s (17m)** | **419s (7m)** | **2.5x** | |

Note: The baseline above is after the first round of optimization (from the original unoptimized 26m24s). Total speedup from the original: **3.8x**.

### Optimization Details

**1. Canny-only edge detection** (`--edge_mode canny`)
- Replaced combined mode (10% Canny + 10% HED + 80% Midas depth) with pure Canny
- Eliminates loading two neural network models (HED ~200MB, Midas ~200MB) on ARM64
- CannyDetector is pure OpenCV — no GPU needed, instant per frame
- Saved ~244s in Stage 1

**2. Lazy torch import for Stage 3**
- Stage 3 (re-preprocessing with MatAnyone masks) only needs numpy/cv2/PIL
- Moved `import torch` and CUDA initialization inside the BiRefNet block
- When `--using_existing_masks` and `--skip_edge_detection` are set, torch is never loaded
- Stage 3 dropped from 87s to 8s

**3. IC-Light: skip highres refinement** (`--highres_denoise 0.0`)
- IC-Light normally runs two passes: lowres (N steps) + highres img2img (N steps)
- When `highres_denoise <= 0` or `highres_scale <= 1.0`, the second pass is skipped entirely
- Combined with reducing steps from 25 to 15, IC-Light does 15 total denoising steps (vs ~50 before)

**4. FlowPortal diffusion acceleration**
- `--num_inference_steps 20`: Reduced from 50 (original) → 25 (round 1) → 20 steps
- `--sampler_name "Flow_DPM++"`: Multi-step solver, higher quality per step than Euler
- `--enable_teacache True --teacache_threshold 0.20`: Caches intermediate transformer activations, skips ~20% of forward passes
- `--cfg_skip_ratio 0.25`: Skips classifier-free guidance for the first 25% of steps
- `--gpu_memory_mode "no_offload"`: GB10 unified memory allows full GPU residency

**5. NVMe caching**
- All models cached on local NVMe (~3GB/s read vs SMB ~200MB/s)
- HuggingFace cache (`HF_HOME`) persisted on NVMe across jobs
- Annotators cached separately for Stage 1 edge detection

### Settings That Did NOT Help on GB10

- **torch.compile**: "Not enough SMs to use max_autotune_gemm mode" — 92s compilation overhead outweighed gains for single runs
- **FP8 quantization**: Not tested; GB10 has limited SMs making the tradeoff uncertain

## How2Sign Batch Processing

### Overview

The How2Sign batch pipeline processes sign language videos with background replacement using 10-GPU parallelism on DGX Spark.

- **Total videos**: 31,047 MP4s (31 GB) in `datasets/How2Sign/`
- **Valid videos**: 29,529 (1,518 too short)
- **Selected set**: 8,000 videos (seed=42 random selection) — processed on Jubail cluster
- **Unselected set**: 21,529 videos — processed on DGX Spark (10 chunks)

### Manifest Generation

```bash
# Generate selected 8000 (original, 4 chunks for Jubail)
python scripts/generate_h2s_manifest.py --mode selected

# Generate unselected remainder (10 chunks for DGX Spark)
python scripts/generate_h2s_manifest.py --mode unselected --num_chunks 10
```

Output files (all regenerable with the same seed):
- `scripts/h2s_unselected_manifest.csv` — full unselected manifest (21,529 rows)
- `scripts/h2s_unselected_videos.txt` — filenames for rsync transfer
- `scripts/h2s_unselected_chunk{0-9}.csv` — 10 chunks (~2,153 each)

### Deployment

```bash
# 1. Transfer unselected videos to DGX (22.3 GB)
rsync -avh --inplace --no-perms --no-owner --no-group \
  --files-from=scripts/h2s_unselected_videos.txt \
  datasets/How2Sign/how2sign/sentence_level/train/rgb_front/raw_videos/ \
  dgx-login:/CVPR/zhewen/FlowPortal/datasets/How2Sign/train_mp4/

# 2. Sync scripts and manifests
rsync -avh --inplace --no-perms --no-owner --no-group \
  scripts/h2s_unselected_*.csv scripts/h2s_prompts.txt run-h2s-batch-dgx.sh \
  dgx-login:/CVPR/zhewen/FlowPortal/

# 3. Move manifests to scripts/ on DGX
ssh dgx-login "mv /CVPR/zhewen/FlowPortal/h2s_unselected_*.csv \
  /CVPR/zhewen/FlowPortal/h2s_prompts.txt /CVPR/zhewen/FlowPortal/scripts/"

# 4. Submit 10-GPU array job
ssh dgx-login "sbatch /CVPR/zhewen/FlowPortal/run-h2s-batch-dgx.sh"
```

### SLURM Script: `run-h2s-batch-dgx.sh`

- `--partition=spark`, `--array=0-9`, `--mem=100G`, `--time=96:00:00`
- Each array task processes one chunk on a separate node
- NVMe model caching (same pattern as cat demo)
- DGX-optimized pipeline (Canny edge, 15-step IC-Light, 20-step FlowPortal with TeaCache)
- Skips already-completed videos (checks `results/flow-edit-{name}/output_video.mp4`)
- Cleans up intermediate files after each video
- Progress logged to `logs/h2s_unselected_progress_{chunk_id}.log`

### Monitoring

```bash
# Job queue
ssh dgx-login "squeue -u cvpr --name=FP-h2s-u"

# Per-chunk progress
ssh dgx-login "for i in 0 1 2 3 4 5 6 7 8 9; do \
  f=/CVPR/zhewen/FlowPortal/logs/h2s_unselected_progress_\${i}.log; \
  ok=\$(grep -c ' OK ' \$f 2>/dev/null || echo 0); \
  fail=\$(grep -c FAILED \$f 2>/dev/null || echo 0); \
  echo \"Chunk \$i: \${ok} OK, \${fail} FAIL\"; done"

# Live output for a specific chunk
ssh dgx-login "tail -20 /CVPR/zhewen/FlowPortal/logs/h2s_unselected_<jobid>_<chunk>.out"
```

### Known Issues

- **SSL certificate errors on compute nodes**: MatAnyone tries to download `resnet50` weights via torch.hub on first run. Fix: set `TORCH_HOME` to a shared path with pre-downloaded weights, and set `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt`.
- **rsync exit code 23**: Harmless "failed to set times" warnings on the SMB mount. All data transfers correctly.

## Troubleshooting

### SMB "Operation not permitted" on chmod/symlink

This is expected. The `/CVPR` SMB mount does not support Unix permission operations. Use `--inplace --no-perms --no-owner --no-group` flags with rsync. Exit code 23 from rsync (failed to set times) is harmless — files are transferred correctly.

### Large file transfer stalls

Files >6 GB may stall during rsync to SMB. Use the split-transfer-reassemble approach described above.

### HuggingFace cache errors (shutil.copymode)

Set `HF_HOME=/tmp/huggingface_cache` to use the compute node's local disk for HF model cache. The SMB mount causes `shutil.copy` failures due to `copymode` calling `chmod`.

### decord ImportError

`decord` has no aarch64 wheel. The codebase has been patched with conditional imports. If you see `ImportError: No module named 'decord'` in new code paths, add:
```python
try:
    from decord import VideoReader
except ImportError:
    VideoReader = None
```

### CUDA/PyTorch compatibility

The GB10 (sm_121) requires CUDA 13.0. If you see errors like:
- `cudaErrorNoKernelImageForDevice` — wrong CUDA version (needs cu130)
- `nvrtc: error: invalid value for --gpu-architecture` — NVRTC doesn't know sm_121 (needs cu130)

Ensure PyTorch is installed with cu130:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

### DOS line endings in scripts

If scripts edited on Windows/local have `\r\n` line endings, SLURM may fail. Fix with:
```bash
sed -i 's/\r$//' run-flow-portal-cat-dgx.sh
```
