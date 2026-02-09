# FlowPortal Jubail Deployment Guide

## Overview

FlowPortal has been deployed on NYU Abu Dhabi's Jubail HPC cluster. This document covers the SSH access, environment setup, job submission, and usage instructions.

## SSH Access

Jubail requires a two-hop SSH connection through a jump host:

```
Local Machine -> chatsign@10.224.35.17 (jump host) -> zl6890@jubail.abudhabi.nyu.edu
```

### Quick Access

```bash
# One-liner to execute commands on jubail
ssh chatsign@10.224.35.17 "ssh jubail '<command>'"

# Interactive session
ssh chatsign@10.224.35.17
# then on jump host:
ssh jubail
```

The jump host (`chatsign@10.224.35.17`) has SSH config with jubail alias pre-configured (`~/.ssh/config`), mapping `jubail` to `zl6890@jubail.abudhabi.nyu.edu`.

### File Transfer

Direct SCP doesn't work (SSH keys are on jump host). Use pipe transfer instead:

```bash
# Local -> Jubail
cat local_file | ssh chatsign@10.224.35.17 "ssh jubail 'cat > /scratch/zl6890/path/to/remote_file'"

# Directory transfer (tar pipe)
cd /local/parent && tar czf - dirname | ssh chatsign@10.224.35.17 "ssh jubail 'cd /scratch/zl6890/path && tar xzf -'"

# Jubail -> Local
ssh chatsign@10.224.35.17 "ssh jubail 'cat /scratch/zl6890/path/to/file'" > local_file
```

## Project Location

```
/scratch/zl6890/Workspace/zhewen/FlowPortal/
```

- Source code: cloned from `https://github.com/openhe-hub/MyFlowPortal.git`
- `/scratch` has 7.5PB total storage, no tight quota (unlike `/home` which has limited quota)

## Conda Environment

### Location

```
/scratch/zl6890/miniconda/envs/flowportal
```

- Python 3.10
- PyTorch 2.2.2 + CUDA 12.1
- All FlowPortal dependencies installed

### Activation

```bash
# Option 1: Direct PATH (recommended for scripts)
export PATH=/scratch/zl6890/miniconda/envs/flowportal/bin:$PATH

# Option 2: Conda activate (for interactive use)
source /scratch/zl6890/miniconda/etc/profile.d/conda.sh
conda activate flowportal
```

### Key Packages

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.2.2+cu121 | CUDA 12.1 |
| transformers | 4.46.2 | |
| diffusers | 0.30.1 | |
| accelerate | 0.29.3 | |
| controlnet_aux | 0.0.10 | HED/Midas detectors |
| kornia | 0.8.2 | Required by MatAnyone |
| huggingface_hub | 0.36.2 | HF CLI with write access configured |

### Installing Additional Packages

```bash
/scratch/zl6890/miniconda/envs/flowportal/bin/pip install <package>
```

## Pretrained Models

All models are stored under `/scratch/zl6890/Workspace/zhewen/FlowPortal/pretrained_models/`:

| Model | Path | Size | Source |
|-------|------|------|--------|
| Wan2.1-Fun-V1.1-1.3B-Control | `models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control/` | 19GB | `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control` |
| IC-Light FC | `IC-Light/models/iclight_sd15_fc.safetensors` | 1.7GB | `lllyasviel/ic-light` |
| IC-Light FBC | `IC-Light/models/iclight_sd15_fbc.safetensors` | 1.7GB | `lllyasviel/ic-light` |
| MatAnyone | `MatAnyone/pretrained_models/matanyone.pth` | 135MB | GitHub releases |
| Annotators (HED/Midas) | `Annotators/` | 470MB | Transferred from local |

Note: The SD1.5 base model (`stablediffusionapi/realistic-vision-v51`) used by IC-Light will be auto-downloaded on first run to HuggingFace cache.

## SLURM Job Submission

### Cluster Info

- **Scheduler**: SLURM
- **sbatch path**: `/opt/slurm/20.11.4-13/bin/sbatch`
- **GPU partition**: `nvidia`
- **Available GPU**: NVIDIA A100-PCIE-40GB
- **CUDA on nodes**: `/share/apps/NYUAD5/cuda/12.2.0/`

### SLURM Script Template

```bash
#!/bin/bash
#SBATCH --job-name=FlowPortal
#SBATCH --output=logs/flowportal_%j.out
#SBATCH --error=logs/flowportal_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=nvidia

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH=/scratch/zl6890/miniconda/envs/flowportal/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

cd /scratch/zl6890/Workspace/zhewen/FlowPortal
mkdir -p logs

# ... your FlowPortal commands here ...
```

### Submitting Jobs

```bash
# On jubail login node
export PATH=/opt/slurm/20.11.4-13/bin:$PATH

# Submit
sbatch run-flow-portal-cat-slurm.sh

# Check status
squeue -u zl6890

# Cancel job
scancel <job_id>

# View output
tail -f logs/flowportal_<job_id>.out
```

### Remote Submission (from local machine)

```bash
ssh chatsign@10.224.35.17 "ssh jubail 'export PATH=/opt/slurm/20.11.4-13/bin:\$PATH; cd /scratch/zl6890/Workspace/zhewen/FlowPortal && sbatch run-flow-portal-cat-slurm.sh'"
```

## HuggingFace Upload

HF CLI is configured with write token (`jubail1`) on jubail:

```bash
# Upload results
/scratch/zl6890/miniconda/envs/flowportal/bin/huggingface-cli upload \
    openhe/flow-portal-trans \
    /scratch/zl6890/Workspace/zhewen/FlowPortal/results/flow-edit-<name>/ \
    <target_folder>/ \
    --repo-type dataset
```

## Performance

| GPU | Cat Demo (49 frames, 480x720) | Notes |
|-----|-------------------------------|-------|
| A6000 (48GB) | ~15-20 min | Local machine |
| A100-40GB | ~7.5 min | Jubail cluster |

## Pipeline Steps

The FlowPortal pipeline runs 5 stages:

1. **Preprocessing** (`run-preprocess-flow-portal.py`): Frame extraction, resizing, mask generation (GroundingDINO + SAM), Canny/HED/Midas edge detection
2. **MatAnyone** (`pretrained_models/MatAnyone/inference_matanyone.py`): Video matting for alpha mask refinement
3. **Second Preprocessing**: Re-run with refined MatAnyone masks
4. **IC-Light** (`run-iclight-fc.py` or `run-iclight-fbc.py`): SD1.5-based relighting to generate reference image
5. **Flow Portal** (`run-flow-portal.py`): Wan2.1-based video generation with control

### Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `height x width` | Output resolution | 480x848 (sign lang), 480x720 (cat) |
| `k_frames` | Number of frames (must be 4n+1) | 41 or 49 |
| `fps` | Target frame rate | 8 |
| `transfer_blurring` | Detail preservation (high=0.4, mid=0.2, low=0.0) | high for quality |
| `partial_edit` | Preserve foreground, replace background only | true |
| `gpu_memory_mode` | VRAM management | no_offload (A100) |

## Troubleshooting

### sbatch not found
```bash
export PATH=/opt/slurm/20.11.4-13/bin:$PATH
```

### Disk quota exceeded on /home
The conda env is on `/scratch` to avoid `/home` quota issues. If HF cache fills up `/home`, set:
```bash
export HF_HOME=/scratch/zl6890/.cache/huggingface
```

### Missing kornia
```bash
/scratch/zl6890/miniconda/envs/flowportal/bin/pip install kornia
```

### Token auth errors for HF upload
Re-login with a write token:
```bash
/scratch/zl6890/miniconda/envs/flowportal/bin/huggingface-cli login --token <your_write_token>
```
