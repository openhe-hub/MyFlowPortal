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

## Phoenix-2014-T Batch Processing

### Dataset

Phoenix-2014-T sign language dataset (7096 train videos) is available on Jubail:

```
/scratch/zl6890/Workspace/zhewen/FlowPortal/datasets/PHOENIX-2014-T/train_mp4/
```

- Original frames (210x260 PNG) were converted to MP4 at 25fps using ffmpeg
- Local symlink: `datasets/PHOENIX-2014-T -> /home/nyuair/junyi/SLRT/PHOENIX-2014-T-release-v3/PHOENIX-2014-T`

### Manifest & Prompts

The batch job uses a manifest CSV (`scripts/phoenix/phoenix_manifest.csv`) with 7039 valid videos (57 skipped as too short). Each video is assigned:

- **Dynamic k_frames** based on frame count: 41 (2531), 21 (3507), 13 (756), 9 (245)
- **Random prompt** from 10 target backgrounds (see `scripts/phoenix/phoenix_prompts.txt`)

| Prompt ID | Scene |
|-----------|-------|
| 0 | Modern office |
| 1 | Library |
| 2 | City street |
| 3 | Green park |
| 4 | Cozy cafe |
| 5 | Classroom |
| 6 | Living room |
| 7 | Beach |
| 8 | Night cityscape |
| 9 | Garden |

Source prompt (shared): `a person in dark clothing is signing in front of a plain gray background in a TV studio`

### Running the Batch Job

```bash
# Submit 4-GPU array job (must use A100 nodes, V100 doesn't support bfloat16)
ssh chatsign@10.224.35.17 "ssh jubail 'export PATH=/opt/slurm/20.11.4-13/bin:\$PATH && \
  cd /scratch/zl6890/Workspace/zhewen/FlowPortal && \
  sbatch --constraint=a100 run-phoenix-batch-slurm.sh'"
```

- **SLURM array**: `--array=0-3`, each task processes ~1760 videos from its chunk
- **Time limit**: 96 hours
- **Resume support**: Completed videos (with `output_video.mp4`) are automatically skipped
- **Disk cleanup**: Intermediate files are removed after each video

### Monitoring Progress

```bash
# Video counts per chunk
ssh chatsign@10.224.35.17 "ssh jubail 'wc -l /scratch/zl6890/Workspace/zhewen/FlowPortal/logs/phoenix_progress_*.log'"

# Check failures
ssh chatsign@10.224.35.17 "ssh jubail 'grep FAILED /scratch/zl6890/Workspace/zhewen/FlowPortal/logs/phoenix_progress_*.log'"

# Job status
ssh chatsign@10.224.35.17 "ssh jubail 'export PATH=/opt/slurm/20.11.4-13/bin:\$PATH && squeue -u zl6890'"
```

### Important Notes

- **Must use A100 nodes** (`--constraint=a100`): V100 (dn* nodes) does not support bfloat16, causing FlowPortal to fail
- **Output**: `results/flow-edit-{video_name}/output_video.mp4` and `compare_output_video.mp4`
- **Estimated time**: ~77 hours total with 4x A100 (~2 min/video on A100-80G, ~4 min on A100-40G)

## How2Sign Batch Processing

### Dataset

How2Sign is a large-scale multimodal dataset for American Sign Language (ASL). We select 8,000 videos from the 31,047 train set for background replacement processing.

- **Source resolution**: 1280x720
- **Output resolution**: 480x848
- **Input FPS**: 24 → target 8 FPS
- **Format**: Already MP4 (no conversion needed)

Local source:
```
datasets/How2Sign/how2sign/sentence_level/train/rgb_front/raw_videos/*.mp4
```

Jubail location:
```
/scratch/zl6890/Workspace/zhewen/FlowPortal/datasets/How2Sign/train_mp4/
```

### Manifest & Prompts

Generated by `scripts/h2s/generate_h2s_manifest.py` with seed=42:

- **8,000 videos** randomly selected from valid pool (videos with ≥27 frames)
- **Dynamic k_frames** based on frame count: 49, 41, 21, 13, or 9
- **Random prompt** from 10 target backgrounds (see `scripts/h2s/h2s_prompts.txt`)

Source prompt: `a person is signing in front of a plain background in a studio`

### Transfer to Jubail

1. Create remote directory:
```bash
ssh chatsign@10.224.35.17 "ssh jubail 'mkdir -p /scratch/zl6890/Workspace/zhewen/FlowPortal/datasets/How2Sign/train_mp4'"
```

2. Transfer 8,000 selected videos (~8GB) via tar pipe:
```bash
cd /home/nyuair/zhewen/FlowPortalRelease/datasets/How2Sign/how2sign/sentence_level/train/rgb_front/raw_videos
tar cf - -T /home/nyuair/zhewen/FlowPortalRelease/scripts/h2s/h2s_selected_videos.txt | \
  ssh chatsign@10.224.35.17 "ssh jubail 'cd /scratch/zl6890/Workspace/zhewen/FlowPortal/datasets/How2Sign/train_mp4 && tar xf -'"
```

3. Transfer scripts:
```bash
for f in scripts/h2s/h2s_manifest.csv scripts/h2s/h2s_manifest_chunk{0,1,2,3}.csv scripts/h2s/h2s_prompts.txt run-h2s-batch-slurm.sh; do
  cat "$f" | ssh chatsign@10.224.35.17 "ssh jubail 'cat > /scratch/zl6890/Workspace/zhewen/FlowPortal/$f'"
done
```

### Running the Batch Job

```bash
# Submit 4-GPU array job (must use A100 nodes)
ssh chatsign@10.224.35.17 "ssh jubail 'export PATH=/opt/slurm/20.11.4-13/bin:\$PATH && \
  cd /scratch/zl6890/Workspace/zhewen/FlowPortal && \
  sbatch --constraint=a100 run-h2s-batch-slurm.sh'"
```

- **SLURM array**: `--array=0-3`, each task processes ~2,000 videos from its chunk
- **Time limit**: 96 hours
- **Resume support**: Completed videos (with `output_video.mp4`) are automatically skipped
- **Disk cleanup**: Intermediate files are removed after each video

### Monitoring Progress

```bash
# Video counts per chunk
ssh chatsign@10.224.35.17 "ssh jubail 'wc -l /scratch/zl6890/Workspace/zhewen/FlowPortal/logs/h2s_progress_*.log'"

# Check failures
ssh chatsign@10.224.35.17 "ssh jubail 'grep FAILED /scratch/zl6890/Workspace/zhewen/FlowPortal/logs/h2s_progress_*.log'"

# Job status
ssh chatsign@10.224.35.17 "ssh jubail 'export PATH=/opt/slurm/20.11.4-13/bin:\$PATH && squeue -u zl6890'"
```

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

### bfloat16 not supported (V100 nodes)
V100 GPUs (dn* nodes) do not support bfloat16. Always add `--constraint=a100` when submitting jobs:
```bash
sbatch --constraint=a100 run-phoenix-batch-slurm.sh
```

### Token auth errors for HF upload
Re-login with a write token:
```bash
/scratch/zl6890/miniconda/envs/flowportal/bin/huggingface-cli login --token <your_write_token>
```
