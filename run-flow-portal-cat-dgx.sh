#!/bin/bash
#SBATCH --job-name=FlowPortal-cat
#SBATCH --output=/CVPR/zhewen/FlowPortal/logs/flowportal_cat_%j.out
#SBATCH --error=/CVPR/zhewen/FlowPortal/logs/flowportal_cat_%j.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00

# ============================================================================
# FlowPortal Cat Demo - DGX Spark Cluster
# Strategy: Cache models on local NVMe (persistent), run code from /CVPR
# ============================================================================

set -e

PROJECT=/CVPR/zhewen/FlowPortal
LOCAL_MODELS=/home/cvpr/zhewen/flowportal_models
PYTHON=/CVPR/python_envs/flowportal/bin/python

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HOME=${LOCAL_MODELS}/.huggingface_cache
export TORCH_CUDA_ARCH_LIST="12.0"
export TORCHINDUCTOR_CACHE_DIR=${LOCAL_MODELS}/.torchinductor_cache

cd $PROJECT
mkdir -p logs

T() { date +%s; }
JOB_T0=$(T)

echo "=============================================================================="
echo "Job started at: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Python: $($PYTHON --version)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=============================================================================="

# --- Cache models on local NVMe (persistent across jobs on same node) ---
WAN_SRC=${PROJECT}/pretrained_models/models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control
WAN_DST=${LOCAL_MODELS}/Wan2.1-Fun-V1.1-1.3B-Control
ICL_SRC=${PROJECT}/pretrained_models/IC-Light/models
ICL_DST=${LOCAL_MODELS}/IC-Light/models
MAT_SRC=${PROJECT}/pretrained_models/MatAnyone/pretrained_models
MAT_DST=${LOCAL_MODELS}/MatAnyone/pretrained_models
ANN_SRC=${PROJECT}/pretrained_models/Annotators
ANN_DST=${LOCAL_MODELS}/Annotators

CACHE_T0=$(T)
if [ -f "${WAN_DST}/models_t5_umt5-xxl-enc-bf16.pth" ]; then
    echo "Models already cached on local NVMe, skipping copy."
else
    echo "Caching models to local NVMe (first run on this node)..."
    mkdir -p ${WAN_DST}/google/umt5-xxl ${WAN_DST}/xlm-roberta-large ${ICL_DST} ${MAT_DST} ${ANN_DST}
    # Wan2.1 model files (largest, ~19GB)
    cp ${WAN_SRC}/diffusion_pytorch_model.safetensors ${WAN_DST}/
    cp ${WAN_SRC}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth ${WAN_DST}/
    cp ${WAN_SRC}/models_t5_umt5-xxl-enc-bf16.pth ${WAN_DST}/
    cp ${WAN_SRC}/Wan2.1_VAE.pth ${WAN_DST}/
    cp ${WAN_SRC}/config.json ${WAN_SRC}/configuration.json ${WAN_DST}/
    cp -r ${WAN_SRC}/google/umt5-xxl/* ${WAN_DST}/google/umt5-xxl/
    cp -r ${WAN_SRC}/xlm-roberta-large/* ${WAN_DST}/xlm-roberta-large/
    # IC-Light models (~3.4GB)
    cp ${ICL_SRC}/*.safetensors ${ICL_DST}/
    # MatAnyone (~135MB)
    cp ${MAT_SRC}/matanyone.pth ${MAT_DST}/
    # Annotators - HED/Midas weights (~470MB)
    cp -r ${ANN_SRC}/* ${ANN_DST}/
    echo "Model caching completed in $(($(T) - CACHE_T0)) seconds"
fi

# Cache Annotators separately (may not exist from earlier jobs)
if [ ! -d "${ANN_DST}" ] || [ -z "$(ls -A ${ANN_DST} 2>/dev/null)" ]; then
    echo "Caching Annotators to local NVMe..."
    mkdir -p ${ANN_DST}
    cp -r ${ANN_SRC}/* ${ANN_DST}/
    echo "Annotators cached."
fi

video_input="./datasets/cat.mp4"
video_name="cat_ic"
prompt_src="A cat is outside on the floor."
prompt_tar="A cat is standing in a broad living room on a wooden floor. The sunlight shine from the window and shine the cat's shadow and create a warm scene. The background is clear and bright."
negative_prompt="blurry, low resolution, poorly drawn face, extra limbs, distorted"

height=480
width=720
k_frames=49
fps=8
partial_edit=true
transfer_blurring="mid"

partial_edit_flag=""
if [ "$partial_edit" = true ]; then
    partial_edit_flag="--partial_edit True"
fi

if [ "$transfer_blurring" = "high" ]; then
    transfer_blurring=0.4
elif [ "$transfer_blurring" = "mid" ]; then
    transfer_blurring=0.2
else
    transfer_blurring=0.0
fi

# Stage 1: Preprocessing
echo "--------------------------------------"
echo "Stage 1: Preprocessing... [$(date +%H:%M:%S)]"
echo "--------------------------------------"
S1=$(T)
$PYTHON run-preprocess-flow-portal.py \
    --video_input ${video_input} \
    --video_name ${video_name} \
    --k_frames ${k_frames} \
    --height ${height} \
    --width ${width} \
    --fps ${fps} \
    --batch_size 7 \
    --edge_mode canny \
    --annotators_path "${ANN_DST}"
echo "Stage 1 done in $(($(T) - S1))s"

# Stage 2: MatAnyone matting
echo "--------------------------------------"
echo "Stage 2: MatAnyone matting... [$(date +%H:%M:%S)]"
echo "--------------------------------------"
S2=$(T)
$PYTHON ./pretrained_models/MatAnyone/inference_matanyone.py \
    -i ./datasets/${video_name}/${video_name}.mp4 \
    -o ./datasets/${video_name}/matanyone \
    -m ./datasets/${video_name}/masks/mask_0000.png \
    -c ${MAT_DST}/matanyone.pth
echo "Stage 2 done in $(($(T) - S2))s"

# Stage 3: Re-preprocessing with MatAnyone masks
echo "--------------------------------------"
echo "Stage 3: Re-preprocessing with masks... [$(date +%H:%M:%S)]"
echo "--------------------------------------"
S3=$(T)
$PYTHON run-preprocess-flow-portal.py \
    --video_input ${video_input} \
    --video_name ${video_name} \
    --k_frames ${k_frames} \
    --height ${height} \
    --width ${width} \
    --fps ${fps} \
    --batch_size 7 \
    --using_existing_masks ./datasets/${video_name}/matanyone/${video_name}_pha.mp4 \
    --skip_edge_detection \
    --skip_frame_extraction
echo "Stage 3 done in $(($(T) - S3))s"

# Stage 4: IC-Light relighting
echo "--------------------------------------"
echo "Stage 4: IC-Light relighting... [$(date +%H:%M:%S)]"
echo "--------------------------------------"
S4=$(T)
$PYTHON run-iclight-fc.py \
    --model_path "${ICL_DST}/iclight_sd15_fc.safetensors" \
    --base_path ${PWD} \
    --video_name ${video_name} \
    --prompt "${prompt_tar}" \
    --height ${height} \
    --width ${width} \
    --seed 0 \
    --steps 15 \
    --highres_scale 1.0 \
    --highres_denoise 0.0 \
    ${partial_edit_flag}
echo "Stage 4 done in $(($(T) - S4))s"

# Stage 5: FlowPortal video generation
echo "--------------------------------------"
echo "Stage 5: FlowPortal generation... [$(date +%H:%M:%S)]"
echo "--------------------------------------"
S5=$(T)
$PYTHON run-flow-portal.py \
    --video_name ${video_name} \
    --k_frames ${k_frames} \
    --fps ${fps} \
    --height ${height} \
    --width ${width} \
    --prompt_tar "${prompt_tar}" \
    --prompt_src "${prompt_src}" \
    --negative_prompt "${negative_prompt}" \
    --src_guidance_scale 6.0 \
    --tar_guidance_scale 6.0 \
    --gpu_memory_mode "no_offload" \
    --seed 42 \
    --edit_amplifier 1.0 \
    --cache_times 10 \
    --n_avg 1 \
    ${partial_edit_flag} \
    --accuracy 16 \
    --src_blurring 0.5 \
    --transfer_blurring ${transfer_blurring} \
    --model_name "${WAN_DST}" \
    --num_inference_steps 20 \
    --sampler_name "Flow_DPM++" \
    --enable_teacache True \
    --teacache_threshold 0.20 \
    --cfg_skip_ratio 0.25

EXIT_CODE=$?
echo "Stage 5 done in $(($(T) - S5))s"

TOTAL=$(($(T) - JOB_T0))
echo "=============================================================================="
echo "Job completed at: $(date)"
echo "Total time: ${TOTAL}s ($((TOTAL/60))m $((TOTAL%60))s)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
    echo "Results: results/flow-edit-${video_name}/"
else
    echo "Status: FAILED (exit code: $EXIT_CODE)"
fi
echo "=============================================================================="

exit $EXIT_CODE
