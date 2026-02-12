#!/bin/bash
#SBATCH --job-name=FP-h2s-uns
#SBATCH --output=/CVPR/zhewen/FlowPortal/logs/h2s_unselected_%A_%a.out
#SBATCH --error=/CVPR/zhewen/FlowPortal/logs/h2s_unselected_%A_%a.err
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=96:00:00
#SBATCH --array=0-9

# ============================================================================
# FlowPortal How2Sign Unselected Batch - DGX Spark Cluster
# 10-GPU array job, each task processes one chunk of unselected videos
# ============================================================================

PROJECT=/CVPR/zhewen/FlowPortal
LOCAL_MODELS=/home/cvpr/zhewen/flowportal_models
PYTHON=/CVPR/python_envs/flowportal/bin/python

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=${LOCAL_MODELS}/.huggingface_cache
export TORCH_HOME=${PROJECT}/pretrained_models/torch_hub_cache
export TORCH_CUDA_ARCH_LIST="12.0"
export TORCHINDUCTOR_CACHE_DIR=${LOCAL_MODELS}/.torchinductor_cache
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

cd $PROJECT
mkdir -p logs

CHUNK_ID=$SLURM_ARRAY_TASK_ID
CHUNK_FILE="${PROJECT}/scripts/h2s_unselected_chunk${CHUNK_ID}.csv"
PROMPTS_FILE="${PROJECT}/scripts/h2s_prompts.txt"
PROGRESS_FILE="${PROJECT}/logs/h2s_unselected_progress_${CHUNK_ID}.log"

T() { date +%s; }
JOB_T0=$(T)

echo "=============================================================================="
echo "Array Job started at: $(date)"
echo "Array Task ID: $CHUNK_ID"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Running on: $(hostname)"
echo "Python: $($PYTHON --version)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "Chunk file: $CHUNK_FILE"
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

# Read source prompt
PROMPT_SRC=$(grep "^SRC|" "$PROMPTS_FILE" | cut -d'|' -f2)
echo "Source prompt: $PROMPT_SRC"

# Read target prompts into array
declare -A PROMPTS_TAR
while IFS='|' read -r pid prompt; do
    if [ "$pid" != "SRC" ]; then
        PROMPTS_TAR[$pid]="$prompt"
    fi
done < "$PROMPTS_FILE"

NEGATIVE_PROMPT="blurry, low resolution, poorly drawn face, extra limbs, distorted"

# Count total videos in chunk
TOTAL=$(tail -n +2 "$CHUNK_FILE" | wc -l)
COUNT=0
SUCCEEDED=0
FAILED=0

echo "Total videos to process: $TOTAL"
echo "=============================================================================="

# Process each video in the chunk
tail -n +2 "$CHUNK_FILE" | while IFS=',' read -r video_name n_frames k_frames prompt_id; do
    COUNT=$((COUNT + 1))
    PROMPT_TAR="${PROMPTS_TAR[$prompt_id]}"
    VIDEO_INPUT="./datasets/How2Sign/train_mp4/${video_name}.mp4"
    VIDEO_NAME="${video_name}"

    echo ""
    echo "======================================================================"
    echo "[${COUNT}/${TOTAL}] Processing: ${video_name}"
    echo "  k_frames=${k_frames}, prompt_id=${prompt_id}"
    echo "  Started at: $(date)"
    echo "======================================================================"

    # Skip if already completed
    RESULT_DIR="./results/flow-edit-${VIDEO_NAME}"
    if [ -f "${RESULT_DIR}/output_video.mp4" ]; then
        echo "  SKIPPED: output already exists"
        SUCCEEDED=$((SUCCEEDED + 1))
        echo "${COUNT}/${TOTAL} SKIPPED ${video_name}" >> "$PROGRESS_FILE"
        continue
    fi

    # Create dataset directory
    mkdir -p "./datasets/${VIDEO_NAME}"

    # Step 1: Preprocessing (first pass) — canny edge mode (pure OpenCV, no HED/Midas)
    echo "  Step 1/5: Preprocessing (first pass)..."
    S1=$(T)
    $PYTHON run-preprocess-flow-portal.py \
        --video_input "${VIDEO_INPUT}" \
        --video_name "${VIDEO_NAME}" \
        --k_frames ${k_frames} \
        --height 480 \
        --width 848 \
        --fps 8 \
        --batch_size 7 \
        --edge_mode canny \
        --annotators_path "${ANN_DST}" 2>&1
    if [ $? -ne 0 ]; then
        echo "  FAILED at preprocessing step 1 ($(($(T) - S1))s)"
        FAILED=$((FAILED + 1))
        echo "${COUNT}/${TOTAL} FAILED ${video_name} preprocess1" >> "$PROGRESS_FILE"
        continue
    fi
    echo "  Step 1 done in $(($(T) - S1))s"

    # Step 2: MatAnyone
    echo "  Step 2/5: MatAnyone..."
    S2=$(T)
    $PYTHON ./pretrained_models/MatAnyone/inference_matanyone.py \
        -i "./datasets/${VIDEO_NAME}/${video_name}.mp4" \
        -o "./datasets/${VIDEO_NAME}/matanyone" \
        -m "./datasets/${VIDEO_NAME}/masks/mask_0000.png" \
        -c ${MAT_DST}/matanyone.pth 2>&1
    if [ $? -ne 0 ]; then
        echo "  FAILED at MatAnyone ($(($(T) - S2))s)"
        FAILED=$((FAILED + 1))
        echo "${COUNT}/${TOTAL} FAILED ${video_name} matanyone" >> "$PROGRESS_FILE"
        continue
    fi
    echo "  Step 2 done in $(($(T) - S2))s"

    # Step 3: Preprocessing (second pass with MatAnyone masks) — skip edge & frame extraction
    echo "  Step 3/5: Preprocessing (second pass)..."
    S3=$(T)
    $PYTHON run-preprocess-flow-portal.py \
        --video_input "${VIDEO_INPUT}" \
        --video_name "${VIDEO_NAME}" \
        --k_frames ${k_frames} \
        --height 480 \
        --width 848 \
        --fps 8 \
        --batch_size 7 \
        --using_existing_masks "./datasets/${VIDEO_NAME}/matanyone/${video_name}_pha.mp4" \
        --skip_edge_detection \
        --skip_frame_extraction 2>&1
    if [ $? -ne 0 ]; then
        echo "  FAILED at preprocessing step 2 ($(($(T) - S3))s)"
        FAILED=$((FAILED + 1))
        echo "${COUNT}/${TOTAL} FAILED ${video_name} preprocess2" >> "$PROGRESS_FILE"
        continue
    fi
    echo "  Step 3 done in $(($(T) - S3))s"

    # Step 4: IC-Light FC — optimized (15 steps, no highres)
    echo "  Step 4/5: IC-Light FC..."
    S4=$(T)
    $PYTHON run-iclight-fc.py \
        --model_path "${ICL_DST}/iclight_sd15_fc.safetensors" \
        --base_path "${PWD}" \
        --video_name "${VIDEO_NAME}" \
        --prompt "${PROMPT_TAR}" \
        --height 480 \
        --width 848 \
        --seed 0 \
        --steps 15 \
        --highres_scale 1.0 \
        --highres_denoise 0.0 \
        --partial_edit True 2>&1
    if [ $? -ne 0 ]; then
        echo "  FAILED at IC-Light ($(($(T) - S4))s)"
        FAILED=$((FAILED + 1))
        echo "${COUNT}/${TOTAL} FAILED ${video_name} iclight" >> "$PROGRESS_FILE"
        continue
    fi
    echo "  Step 4 done in $(($(T) - S4))s"

    # Step 5: FlowPortal — optimized (20 steps, Flow_DPM++, TeaCache)
    echo "  Step 5/5: FlowPortal..."
    S5=$(T)
    $PYTHON run-flow-portal.py \
        --video_name "${VIDEO_NAME}" \
        --k_frames ${k_frames} \
        --fps 8 \
        --height 480 \
        --width 848 \
        --prompt_tar "${PROMPT_TAR}" \
        --prompt_src "${PROMPT_SRC}" \
        --negative_prompt "${NEGATIVE_PROMPT}" \
        --src_guidance_scale 6.0 \
        --tar_guidance_scale 6.0 \
        --gpu_memory_mode "no_offload" \
        --seed 42 \
        --edit_amplifier 1.0 \
        --cache_times 10 \
        --n_avg 1 \
        --partial_edit True \
        --accuracy 16 \
        --src_blurring 0.5 \
        --transfer_blurring 0.4 \
        --model_name "${WAN_DST}" \
        --num_inference_steps 20 \
        --sampler_name "Flow_DPM++" \
        --enable_teacache True \
        --teacache_threshold 0.20 \
        --cfg_skip_ratio 0.25 2>&1
    if [ $? -ne 0 ]; then
        echo "  FAILED at FlowPortal ($(($(T) - S5))s)"
        FAILED=$((FAILED + 1))
        echo "${COUNT}/${TOTAL} FAILED ${video_name} flowportal" >> "$PROGRESS_FILE"
        continue
    fi
    echo "  Step 5 done in $(($(T) - S5))s"

    SUCCEEDED=$((SUCCEEDED + 1))
    echo "  DONE at: $(date)"
    echo "${COUNT}/${TOTAL} OK ${video_name}" >> "$PROGRESS_FILE"

    # Cleanup intermediate files to save disk space
    rm -rf "./datasets/${VIDEO_NAME}/video_frames"
    rm -rf "./datasets/${VIDEO_NAME}/video_canny"
    rm -rf "./datasets/${VIDEO_NAME}/masked_canny"
    rm -rf "./datasets/${VIDEO_NAME}/matanyone"
    rm -rf "./datasets/${VIDEO_NAME}/masks"

done

TOTAL_TIME=$(($(T) - JOB_T0))
echo ""
echo "=============================================================================="
echo "Chunk ${CHUNK_ID} completed at: $(date)"
echo "Succeeded: ${SUCCEEDED}, Failed: ${FAILED}, Total: ${TOTAL}"
echo "Total time: ${TOTAL_TIME}s ($((TOTAL_TIME/60))m $((TOTAL_TIME%60))s)"
echo "=============================================================================="
