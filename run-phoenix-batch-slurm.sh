#!/bin/bash
#SBATCH --job-name=FP-phoenix
#SBATCH --output=logs/phoenix_batch_%A_%a.out
#SBATCH --error=logs/phoenix_batch_%A_%a.err
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=nvidia
#SBATCH --array=0-3

# ============================================================================
# FlowPortal Phoenix Batch - SLURM Array Job
# Each array task processes ~1760 videos from its chunk
# ============================================================================

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH=/scratch/zl6890/miniconda/envs/flowportal/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

PROJ_DIR=/scratch/zl6890/Workspace/zhewen/FlowPortal
cd $PROJ_DIR
mkdir -p logs

CHUNK_ID=$SLURM_ARRAY_TASK_ID
CHUNK_FILE="${PROJ_DIR}/scripts/phoenix_manifest_chunk${CHUNK_ID}.csv"
PROMPTS_FILE="${PROJ_DIR}/scripts/phoenix_prompts.txt"
PROGRESS_FILE="${PROJ_DIR}/logs/phoenix_progress_${CHUNK_ID}.log"

echo "=============================================================================="
echo "Array Job started at: $(date)"
echo "Array Task ID: $CHUNK_ID"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Running on: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "Chunk file: $CHUNK_FILE"
echo "=============================================================================="

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
    VIDEO_INPUT="./datasets/PHOENIX-2014-T/train_mp4/${video_name}.mp4"
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

    # Step 1: Preprocessing (first pass)
    echo "  Step 1/5: Preprocessing (first pass)..."
    python run-preprocess-flow-portal.py \
        --video_input "${VIDEO_INPUT}" \
        --video_name "${VIDEO_NAME}" \
        --k_frames ${k_frames} \
        --height 480 \
        --width 384 \
        --fps 8 \
        --batch_size 7 2>&1
    if [ $? -ne 0 ]; then
        echo "  FAILED at preprocessing step 1"
        FAILED=$((FAILED + 1))
        echo "${COUNT}/${TOTAL} FAILED ${video_name} preprocess1" >> "$PROGRESS_FILE"
        continue
    fi

    # Step 2: MatAnyone
    echo "  Step 2/5: MatAnyone..."
    python ./pretrained_models/MatAnyone/inference_matanyone.py \
        -i "./datasets/${VIDEO_NAME}/${video_name}.mp4" \
        -o "./datasets/${VIDEO_NAME}/matanyone" \
        -m "./datasets/${VIDEO_NAME}/masks/mask_0000.png" \
        -c ./pretrained_models/MatAnyone/pretrained_models/matanyone.pth 2>&1
    if [ $? -ne 0 ]; then
        echo "  FAILED at MatAnyone"
        FAILED=$((FAILED + 1))
        echo "${COUNT}/${TOTAL} FAILED ${video_name} matanyone" >> "$PROGRESS_FILE"
        continue
    fi

    # Step 3: Preprocessing (second pass with MatAnyone masks)
    echo "  Step 3/5: Preprocessing (second pass)..."
    python run-preprocess-flow-portal.py \
        --video_input "${VIDEO_INPUT}" \
        --video_name "${VIDEO_NAME}" \
        --k_frames ${k_frames} \
        --height 480 \
        --width 384 \
        --fps 8 \
        --batch_size 7 \
        --using_existing_masks "./datasets/${VIDEO_NAME}/matanyone/${video_name}_pha.mp4" 2>&1
    if [ $? -ne 0 ]; then
        echo "  FAILED at preprocessing step 2"
        FAILED=$((FAILED + 1))
        echo "${COUNT}/${TOTAL} FAILED ${video_name} preprocess2" >> "$PROGRESS_FILE"
        continue
    fi

    # Step 4: IC-Light (text-prompt based, FC mode)
    echo "  Step 4/5: IC-Light FC..."
    python run-iclight-fc.py \
        --model_path "./pretrained_models/IC-Light/models/iclight_sd15_fc.safetensors" \
        --base_path "${PWD}" \
        --video_name "${VIDEO_NAME}" \
        --prompt "${PROMPT_TAR}" \
        --height 480 \
        --width 384 \
        --seed 0 \
        --steps 50 \
        --partial_edit True 2>&1
    if [ $? -ne 0 ]; then
        echo "  FAILED at IC-Light"
        FAILED=$((FAILED + 1))
        echo "${COUNT}/${TOTAL} FAILED ${video_name} iclight" >> "$PROGRESS_FILE"
        continue
    fi

    # Step 5: FlowPortal
    echo "  Step 5/5: FlowPortal..."
    python run-flow-portal.py \
        --video_name "${VIDEO_NAME}" \
        --k_frames ${k_frames} \
        --fps 8 \
        --height 480 \
        --width 384 \
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
        --model_name "./pretrained_models/models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control" 2>&1
    if [ $? -ne 0 ]; then
        echo "  FAILED at FlowPortal"
        FAILED=$((FAILED + 1))
        echo "${COUNT}/${TOTAL} FAILED ${video_name} flowportal" >> "$PROGRESS_FILE"
        continue
    fi

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

echo ""
echo "=============================================================================="
echo "Chunk ${CHUNK_ID} completed at: $(date)"
echo "Succeeded: ${SUCCEEDED}, Failed: ${FAILED}, Total: ${TOTAL}"
echo "=============================================================================="
