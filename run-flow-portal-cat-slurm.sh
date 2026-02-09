#!/bin/bash
#SBATCH --job-name=FlowPortal-cat
#SBATCH --output=logs/flowportal_cat_%j.out
#SBATCH --error=logs/flowportal_cat_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=nvidia

# ============================================================================
# FlowPortal Cat Demo - SLURM Job
# ============================================================================

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH=/scratch/zl6890/miniconda/envs/flowportal/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

echo "=============================================================================="
echo "Job started at: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=============================================================================="

cd /scratch/zl6890/Workspace/zhewen/FlowPortal
mkdir -p logs

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
do_preprocess=false
provided_bg=false
provided_bg_path=""
transfer_blurring="mid"

if [ "$partial_edit" = true ]; then
    partial_edit_flag="--partial_edit True"
else
    partial_edit_flag=""
fi

if [ "$transfer_blurring" = "high" ]; then
    transfer_blurring=0.4
elif [ "$transfer_blurring" = "mid" ]; then
    transfer_blurring=0.2
else
    transfer_blurring=0.0
fi

echo "--------------------------------------"
echo "Preprocessing..."
echo "--------------------------------------"
python run-preprocess-flow-portal.py \
    --video_input ${video_input} \
    --video_name ${video_name} \
    --k_frames ${k_frames} \
    --height ${height} \
    --width ${width} \
    --fps ${fps} \
    --batch_size 7
python ./pretrained_models/MatAnyone/inference_matanyone.py \
    -i ./datasets/${video_name}/${video_name}.mp4 \
    -o ./datasets/${video_name}/matanyone \
    -m ./datasets/${video_name}/masks/mask_0000.png \
    -c ./pretrained_models/MatAnyone/pretrained_models/matanyone.pth
python run-preprocess-flow-portal.py \
    --video_input ${video_input} \
    --video_name ${video_name} \
    --k_frames ${k_frames} \
    --height ${height} \
    --width ${width} \
    --fps ${fps} \
    --batch_size 7 \
    --using_existing_masks ./datasets/${video_name}/matanyone/${video_name}_pha.mp4
echo "Preprocessing completed."
echo "--------------------------------------"

python run-iclight-fc.py \
    --model_path "./pretrained_models/IC-Light/models/iclight_sd15_fc.safetensors" \
    --base_path ${PWD} \
    --video_name ${video_name} \
    --prompt "${prompt_tar}" \
    --height ${height} \
    --width ${width} \
    --seed 0 \
    --steps 50 \
    ${partial_edit_flag}

echo "--------------------------------------"
echo "Running flow portal..."
echo "--------------------------------------"
python run-flow-portal.py \
    --video_name ${video_name} \
    --k_frames ${k_frames} \
    --fps ${fps} \
    --height ${height} \
    --width ${width} \
    --prompt_tar "${prompt_tar}" \
    --prompt_src "${prompt_src}" \
    --negative_prompt "blurry, low resolution, poorly drawn face, extra limbs, distorted" \
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
    --model_name "./pretrained_models/models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"

EXIT_CODE=$?

echo "=============================================================================="
echo "Job completed at: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
    echo "Results: results/flow-edit-${video_name}/"
else
    echo "Status: FAILED (exit code: $EXIT_CODE)"
fi
echo "=============================================================================="

exit $EXIT_CODE
