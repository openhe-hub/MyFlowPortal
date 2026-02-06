export CUDA_VISIBLE_DEVICES=0

echo "Cuda visible device: $CUDA_VISIBLE_DEVICES"

video_input="./datasets/video_part1/03gehw09sn.mp4"
video_name="sign_classroom"
prompt_src="a woman in a black shirt is signing in front of a blue curtain"
prompt_tar="a woman in a black shirt is signing in a bright classroom with a clean whiteboard behind her, soft fluorescent lighting, educational environment"
negative_prompt="blurry, low resolution, poorly drawn face, extra limbs, distorted"

height=480
width=848
k_frames=41
fps=8
partial_edit=true
do_preprocess=true
provided_bg=false
provided_bg_path=""
transfer_blurring="high"

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

cd "$(cd "$(dirname "$0")" && pwd)"
echo "Changed directory to $(pwd)"

if [ "$do_preprocess" = true ]; then
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
fi

if [ "$provided_bg" = true ]; then
    python run-iclight-fbc.py \
        --model_path "./pretrained_models/IC-Light/models/iclight_sd15_fbc.safetensors" \
        --base_path ${PWD} \
        --bg_path ${provided_bg_path} \
        --video_name ${video_name} \
        --prompt "${prompt_tar}" \
        --height ${height} \
        --width ${width} \
        --seed 0 \
        --steps 50 \
        ${partial_edit_flag}
else
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
fi

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
echo "Flow portal completed."
echo "--------------------------------------"
