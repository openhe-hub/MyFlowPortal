import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 选择显卡使用
import sys

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer
import imageio
# current_file_path = os.path.abspath(__file__)
current_file_path = os.path.abspath("")
# print(current_file_path)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                               WanT5EncoderModel, WanTransformer3DModel)
from videox_fun.data.dataset_image_video import process_pose_file
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import WanFunControlPipeline # , WanPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
# from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent, get_image_latent,
                                    get_video_to_video_latent,
                                    save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

import argparse
import cv2
def parse_arg():
    parser = argparse.ArgumentParser(description="Preprocess video for flow editing")
    parser.add_argument("--direct_inference", action='store_true', help="Run direct inference without flow editing")
    parser.add_argument("--video_name", type=str, help="The name of input video in datasets", required=True)
    parser.add_argument("--k_frames", type=int,  required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=7)
    parser.add_argument("--model_name", type=str, required=True, help="The path of model")

    parser.add_argument("--height", type=int, required=True, help="Height of the video frames")
    parser.add_argument("--width", type=int, required=True, help="Width of the video frames")

    parser.add_argument("--prompt_tar", type=str, help="Target prompt for flow editing", required=True)
    parser.add_argument("--prompt_src", type=str, help="Source prompt for flow editing", required=True)
    parser.add_argument("--negative_prompt", type=str, help="Negative prompt for both", required=True)
    
    parser.add_argument("--src_guidance_scale", type=float, required=True, help="Guidance scale for source video")
    parser.add_argument("--tar_guidance_scale", type=float, required=True, help="Guidance scale for target video")

    parser.add_argument("--gpu_memory_mode", type=str, default="sequential_cpu_offload", choices=["no_offload", "sequential_cpu_offload"], help="GPU memory mode for the pipeline")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--edit_amplifier", type=float, default=1.0, help="Amplifier for the editing strength")
    parser.add_argument("--cache_times", type=int, default=0, help="Cache steps for source side generation")
    parser.add_argument("--n_avg", type=int, default=1, help="Number of averages for noise")
    parser.add_argument("--partial_edit", type=bool, default=False, help="Whether to use partial editing with mask")
    parser.add_argument("--accuracy", type=int, default=16, choices=[16, 32], help="Model accuracy, 16 for torch.bfloat16, 32 for torch.float32")
    parser.add_argument("--src_blurring", type=float, default=0.0, help="Blurring for source side generation, good for detail")
    parser.add_argument("--transfer_blurring", type=float, default=0.0, help="Blurring for transfer delta, good for detail")
    return parser.parse_args()
args = parse_arg()

print("Partial edit:", args.partial_edit)

video_name = args.video_name

# 使用更长的neg prompt如"模糊，突变，变形，失真，画面暗，文本字幕，画面固定，连环画，漫画，线稿，没有主体。"，可以增加稳定性
# 在neg prompt中添加"安静，固定"等词语可以增加动态性。
prompt_tar = args.prompt_tar
prompt_src = args.prompt_src

negative_prompt_tar = negative_prompt_src = args.negative_prompt

control_video_tar = os.path.join(current_file_path, "datasets", video_name, "masked_canny", "masked_canny_video.mp4")
control_video_src = os.path.join(current_file_path, "datasets", video_name, "masked_canny", "masked_canny_video.mp4")
if not args.partial_edit:
    control_video_tar = os.path.join(current_file_path, "datasets", video_name, "video_canny", "canny_video.mp4")
    control_video_src = os.path.join(current_file_path, "datasets", video_name, "video_canny", "canny_video.mp4")
partial_edit_mask = os.path.join(current_file_path, "datasets", video_name, "masks", "mask_video.mp4")
ref_image_tar_path      = os.path.join(current_file_path, "datasets", video_name, f"{video_name}_reference.png")
ref_image_src_path      = os.path.join(current_file_path, "datasets", video_name, "video_frames", "frame_0000.png")

src_video_path          = os.path.join(current_file_path, "datasets", video_name, f"{video_name}.mp4")
save_path               = os.path.join(current_file_path, "results", f"flow-edit-{video_name}/")

src_guidance_scale = args.src_guidance_scale
tar_guidance_scale = args.tar_guidance_scale

sample_size         = [args.height, args.width]
video_length        = args.k_frames
fps                 = args.fps



GPU_memory_mode     = args.gpu_memory_mode # "sequential_cpu_offload"  # "no offload" 

ulysses_degree      = 1
ring_degree         = 1 # 多卡运行（ 需要满足 ulysses_degree * ring_degree = gpu-nums ）
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# Support TeaCache.
enable_teacache     = False
# Recommended to be set between 0.05 and 0.30. A larger threshold can cache more steps, speeding up the inference process, 
# but it may cause slight differences between the generated content and the original content.
# # --------------------------------------------------------------------------------------------------- #
# | Model Name          | threshold | Model Name          | threshold | Model Name          | threshold |
# | Wan2.1-T2V-1.3B     | 0.05~0.10 | Wan2.1-T2V-14B      | 0.10~0.15 | Wan2.1-I2V-14B-720P | 0.20~0.30 |
# | Wan2.1-I2V-14B-480P | 0.20~0.25 | Wan2.1-Fun-*-1.3B-* | 0.05~0.10 | Wan2.1-Fun-*-14B-*  | 0.20~0.30 |
# # --------------------------------------------------------------------------------------------------- #
teacache_threshold  = 0.10
# The number of steps to skip TeaCache at the beginning of the inference process, which can
# reduce the impact of TeaCache on generated video quality.
num_skip_start_steps = 5
# Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
teacache_offload    = False

# Skip some cfg steps in inference for acceleration
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
# Index of intrinsic frequency
riflex_k            = 6

# Config and model path
config_path         = "config/wan2.1/wan_civitai.yaml"
# print(config_path)
# model path
model_name          = args.model_name

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
# If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
# If you want to generate a 720p video, it is recommended to set the shift value to 5.0.
shift               = 3 

# # Load pretrained model if need
# transformer_path    = None
# vae_path            = None
# lora_path           = None


accuracy = args.accuracy  # 16 for torch.bfloat16, 32 for torch.float32
# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16 if accuracy == 16 else (torch.float32 if accuracy == 32 else torch.float16)
control_camera_txt      = None
start_image             = None


# Using longer neg prompt such as "Blurring, mutation, deformation, distortion, dark and solid, comics, text subtitles, line art." can increase stability
# Adding words such as "quiet, solid" to the neg prompt can increase dynamism.
# prompt                  = "A young woman with beautiful, clear eyes and blonde hair stands in the forest, wearing a white dress and a crown. Her expression is serene, reminiscent of a movie star, with fair and youthful skin. Her brown long hair flows in the wind. The video quality is very high, with a clear view. High quality, masterpiece, best quality, high resolution, ultra-fine, fantastical."
# negative_prompt         = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code."
guidance_scale          = 6.0
seed                    = args.seed
num_inference_steps     = 50

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)



transformer = WanTransformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True if accuracy == 16 else (False if accuracy == 32 else True),
    torch_dtype=weight_dtype,
).requires_grad_(False)

# Get Vae
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype).requires_grad_(False)

tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True if accuracy == 16 else (False if accuracy == 32 else True),
    torch_dtype=weight_dtype,
).requires_grad_(False)
text_encoder = text_encoder.eval()

clip_image_encoder = CLIPModel.from_pretrained(
    os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
).to(weight_dtype).requires_grad_(False)
clip_image_encoder = clip_image_encoder.eval()

# Get Scheduler
scheduler = FlowMatchEulerDiscreteScheduler(
    **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = WanFunControlPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    clip_image_encoder=clip_image_encoder
)

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device) # 完全不优化，22g显存占用，1min 50step

coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )


if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)




generator = torch.Generator(device=device).manual_seed(seed)

video = None

with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1



    control_video_input_tar, _, _, _ = get_video_to_video_latent(control_video_tar, video_length=video_length, sample_size=sample_size, fps=None, ref_image=None)
    control_video_input_src, _, _, _ = get_video_to_video_latent(control_video_src, video_length=video_length, sample_size=sample_size, fps=None, ref_image=None)
    control_camera_video = None
    print("control_video_input_tar shape:", control_video_input_tar.shape)
    print("control_video_input_src shape:", control_video_input_src.shape)

    partial_edit_mask_input, _, _, _ = get_video_to_video_latent(partial_edit_mask, video_length=video_length, sample_size=sample_size, fps=None, ref_image=None)
    print("partial_edit_mask_input shape:", partial_edit_mask_input.shape)

    ref_image_tar = get_image_latent(ref_image_tar_path, sample_size=sample_size)
    ref_image_src = get_image_latent(ref_image_src_path, sample_size=sample_size)

    src_video, _, _, _ = get_video_to_video_latent(src_video_path, video_length=video_length, sample_size=sample_size, fps=None, ref_image=None)
    print("src_video shape:", src_video.shape)
    print("ref_image_tar shape:", ref_image_tar.shape)
    print("ref_image_src shape:", ref_image_src.shape)

    if not args.direct_inference:
        video = pipeline.flow_portal(
            prompt_tar = prompt_tar,
            prompt_src = prompt_src,
            negative_prompt_tar = negative_prompt_tar,
            negative_prompt_src = negative_prompt_src,
            sampling_steps = num_inference_steps,

            height = sample_size[0],
            width = sample_size[1],

            control_video_tar = control_video_input_tar,
            control_video_src = control_video_input_src,
            src_video = src_video,

            ref_image_tar = ref_image_tar,
            ref_image_src = ref_image_src,

            ref_image_tar_path = ref_image_tar_path,
            ref_image_src_path = ref_image_src_path,

            src_guide_scale= src_guidance_scale,
            tar_guide_scale= tar_guidance_scale,

            generator = generator,

            visualize = False, #True,
            visualize_dir = save_path,

            partial_edit = args.partial_edit,
            partial_edit_mask = partial_edit_mask_input,

            edit_amplifier = args.edit_amplifier,
            cache_times = args.cache_times,
            n_avg= args.n_avg,
            src_blurring = args.src_blurring,
            transfer_blurring = args.transfer_blurring,
        )
        video = video.squeeze(0)  # Remove batch dimension
        frames = []
        for i in range(video.shape[1]):  # Iterate through frames
            frame = video[:, i, :, :]  # Get frame (C, H, W)
            frame = (frame * 255).byte().cpu().numpy().transpose(1, 2, 0)  # Convert to numpy uint8
            frames.append(frame)
        
        output_path = save_path + "output_video.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimsave(output_path, frames, fps=fps)

        # 拼接生成视频和原视频为横向对比视频

        # 读取原视频帧
        src_video_cv = cv2.VideoCapture(src_video_path)
        src_frames = []
        while True:
            ret, frame = src_video_cv.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            src_frames.append(frame)
        src_video_cv.release()

        # 对齐帧数
        min_len = min(len(frames), len(src_frames))
        gen_frames = frames[:min_len]
        src_frames = src_frames[:min_len]

        # resize 原视频帧到生成帧大小
        h, w, c = gen_frames[0].shape
        resized_src_frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in src_frames]

        # 横向拼接
        concat_frames = [np.concatenate([src, gen], axis=1) for src, gen in zip(resized_src_frames, gen_frames)]

        # 保存对比视频
        compare_output_path = save_path + "compare_output_video.mp4"
        imageio.mimsave(compare_output_path, concat_frames, fps=fps)
        print(f"对比视频已保存到: {compare_output_path}")
    else:
        video = pipeline.flow_portal(
            prompt_tar = prompt_tar,
            prompt_src = prompt_src,
            negative_prompt_tar = negative_prompt_tar,
            negative_prompt_src = negative_prompt_src,
            sampling_steps = num_inference_steps,

            height = sample_size[0],
            width = sample_size[1],

            control_video_tar = control_video_input_tar,
            control_video_src = control_video_input_src,
            src_video = src_video,

            ref_image_tar = ref_image_tar,
            ref_image_src = ref_image_src,

            ref_image_tar_path = ref_image_tar_path,
            ref_image_src_path = ref_image_src_path,

            src_guide_scale= src_guidance_scale,
            tar_guide_scale= tar_guidance_scale,

            generator = generator,

            visualize = False, #True,
            visualize_dir = save_path,

            partial_edit = args.partial_edit,
            partial_edit_mask = partial_edit_mask_input,

            edit_amplifier = 0.0,
            cache_times = 50,
        )
        video = video.squeeze(0)  # Remove batch dimension
        frames = []
        for i in range(video.shape[1]):  # Iterate through frames
            frame = video[:, i, :, :]  # Get frame (C, H, W)
            frame = (frame * 255).byte().cpu().numpy().transpose(1, 2, 0)  # Convert to numpy uint8
            frames.append(frame)
        
        output_path = save_path + "direct_inference_output_video.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimsave(output_path, frames, fps=fps)

        # 拼接生成视频和原视频为横向对比视频

        # 读取原视频帧
        src_video_cv = cv2.VideoCapture(src_video_path)
        src_frames = []
        while True:
            ret, frame = src_video_cv.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            src_frames.append(frame)
        src_video_cv.release()

        # 对齐帧数
        min_len = min(len(frames), len(src_frames))
        gen_frames = frames[:min_len]
        src_frames = src_frames[:min_len]

        # resize 原视频帧到生成帧大小
        h, w, c = gen_frames[0].shape
        resized_src_frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in src_frames]

        # 横向拼接
        concat_frames = [np.concatenate([src, gen], axis=1) for src, gen in zip(resized_src_frames, gen_frames)]

        # 保存对比视频
        compare_output_path = save_path + "direct_inference_compare_output_video.mp4"
        imageio.mimsave(compare_output_path, concat_frames, fps=fps)
        print(f"对比视频已保存到: {compare_output_path}")