import os
import math

import numpy as np
import torch
import safetensors.torch as sf

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from torch.hub import download_url_to_file

# 'stablediffusionapi/realistic-vision-v51'
# 'runwayml/stable-diffusion-v1-5'
sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")


# Change UNet

with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(12, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward


def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


unet.forward = hooked_unet_forward



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='./models/iclight_sd15_fbc.safetensors')
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--bg_path", type=str, required=True)
parser.add_argument("--video_name", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--width", type=int, required=True, help="生成宽度")
parser.add_argument("--height", type=int, required=True, help="生成高度")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--partial_edit", type=bool, default=False)

args = parser.parse_args()

# Load

model_path = args.model_path

if not os.path.exists(model_path):
    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors', dst=model_path)

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device

device = torch.device('cuda')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

# Pipelines

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h

def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise):

    rng = torch.Generator(device=device).manual_seed(int(seed))

    
    fg = resize_without_crop(input_fg, image_width, image_height)
    bg = resize_without_crop(input_bg, image_width, image_height)

    concat_conds = numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

    latents = t2i_pipe(
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=steps,
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [resize_without_crop(
        image=p,
        target_width=int(math.ceil(image_width * highres_scale / 64.0) * 64),
        target_height=int(math.ceil(image_height * highres_scale / 64.0) * 64))
    for p in pixels]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

    fg = resize_without_crop(input_fg, image_width, image_height)
    bg = resize_without_crop(input_bg, image_width, image_height)
    concat_conds = numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample

    return pytorch2numpy(pixels)


@torch.inference_mode()
def process_relight(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise):
    results = process(input_fg, input_bg, prompt, image_width, 
        image_height, num_samples, seed, steps, a_prompt, 
        n_prompt, cfg, highres_scale, highres_denoise, 
        lowres_denoise)
    return input_fg, results

def save_np_images(np_list, out_path, prefix="out"):
    paths = []
    for i, arr in enumerate(np_list, 1):
        Image.fromarray(arr).save(out_path)
    return out_path


video_name = args.video_name
base_path = args.base_path
ref_image_tar_path = os.path.join(base_path, "datasets", video_name, f"{video_name}_reference.png")
input_fg_path = os.path.join(base_path, "datasets", video_name, "video_frames", "frame_0000.png")
input_bg_path = args.bg_path
out_path = os.path.join(base_path, "datasets", video_name, f"{video_name}_reference.png")
mask_path = os.path.join(base_path, "datasets", video_name, "masks", "mask_0000.png")
num_samples   = 1
seed          = args.seed
steps         = args.steps
cfg           = 2.0
image_width   = int(math.ceil(args.width / 64.0) * 64)
image_height  = int(math.ceil(args.height / 64.0) * 64)
lowres_denoise   = 0.9
highres_scale    = 1.5
highres_denoise  = 0.5
a_prompt      = "highres, best quality, enhanced lighting, detailed face"
n_prompt      = "lowres, bad anatomy, bad hands, cropped, worst quality, furry, hairy, branchy, rainy, low quality, jpeg artifacts, ugly, duplicate"
# 这个 demo 版本用“前景条件”重打光；背景由模型生成（不指定外部背景图）
prompt        = args.prompt

# 读图（numpy 数组）
input_fg_np = np.array(Image.open(input_fg_path).convert("RGB"))
input_bg_np = np.array(Image.open(input_bg_path).convert("RGB"))

# 读 mask（灰度，范围 0–255）
print("Partial edit:", args.partial_edit)
if args.partial_edit:
    mask = np.array(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0
else:
    mask = np.ones((input_fg_np.shape[0], input_fg_np.shape[1]), dtype=np.float32)
# 确保尺寸一致
mask = np.array(Image.fromarray((mask*255).astype(np.uint8)).resize(input_fg_np.shape[1::-1], Image.BILINEAR)) / 255.0
# 只保留 mask 区域，其余背景置灰(127)或黑(0)
fg_only = input_fg_np.astype(np.float32) * mask[..., None] + 127.0 * (1 - mask[..., None])
input_fg_np_aftermatting = fg_only.clip(0,255).astype(np.uint8)

# 调用你的核心函数（会自动抠图？？？ + 低清/高清两阶段 + 前景条件引导）
pre_fg, results = process_relight(
    input_fg=input_fg_np_aftermatting,
    input_bg=input_bg_np,
    prompt=prompt,
    image_width=image_width,
    image_height=image_height,
    num_samples=num_samples,
    seed=seed,
    steps=steps,
    a_prompt=a_prompt,
    n_prompt=n_prompt,
    cfg=cfg,
    highres_scale=highres_scale,
    highres_denoise=highres_denoise,
    lowres_denoise=lowres_denoise,
)

for i in range(len(results)):
    results[i] = resize_without_crop(results[i], args.width, args.height)

# 保存输出
saved = save_np_images(results, out_path)
print("Saved files:", saved)