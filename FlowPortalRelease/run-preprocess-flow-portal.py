import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 选择显卡使用, set in .sh/
import sys

import warnings
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='imageio')

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

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
from videox_fun.pipeline import WanFunControlPipeline, WanPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent, get_image_latent,
                                    get_video_to_video_latent,
                                    save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from IPython.display import display, Image as IPyImage, Video as IPyVideo

import argparse
import imageio

def parse_arg():
    parser = argparse.ArgumentParser(description="Preprocess video for flow editing")
    parser.add_argument("--video_name", type=str, help="The name of input video in datasets", required=True)
    parser.add_argument("--k_frames", type=int,  required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=7)
    parser.add_argument("--video_input", type=str, default=None, help="Path to the input video file")
    parser.add_argument("--height", type=int, required=True, help="Height of the video frames")
    parser.add_argument("--width", type=int, required=True, help="Width of the video frames")
    parser.add_argument("--using_existing_masks", type=str, default=None, help="Path to existing masks directory, if any")
    return parser.parse_args()

args = parse_arg()
video_name = args.video_name

video_input = args.video_input
input_video_path = os.path.join(current_file_path, "datasets", video_name, f"{video_name}.mp4")
first_frame_path = os.path.join(current_file_path, "datasets", video_name, "video_frames", "frame_0000.png")
video_frames_dir = os.path.join(current_file_path, "datasets", video_name, "video_frames")
video_canny_dir = os.path.join(current_file_path, "datasets", video_name, "video_canny")
masks_path = os.path.join(current_file_path, "datasets", video_name, "masks")
masked_canny_dir = os.path.join(current_file_path, "datasets", video_name, "masked_canny")

import cv2

k_frames, fps = args.k_frames, args.fps
height, width = args.height, args.width

# 检查输入视频是否存在
if not video_input or not os.path.exists(video_input):
    raise FileNotFoundError(f"Input video file not found: {video_input}")

os.makedirs(video_frames_dir, exist_ok=True)
print(f"Extracting and resizing first {k_frames} frames to {width}x{height}...")
cap = cv2.VideoCapture(video_input)
orig_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames < args.k_frames:
    raise ValueError(f"Input video has only {total_frames} frames, but k_frames={args.k_frames} is required.")

print(f"Original video size: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, FPS: {orig_fps}, Total frames: {total_frames}")

# 取前面k帧的索引
frame_indices = np.arange(args.k_frames)

frames = []
for idx, frame_idx in enumerate(frame_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx} from video.")
    # resize到目标大小
    frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
    frame_path = os.path.join(video_frames_dir, f"frame_{idx:04d}.png")
    cv2.imwrite(frame_path, frame_resized)
    frames.append(frame_resized)
cap.release()

# 保存采样并resize后的视频，fps为目标fps，使用 imageio 替代 cv2.VideoWriter
if frames:
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    imageio.mimsave(input_video_path, frames_rgb, fps=args.fps)

print("Frame extraction and resizing completed.")

print(f"INPUT VIDEO INFO --frame_num: {k_frames} --fps: {fps}" )
print(f"USING HYPERPARAMS --k_frames: {args.k_frames} --fps: {args.fps} --batch_size: {args.batch_size}")

batch_size = args.batch_size

# display(IPyImage(first_frame_path)) # in .ipynb

if not os.path.exists(masks_path):
    os.makedirs(masks_path)
from transformers import AutoModelForImageSegmentation
rmbg = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)

rmbg.eval()
device = torch.device('cuda')
rmbg = rmbg.to(device=device, dtype=torch.float32)

def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

@torch.inference_mode()
def run_rmbg(imgs):
    H, W, C = imgs[0].shape
    assert C == 3
    feeds = []
    for img in imgs:
        assert img.shape == (H, W, C)
        k = (256.0 / float(H * W)) ** 0.5
        feed = resize_without_crop(img, int(64 * round(W // 64 / 2)), int(64 * round(H // 64 / 2)))
        feeds.append(feed)
    feedh = torch.from_numpy(np.stack(feeds, axis=0)).float() / 255.0
    #print(feedh.shape)
    feedh = feedh.movedim(-1, 1).to(device=device, dtype=torch.float32)
    #print(feedh.shape)
    with torch.no_grad():
        alpha = rmbg(feedh)[-1].sigmoid()
        #print(alpha.shape)
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    #print(alpha.shape)
    return alpha

print("Generating masks for video frames...")

frames = []

input_frames = []
for i in range(k_frames):
    input_source = Image.open(os.path.join(video_frames_dir, f'frame_{i:04d}.png'))
    # input_source = Image.open(first_frame_path)
    input_frames.append(np.array(input_source))
frame_count = len(input_frames)

if args.using_existing_masks is None:
    # batch_size = args.batch_size # the maximum batch size is around 7
    mask_frames = []
    for i in range(0, frame_count, batch_size):
        endpoint = min(i + batch_size, frame_count + 1)
        batch_frames = input_frames[i:endpoint]
        mask = run_rmbg(batch_frames)
        mask_frames.extend(mask)

    mask_frame_list = []
    for i in range(k_frames):
        mask = mask_frames[i]
        mask = np.repeat(mask, 3, axis=2)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image.save(os.path.join(masks_path, f"mask_{i:04d}.png"))
        mask_image = (mask * 255).astype(np.uint8)
        mask_frame_list.append(mask_image)
    mask_video_path = os.path.join(masks_path, "mask_video.mp4")
    imageio.mimsave(mask_video_path, mask_frame_list, fps=fps)
else:
    if not os.path.exists(args.using_existing_masks):
        raise FileNotFoundError(f"Existing mask video not found: {args.using_existing_masks}")
    print(f"Using existing masks from: {args.using_existing_masks}")
    mask_frames = []
    mask_video = imageio.mimread(args.using_existing_masks)  # list of frames
    for i in range(k_frames):
        mask_image = mask_video[i]
        # If mask is RGB, convert to grayscale
        if mask_image.ndim == 3 and mask_image.shape[2] == 3:
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
        mask_image = mask_image.astype(np.float32) / 255.0
        mask_frames.append(mask_image)
    mask_frame_list = []
    for i in range(k_frames):
        mask = mask_frames[i]
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_image.save(os.path.join(masks_path, f"mask_{i:04d}.png"))
        mask_image = (mask * 255).astype(np.uint8)
        mask_frame_list.append(mask_image)
    mask_video_path = os.path.join(masks_path, "mask_video.mp4")
    imageio.mimsave(mask_video_path, mask_frame_list, fps=fps)

print("Mask generation completed.")
print("Applying Canny edge detection to video frames...")

from controlnet_aux import CannyDetector, HEDdetector, MidasDetector

control = "combined"  # "Canny", "HED", "depth", "combined"
if control == "HED":
    canny = HEDdetector.from_pretrained("lllyasviel/Annotators")
elif control == "Canny":
    canny = CannyDetector()
elif control == "depth":
    canny = MidasDetector.from_pretrained("lllyasviel/Annotators")
elif control == "combined":
    canny = CannyDetector()
    canny_depth = MidasDetector.from_pretrained("lllyasviel/Annotators")
    canny_hed = HEDdetector.from_pretrained("lllyasviel/Annotators")

def apply_canny_to_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    frame_list = []
    f_count = 0
    for fname in sorted(os.listdir(input_dir)):
        if f_count >= k_frames:
            break
        f_count += 1
        if not fname.endswith(".png"):
            continue
        img = cv2.imread(os.path.join(input_dir, fname))
        H, W = img.shape[0], img.shape[1]
        if control == "HED":
            result = canny(img)
            result = np.array(result)
        elif control == "Canny":
            result = canny(img, low_threshold=50, high_threshold=100)
        elif control == "depth":
            result = canny(img)
            result = np.array(result)
        elif control == "combined":
            canny_result = canny(img, low_threshold=50, high_threshold=100)
            hed_result = canny_hed(img)
            depth_result = canny_depth(img)
            canny_result = np.array(canny_result)
            hed_result = np.array(hed_result)
            depth_result = np.array(depth_result)
            result = canny_result * 0.1 + hed_result * 0.1 + depth_result * 0.8
            result = np.clip(result, 0, 255).astype(np.uint8)
        result = resize_without_crop(result, W, H)
        cv2.imwrite(os.path.join(output_dir, fname), result)
        frame_list.append(result)

    # 生成视频，使用 imageio 替代 cv2.VideoWriter
    video_path = os.path.join(output_dir, "canny_video.mp4")
    imageio.mimsave(video_path, frame_list, fps=fps)

apply_canny_to_folder(video_frames_dir, video_canny_dir)

os.makedirs(masked_canny_dir, exist_ok=True)

frame_list = []

for i in range(k_frames):
    frame_path = os.path.join(video_canny_dir, f"frame_{i:04d}.png")
    mask_path = os.path.join(masks_path, f"mask_{i:04d}.png")
    out_path = os.path.join(masked_canny_dir, f"masked_{i:04d}.png")

    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)

    masked_canny = (frame * mask / 255).astype(np.uint8)
    cv2.imwrite(out_path, masked_canny)
    frame_list.append(masked_canny)


# 生成视频，使用 imageio 替代 cv2.VideoWriter
video_path = os.path.join(masked_canny_dir, "masked_canny_video.mp4")
imageio.mimsave(video_path, frame_list, fps=fps)

print("Canny edge detection completed.")