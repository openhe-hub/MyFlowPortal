import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 选择显卡使用, set in .sh/
import sys

import warnings
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='imageio')

import numpy as np
from PIL import Image

# current_file_path = os.path.abspath(__file__)
current_file_path = os.path.abspath("")

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
    parser.add_argument("--skip_edge_detection", action='store_true', help="Skip canny/HED/Midas computation, reuse existing video_canny/ from a prior run")
    parser.add_argument("--skip_frame_extraction", action='store_true', help="Skip video frame extraction, reuse existing video_frames/ from a prior run")
    parser.add_argument("--annotators_path", type=str, default=None, help="Path to Annotators directory (HED/Midas weights). Overrides default.")
    parser.add_argument("--edge_mode", type=str, default="combined", choices=["combined", "canny", "hed", "depth"], help="Edge detection mode. 'canny' is fastest (no neural network).")
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

if not args.skip_frame_extraction:
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

    # 按目标fps均匀采样，覆盖整个视频时长
    sample_interval = orig_fps / args.fps  # e.g. 29.97/8 ≈ 3.75
    max_sampled = int(total_frames / sample_interval)  # how many frames at target fps
    n_sample = min(args.k_frames, max_sampled)
    frame_indices = np.round(np.arange(n_sample) * sample_interval).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    print(f"Sampling {n_sample} frames at {args.fps}fps (interval={sample_interval:.1f}), covering {n_sample/args.fps:.2f}s of {total_frames/orig_fps:.2f}s video")

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
else:
    print("Skipping frame extraction (reusing existing video_frames/)...")

print(f"INPUT VIDEO INFO --frame_num: {k_frames} --fps: {fps}" )
print(f"USING HYPERPARAMS --k_frames: {args.k_frames} --fps: {args.fps} --batch_size: {args.batch_size}")

batch_size = args.batch_size

if not os.path.exists(masks_path):
    os.makedirs(masks_path)

def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

if args.using_existing_masks is None:
    import torch
    device = torch.device('cuda')
    from transformers import AutoModelForImageSegmentation
    rmbg = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
    rmbg.eval()
    rmbg = rmbg.to(device=device, dtype=torch.float32)

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
        feedh = feedh.movedim(-1, 1).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            alpha = rmbg(feedh)[-1].sigmoid()
        alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
        alpha = alpha.movedim(1, -1)
        alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
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

if not args.skip_edge_detection:
    print(f"Applying edge detection (mode={args.edge_mode}) to video frames...")

    control = args.edge_mode  # "canny", "hed", "depth", "combined"
    _annotators_path = args.annotators_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models", "Annotators")
    if control == "canny":
        from controlnet_aux import CannyDetector
        canny = CannyDetector()
    elif control == "hed":
        from controlnet_aux import HEDdetector
        canny = HEDdetector.from_pretrained(_annotators_path)
    elif control == "depth":
        from controlnet_aux import MidasDetector
        canny = MidasDetector.from_pretrained(_annotators_path)
    elif control == "combined":
        from controlnet_aux import CannyDetector, HEDdetector, MidasDetector
        canny = CannyDetector()
        canny_depth = MidasDetector.from_pretrained(_annotators_path)
        canny_hed = HEDdetector.from_pretrained(_annotators_path)

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
            if control == "hed":
                result = canny(img)
                result = np.array(result)
            elif control == "canny":
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
else:
    print("Skipping edge detection (reusing existing video_canny/)...")

os.makedirs(masked_canny_dir, exist_ok=True)

frame_list = []

for i in range(k_frames):
    frame_path = os.path.join(video_canny_dir, f"frame_{i:04d}.png")
    mask_path = os.path.join(masks_path, f"mask_{i:04d}.png")
    out_path = os.path.join(masked_canny_dir, f"masked_{i:04d}.png")

    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)
    if mask.shape != frame.shape:
        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    masked_canny = (frame * mask / 255).astype(np.uint8)
    cv2.imwrite(out_path, masked_canny)
    frame_list.append(masked_canny)


# 生成视频，使用 imageio 替代 cv2.VideoWriter
video_path = os.path.join(masked_canny_dir, "masked_canny_video.mp4")
imageio.mimsave(video_path, frame_list, fps=fps)

print("Canny edge detection completed.")