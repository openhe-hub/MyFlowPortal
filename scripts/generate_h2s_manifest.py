"""Generate manifest for How2Sign batch processing.

Scans How2Sign train MP4s, uses ffprobe to get frame counts, assigns k_frames
based on video length, selects 8000 random videos, and splits into 4 chunks
for SLURM array jobs.
"""
import os
import csv
import random
import math
import subprocess
import json

random.seed(42)

VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "datasets", "How2Sign", "how2sign", "sentence_level", "train",
                         "rgb_front", "raw_videos")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")

NUM_CHUNKS = 4
NUM_SELECT = 8000
INPUT_FPS = 24
TARGET_FPS = 8

PROMPTS_SRC = "a person is signing in front of a plain background in a studio"

PROMPTS_TAR = [
    "a person is signing in a bright modern office with large windows, natural daylight, clean white walls, professional setting",
    "a person is signing in a quiet library with wooden bookshelves, warm ambient lighting, academic atmosphere",
    "a person is signing on a city street with buildings, shops, and pedestrians in the background, daytime urban scene",
    "a person is signing in a green park with trees, grass, and a walking path, sunny day, natural environment",
    "a person is signing in a cozy cafe with warm lighting, wooden tables, coffee cups, and indoor plants",
    "a person is signing in a classroom with a whiteboard, desks, and educational posters on the wall",
    "a person is signing in a modern living room with a sofa, bookshelf, floor lamp, and soft warm lighting",
    "a person is signing at a beach with ocean waves, sandy shore, and blue sky in the background",
    "a person is signing in front of a cityscape at night with neon lights, skyscrapers, and colorful reflections",
    "a person is signing in a beautiful garden with colorful flowers, stone path, and soft sunlight filtering through",
]


def get_frame_counts(video_dir):
    """Get frame counts for all MP4 files using ffprobe."""
    mp4_files = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
    print(f"Found {len(mp4_files)} MP4 files, probing frame counts...")

    frame_counts = {}
    for i, fname in enumerate(mp4_files):
        if (i + 1) % 1000 == 0:
            print(f"  Probed {i + 1}/{len(mp4_files)}...")
        filepath = os.path.join(video_dir, fname)
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", filepath],
                capture_output=True, text=True, timeout=10
            )
            data = json.loads(result.stdout)
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    n_frames = int(stream.get("nb_frames", 0))
                    if n_frames > 0:
                        frame_counts[fname] = n_frames
                    break
        except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
            pass

    print(f"  Successfully probed {len(frame_counts)} videos")
    return frame_counts


def get_valid_k_frames(n_frames):
    """Return largest valid k_frames (4n+1) for given frame count."""
    sample_interval = INPUT_FPS / TARGET_FPS
    max_sampled = int(n_frames / sample_interval)
    # Valid k_frames: 4n+1 values
    for k in [49, 41, 21, 13, 9]:
        if max_sampled >= k:
            return k
    return None  # too short


def main():
    frame_counts = get_frame_counts(VIDEO_DIR)

    entries = []
    skipped = 0

    for fname in sorted(frame_counts.keys()):
        n_frames = frame_counts[fname]
        k = get_valid_k_frames(n_frames)
        if k is None:
            skipped += 1
            continue
        video_name = fname.rsplit(".", 1)[0]  # strip .mp4
        entries.append({
            "video_name": video_name,
            "n_frames": n_frames,
            "k_frames": k,
        })

    print(f"Total valid: {len(entries)}, Skipped (too short): {skipped}")

    # Select 8000 random videos
    random.shuffle(entries)
    if len(entries) > NUM_SELECT:
        entries = entries[:NUM_SELECT]
        print(f"Selected {NUM_SELECT} videos from valid pool")
    else:
        print(f"Warning: only {len(entries)} valid videos (requested {NUM_SELECT})")

    # Assign random prompt_id
    for e in entries:
        e["prompt_id"] = random.randint(0, len(PROMPTS_TAR) - 1)

    # k_frames distribution
    from collections import Counter
    k_dist = Counter(e["k_frames"] for e in entries)
    for k in sorted(k_dist):
        print(f"  k_frames={k}: {k_dist[k]} videos")

    # prompt distribution
    p_dist = Counter(e["prompt_id"] for e in entries)
    for p in sorted(p_dist):
        print(f"  prompt_{p}: {p_dist[p]} videos")

    # Write full manifest
    manifest_path = os.path.join(OUTPUT_DIR, "h2s_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_name", "n_frames", "k_frames", "prompt_id"])
        writer.writeheader()
        writer.writerows(entries)
    print(f"Written: {manifest_path}")

    # Write prompts file
    prompts_path = os.path.join(OUTPUT_DIR, "h2s_prompts.txt")
    with open(prompts_path, "w") as f:
        f.write(f"SRC|{PROMPTS_SRC}\n")
        for i, p in enumerate(PROMPTS_TAR):
            f.write(f"{i}|{p}\n")
    print(f"Written: {prompts_path}")

    # Write selected video filenames (for transfer)
    selected_path = os.path.join(OUTPUT_DIR, "h2s_selected_videos.txt")
    with open(selected_path, "w") as f:
        for e in entries:
            f.write(f"{e['video_name']}.mp4\n")
    print(f"Written: {selected_path}")

    # Split into chunks
    chunk_size = math.ceil(len(entries) / NUM_CHUNKS)
    for i in range(NUM_CHUNKS):
        chunk = entries[i * chunk_size : (i + 1) * chunk_size]
        chunk_path = os.path.join(OUTPUT_DIR, f"h2s_manifest_chunk{i}.csv")
        with open(chunk_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_name", "n_frames", "k_frames", "prompt_id"])
            writer.writeheader()
            writer.writerows(chunk)
        print(f"Written: {chunk_path} ({len(chunk)} videos)")


if __name__ == "__main__":
    main()
