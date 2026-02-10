"""Generate manifest for Phoenix batch processing.

Assigns k_frames based on video frame count, random prompt from 10 options,
and splits into N chunks for SLURM array jobs.
"""
import os
import csv
import random
import math

random.seed(42)

TRAIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "datasets", "PHOENIX-2014-T", "features", "fullFrame-210x260px", "train")
MP4_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "datasets", "PHOENIX-2014-T", "train_mp4")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")

NUM_CHUNKS = 4
INPUT_FPS = 25
TARGET_FPS = 8

PROMPTS_SRC = "a person in dark clothing is signing in front of a plain gray background in a TV studio"

PROMPTS_TAR = [
    "a person in dark clothing is signing in a bright modern office with large windows, natural daylight, clean white walls, professional setting",
    "a person in dark clothing is signing in a quiet library with wooden bookshelves, warm ambient lighting, academic atmosphere",
    "a person in dark clothing is signing on a city street with buildings, shops, and pedestrians in the background, daytime urban scene",
    "a person in dark clothing is signing in a green park with trees, grass, and a walking path, sunny day, natural environment",
    "a person in dark clothing is signing in a cozy cafe with warm lighting, wooden tables, coffee cups, and indoor plants",
    "a person in dark clothing is signing in a classroom with a whiteboard, desks, and educational posters on the wall",
    "a person in dark clothing is signing in a modern living room with a sofa, bookshelf, floor lamp, and soft warm lighting",
    "a person in dark clothing is signing at a beach with ocean waves, sandy shore, and blue sky in the background",
    "a person in dark clothing is signing in front of a cityscape at night with neon lights, skyscrapers, and colorful reflections",
    "a person in dark clothing is signing in a beautiful garden with colorful flowers, stone path, and soft sunlight filtering through",
]

def get_valid_k_frames(n_frames):
    """Return largest valid k_frames (4n+1) for given frame count."""
    sample_interval = INPUT_FPS / TARGET_FPS
    max_sampled = int(n_frames / sample_interval)
    # Valid k_frames: 4n+1 values
    for k in [41, 21, 13, 9]:
        if max_sampled >= k:
            return k
    return None  # too short


def main():
    entries = []
    skipped = 0

    for name in sorted(os.listdir(TRAIN_DIR)):
        frame_dir = os.path.join(TRAIN_DIR, name)
        if not os.path.isdir(frame_dir):
            continue
        n_frames = len(os.listdir(frame_dir))
        k = get_valid_k_frames(n_frames)
        if k is None:
            skipped += 1
            continue
        prompt_id = random.randint(0, len(PROMPTS_TAR) - 1)
        entries.append({
            "video_name": name,
            "n_frames": n_frames,
            "k_frames": k,
            "prompt_id": prompt_id,
        })

    random.shuffle(entries)
    print(f"Total valid: {len(entries)}, Skipped (too short): {skipped}")

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
    manifest_path = os.path.join(OUTPUT_DIR, "phoenix_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_name", "n_frames", "k_frames", "prompt_id"])
        writer.writeheader()
        writer.writerows(entries)
    print(f"Written: {manifest_path}")

    # Write prompts file
    prompts_path = os.path.join(OUTPUT_DIR, "phoenix_prompts.txt")
    with open(prompts_path, "w") as f:
        f.write(f"SRC|{PROMPTS_SRC}\n")
        for i, p in enumerate(PROMPTS_TAR):
            f.write(f"{i}|{p}\n")
    print(f"Written: {prompts_path}")

    # Split into chunks
    chunk_size = math.ceil(len(entries) / NUM_CHUNKS)
    for i in range(NUM_CHUNKS):
        chunk = entries[i * chunk_size : (i + 1) * chunk_size]
        chunk_path = os.path.join(OUTPUT_DIR, f"phoenix_manifest_chunk{i}.csv")
        with open(chunk_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_name", "n_frames", "k_frames", "prompt_id"])
            writer.writeheader()
            writer.writerows(chunk)
        print(f"Written: {chunk_path} ({len(chunk)} videos)")


if __name__ == "__main__":
    main()
