"""Generate manifest for How2Sign batch processing.

Scans How2Sign train MP4s, uses ffprobe to get frame counts, assigns k_frames
based on video length, selects 8000 random videos, and splits into chunks
for SLURM array jobs.

Modes:
  selected   (default) — pick 8000 videos, split into 4 chunks
  unselected           — everything *except* those 8000, split into --num_chunks
"""
import os
import csv
import random
import math
import subprocess
import json
import argparse
from collections import Counter

VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "datasets", "How2Sign", "how2sign", "sentence_level", "train",
                         "rgb_front", "raw_videos")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

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


def print_distributions(entries):
    """Print k_frames and prompt_id distributions."""
    k_dist = Counter(e["k_frames"] for e in entries)
    for k in sorted(k_dist):
        print(f"  k_frames={k}: {k_dist[k]} videos")
    p_dist = Counter(e["prompt_id"] for e in entries)
    for p in sorted(p_dist):
        print(f"  prompt_{p}: {p_dist[p]} videos")


def write_manifest(entries, manifest_path):
    """Write CSV manifest."""
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_name", "n_frames", "k_frames", "prompt_id"])
        writer.writeheader()
        writer.writerows(entries)
    print(f"Written: {manifest_path} ({len(entries)} videos)")


def write_chunks(entries, num_chunks, prefix):
    """Split entries into num_chunks CSV files."""
    chunk_size = math.ceil(len(entries) / num_chunks)
    for i in range(num_chunks):
        chunk = entries[i * chunk_size : (i + 1) * chunk_size]
        chunk_path = os.path.join(OUTPUT_DIR, f"{prefix}_chunk{i}.csv")
        write_manifest(chunk, chunk_path)


def mode_selected(all_valid):
    """Original mode: select 8000 random videos, split into 4 chunks."""
    num_chunks = 4

    random.seed(42)
    random.shuffle(all_valid)

    if len(all_valid) > NUM_SELECT:
        entries = all_valid[:NUM_SELECT]
        print(f"Selected {NUM_SELECT} videos from valid pool")
    else:
        entries = all_valid[:]
        print(f"Warning: only {len(entries)} valid videos (requested {NUM_SELECT})")

    # Assign random prompt_id
    for e in entries:
        e["prompt_id"] = random.randint(0, len(PROMPTS_TAR) - 1)

    print_distributions(entries)

    # Write full manifest
    write_manifest(entries, os.path.join(OUTPUT_DIR, "h2s_manifest.csv"))

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
    write_chunks(entries, num_chunks, "h2s_manifest")


def mode_unselected(all_valid, num_chunks):
    """Select everything EXCEPT the 8000 already-selected videos."""
    # Reproduce the same seed=42 shuffle to identify the selected 8000
    random.seed(42)
    shuffled = all_valid[:]
    random.shuffle(shuffled)

    if len(shuffled) <= NUM_SELECT:
        print(f"Error: only {len(shuffled)} valid videos, nothing unselected")
        return

    selected_names = set(e["video_name"] for e in shuffled[:NUM_SELECT])
    unselected = [e for e in shuffled[NUM_SELECT:]]
    print(f"Selected (excluded): {len(selected_names)}, Unselected: {len(unselected)}")

    # Assign random prompt_id (fresh RNG after the shuffle consumed seed=42 state)
    random.seed(42)
    for e in unselected:
        e["prompt_id"] = random.randint(0, len(PROMPTS_TAR) - 1)

    print_distributions(unselected)

    # Write full unselected manifest
    write_manifest(unselected, os.path.join(OUTPUT_DIR, "h2s_unselected_manifest.csv"))

    # Write unselected video filenames (for transfer)
    videos_path = os.path.join(OUTPUT_DIR, "h2s_unselected_videos.txt")
    with open(videos_path, "w") as f:
        for e in unselected:
            f.write(f"{e['video_name']}.mp4\n")
    print(f"Written: {videos_path}")

    # Split into chunks
    write_chunks(unselected, num_chunks, "h2s_unselected")


def main():
    parser = argparse.ArgumentParser(description="Generate H2S manifest")
    parser.add_argument("--mode", choices=["selected", "unselected"], default="selected",
                        help="selected = pick 8000; unselected = everything else")
    parser.add_argument("--num_chunks", type=int, default=10,
                        help="Number of chunks for unselected mode (default: 10)")
    args = parser.parse_args()

    frame_counts = get_frame_counts(VIDEO_DIR)

    all_valid = []
    skipped = 0
    for fname in sorted(frame_counts.keys()):
        n_frames = frame_counts[fname]
        k = get_valid_k_frames(n_frames)
        if k is None:
            skipped += 1
            continue
        video_name = fname.rsplit(".", 1)[0]
        all_valid.append({
            "video_name": video_name,
            "n_frames": n_frames,
            "k_frames": k,
        })

    print(f"Total valid: {len(all_valid)}, Skipped (too short): {skipped}")

    if args.mode == "selected":
        mode_selected(all_valid)
    else:
        mode_unselected(all_valid, args.num_chunks)


if __name__ == "__main__":
    main()
