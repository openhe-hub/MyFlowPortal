#!/usr/bin/env python3
"""Concatenate word-level ASL videos into sentence-level videos for ASL 50K dataset.

Reads sentences.csv, finds word videos by ref_id in SignASL_New, and concatenates
them into sentence-level MP4 files using ffmpeg concat demuxer with stream copy
(no re-encoding, near-instant per sentence).

Usage:
    python scripts/concat_asl50k_sentences.py
    python scripts/concat_asl50k_sentences.py --workers 8 --max-rows 100  # test run
"""

import argparse
import csv
import os
import subprocess
import tempfile
import time
from multiprocessing import Pool

# Defaults
CSV_PATH = "/home/nyuair/118-data-003/spamo/asl_50k/dict/sentences.csv"
WORD_VIDEO_DIR = "/home/nyuair/118-data-003/SignASL_New"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "datasets", "asl_50k")
MAX_ROWS = 20000
NUM_WORKERS = 16
FFMPEG = "/usr/bin/ffmpeg"


def parse_ref_ids(ref_ids_str):
    """Parse comma-separated ref_ids, filtering out NO_REF entries."""
    ids = [rid.strip() for rid in ref_ids_str.split(",")]
    return [rid for rid in ids if rid and not rid.startswith("NO_REF")]


def concat_sentence(args):
    """Concatenate word videos for a single sentence using concat demuxer + stream copy."""
    sentence_id, ref_ids, word_video_dir, output_dir = args
    output_path = os.path.join(output_dir, f"{sentence_id}.mp4")

    # Skip if already exists (resume support)
    if os.path.exists(output_path):
        return ("skipped", sentence_id, "already exists")

    if not ref_ids:
        return ("failed", sentence_id, "no valid ref_ids")

    # Check which word videos exist
    video_paths = []
    for rid in ref_ids:
        vpath = os.path.join(word_video_dir, f"{rid}.mp4")
        if os.path.isfile(vpath):
            video_paths.append(vpath)

    if not video_paths:
        return ("failed", sentence_id, "no word videos found")

    # Single video: just copy
    if len(video_paths) == 1:
        cmd = [FFMPEG, "-y", "-i", video_paths[0], "-c", "copy", "-an", output_path]
        try:
            subprocess.run(cmd, capture_output=True, timeout=30, check=True)
            return ("ok", sentence_id, "1 clip")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            stderr = e.stderr.decode(errors="replace")[-200:] if hasattr(e, "stderr") and e.stderr else str(e)
            return ("failed", sentence_id, stderr)

    # Multiple videos: concat demuxer + stream copy
    tmp_list = None
    try:
        tmp_list = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix=f"concat_{sentence_id}_"
        )
        for vp in video_paths:
            tmp_list.write(f"file '{vp}'\n")
        tmp_list.close()

        cmd = [
            FFMPEG, "-y",
            "-f", "concat", "-safe", "0", "-i", tmp_list.name,
            "-c", "copy", "-an",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=30, check=True)
        return ("ok", sentence_id, f"{len(video_paths)} clips")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        stderr = e.stderr.decode(errors="replace")[-200:] if hasattr(e, "stderr") and e.stderr else str(e)
        return ("failed", sentence_id, stderr)
    finally:
        if tmp_list and os.path.exists(tmp_list.name):
            os.remove(tmp_list.name)


def main():
    parser = argparse.ArgumentParser(description="Concatenate ASL word videos into sentence videos")
    parser.add_argument("--csv", default=CSV_PATH, help="Path to sentences.csv")
    parser.add_argument("--word-dir", default=WORD_VIDEO_DIR, help="Directory with word-level MP4s")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory for sentence MP4s")
    parser.add_argument("--max-rows", type=int, default=MAX_ROWS, help="Max rows to process from CSV")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read CSV
    print(f"Reading {args.csv} ...")
    tasks = []
    with open(args.csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= args.max_rows:
                break
            sentence_id = row["sentence_id"]
            ref_ids = parse_ref_ids(row["ref_ids"])
            tasks.append((sentence_id, ref_ids, args.word_dir, args.output_dir))

    print(f"Loaded {len(tasks)} sentences, output → {args.output_dir}")
    print(f"Starting {args.workers} workers ...")

    t0 = time.time()
    ok_count = 0
    skip_count = 0
    fail_count = 0
    fail_examples = []

    with Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(concat_sentence, tasks), 1):
            status, sid, detail = result
            if status == "ok":
                ok_count += 1
            elif status == "skipped":
                skip_count += 1
            else:
                fail_count += 1
                if len(fail_examples) < 10:
                    fail_examples.append((sid, detail))

            if i % 500 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"[{i}/{len(tasks)}] ok={ok_count} skip={skip_count} fail={fail_count} "
                      f"({rate:.1f} sent/s, {elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  OK:      {ok_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed:  {fail_count}")
    if fail_examples:
        print(f"\nFirst {len(fail_examples)} failures:")
        for sid, detail in fail_examples:
            print(f"  {sid}: {detail}")


if __name__ == "__main__":
    main()
