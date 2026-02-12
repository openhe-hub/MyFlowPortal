#!/usr/bin/env python3
"""10x temporal downsampling of ASL 50K sentence videos using OpenCV.

Reads every Nth frame from source videos and writes to output directory.
Uses OpenCV to handle mixed-timebase concat files reliably.

Usage:
    python scripts/subsample_asl50k.py
    python scripts/subsample_asl50k.py --factor 5 --workers 16
"""

import argparse
import glob
import os
import time
from multiprocessing import Pool

import cv2

INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "datasets", "asl_50k")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "datasets", "asl_50k_10x")
NUM_WORKERS = 16
FACTOR = 10
FPS_OUT = 30
OUT_W, OUT_H = 640, 360


def subsample(args):
    input_path, output_path, factor = args

    if os.path.exists(output_path):
        return ("skipped", os.path.basename(input_path), 0)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return ("failed", os.path.basename(input_path), 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS_OUT, (OUT_W, OUT_H))

    idx = 0
    written = 0
    while True:
        if idx % factor == 0:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != OUT_W or frame.shape[0] != OUT_H:
                frame = cv2.resize(frame, (OUT_W, OUT_H))
            writer.write(frame)
            written += 1
        else:
            if not cap.grab():
                break
        idx += 1

    cap.release()
    writer.release()

    if written == 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        return ("failed", os.path.basename(input_path), 0)

    return ("ok", os.path.basename(input_path), written)


def main():
    parser = argparse.ArgumentParser(description="Temporal downsampling of ASL 50K videos")
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--factor", type=int, default=FACTOR, help="Keep every Nth frame")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.mp4")))
    print(f"Found {len(files)} videos, {args.factor}x downsample → {args.output_dir}")

    tasks = [
        (f, os.path.join(args.output_dir, os.path.basename(f)), args.factor)
        for f in files
    ]

    t0 = time.time()
    ok = skip = fail = 0
    total_frames = 0

    with Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(subsample, tasks), 1):
            status, name, nframes = result
            if status == "ok":
                ok += 1
                total_frames += nframes
            elif status == "skipped":
                skip += 1
            else:
                fail += 1
                if fail <= 5:
                    print(f"  FAIL: {name}")

            if i % 1000 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                print(f"[{i}/{len(tasks)}] ok={ok} skip={skip} fail={fail} "
                      f"({i/elapsed:.1f}/s, {elapsed:.0f}s)")

    elapsed = time.time() - t0
    avg = total_frames / ok if ok else 0
    print(f"\nDone in {elapsed:.0f}s — ok={ok} skip={skip} fail={fail}")
    print(f"Avg frames per video: {avg:.1f}")


if __name__ == "__main__":
    main()
