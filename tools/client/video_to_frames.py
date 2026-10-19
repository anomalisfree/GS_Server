#!/usr/bin/env python
"""Turn walking/phone videos into a photo set for the server's colmap_images mode.

For every window of --block consecutive video frames the SHARPEST one (variance of the
Laplacian on a 540x960 downscale) is kept, so motion-blurred frames are dropped. Output files are
named <prefix>_<NNNN>.jpg; use a different prefix per video so several videos can be merged into one
job without file-name collisions (the server keeps images in one flat directory).

Example (what the CameraSet experiments used):
  python tools/client/video_to_frames.py --out frames8 --block 8 \
      --video "data/CameraSet/scan01.MOV=scan01" --video "data/CameraSet/scan02.MOV=scan02"

Needs: opencv-python, numpy.   Rule of thumb for --block at 30 fps walking video:
  10 -> ~3 fps (used for the 3 first CameraSet models), 8 -> 3.75 fps, 5 -> 6 fps.
Denser is NOT automatically better: with the server's default 32k SIFT features per image, 1145 frames
made the COLMAP mapper crawl (13 images/hour); see CameraSet_Training_Report_2026-10-19.md.
"""
import argparse
import os
import sys

import cv2


def sharpness(frame):
    small = cv2.resize(frame, (540, 960), interpolation=cv2.INTER_AREA)
    return cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def extract(video, prefix, out_dir, block, quality, max_windows):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{video}: {total} frames @ {fps:.1f} fps ({total / fps:.0f} s), "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    k, window = 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        window.append((sharpness(frame), frame))
        if len(window) == block:
            _, best = max(window, key=lambda t: t[0])
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_{k:04d}.jpg"), best,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            k += 1
            window = []
            if max_windows and k >= max_windows:
                break
    print(f"  -> {k} images written as {prefix}_NNNN.jpg")
    return k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", required=True, metavar="PATH=PREFIX",
                    help="video file and the file-name prefix for its frames (repeatable)")
    ap.add_argument("--out", required=True, help="output directory (created if missing)")
    ap.add_argument("--block", type=int, default=10, help="keep the sharpest frame of every N frames (default 10)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    ap.add_argument("--max-windows", type=int, default=0, help="debug: stop after N output frames per video")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    total = 0
    for spec in a.video:
        if "=" not in spec:
            sys.exit(f"--video must look like PATH=PREFIX, got: {spec}")
        path, prefix = spec.rsplit("=", 1)
        total += extract(path, prefix, a.out, a.block, a.quality, a.max_windows)
    print(f"done: {total} images in {a.out}")


if __name__ == "__main__":
    main()
