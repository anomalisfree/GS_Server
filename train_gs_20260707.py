"""
train_gs_20260707.py
=====================
Launches Gaussian Splatting training for data/GS_20260707_170700.

The dataset already has:
  - transforms.json  – camera poses from phone ARKit/ARCore (nerfstudio format)
  - images/          – 189 video frames
  - points3d_lidar.ply – dense LiDAR point cloud (run lidar_to_pointcloud.py first)

COLMAP is NOT needed because camera poses are already computed by the phone.

Usage:
  python train_gs_20260707.py [--steps N] [--max-res R] [--dry-run]

Options:
  --steps     Total training iterations (default: 30000)
  --max-res   Max image resolution (default: 1920)
  --dry-run   Print the command without running it
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "GS_20260707_170700"
BRUSH_EXE = ROOT / "brush" / "target" / "release" / "brush.exe"
LIDAR_SCRIPT = ROOT / "lidar_to_pointcloud.py"
PYTHON = sys.executable


def ensure_lidar_pointcloud() -> None:
    ply_path = DATA_DIR / "points3d_lidar.ply"
    if ply_path.exists():
        print(f"[OK] LiDAR point cloud exists: {ply_path}")
        return

    print("[INFO] LiDAR point cloud not found – generating it now …")
    result = subprocess.run(
        [PYTHON, str(LIDAR_SCRIPT),
         "--data-dir", str(DATA_DIR),
         "--voxel", "0.02",
         "--max-pts", "500000",
         "--max-range", "8.0"],
        check=False,
    )
    if result.returncode != 0:
        print("[WARN] LiDAR point cloud generation failed. Training will proceed "
              "with the existing points.ply (20K points).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps",   type=int, default=30_000, help="Total training iterations")
    ap.add_argument("--max-res", type=int, default=1920,   help="Max image resolution")
    ap.add_argument("--dry-run", action="store_true",      help="Print command without running")
    args = ap.parse_args()

    if not BRUSH_EXE.exists():
        print(f"[ERROR] Brush executable not found: {BRUSH_EXE}")
        print("  Build it with: cd brush && cargo build --release --bin brush")
        sys.exit(1)

    if not DATA_DIR.exists():
        print(f"[ERROR] Dataset not found: {DATA_DIR}")
        sys.exit(1)

    # Make sure we have a dense LiDAR point cloud
    ensure_lidar_pointcloud()

    # Export directory relative to dataset's parent (= data/)
    # Using {dataset} interpolation: data/GS_20260707_170700_exports/
    export_path = "./{dataset}_exports/"

    cmd = [
        str(BRUSH_EXE),
        str(DATA_DIR),
        "--total-train-iters",   str(args.steps),
        "--max-resolution",      str(args.max_res),
        "--refine-every",        "200",   # ~equal to number of images (189)
        "--eval-split-every",    "8",     # hold out every 8th frame for eval
        "--eval-every",          "1000",  # compute eval metrics every 1000 steps
        "--export-every",        "5000",  # export .ply checkpoint every 5000 steps
        "--export-path",         export_path,
        "--export-name",         "model_{iter}.ply",
    ]

    print("\n=== Brush Training Command ===")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    print()

    if args.dry_run:
        print("[DRY RUN] Not executing.")
        return

    print(f"[INFO] Training {args.steps} iterations on {DATA_DIR.name}")
    print(f"[INFO] Output: data/{DATA_DIR.name}_exports/")
    print("[INFO] Press Ctrl+C to stop early (model will be exported at last checkpoint)")
    print()

    try:
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode == 0:
            print("\n[DONE] Training completed successfully.")
            print(f"  Model files in: data/{DATA_DIR.name}_exports/")
        else:
            print(f"\n[ERROR] Training exited with code {result.returncode}")
            sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training stopped by user.")


if __name__ == "__main__":
    main()
