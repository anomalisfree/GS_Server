"""
Leica BLK2GO SensorFrames → Nerfstudio dataset for Brush (Gaussian Splatting)

Reads Leica export folder containing:
  images/          RGB frames (1920×1440 JPEG)
  depth/           depth PNGs (same index)
  *_intrinsics.txt per-frame camera intrinsics (CSV)
  *_SensorFrames.txt per-frame 4×4 c2w matrices (row-major, OpenCV convention)

Also extracts colored point cloud from the E57 file for GS initialization
(replaces COLMAP's points3D).

Coordinate convention:
  Leica SensorFrames: c2w in OpenCV convention (X=right, Y=down, Z=fwd), world Z=up
  Nerfstudio/Brush:   NeRF convention (flip Y,Z columns of OpenCV c2w)

Output (written into <out_dir>):
  images/           copy of (or symlink to) source images
  transforms.json   Nerfstudio format, per-frame intrinsics
  points3d.ply      Colored point cloud from E57 (GS init, replaces COLMAP)

Usage:
  python leica_to_nerfstudio.py [--data-dir DIR] [--e57 PATH] [--out DIR]
                                [--no-ply] [--step N] [--max-pts N]
"""

import argparse
import json
import os
import shutil
import struct
import time
from pathlib import Path

import numpy as np

try:
    import pye57
    from pyquaternion import Quaternion
    HAS_E57 = True
except ImportError:
    HAS_E57 = False
    print("[WARN] pye57/pyquaternion not installed – skipping point cloud generation")


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_intrinsics(path: Path) -> dict:
    """Return dict: filename → {w, h, fx, fy, cx, cy}"""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            fname   = parts[0].strip()          # e.g. "images/foo.jpg"
            w       = int(parts[1])
            h       = int(parts[2])
            fx      = float(parts[3])
            fy      = float(parts[4])
            cx      = float(parts[5])
            cy      = float(parts[6])
            result[fname] = dict(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy)
    return result


def parse_sensor_frames(path: Path) -> list:
    """Return list of (filename, 4×4 ndarray) in file order."""
    frames = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            idx    = line.index(",")
            fname  = line[:idx].strip()
            values = list(map(float, line[idx + 1:].split()))
            if len(values) != 16:
                print(f"  [WARN] unexpected matrix size for {fname}: {len(values)} values")
                continue
            mat = np.array(values, dtype=np.float64).reshape(4, 4)
            frames.append((fname, mat))
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate conversion
# ─────────────────────────────────────────────────────────────────────────────

def opencv_c2w_to_nerf(mat: np.ndarray) -> np.ndarray:
    """
    Convert Leica c2w (OpenCV sensor: X=right, Y=down, Z=fwd)
    to Nerfstudio/Brush NeRF convention (flip Y and Z columns).

    Brush internally converts back to OpenCV:
        transform.y_axis *= -1;  transform.z_axis *= -1;
    So we pre-flip those columns.
    """
    m = mat.copy()
    m[:3, 1] *= -1.0   # Y column: down → up
    m[:3, 2] *= -1.0   # Z column: fwd  → backward (NeRF convention)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# PLY writer
# ─────────────────────────────────────────────────────────────────────────────

def write_ply(path: Path, pts: np.ndarray, r: np.ndarray,
              g: np.ndarray, b: np.ndarray):
    n = len(pts)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    data = np.zeros(n, dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    data['x'] = pts[:, 0].astype(np.float32)
    data['y'] = pts[:, 1].astype(np.float32)
    data['z'] = pts[:, 2].astype(np.float32)
    data['red']   = r
    data['green'] = g
    data['blue']  = b
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())
    print(f"  PLY written: {path}  ({path.stat().st_size / 1e6:.1f} MB, {n:,} pts)")


# ─────────────────────────────────────────────────────────────────────────────
# E57 point cloud extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_ply_from_e57(e57_path: str, ply_path: Path,
                         step: int = 1, max_pts_per_scan: int = 50_000):
    """Extract world-frame colored point cloud from E57 → PLY."""
    if not HAS_E57:
        print("  [SKIP] pye57 not available")
        return

    print(f"\nExtracting point cloud from E57: {e57_path}")
    e57 = pye57.E57(e57_path)
    total = e57.scan_count
    scan_indices = list(range(0, total, step))
    print(f"  {total} scans → using {len(scan_indices)} (step={step})")

    all_pts, all_r, all_g, all_b = [], [], [], []

    t0 = time.time()
    for i, idx in enumerate(scan_indices):
        try:
            raw    = e57.read_scan_raw(idx)
            header = e57.get_header(idx)
        except Exception as ex:
            print(f"  [WARN] scan {idx}: {ex}")
            continue

        x = np.array(raw['cartesianX'], dtype=np.float32)
        y = np.array(raw['cartesianY'], dtype=np.float32)
        z = np.array(raw['cartesianZ'], dtype=np.float32)

        valid = (z > 0.05) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[valid], y[valid], z[valid]

        has_color = all(k in raw for k in ('colorRed', 'colorGreen', 'colorBlue'))
        if has_color:
            r = np.array(raw['colorRed'],   dtype=np.uint8)[valid]
            g = np.array(raw['colorGreen'], dtype=np.uint8)[valid]
            b = np.array(raw['colorBlue'],  dtype=np.uint8)[valid]
        else:
            r = g = b = np.full(valid.sum(), 128, dtype=np.uint8)

        if len(x) == 0:
            continue

        # Sub-sample
        if len(x) > max_pts_per_scan:
            sel = np.random.choice(len(x), max_pts_per_scan, replace=False)
            x, y, z = x[sel], y[sel], z[sel]
            if has_color:
                r, g, b = r[sel], g[sel], b[sel]

        # Transform sensor→world: P_world = R @ P_local + T
        rot_wxyz = header.rotation
        w_, x_, y_, z_ = rot_wxyz
        q = Quaternion(w=float(w_), x=float(x_), y=float(y_), z=float(z_))
        R = q.rotation_matrix.astype(np.float32)
        T = np.array(header.translation, dtype=np.float32)

        pts_local = np.stack([x, y, z], axis=1)  # (N,3)
        pts_world = (R @ pts_local.T).T + T

        all_pts.append(pts_world)
        all_r.append(r); all_g.append(g); all_b.append(b)

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(scan_indices) - i - 1)
        print(f"  scan {idx:3d}/{total-1}  pts={len(x):,}  ETA {eta:.0f}s    ", end='\r')

    e57.close()
    print(f"\n  Done in {time.time()-t0:.1f}s")

    if not all_pts:
        print("  [WARN] No valid points extracted from E57")
        return

    pts = np.concatenate(all_pts)
    R_  = np.concatenate(all_r)
    G_  = np.concatenate(all_g)
    B_  = np.concatenate(all_b)
    write_ply(ply_path, pts, R_, G_, B_)


# ─────────────────────────────────────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert(data_dir: str, e57_path: str, out_dir: str,
            write_ply_flag: bool = True,
            e57_step: int = 2, max_pts_per_scan: int = 50_000,
            copy_images: bool = True):

    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Find metadata files ──
    intrinsics_file = next(data_dir.glob("*_intrinsics.txt"), None)
    poses_file      = next(data_dir.glob("*_SensorFrames.txt"), None)

    if intrinsics_file is None:
        raise FileNotFoundError(f"No *_intrinsics.txt found in {data_dir}")
    if poses_file is None:
        raise FileNotFoundError(f"No *_SensorFrames.txt found in {data_dir}")

    print(f"Intrinsics: {intrinsics_file.name}")
    print(f"Poses:      {poses_file.name}")

    intrinsics_map = parse_intrinsics(intrinsics_file)
    pose_frames    = parse_sensor_frames(poses_file)

    print(f"Loaded {len(intrinsics_map)} intrinsic entries, {len(pose_frames)} pose entries")

    # ── Image output folder ──
    out_images_dir = out_dir / "images"
    out_images_dir.mkdir(exist_ok=True)

    frames = []
    missing = 0

    t0 = time.time()
    for i, (rel_path, mat_opencv) in enumerate(pose_frames):
        # rel_path is like "images/2026-06-28 Nico Office_SensorFrame_00000.jpg"
        src_img = data_dir / rel_path

        if not src_img.exists():
            print(f"  [WARN] missing: {src_img}")
            missing += 1
            continue

        # Copy / ensure image in output
        dest_img = out_images_dir / src_img.name
        if copy_images and not dest_img.exists():
            shutil.copy2(src_img, dest_img)

        # Intrinsics for this frame
        intr = intrinsics_map.get(rel_path)
        if intr is None:
            # Try matching just filename
            intr = next((v for k, v in intrinsics_map.items()
                         if Path(k).name == src_img.name), None)
        if intr is None:
            print(f"  [WARN] no intrinsics for {rel_path}")
            continue

        # Convert c2w: OpenCV → NeRF convention
        c2w_nerf = opencv_c2w_to_nerf(mat_opencv)

        frame_entry = {
            "file_path": f"images/{src_img.name}",
            "fl_x": intr["fx"],
            "fl_y": intr["fy"],
            "cx":   intr["cx"],
            "cy":   intr["cy"],
            "w":    intr["w"],
            "h":    intr["h"],
            "transform_matrix": c2w_nerf.tolist(),
        }
        frames.append(frame_entry)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(pose_frames)} frames ...", end='\r')

    print(f"\nBuilt {len(frames)} frames  ({missing} missing images)")

    # Compute median intrinsics for global fields
    fx_vals = [f["fl_x"] for f in frames]
    fy_vals = [f["fl_y"] for f in frames]
    cx_vals = [f["cx"]   for f in frames]
    cy_vals = [f["cy"]   for f in frames]
    med_fx = float(np.median(fx_vals))
    med_fy = float(np.median(fy_vals))
    med_cx = float(np.median(cx_vals))
    med_cy = float(np.median(cy_vals))
    w_val  = frames[0]["w"] if frames else 1920
    h_val  = frames[0]["h"] if frames else 1440

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": med_fx,
        "fl_y": med_fy,
        "cx":   med_cx,
        "cy":   med_cy,
        "w":    w_val,
        "h":    h_val,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "frames": frames,
    }

    tf_path = out_dir / "transforms.json"
    with open(tf_path, "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)
    print(f"Wrote: {tf_path}  ({len(frames)} frames)")
    print(f"  Median intrinsics: fx={med_fx:.1f}, fy={med_fy:.1f}, "
          f"cx={med_cx:.1f}, cy={med_cy:.1f}, {w_val}x{h_val}")
    print(f"  Images copied to: {out_images_dir}")
    print(f"  Total prep time: {time.time()-t0:.1f}s")

    # ── Point cloud from E57 ──
    if write_ply_flag and e57_path:
        ply_out = out_dir / "points3d.ply"
        extract_ply_from_e57(e57_path, ply_out,
                             step=e57_step, max_pts_per_scan=max_pts_per_scan)
    elif write_ply_flag:
        print("\n[INFO] No E57 path specified – skipping point cloud")

    print(f"\n{'='*55}")
    print(f"Dataset ready: {out_dir}")
    print(f"\nTo train with Brush:")
    print(f'  brush.exe train "{out_dir}"')
    print(f'                   --export-path "./NicoOffice_2028_exports/"')
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEFAULT_DATA = r"D:\VSProjects\GS_Sever\data\2026-06-28 Nico Office"
    DEFAULT_E57  = (r"D:\VSProjects\GS_Sever\data\2026-06-28 Nico Office"
                    r"\2026-06-28 E57 Nico Office w HighRes RGB 1.51min 5m MD 5m Thresh.e57")
    DEFAULT_OUT  = r"D:\VSProjects\GS_Sever\data\NicoOffice_2028_nerfstudio"

    parser = argparse.ArgumentParser(
        description="Convert Leica SensorFrames + E57 to Nerfstudio dataset for Brush")
    parser.add_argument("--data-dir",    default=DEFAULT_DATA,
                        help="Folder with images/, depth/, *_intrinsics.txt, *_SensorFrames.txt")
    parser.add_argument("--e57",         default=DEFAULT_E57,
                        help="Path to E57 file for point cloud initialization")
    parser.add_argument("--out",         default=DEFAULT_OUT,
                        help="Output directory for Nerfstudio dataset")
    parser.add_argument("--no-ply",      action="store_true",
                        help="Skip E57 point cloud extraction")
    parser.add_argument("--e57-step",    type=int, default=2,
                        help="Use every N-th scan from E57 (default 2)")
    parser.add_argument("--max-pts",     type=int, default=50_000,
                        help="Max points per scan in PLY (default 50000)")
    parser.add_argument("--no-copy",     action="store_true",
                        help="Do not copy images (assumes out-dir == data-dir or images already there)")
    args = parser.parse_args()

    convert(
        data_dir       = args.data_dir,
        e57_path       = None if args.no_ply else args.e57,
        out_dir        = args.out,
        write_ply_flag = not args.no_ply,
        e57_step       = args.e57_step,
        max_pts_per_scan = args.max_pts,
        copy_images    = not args.no_copy,
    )
