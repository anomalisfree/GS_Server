"""
E57 → Nerfstudio dataset converter for Brush (Gaussian Splatting training)

Converts a Leica/FARO E57 file with structured scans (rowIndex/columnIndex + RGB + pose)
into a Nerfstudio-compatible dataset (transforms.json + images).

Camera model determined from data:
  - fx = fy = 532.63 px  (HFOV=62°, VFOV=48°, image 639×479)
  - Sensor frame: X=right, Y=down, Z=forward (OpenCV convention)
  - Pose: R,T convert sensor→world (P_world = R @ P_sensor + T)

Output:
  <out_dir>/
    images/          JPEG frames
    transforms.json  Nerfstudio format (compatible with Brush)
    points3d.ply     Merged colored point cloud for GS init (optional)

Usage:
  python e57_to_nerfstudio.py [--e57 PATH] [--out DIR] [--step N] [--no-ply]
"""

import argparse
import json
import math
import os
import struct
import time
from pathlib import Path

import numpy as np
import pye57
from PIL import Image
from pyquaternion import Quaternion

# ── Camera intrinsics (computed from data – see compute_intrinsics.py) ──
FX = 532.63
FY = 532.63
CX = 320.29
CY = 239.81
W  = 639
H  = 479


def build_c2w_nerf(rotation_wxyz: np.ndarray, translation: np.ndarray) -> list:
    """
    Convert E57 pose (sensor→world, OpenCV sensor frame X=right,Y=down,Z=fwd)
    to a NeRF-convention camera-to-world 4×4 matrix.

    Brush reads NeRF c2w and internally flips Y and Z axes to recover OpenCV:
        transform.y_axis *= -1;  transform.z_axis *= -1;
    So we provide R with those columns pre-flipped (NeRF convention = flip Y,Z of OpenCV).
    """
    w, x, y, z = rotation_wxyz
    q = Quaternion(w=float(w), x=float(x), y=float(y), z=float(z))
    R = q.rotation_matrix  # 3×3, sensor→world (OpenCV sensor convention)

    # Convert to NeRF convention: negate Y and Z columns
    R_nerf = R.copy()
    R_nerf[:, 1] *= -1.0  # Y_world_of_sensor_up  = -Y_world_of_sensor_down
    R_nerf[:, 2] *= -1.0  # Z_nerf = toward viewer = -Z_fwd

    c2w = np.eye(4, dtype=float)
    c2w[:3, :3] = R_nerf
    c2w[:3, 3]  = translation
    return c2w.tolist()


def render_image(raw: dict, alpha_mask: bool = False) -> np.ndarray:
    """Build an H×W RGB(A) uint8 image from raw E57 scan data.

    If alpha_mask=True, returns RGBA where unmeasured pixels have alpha=0.
    This allows Brush (--alpha-mode masked) to ignore black background pixels.
    """
    row = np.array(raw['rowIndex'],    dtype=np.int32)
    col = np.array(raw['columnIndex'], dtype=np.int32)
    r   = np.array(raw['colorRed'],    dtype=np.uint8)
    g   = np.array(raw['colorGreen'],  dtype=np.uint8)
    b   = np.array(raw['colorBlue'],   dtype=np.uint8)

    h = int(row.max()) + 1
    w = int(col.max()) + 1

    if alpha_mask:
        img = np.zeros((h, w, 4), dtype=np.uint8)          # RGBA, alpha=0 by default
        img[row, col, 0] = r
        img[row, col, 1] = g
        img[row, col, 2] = b
        img[row, col, 3] = 255                              # measured pixels → opaque
    else:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[row, col] = np.stack([r, g, b], axis=1)
    return img


def write_ply_header(f, n_points: int):
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    f.write(header.encode('ascii'))


def convert(e57_path: str, out_dir: str, step: int = 1, write_ply: bool = True,
            max_pts_per_scan: int = 50_000, alpha_mask: bool = False):
    out_dir   = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    e57 = pye57.E57(e57_path)
    total = e57.scan_count
    scan_indices = list(range(0, total, step))
    print(f"E57: {total} scans → using {len(scan_indices)} (step={step})")

    frames   = []
    ply_path = out_dir / "points3d.ply"
    ply_pts  = []  # list of (x,y,z,r,g,b) arrays in world frame

    t0 = time.time()
    for i, idx in enumerate(scan_indices):
        raw    = e57.read_scan_raw(idx)
        header = e57.get_header(idx)

        # ── Image ──
        img_arr = render_image(raw, alpha_mask=alpha_mask)
        if alpha_mask:
            img_filename = f"frame_{idx:04d}.png"
            img_path     = images_dir / img_filename
            Image.fromarray(img_arr, mode="RGBA").save(img_path)
        else:
            img_filename = f"frame_{idx:04d}.jpg"
            img_path     = images_dir / img_filename
            Image.fromarray(img_arr).save(img_path, quality=92)

        # ── Pose ──
        c2w = build_c2w_nerf(header.rotation, header.translation)

        frames.append({
            "file_path": f"images/{img_filename}",
            "transform_matrix": c2w,
        })

        # ── Point cloud (world frame) for PLY ──
        if write_ply:
            x = np.array(raw['cartesianX'], dtype=np.float32)
            y = np.array(raw['cartesianY'], dtype=np.float32)
            z_cam = np.array(raw['cartesianZ'], dtype=np.float32)
            r_col = np.array(raw['colorRed'],   dtype=np.uint8)
            g_col = np.array(raw['colorGreen'], dtype=np.uint8)
            b_col = np.array(raw['colorBlue'],  dtype=np.uint8)

            valid = (z_cam > 0.05) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z_cam)
            x, y, z_cam = x[valid], y[valid], z_cam[valid]
            r_col, g_col, b_col = r_col[valid], g_col[valid], b_col[valid]

            # Transform sensor→world:  P_world = R @ P_sensor + T
            rot_wxyz = header.rotation
            w_, x_, y_, z_ = rot_wxyz
            q = Quaternion(w=float(w_), x=float(x_), y=float(y_), z=float(z_))
            R = q.rotation_matrix  # 3×3

            pts_local = np.stack([x, y, z_cam], axis=1)  # (N, 3)
            T = np.array(header.translation, dtype=np.float32)

            # Subsample to keep PLY manageable
            if len(pts_local) > max_pts_per_scan:
                idx_sub = np.random.choice(len(pts_local), max_pts_per_scan, replace=False)
                pts_local = pts_local[idx_sub]
                r_col = r_col[idx_sub]
                g_col = g_col[idx_sub]
                b_col = b_col[idx_sub]

            pts_world = (R @ pts_local.T).T + T  # (N, 3)
            ply_pts.append((pts_world, r_col, g_col, b_col))

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(scan_indices) - i - 1)
        print(f"  [{i+1:3d}/{len(scan_indices)}] scan {idx:3d}  ETA {eta:.0f}s", end='\r')

    e57.close()
    print(f"\nAll scans processed in {time.time()-t0:.1f}s")

    # ── Write transforms.json ──
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": FX,
        "fl_y": FY,
        "cx":   CX,
        "cy":   CY,
        "w":    W,
        "h":    H,
        "frames": frames,
    }
    tf_path = out_dir / "transforms.json"
    with open(tf_path, "w") as f:
        json.dump(transforms, f, indent=2)
    print(f"Wrote: {tf_path}  ({len(frames)} frames)")

    # ── Write PLY ──
    if write_ply and ply_pts:
        all_pts  = np.concatenate([p[0] for p in ply_pts], axis=0)
        all_r    = np.concatenate([p[1] for p in ply_pts], axis=0)
        all_g    = np.concatenate([p[2] for p in ply_pts], axis=0)
        all_b    = np.concatenate([p[3] for p in ply_pts], axis=0)

        n = len(all_pts)
        print(f"Writing PLY: {n:,} points → {ply_path}")
        with open(ply_path, 'wb') as f:
            write_ply_header(f, n)
            # Pack: 3×float32 + 3×uint8 = 15 bytes per point
            data = np.zeros(n, dtype=[
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ])
            data['x'] = all_pts[:, 0]
            data['y'] = all_pts[:, 1]
            data['z'] = all_pts[:, 2]
            data['red']   = all_r
            data['green'] = all_g
            data['blue']  = all_b
            f.write(data.tobytes())
        print(f"PLY written: {ply_path.stat().st_size / 1e6:.1f} MB")

    print("\nDataset ready:")
    print(f"  Images:         {images_dir}")
    print(f"  transforms.json:{tf_path}")
    if write_ply:
        print(f"  points3d.ply:   {ply_path}")
    print(f"\nTo train with Brush:")
    print(f"  brush train --source \"{out_dir}\" --export-path model.ply")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert E57 to Nerfstudio dataset for Brush")
    parser.add_argument("--e57",  default=r"d:\VSProjects\GS_Sever\data\2026-06-23 e57 NicoOffice 1.51min 5m TD 5m Threshold export 1.e57")
    parser.add_argument("--out",  default=r"d:\VSProjects\GS_Sever\data\NicoOffice_nerfstudio")
    parser.add_argument("--step", type=int, default=1,
                        help="Use every N-th scan (1=all, 2=half, etc.)")
    parser.add_argument("--no-ply", action="store_true",
                        help="Skip point cloud PLY generation")
    parser.add_argument("--max-pts", type=int, default=50_000,
                        help="Max points per scan in PLY (default 50000)")
    parser.add_argument("--alpha-mask", action="store_true",
                        help="Save images as RGBA PNG with alpha=0 for unmeasured pixels. "
                             "Use with: brush --alpha-mode masked")
    args = parser.parse_args()

    convert(
        e57_path=args.e57,
        out_dir=args.out,
        step=args.step,
        write_ply=not args.no_ply,
        max_pts_per_scan=args.max_pts,
        alpha_mask=args.alpha_mask,
    )
