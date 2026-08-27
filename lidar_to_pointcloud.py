"""
lidar_to_pointcloud.py
=======================
Converts per-frame phone LiDAR data into a single dense world-space point cloud
that can be used as initialization for Gaussian Splatting training with Brush.

Binary frame format (FD3Q v1):
  [0]  4 bytes  magic = b'FD3Q'
  [4]  4 bytes  version = 1 (uint32 LE)
  [8]  8 bytes  unix timestamp (int64 LE, microseconds)
  [16] 4 bytes  point count (uint32 LE)
  [20] N*12     float32 XYZ, OpenCV camera space (x-right, y-down, z-forward)

Coordinate conventions:
  LiDAR local space : OpenCV  (x right, y down,  z forward into scene)
  transforms.json   : NeRF    (x right, y up,    z toward viewer, i.e. OpenGL)
  Conversion        : flip y and z before applying the c2w matrix

Timestamp sync:
  Photo 'time' field  – seconds from device boot (or session epoch)
  LiDAR timestamp     – unix microseconds (same physical clock, different epoch)
  Offset              = first_lidar_unix_s - first_photo_time_s

Usage:
  python lidar_to_pointcloud.py [--data-dir DIR] [--max-pts N] [--voxel V]

  --data-dir    folder containing transforms.json + lidar_frames/ + lidar_frames.csv
                (default: data/GS_20260707_170700)
  --max-pts     maximum output points after voxel filtering (default: 500000)
  --voxel       voxel grid size in meters for downsampling (default: 0.02)
  --max-range   maximum per-point distance from sensor in meters (default: 8.0)
                Removes lidar noise / sky hits that are far from the sensor.
  --out-ply     output PLY filename (default: points3d_lidar.ply in data-dir)
  --no-update   do NOT patch ply_file_path into transforms.json
"""

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Binary frame reader
# ──────────────────────────────────────────────────────────────────────────────

MAGIC = b"FD3Q"
HEADER_SIZE = 20  # bytes


def read_lidar_frame(path: Path, max_range: float = 8.0) -> tuple[int, np.ndarray]:
    """Return (unix_timestamp_us, Nx3 float32 array) from a .bin lidar frame.

    Points beyond `max_range` metres from the sensor are discarded — they are
    typically spurious sky/specular returns and would corrupt the world-space
    point cloud.
    """
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < HEADER_SIZE or data[:4] != MAGIC:
        raise ValueError(f"Not a valid FD3Q file: {path}")

    timestamp_us: int = struct.unpack_from("<q", data, 8)[0]
    npoints: int = struct.unpack_from("<I", data, 16)[0]

    expected = HEADER_SIZE + npoints * 12
    if len(data) < expected:
        raise ValueError(
            f"Truncated file {path}: expected {expected} bytes, got {len(data)}"
        )

    pts = np.frombuffer(data[HEADER_SIZE : HEADER_SIZE + npoints * 12], dtype=np.float32).reshape(-1, 3)

    # Remove outlier / far returns
    if max_range > 0:
        dist_sq = (pts ** 2).sum(axis=1)
        pts = pts[dist_sq <= max_range ** 2]

    return timestamp_us, pts


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate transform helpers
# ──────────────────────────────────────────────────────────────────────────────

def c2w_from_json_row(matrix_rows: list) -> np.ndarray:
    """
    Parse a 4x4 transform_matrix from nerfstudio transforms.json.

    The JSON stores the matrix in row-major order where each row is
    [r0, r1, r2, 0] for the rotation part and [tx, ty, tz, 1] for
    the translation.  numpy reads these rows directly so
    `np.array(matrix_rows)` already gives the conventional c2w in
    column-vector form (same as what Brush reconstructs after its
    `from_cols_slice + transpose` reading).

    Returns a 4x4 float64 ndarray (c2w, column-vector convention,
    NeRF / OpenGL coordinate system).
    """
    return np.array(matrix_rows, dtype=np.float64)


def lidar_opencv_to_world(pts_opencv: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """
    Transform LiDAR points from OpenCV camera space to NeRF world space.

    LiDAR local  : x right, y down,  z forward  (OpenCV)
    NeRF world   : x right, y up,    z toward viewer (OpenGL)

    Step 1 – convert OpenCV → OpenGL: flip y and z.
    Step 2 – apply c2w matrix.

    Parameters
    ----------
    pts_opencv : (N, 3) float32/64 in OpenCV camera space
    c2w        : (4, 4) float64 camera-to-world (NeRF convention)

    Returns
    -------
    (N, 3) float64 world-space XYZ
    """
    pts = pts_opencv.astype(np.float64)
    # Flip y and z (OpenCV → OpenGL)
    pts[:, 1] *= -1.0
    pts[:, 2] *= -1.0

    # Homogeneous transform.
    # The JSON matrix M is the transpose of the standard column-vector c2w, i.e.
    # translation is in the LAST ROW (not last column).  Row-vector convention:
    #   p_world_row = p_cam_row @ M_json   (do NOT transpose M_json here)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])          # (N, 4)
    pts_world_h = pts_h @ c2w               # (N, 4)  – row-vector convention
    return pts_world_h[:, :3]


# ──────────────────────────────────────────────────────────────────────────────
# Timestamp synchronisation
# ──────────────────────────────────────────────────────────────────────────────

def compute_clock_offset(photo_times: list[float], lidar_first_us: int) -> float:
    """
    Estimate the constant offset between the LiDAR unix clock and the photo
    session clock so that:
        photo_equivalent_time = lidar_unix_s - clock_offset

    Both sensors are on the same phone and share the same physical oscillator;
    only the epoch differs.  We align the first frames as the reference.
    """
    lidar_first_s = lidar_first_us / 1_000_000.0
    return lidar_first_s - photo_times[0]


def find_nearest_photo(photo_times: list[float], lidar_unix_us: int, clock_offset: float) -> int:
    """Return the index of the photo frame closest in time to the given lidar frame."""
    target = lidar_unix_us / 1_000_000.0 - clock_offset
    times = np.asarray(photo_times)
    return int(np.argmin(np.abs(times - target)))


# ──────────────────────────────────────────────────────────────────────────────
# Voxel-grid downsampling
# ──────────────────────────────────────────────────────────────────────────────

def voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """Thin-plate voxel grid: keep one random point per voxel."""
    if voxel_size <= 0 or len(pts) == 0:
        return pts

    mins = pts.min(axis=0)
    indices = ((pts - mins) / voxel_size).astype(np.int64)

    # Encode 3-D voxel index as a single int for fast dedup
    max_idx = indices.max(axis=0) + 1
    flat = indices[:, 0] * (max_idx[1] * max_idx[2]) + indices[:, 1] * max_idx[2] + indices[:, 2]

    # Shuffle then unique → random representative per voxel
    order = np.random.permutation(len(flat))
    flat_shuf = flat[order]
    _, keep = np.unique(flat_shuf, return_index=True)
    return pts[order[keep]]


# ──────────────────────────────────────────────────────────────────────────────
# PLY writer (binary little-endian, XYZ only)
# ──────────────────────────────────────────────────────────────────────────────

def write_ply(path: Path, pts: np.ndarray) -> None:
    pts32 = pts.astype(np.float32)
    n = len(pts32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode()
    with open(path, "wb") as f:
        f.write(header)
        f.write(pts32.tobytes())
    print(f"  Wrote {n:,} points → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir",  default=r"data\GS_20260707_170700", help="Dataset directory")
    ap.add_argument("--max-pts",   type=int,   default=500_000, help="Max output points")
    ap.add_argument("--voxel",     type=float, default=0.02,    help="Voxel grid size (m)")
    ap.add_argument("--max-range", type=float, default=8.0,     help="Max point distance from sensor (m)")
    ap.add_argument("--out-ply",   default=None, help="Output PLY path (default: <data-dir>/points3d_lidar.ply)")
    ap.add_argument("--no-update", action="store_true", help="Do not patch transforms.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    transforms_path = data_dir / "transforms.json"
    lidar_csv_path  = data_dir / "lidar_frames.csv"
    lidar_dir       = data_dir / "lidar_frames"
    out_ply         = Path(args.out_ply) if args.out_ply else data_dir / "points3d_lidar.ply"

    # ── Load transforms ───────────────────────────────────────────────────────
    print(f"Loading transforms from {transforms_path} …")
    with open(transforms_path, encoding="utf-8") as f:
        transforms = json.load(f)

    frames   = transforms["frames"]
    photo_times = [fr["time"] for fr in frames]
    c2w_list    = [c2w_from_json_row(fr["transform_matrix"]) for fr in frames]
    print(f"  {len(frames)} photo frames, time range [{photo_times[0]:.3f}, {photo_times[-1]:.3f}] s")

    # ── Load lidar CSV ────────────────────────────────────────────────────────
    print(f"Loading lidar index from {lidar_csv_path} …")
    with open(lidar_csv_path, newline="", encoding="utf-8") as f:
        lidar_rows = list(csv.DictReader(f))
    print(f"  {len(lidar_rows)} lidar frames")

    # Compute clock offset using first frame pair
    lidar_first_us = int(lidar_rows[0]["unix_timestamp_us"])
    clock_offset   = compute_clock_offset(photo_times, lidar_first_us)
    print(f"  Clock offset (lidar_unix_s - photo_s): {clock_offset:.3f} s")

    # ── Merge all lidar frames ────────────────────────────────────────────────
    all_world_pts: list[np.ndarray] = []
    max_dt_ms = 250.0  # reject match if nearest photo is >250 ms away

    print(f"Transforming lidar frames to world space …")
    for row in lidar_rows:
        frame_idx = int(row["frame_index"])
        unix_us   = int(row["unix_timestamp_us"])
        bin_path  = data_dir / row["file"]

        if not bin_path.exists():
            print(f"  [WARN] missing {bin_path}, skipping")
            continue

        try:
            _, pts_local = read_lidar_frame(bin_path, max_range=args.max_range)
        except ValueError as e:
            print(f"  [WARN] {e}, skipping")
            continue

        # Find nearest photo pose
        nearest_idx = find_nearest_photo(photo_times, unix_us, clock_offset)
        dt_s = abs(unix_us / 1e6 - clock_offset - photo_times[nearest_idx])

        if dt_s * 1000 > max_dt_ms:
            print(f"  [SKIP] lidar frame {frame_idx}: nearest photo is {dt_s*1000:.0f} ms away")
            continue

        c2w = c2w_list[nearest_idx]
        pts_world = lidar_opencv_to_world(pts_local, c2w)
        all_world_pts.append(pts_world)

    if not all_world_pts:
        print("[ERROR] No lidar frames were successfully processed.")
        sys.exit(1)

    pts_merged = np.vstack(all_world_pts)
    print(f"  Merged {len(pts_merged):,} raw points from {len(all_world_pts)} frames")

    # ── Voxel downsample ──────────────────────────────────────────────────────
    print(f"Voxel downsampling (voxel={args.voxel} m) …")
    pts_down = voxel_downsample(pts_merged, args.voxel)
    print(f"  After voxel filter: {len(pts_down):,} points")

    if len(pts_down) > args.max_pts:
        idx = np.random.choice(len(pts_down), args.max_pts, replace=False)
        pts_down = pts_down[idx]
        print(f"  After max-pts cap:  {len(pts_down):,} points")

    print(f"Point cloud extents:")
    print(f"  X: [{pts_down[:,0].min():.3f}, {pts_down[:,0].max():.3f}]")
    print(f"  Y: [{pts_down[:,1].min():.3f}, {pts_down[:,1].max():.3f}]")
    print(f"  Z: [{pts_down[:,2].min():.3f}, {pts_down[:,2].max():.3f}]")

    # ── Write PLY ─────────────────────────────────────────────────────────────
    write_ply(out_ply, pts_down)

    # ── Patch transforms.json ─────────────────────────────────────────────────
    if not args.no_update:
        rel_ply = out_ply.name  # same directory as transforms.json
        if transforms.get("ply_file_path") != rel_ply:
            transforms["ply_file_path"] = rel_ply
            with open(transforms_path, "w", encoding="utf-8") as f:
                json.dump(transforms, f, indent=2, ensure_ascii=False)
            print(f"  Updated transforms.json: ply_file_path = {rel_ply!r}")
        else:
            print(f"  transforms.json already has ply_file_path = {rel_ply!r}")


if __name__ == "__main__":
    main()
