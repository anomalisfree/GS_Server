"""
Second-generation external-LiDAR calibration: replaces the interpolated-camera-pose
+ hand-eye-ICP approach (external_lidar_to_pointcloud.py) with KISS-ICP LiDAR
odometry.

Why: the previous approach placed each LiDAR frame using the CAMERA's pose,
interpolated between sparse camera frames, plus an estimated fixed camera->LiDAR
offset. That makes every source of camera-pose error (COLMAP/ARKit pose noise,
SLERP interpolation between camera frames that are ~0.5s apart while LiDAR
samples faster, hand-eye offset estimation error) leak directly into the LiDAR
cloud's geometry. A self-consistency check (each frame's points vs. a dense
reference built from OTHER frames placed the same way) looked fine on average,
but the user's own visual inspection of the resulting merged cloud/trained model
still showed it as wrong.

KISS-ICP estimates the LiDAR's OWN trajectory directly from frame-to-frame /
frame-to-map scan registration - no camera pose, no interpolation, no hand-eye
estimate involved at all. This produces a self-consistent map purely from LiDAR
geometry. The only remaining step is a single similarity (rotation+translation+
scale) alignment between the resulting LiDAR-SLAM trajectory and the camera
trajectory (matched by timestamp), which is a much better-posed problem than
point-cloud ICP against a sparse target: it's a direct-correspondence Umeyama fit
between two full 3D trajectories, not nearest-neighbor matching against a sparse,
non-uniformly-covered COLMAP point cloud.

Run on the server (not Windows) because kiss-icp only ships manylinux wheels.
"""
import sys
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from external_lidar_to_pointcloud import (
    parse_lidar_frame, index_lidar_frames, load_arkit_transforms,
    parse_colmap_images_txt, parse_colmap_points3d_txt, undo_nerf_flip,
    apply_nerf_flip, recover_scale, interpolate_pose, write_ply_xyzrgb,
)

from kiss_icp.config.parser import KISSConfig
from kiss_icp.kiss_icp import KissICP


def run_kiss_icp_odometry(frame_local_xyz: list, max_range: float, voxel_size: float):
    """Runs KISS-ICP sequentially over a list of per-frame local-XYZ point arrays.
    Returns a list of 4x4 poses (frame_k -> "SLAM frame", frame 0 = identity)."""
    cfg = KISSConfig()
    cfg.data.max_range = max_range
    cfg.data.min_range = 0.0
    cfg.data.deskew = False  # no per-point timestamps available within a frame
    cfg.mapping.voxel_size = voxel_size
    icp = KissICP(cfg)

    poses = []
    for i, local_xyz in enumerate(frame_local_xyz):
        ts = np.zeros(len(local_xyz))
        icp.register_frame(local_xyz.astype(np.float64), ts)
        poses.append(icp.last_pose.copy())
        if i % 20 == 0:
            print(f"  KISS-ICP frame {i}/{len(frame_local_xyz)}: "
                  f"pos={icp.last_pose[:3,3]}")
    return poses


def umeyama_similarity(P: np.ndarray, Q: np.ndarray):
    """Least-squares similarity transform (scale, rotation, translation) mapping
    P onto Q: Q ~= scale * R @ P + t. Standard Umeyama (1991) closed form."""
    mu_p = P.mean(axis=0)
    mu_q = Q.mean(axis=0)
    X = P - mu_p
    Y = Q - mu_q
    n = len(P)
    Sxy = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(Sxy)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_x = (X ** 2).sum(axis=1).mean()
    scale = np.trace(np.diag(D) @ S) / var_x
    t = mu_q - scale * R @ mu_p
    return R, t, scale


def main():
    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/uploads/GS_20260805_105828_raw")
    colmap_cache = dataset_dir / "colmap_cache" / "sparse_txt"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else dataset_dir / "lidar_calibrated_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. Loading ARKit transforms.json + cached COLMAP result ===")
    arkit_json, arkit_frames = load_arkit_transforms(dataset_dir / "transforms.json")
    colmap_poses = parse_colmap_images_txt(colmap_cache / "images.txt")
    colmap_xyz, colmap_rgb = parse_colmap_points3d_txt(colmap_cache / "points3D.txt")
    print(f"  {len(arkit_frames)} camera frames, COLMAP registered {len(colmap_poses)}, "
          f"{len(colmap_xyz)} sparse points")

    print("=== 2. Recovering metric scale (COLMAP vs ARKit relative motion) ===")
    scale = recover_scale(colmap_poses, arkit_frames)
    print(f"  scale factor = {scale:.4f}")
    colmap_poses_scaled = {n: m.copy() for n, m in colmap_poses.items()}
    for m in colmap_poses_scaled.values():
        m[:3, 3] *= scale
    colmap_xyz_scaled = colmap_xyz * scale

    ts_lookup = {f["name"]: f["ts"] for f in arkit_frames}
    pose_frames = sorted(
        [{"ts": ts_lookup[n], "c2w": m} for n, m in colmap_poses_scaled.items() if n in ts_lookup],
        key=lambda f: f["ts"],
    )

    print("=== 3. Parsing raw LiDAR frames ===")
    lidar_index = index_lidar_frames(dataset_dir)
    frame_local_xyz = [parse_lidar_frame(path) for ts, path in lidar_index]
    frame_ts = [ts for ts, path in lidar_index]
    n_total = sum(len(a) for a in frame_local_xyz)
    print(f"  {len(lidar_index)} frames, {n_total:,} points")

    print("=== 4. KISS-ICP LiDAR odometry (camera-independent) ===")
    # tof ranged ~1.3-22m in this dataset; voxel_size ~ typical point spacing at
    # mid-range, small enough to preserve room-scale detail.
    slam_poses = run_kiss_icp_odometry(frame_local_xyz, max_range=25.0, voxel_size=0.05)

    print("=== 5. Aligning LiDAR-SLAM trajectory to camera trajectory (Umeyama) ===")
    slam_positions = np.array([p[:3, 3] for p in slam_poses])
    camera_positions = []
    for ts in frame_ts:
        pose, _ = interpolate_pose(pose_frames, ts)
        camera_positions.append(pose[:3, 3])
    camera_positions = np.array(camera_positions)

    R_align, t_align, s_align = umeyama_similarity(slam_positions, camera_positions)
    fit_err = np.linalg.norm((s_align * (R_align @ slam_positions.T).T + t_align) - camera_positions, axis=1)
    print(f"  scale={s_align:.4f} rotation={np.degrees(Rotation.from_matrix(R_align).magnitude()):.2f}deg "
          f"translation_norm={np.linalg.norm(t_align):.4f}")
    print(f"  trajectory fit residual: median={np.median(fit_err):.4f} p95={np.percentile(fit_err,95):.4f} "
          f"max={fit_err.max():.4f}")

    print("=== 6. Building final calibrated cloud ===")
    all_world_pts = []
    for local_xyz, slam_pose in zip(frame_local_xyz, slam_poses):
        slam_world = (slam_pose[:3, :3] @ local_xyz.T).T + slam_pose[:3, 3]
        final_world = s_align * (R_align @ slam_world.T).T + t_align
        all_world_pts.append(final_world)
    corrected_lidar_world = np.concatenate(all_world_pts, axis=0)
    print(f"  {len(corrected_lidar_world):,} points")

    print("=== 7. Writing outputs ===")
    rgb = np.full((len(corrected_lidar_world), 3), 180, dtype=np.uint8)
    write_ply_xyzrgb(out_dir / "points3d.ply", corrected_lidar_world, rgb)
    write_ply_xyzrgb(out_dir / "colmap_sparse_reference.ply", colmap_xyz_scaled, colmap_rgb)

    global_intrinsics = {k: arkit_json[k] for k in ("fl_x", "fl_y", "cx", "cy", "w", "h") if k in arkit_json}
    out_frames = []
    for fr in arkit_frames:
        name = fr["name"]
        if name not in colmap_poses_scaled:
            continue
        c2w_nerf = apply_nerf_flip(colmap_poses_scaled[name])
        out_frames.append({
            "file_path": f"images/{name}",
            "unix_timestamp_us": fr["ts"],
            "transform_matrix": c2w_nerf.tolist(),
            **global_intrinsics,
        })
    out_json = {k: v for k, v in arkit_json.items() if k != "frames"}
    out_json["frames"] = out_frames
    (out_dir / "transforms.json").write_text(json.dumps(out_json, indent=2))

    print(f"  wrote {out_dir}/points3d.ply ({len(corrected_lidar_world):,} pts)")
    print(f"  wrote {out_dir}/transforms.json ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()
