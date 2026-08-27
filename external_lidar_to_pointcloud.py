"""
Build a metric, world-aligned point cloud from an external LiDAR's raw per-frame
scans, for datasets shaped like GS_20260805_105828 / GS_20260805_135900:
  images/            - camera frames, referenced by transforms.json
  transforms.json    - ARKit camera poses (NOT trustworthy - only used as a rough
                        scale/seed reference), per-frame unix_timestamp_us
  points.ply         - sparse ARKit point cloud (xyz only, no color)
  lidar_frames/*.txt - raw external-LiDAR frames: for each point, spherical
                        (tof, theta, phi) in the LiDAR's own local frame at
                        capture time. No pose - axes/scale relative to the phone
                        are unknown and uncalibrated.
  lidar_frames.csv   - index of lidar_frames/ with unix_timestamp_us per frame

The external LiDAR has no pose of its own, so we can't place its frames in world
space directly. The approach:

  1. Run real COLMAP SfM on the images -> trustworthy camera poses + sparse
     points, but in COLMAP's own ARBITRARY scale (monocular SfM has no metric
     scale).
  2. Recover metric scale by comparing COLMAP's inter-camera translations to
     ARKit's for the same frame pairs (ARKit is not trusted for absolute pose,
     but its *relative* motion over a short capture is usually metrically sane)
     and rescaling the whole COLMAP reconstruction by the median ratio.
  3. Parse each LiDAR frame's local (tof, theta, phi) into local Cartesian points
     (standard physics colatitude/azimuth convention - validated by inspection),
     interpolate the (now-metric) COLMAP camera pose at that frame's timestamp
     (SLERP rotation + lerp translation), and place the frame naively as if the
     LiDAR were co-located with the camera.
  4. That naive placement ignores the real (unknown, fixed) rigid offset between
     the external LiDAR and the phone camera, which the pilot user warned about
     ("axes and scale may not match"). Correct the resulting systematic bias with
     a single global rotation+translation ICP registration of the aggregated
     LiDAR cloud onto COLMAP's own sparse points (scale is already fixed in step
     2, so ICP here only solves rotation+translation - a 6-DoF hand-eye offset,
     not the full time-varying hand-eye problem, but a reasonable first pass).
  5. Write the corrected LiDAR cloud as points3d.ply, and a transforms.json using
     the metric COLMAP poses (NeRF axis convention, matching the rest of the
     gs_server pipeline) - both usable directly by pointcloud_only/
     hybrid_known_poses training.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

COLMAP_EXE = r"D:\VSProjects\GS_Sever\colmap\bin\colmap.exe"


# ── LiDAR frame parsing ────────────────────────────────────────────────────

def parse_lidar_frame(path: Path, max_range: float = None) -> np.ndarray:
    """max_range: drop points beyond this tof (meters). The reference photo shows a
    compact, enclosed, highly reflective metal corridor with no plausible 15-20m+
    sightline - a "far cluster" of points at 15-22m (vs. a dense 1.3-9.5m cluster
    matching the room's real scale) is almost certainly a multipath/reflection
    artifact off the metal surfaces, not real geometry."""
    tof, theta, phi = [], [], []
    with open(path) as f:
        lines = f.readlines()
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        parts = dict(p.strip().split(" = ") for p in line.split(", "))
        tof.append(float(parts["tof"]))
        theta.append(float(parts["theta"]))
        phi.append(float(parts["phi"]))
    tof = np.array(tof); theta = np.array(theta); phi = np.array(phi)
    if max_range is not None:
        keep = tof <= max_range
        tof, theta, phi = tof[keep], theta[keep], phi[keep]
    x = tof * np.sin(theta) * np.cos(phi)
    y = tof * np.sin(theta) * np.sin(phi)
    z = tof * np.cos(theta)
    return np.stack([x, y, z], axis=1)


def index_lidar_frames(dataset_dir: Path):
    lidar_dir = dataset_dir / "lidar_frames"
    out = []
    for f in sorted(lidar_dir.glob("*.txt")):
        ts = int(f.stem.split("_")[1])
        out.append((ts, f))
    return out


# ── transforms.json (ARKit, seed only) ─────────────────────────────────────

def load_arkit_transforms(path: Path):
    """transform_matrix is stored column-major (ARKit's native simd_float4x4.columns
    layout - each inner list is a COLUMN, e.g. columns[3] = [tx, ty, tz, 1]), not the
    row-major layout the rest of this codebase's transforms.json use. Transposing here
    once means every downstream consumer (interpolation, NeRF flip, ICP) can keep
    assuming standard row-major c2w."""
    d = json.load(open(path))
    frames = []
    for fr in d["frames"]:
        name = Path(fr["file_path"]).name
        m = np.array(fr["transform_matrix"], dtype=np.float64).T
        frames.append({"name": name, "ts": fr["unix_timestamp_us"], "c2w_nerf": m})
    frames.sort(key=lambda f: f["ts"])
    return d, frames


def undo_nerf_flip(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    out[:3, 1] *= -1.0
    out[:3, 2] *= -1.0
    return out


def apply_nerf_flip(m: np.ndarray) -> np.ndarray:
    return undo_nerf_flip(m)  # self-inverse


# ── Pose interpolation ──────────────────────────────────────────────────────

def interpolate_pose(frames_sorted, ts):
    """frames_sorted: list of dict(ts=..., c2w=4x4 opencv), sorted by ts."""
    ts_arr = np.array([f["ts"] for f in frames_sorted])
    idx = np.searchsorted(ts_arr, ts)
    if idx <= 0:
        return frames_sorted[0]["c2w"], False
    if idx >= len(frames_sorted):
        return frames_sorted[-1]["c2w"], False
    t0, m0 = frames_sorted[idx - 1]["ts"], frames_sorted[idx - 1]["c2w"]
    t1, m1 = frames_sorted[idx]["ts"], frames_sorted[idx]["c2w"]
    if t1 == t0:
        return m0, True
    alpha = (ts - t0) / (t1 - t0)
    R0 = Rotation.from_matrix(m0[:3, :3])
    R1 = Rotation.from_matrix(m1[:3, :3])
    slerp = Slerp([0, 1], Rotation.concatenate([R0, R1]))
    R_interp = slerp([alpha])[0]
    trans = (1 - alpha) * m0[:3, 3] + alpha * m1[:3, 3]
    m = np.eye(4)
    m[:3, :3] = R_interp.as_matrix()
    m[:3, 3] = trans
    return m, True


# ── COLMAP SfM (standard, unknown poses) ───────────────────────────────────

def run_colmap_sfm(images_dir: Path, work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    txt_dir = work_dir / "sparse_txt"

    if (txt_dir / "images.txt").exists() and (txt_dir / "points3D.txt").exists():
        print("  (reusing cached COLMAP result)")
        return txt_dir

    def run(args, **kw):
        print("[COLMAP]", " ".join(args))
        subprocess.run([COLMAP_EXE] + args, check=True, **kw)

    run(["feature_extractor",
         "--database_path", str(db_path),
         "--image_path", str(images_dir),
         "--ImageReader.camera_model", "OPENCV",
         "--ImageReader.single_camera", "1",
         "--FeatureExtraction.use_gpu", "1"])
    run(["exhaustive_matcher",
         "--database_path", str(db_path),
         "--FeatureMatching.use_gpu", "1"])
    run(["mapper",
         "--database_path", str(db_path),
         "--image_path", str(images_dir),
         "--output_path", str(sparse_dir)])

    # mapper may produce multiple models (0, 1, ...) if it fragments; use the largest
    models = sorted(sparse_dir.glob("*"), key=lambda p: p.name)
    models = [m for m in models if m.is_dir()]
    if not models:
        raise RuntimeError("COLMAP mapper produced no model")
    best = max(models, key=lambda m: (m / "points3D.bin").stat().st_size)

    txt_dir.mkdir(exist_ok=True)
    run(["model_converter", "--input_path", str(best), "--output_path", str(txt_dir), "--output_type", "TXT"])
    return txt_dir


def parse_colmap_images_txt(path: Path):
    """Returns dict: name -> 4x4 c2w (OpenCV convention). Each image is two lines
    (pose, then a POINTS2D line that may legitimately be blank) - only drop '#'
    comment lines, so blank POINTS2D lines don't shift the pairing."""
    out = {}
    lines = [l.rstrip("\n") for l in open(path) if not l.startswith("#")]
    while lines and lines[-1].strip() == "":
        lines.pop()  # trailing blank lines from EOF only
    for i in range(0, len(lines), 2):  # every image is 2 lines (pose line + points2D line)
        parts = lines[i].split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        t_w2c = np.array([tx, ty, tz])
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        c2w = np.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = t_c2w
        out[name] = c2w
    return out


def parse_colmap_points3d_txt(path: Path):
    xyz, rgb = [], []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
        rgb.append([int(p[4]), int(p[5]), int(p[6])])
    return np.array(xyz, dtype=np.float64), np.array(rgb, dtype=np.uint8)


# ── Scale recovery (COLMAP arbitrary units -> metric) ──────────────────────

def recover_scale(colmap_poses: dict, arkit_frames: list) -> float:
    """Compare inter-camera translation magnitudes between COLMAP and ARKit for
    the same frame pairs; ARKit's absolute pose is untrusted but its short-range
    relative motion is a reasonable metric-scale reference."""
    arkit_by_name = {f["name"]: undo_nerf_flip(f["c2w_nerf"]) for f in arkit_frames}
    common = [n for n in colmap_poses if n in arkit_by_name]
    common.sort()
    ratios = []
    for i in range(len(common) - 1):
        n0, n1 = common[i], common[i + 1]
        d_colmap = np.linalg.norm(colmap_poses[n1][:3, 3] - colmap_poses[n0][:3, 3])
        d_arkit = np.linalg.norm(arkit_by_name[n1][:3, 3] - arkit_by_name[n0][:3, 3])
        if d_colmap > 1e-6:
            ratios.append(d_arkit / d_colmap)
    ratios = np.array(ratios)
    print(f"scale ratios: n={len(ratios)} median={np.median(ratios):.4f} "
          f"mean={ratios.mean():.4f} std={ratios.std():.4f}")
    return float(np.median(ratios))


# ── Rigid ICP (point-to-point, Kabsch, no scale) ───────────────────────────

def voxel_downsample(xyz, voxel_size):
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    return xyz[unique_idx]


def icp_rigid(source_xyz, target_xyz, max_iters=150, tol=1e-6, keep_fraction=0.5):
    """Aligns source onto target. Returns (R, t) such that target ≈ R @ source + t,
    and the aligned source cloud. keep_fraction: only the best-matching fraction of
    correspondences drive each fit (robust to a sparse/non-corresponding target)."""
    target_tree = cKDTree(target_xyz)
    R = np.eye(3)
    t = np.zeros(3)
    src = source_xyz.copy()
    prev_err = None
    for it in range(max_iters):
        dists, idx = target_tree.query(src, k=1, workers=-1)
        thresh = np.percentile(dists, keep_fraction * 100)
        mask = dists <= thresh
        src_m, tgt_m = src[mask], target_xyz[idx[mask]]

        src_centroid = src_m.mean(axis=0)
        tgt_centroid = tgt_m.mean(axis=0)
        src_c = src_m - src_centroid
        tgt_c = tgt_m - tgt_centroid
        H = src_c.T @ tgt_c
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1, 1, d])
        R_step = Vt.T @ D @ U.T
        t_step = tgt_centroid - R_step @ src_centroid

        src = (R_step @ src.T).T + t_step
        R = R_step @ R
        t = R_step @ t + t_step

        err = np.mean(dists[mask])
        print(f"  ICP iter {it}: mean_dist={err:.4f} (kept {mask.sum()}/{len(mask)})")
        if prev_err is not None and abs(prev_err - err) < tol:
            break
        prev_err = err
    return R, t, src


def calibrate_camera_to_lidar_extrinsic(
    frame_points, frame_poses, target_xyz, max_iters=500, tol=1e-7, keep_fraction=0.5,
):
    """Solves for the single fixed camera->LiDAR extrinsic (R_ext, t_ext) such that
    world_pt = R_cam(t) @ (R_ext @ local_pt + t_ext) + T_cam(t)
    best explains target_xyz, given each frame's own (already-known, time-varying)
    camera pose. This is a proper hand-eye-style calibration - NOT the same as
    icp_rigid() applied to the naively-aggregated cloud afterwards: that shortcut
    implicitly assumes R_ext=I while aggregating, then tries to fix the whole cloud
    with one constant post-hoc correction. Since the camera rotates over the
    capture, that missing/wrong R_ext produces a world-frame error that itself
    rotates over time - no single after-the-fact correction can undo it (confirmed
    empirically: residual error to the COLMAP reference varied 5x+ depending on
    camera orientation at capture time, from ~0.4 to ~2.9 scene units, when using
    the naive+post-hoc-ICP approach).

    Instead, each iteration re-expresses the current correspondences in the
    CAMERA's own local frame at each point's capture time (a fixed frame, not a
    rotating one) via q' = R_cam(t)^-1 @ (q - T_cam(t)), turning the problem into a
    standard fixed-frame rigid alignment (local_pt -> q') solvable by Kabsch/SVD
    per iteration, alternating with nearest-neighbor correspondence search - i.e.
    ICP with the extrinsic correctly placed in the transform chain instead of
    tacked on afterwards.

    frame_points: list of (camera_pose_4x4, local_xyz) per LiDAR frame.
    """
    target_tree = cKDTree(target_xyz)
    R_ext = np.eye(3)
    t_ext = np.zeros(3)
    prev_err = None
    for it in range(max_iters):
        all_src = []
        all_tgt = []
        all_dists = []
        for pose, local_xyz in frame_points:
            R_cam = pose[:3, :3]
            T_cam = pose[:3, 3]
            world = (R_cam @ (R_ext @ local_xyz.T + t_ext[:, None])).T + T_cam
            dists, idx = target_tree.query(world, k=1, workers=-1)
            q = target_xyz[idx]
            # re-express the matched target in this frame's camera-local frame -
            # a fixed-frame target that doesn't rotate with the camera
            q_local = (R_cam.T @ (q - T_cam).T).T
            all_src.append(local_xyz)
            all_tgt.append(q_local)
            all_dists.append(dists)
        src = np.concatenate(all_src, axis=0)
        tgt = np.concatenate(all_tgt, axis=0)
        dists = np.concatenate(all_dists, axis=0)

        thresh = np.percentile(dists, keep_fraction * 100)
        mask = dists <= thresh
        src_m, tgt_m = src[mask], tgt[mask]

        src_centroid = src_m.mean(axis=0)
        tgt_centroid = tgt_m.mean(axis=0)
        H = (src_m - src_centroid).T @ (tgt_m - tgt_centroid)
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1, 1, d])
        R_ext = Vt.T @ D @ U.T
        t_ext = tgt_centroid - R_ext @ src_centroid

        err = np.mean(dists[mask])
        print(f"  hand-eye iter {it}: mean_dist={err:.4f} (kept {mask.sum()}/{len(mask)}) "
              f"rot={np.degrees(Rotation.from_matrix(R_ext).magnitude()):.1f}deg")
        if prev_err is not None and abs(prev_err - err) < tol:
            break
        prev_err = err
    return R_ext, t_ext


# ── PLY I/O ──────────────────────────────────────────────────────────────

def write_ply_xyzrgb(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    n = len(xyz)
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data = np.zeros(n, dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        data.tofile(f)


# ── Main pipeline ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--work_dir", type=Path, default=None)
    args = ap.parse_args()

    dataset_dir: Path = args.dataset_dir
    work_dir = args.work_dir or (dataset_dir / "_lidar_calib")
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. Loading ARKit transforms.json (seed only) ===")
    arkit_json, arkit_frames = load_arkit_transforms(dataset_dir / "transforms.json")
    print(f"  {len(arkit_frames)} camera frames")

    print("=== 2. Running COLMAP SfM on images ===")
    txt_dir = run_colmap_sfm(dataset_dir / "images", work_dir / "colmap")
    colmap_poses = parse_colmap_images_txt(txt_dir / "images.txt")
    colmap_xyz, colmap_rgb = parse_colmap_points3d_txt(txt_dir / "points3D.txt")
    print(f"  COLMAP registered {len(colmap_poses)}/{len(arkit_frames)} images, "
          f"{len(colmap_xyz)} sparse points")

    print("=== 3. Recovering metric scale (COLMAP vs ARKit relative motion) ===")
    scale = recover_scale(colmap_poses, arkit_frames)
    print(f"  scale factor = {scale:.4f}")

    colmap_poses_scaled = {n: m.copy() for n, m in colmap_poses.items()}
    for m in colmap_poses_scaled.values():
        m[:3, 3] *= scale
    colmap_xyz_scaled = colmap_xyz * scale

    print("=== 4. Interpolating metric camera pose at each LiDAR frame timestamp ===")
    ts_lookup = {f["name"]: f["ts"] for f in arkit_frames}
    pose_frames = sorted(
        [{"ts": ts_lookup[n], "c2w": m} for n, m in colmap_poses_scaled.items() if n in ts_lookup],
        key=lambda f: f["ts"],
    )
    print(f"  {len(pose_frames)} timestamped COLMAP poses available for interpolation")

    lidar_index = index_lidar_frames(dataset_dir)
    print(f"  {len(lidar_index)} raw LiDAR frames")

    frame_local_xyz = []
    frame_poses = []
    n_extrapolated = 0
    for ts, path in lidar_index:
        local_xyz = parse_lidar_frame(path, max_range=10.0)
        pose, interpolated = interpolate_pose(pose_frames, ts)
        if not interpolated:
            n_extrapolated += 1
        frame_local_xyz.append(local_xyz)
        frame_poses.append(pose)
    n_raw_total = sum(len(a) for a in frame_local_xyz)
    print(f"  parsed {n_raw_total:,} raw LiDAR points across {len(lidar_index)} frames "
          f"({n_extrapolated} frames used nearest-pose extrapolation)")

    print("=== 5. Hand-eye calibration: solving for the fixed camera<->LiDAR extrinsic ===")
    # A naive aggregate-then-globally-ICP-correct approach was tried first and
    # empirically shown wrong: residual error to the COLMAP reference varied ~5x+
    # (0.4 to 2.9 scene units) depending on the camera's rotation at capture time -
    # the signature of an extrinsic offset applied *after* composing with a
    # rotating camera pose instead of before. calibrate_camera_to_lidar_extrinsic
    # solves for the offset in the correct place in the transform chain.
    rng = np.random.default_rng(0)
    calib_frame_points = []
    for local_xyz, pose in zip(frame_local_xyz, frame_poses):
        n_sub = min(1500, len(local_xyz))
        sub = local_xyz[rng.choice(len(local_xyz), size=n_sub, replace=False)]
        calib_frame_points.append((pose, sub))
    R_ext, t_ext = calibrate_camera_to_lidar_extrinsic(calib_frame_points, frame_poses, colmap_xyz_scaled)
    print(f"  camera->LiDAR extrinsic: rotation={np.degrees(Rotation.from_matrix(R_ext).magnitude()):.2f} deg, "
          f"translation norm={np.linalg.norm(t_ext):.4f}")

    print("=== 6. Applying calibrated extrinsic to the full-resolution cloud ===")
    all_world_pts = []
    all_raw_world_pts = []
    for local_xyz, pose in zip(frame_local_xyz, frame_poses):
        R_cam = pose[:3, :3]
        T_cam = pose[:3, 3]
        world = (R_cam @ (R_ext @ local_xyz.T + t_ext[:, None])).T + T_cam
        all_world_pts.append(world)
        all_raw_world_pts.append((R_cam @ local_xyz.T).T + T_cam)  # naive (R_ext=I) for comparison
    corrected_lidar_world = np.concatenate(all_world_pts, axis=0)
    raw_lidar_world = np.concatenate(all_raw_world_pts, axis=0)

    print("=== 7. Writing outputs ===")
    out_dir = dataset_dir / "lidar_calibrated"
    out_dir.mkdir(exist_ok=True)

    rgb = np.full((len(corrected_lidar_world), 3), 180, dtype=np.uint8)
    write_ply_xyzrgb(out_dir / "points3d.ply", corrected_lidar_world, rgb)
    write_ply_xyzrgb(out_dir / "points3d_raw_uncorrected.ply", raw_lidar_world, rgb)
    write_ply_xyzrgb(out_dir / "colmap_sparse_reference.ply", colmap_xyz_scaled, colmap_rgb)

    # final transforms.json using metric COLMAP poses, NeRF convention.
    # known_pose_triangulation.py (the hybrid_known_poses pipeline) reads intrinsics
    # per-frame (fr["fl_x"] etc.), not from the top-level json - this source
    # dataset's original transforms.json only had them at top level, so copy them
    # into every frame here.
    global_intrinsics = {k: arkit_json[k] for k in ("fl_x", "fl_y", "cx", "cy", "w", "h") if k in arkit_json}
    out_frames = []
    for fr in arkit_frames:
        name = fr["name"]
        if name not in colmap_poses_scaled:
            continue
        c2w_opencv = colmap_poses_scaled[name]
        c2w_nerf = apply_nerf_flip(c2w_opencv)
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
    print(f"  wrote {out_dir}/transforms.json ({len(out_frames)} frames, metric COLMAP poses)")
    print(f"  wrote {out_dir}/colmap_sparse_reference.ply (for visual QC)")
    print(f"  wrote {out_dir}/points3d_raw_uncorrected.ply (pre-ICP, for comparison)")


if __name__ == "__main__":
    main()
