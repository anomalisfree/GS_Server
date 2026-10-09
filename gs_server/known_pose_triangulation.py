"""
Known-pose COLMAP triangulation — hybrid_known_poses training mode.

Motivation: for scenes with low SIFT-matchable texture (e.g. reflective/repetitive
industrial surfaces), COLMAP's incremental mapper fails to *estimate* camera poses
from images alone — it can't find enough connected matches to chain registration,
and the reconstruction fragments into many disconnected sub-models (see
Batch12_Scans_Training_Report_2026-08-13.md for the diagnosis).

But we don't need COLMAP to estimate poses at all: we already have accurate camera
poses from the scanner's own LiDAR/IMU tracking (the same poses used by
pointcloud_only mode's transforms.json). This module feeds those *known* poses to
COLMAP directly and only asks it to triangulate 3D points from feature matches
(`colmap point_triangulator`), which works even with sparse/weak pairwise matches —
unlike incremental mapping, triangulation doesn't need a fully connected match graph,
just 2+ observations per point. The result is a photogrammetric point cloud in the
*same* coordinate frame as the input poses (no COLMAP-frame alignment needed, since
poses were never re-estimated), which we merge with the original LiDAR point cloud
for a denser, more complete initialization than either source alone.

Pipeline:
  1. Recover original (OpenCV/COLMAP-convention) camera poses from transforms.json
     (undoing the NeRF-convention axis flip applied when it was created).
  2. colmap database_creator + feature_extractor + matcher (exhaustive by default —
     unlike incremental mapping we don't need connectivity/chaining, just as many
     pairwise matches as possible for triangulation coverage).
  3. Write a "known pose" sparse model (cameras.txt/images.txt/points3D.txt) and run
     `colmap point_triangulator` (re-associates by image filename, not by ID).
  4. Optionally refine with `colmap bundle_adjuster`.
  5. Export triangulated points, merge with the original points3d.ply.
  6. Point the job at a new prepared dataset (same images + transforms.json + merged
     cloud) so brush_runner.py's existing pointcloud_only-style path trains on it
     unmodified.
"""

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial import cKDTree

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False

from .models import JobStatus, ColmapProgress, ColmapStage
from .config import get_config

if TYPE_CHECKING:
    from .job_manager import JobManager, Job

logger = logging.getLogger(__name__)


# ── Pose conversion ──────────────────────────────────────────────────────────

def _undo_nerf_flip(m: np.ndarray) -> np.ndarray:
    """NeRF-convention c2w -> OpenCV/COLMAP-convention c2w. Self-inverse (same op
    used to create the NeRF matrix in the first place — flipping Y,Z columns twice
    is the identity), see leica_to_nerfstudio.py:opencv_c2w_to_nerf."""
    out = m.copy()
    out[:3, 1] *= -1.0
    out[:3, 2] *= -1.0
    return out


def _c2w_to_w2c(c2w: np.ndarray):
    R_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    return R_w2c, t_w2c


def _rotmat_to_quat(R: np.ndarray):
    """3x3 rotation matrix -> (qw, qx, qy, qz), COLMAP's quaternion convention."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return float(qw), float(qx), float(qy), float(qz)


# ── PLY helpers (mirrors downsample_ply_memmap.py's header parsing) ──────────

_PLY_TYPE_MAP = {
    'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1',
    'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
    'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
}


def _read_ply_xyzrgb(path: Path):
    with path.open('rb') as f:
        header = bytearray()
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f'Unexpected EOF reading PLY header: {path}')
            header.extend(line)
            if line.strip() == b'end_header':
                break
    text = header.decode('ascii', errors='strict')
    fields, nverts, in_vertex = [], None, False
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'element':
            in_vertex = parts[1] == 'vertex'
            if in_vertex:
                nverts = int(parts[2])
        elif parts[0] == 'property' and in_vertex:
            fields.append((parts[2], _PLY_TYPE_MAP[parts[1]]))
    dtype = np.dtype([(n, f'<{t}') for n, t in fields])
    data = np.memmap(path, dtype=dtype, mode='r', offset=len(header), shape=(nverts,))
    xyz = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)
    rgb = np.stack([data['red'], data['green'], data['blue']], axis=1).astype(np.uint8)
    return xyz, rgb


def _compute_density_clamped_log_scales(xyz: np.ndarray, clamp_percentile: float = 90.0) -> np.ndarray:
    """Per-point initial gaussian scale from local point density, for writing
    directly into points3d.ply's scale_0/1/2 fields.

    Same distance formula Brush's own init uses - average of the 2 nearest-
    neighbor distances, halved (see brush/crates/brush-train/src/
    splat_init.rs::compute_knn_scales) - but with a MUCH tighter clamp
    ceiling. Brush's own clamp (10% of the scene's median bounding-box size)
    is generous enough that a genuinely isolated point can still start out
    tens of times larger than a typical nearby point: measured on a real
    hybrid job's merged cloud, Brush's ceiling was 0.68 against a median
    point spacing of 0.014 (a ~48x gap) - a big head start toward becoming a
    visible "needle" once training stretches it further along one axis.
    Capping at a percentile of the cloud's OWN nearest-neighbor-distance
    distribution instead keeps the ceiling self-calibrating (same rationale
    as floater_filter.py's median-relative size cutoff) while staying close
    to the bulk of the data - the same measurement gave a ceiling of 0.030
    (~2x the median) with this approach, a 22x tighter cap.

    If points3d.ply has scale_0/1/2 present, Brush uses them directly and
    skips its own KNN computation entirely (brush-serde/src/import.rs -
    `log_scales: vertex.has_property("scale_0").then(...)`).

    Returns (N,) natural-log scale values (Brush's own pre-activation
    convention - actual linear scale is exp(this)); isotropic init (same
    value goes into scale_0/1/2), matching Brush's own default.
    """
    n = len(xyz)
    if n < 3:
        return np.zeros(n, dtype=np.float32)
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=3, workers=-1)  # self, 1st, 2nd nearest
    dist = (d[:, 1] + d[:, 2]) / 4.0
    cap = float(np.percentile(dist, clamp_percentile))
    clamped = np.clip(dist, 1e-4, max(cap, 1e-4))
    return np.log(clamped).astype(np.float32)


def _generate_overexposure_masks(images_dir: Path, masks_dir: Path, threshold: int = 250) -> dict:
    """Generate a per-image training-loss mask that excludes fully overexposed
    (blown-highlight) pixels - diagnosed as the root cause of a persistent
    "sparkle" artifact that survived both geometric floater filtering and
    anisotropy regularization (see git history / CLAUDE.md). In these dark
    industrial-interior scans, bright light sources/sky visible through gaps
    blow out to pure white in the source photos (min(R,G,B) > threshold, no
    gradient at all) - that carries zero recoverable color or geometry
    information, so the 3DGS optimizer fills the gap with a cluster of
    poorly-conditioned, brightly-colored splats that fragment into a visible
    "sparkle" when viewed from other angles.

    Brush's nerfstudio-format loader already looks for a same-named file
    under a sibling `masks/` directory and treats it as a hard per-pixel loss
    mask (0 = excluded, 255 = kept) - see brush-dataset/src/formats/mod.rs
    find_mask_path and brush-train/src/train.rs's alpha_is_mask handling. No
    Brush/Rust changes needed, just generating these mask files here.

    Uses a pure brightness threshold, not semantic segmentation (unlike
    mask_generator.py's DeepLabV3 masks, which are COLMAP-only and don't
    reach Brush's training loss at all) - this is deliberately dumb and
    cheap, since "fully blown to white" is a well-defined, unambiguous
    signal regardless of scene content.
    """
    from PIL import Image

    masks_dir.mkdir(parents=True, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions)

    blown_fracs = []
    for img_path in image_files:
        img = np.array(Image.open(img_path).convert("RGB"))
        blown = img.min(axis=2) > threshold
        mask = np.where(blown, 0, 255).astype(np.uint8)
        Image.fromarray(mask, mode="L").save(masks_dir / f"{img_path.stem}.png")
        blown_fracs.append(float(blown.mean()))

    return {
        "n_images": len(image_files),
        "mean_blown_pct": round(100 * float(np.mean(blown_fracs)), 3) if blown_fracs else 0.0,
        "max_blown_pct": round(100 * float(np.max(blown_fracs)), 3) if blown_fracs else 0.0,
    }


def _generate_person_masks(images_dir: Path, masks_dir: Path, dilate_px: int = 25,
                           max_side: int = 1024, batch_size: int = 4) -> dict:
    """Mask out people (DeepLabV3, VOC class 15) from Brush's training loss.

    Moving people appear in different places in different photos, so they can't be a consistent
    3D surface; the optimizer explains them with semi-transparent haze and smeared ghost splats
    (visible in AU_Las_Vegas renders vs. real photos). Same Brush convention as
    _generate_overexposure_masks: `masks/<stem>.png`, 0 = excluded from the loss, 255 = kept,
    full image resolution. mask_generator.py's DeepLabV3 masks are COLMAP-only and never reach
    Brush's loss, hence this separate path.

    Any mask already at masks_dir/<stem>.png (e.g. the overexposure mask) is AND-ed with the
    person mask rather than overwritten, so run this AFTER _generate_overexposure_masks.
    Segmentation runs at `max_side` (more reliable for small/distant people than torchvision's
    default 520) and the person mask is upsampled to full resolution and dilated by `dilate_px`
    to cover motion blur and edge pixels. Blocking - call via run_in_executor.
    """
    import cv2
    import torch
    from PIL import Image
    from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

    masks_dir.mkdir(parents=True, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    kernel = None
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))

    person_fracs = []
    n_with_person = 0
    try:
        for start in range(0, len(image_files), batch_size):
            batch = image_files[start:start + batch_size]
            full_sizes, tensors = [], []
            for p in batch:
                im = Image.open(p).convert("RGB")
                full_sizes.append(im.size)  # (w, h)
                scale = max_side / max(im.size)
                small = im.resize((max(1, round(im.size[0] * scale)), max(1, round(im.size[1] * scale))),
                                  Image.BILINEAR)
                tensors.append(torch.from_numpy(np.asarray(small)).permute(2, 0, 1).float() / 255.0)
            # Batches only stack when all resized images share a shape; otherwise run one by one.
            groups = [list(range(len(batch)))] if len({t.shape for t in tensors}) == 1 else [[i] for i in range(len(batch))]
            for g in groups:
                x = torch.stack([tensors[i] for i in g]).to(device)
                x = (x - mean) / std
                with torch.no_grad():
                    pred = model(x)["out"].argmax(1).cpu().numpy()
                for k, i in enumerate(g):
                    w, h = full_sizes[i]
                    person = cv2.resize((pred[k] == 15).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    if kernel is not None:
                        person = cv2.dilate(person, kernel)
                    mask_path = masks_dir / f"{batch[i].stem}.png"
                    keep = np.full((h, w), 255, dtype=np.uint8)
                    if mask_path.exists():
                        existing = np.array(Image.open(mask_path).convert("L"))
                        if existing.shape == keep.shape:
                            keep = existing
                    keep[person > 0] = 0
                    Image.fromarray(keep, mode="L").save(mask_path)
                    frac = float((person > 0).mean())
                    person_fracs.append(frac)
                    n_with_person += int(frac > 0)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "n_images": len(image_files),
        "n_with_person": n_with_person,
        "mean_person_pct": round(100 * float(np.mean(person_fracs)), 3) if person_fracs else 0.0,
        "max_person_pct": round(100 * float(np.max(person_fracs)), 3) if person_fracs else 0.0,
    }


def _write_ply_xyzrgb(path: Path, xyz: np.ndarray, rgb: np.ndarray, log_scales: Optional[np.ndarray] = None):
    n = len(xyz)
    scale_props = (
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        if log_scales is not None else ""
    )
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"{scale_props}"
        "end_header\n"
    )
    dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if log_scales is not None:
        dtype += [('scale_0', '<f4'), ('scale_1', '<f4'), ('scale_2', '<f4')]
    out = np.empty(n, dtype=dtype)
    out['x'], out['y'], out['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out['red'], out['green'], out['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if log_scales is not None:
        out['scale_0'] = out['scale_1'] = out['scale_2'] = log_scales
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        f.write(header.encode('ascii'))
        f.write(out.tobytes())


def _parse_points3d_txt(path: Path):
    """Returns (xyz, rgb, track_length). track_length is the number of images whose
    feature matches support each point — COLMAP's points3D.txt line format is
    POINT3D_ID X Y Z R G B ERROR TRACK[](IMAGE_ID, POINT2D_IDX). A track length of 2
    (the minimum possible) means the point was triangulated from just two rays; at a
    near-degenerate/grazing baseline (common right at silhouette edges) that's barely
    constrained even when COLMAP's own reprojection error looks small — see the
    "confidently wrong" comment below on why the error field alone doesn't catch this.
    """
    xyz, rgb, track_len = [], [], []
    if not path.exists():
        return (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8),
                np.zeros((0,), dtype=np.int32))
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
            rgb.append([int(parts[4]), int(parts[5]), int(parts[6])])
            track_len.append((len(parts) - 8) // 2)
    if not xyz:
        return (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8),
                np.zeros((0,), dtype=np.int32))
    return np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.uint8), np.array(track_len, dtype=np.int32)


# ── LiDAR cloud cleanup ────────────────────────────────────────────────────

def _denoise_and_prioritize_colmap(
    lidar_xyz: np.ndarray, lidar_rgb: np.ndarray,
    colmap_xyz: np.ndarray,
    sor_k: int = 12, sor_std_ratio: float = 2.5,
    colmap_priority_radius: float = None,
    colmap_priority_radius_multiplier: float = 1.5,
):
    """Clean up the raw LiDAR cloud before merging with COLMAP points:

    1. Statistical outlier removal — for each LiDAR point, compare its mean
       distance to its `sor_k` nearest neighbors against the cloud-wide
       mean/std of that quantity; discard points sitting in abnormally sparse
       regions (typically scanner noise / multipath reflections off the same
       reflective metal surfaces that make these scenes hard for SIFT).
    2. Local COLMAP priority — discard any LiDAR point within
       `colmap_priority_radius` of a COLMAP-triangulated point, since COLMAP
       points are photogrammetrically verified (fixed known poses + real
       feature observations) and more locally precise where available. LiDAR
       remains the sole source everywhere COLMAP has no coverage.

       If `colmap_priority_radius` is not given explicitly, it's derived from
       the LiDAR cloud's own point spacing (mean nearest-neighbor distance x
       `colmap_priority_radius_multiplier`) instead of a fixed absolute value.
       A fixed radius doesn't adapt across datasets of very different scan
       density: at 0.05 units against a cloud with ~0.01-unit point spacing,
       each COLMAP point wiped out a ~25-point neighborhood instead of just
       ceding its own immediate spot, cutting the LiDAR cloud by ~38% in
       testing — a much bigger effect than intended.

    Returns (kept_xyz, kept_rgb, stats_dict).
    """
    n_before = len(lidar_xyz)
    if n_before == 0:
        return lidar_xyz, lidar_rgb, {"n_before": 0, "n_after_sor": 0, "n_after_priority": 0,
                                       "colmap_priority_radius": None}

    tree = cKDTree(lidar_xyz)
    dists, _ = tree.query(lidar_xyz, k=sor_k + 1, workers=-1)
    mean_nn_dist = dists[:, 1:].mean(axis=1)  # exclude the point itself (dist 0)
    threshold = mean_nn_dist.mean() + sor_std_ratio * mean_nn_dist.std()
    keep = mean_nn_dist <= threshold

    xyz, rgb = lidar_xyz[keep], lidar_rgb[keep]
    n_after_sor = len(xyz)

    if colmap_priority_radius is None:
        colmap_priority_radius = colmap_priority_radius_multiplier * float(mean_nn_dist.mean())

    if len(colmap_xyz):
        colmap_tree = cKDTree(colmap_xyz)
        dist_to_colmap, _ = colmap_tree.query(xyz, k=1, workers=-1)
        keep2 = dist_to_colmap > colmap_priority_radius
        xyz, rgb = xyz[keep2], rgb[keep2]
    n_after_priority = len(xyz)

    return xyz, rgb, {
        "n_before": n_before, "n_after_sor": n_after_sor, "n_after_priority": n_after_priority,
        "colmap_priority_radius": colmap_priority_radius,
    }


# ── Selective bundle adjustment ───────────────────────────────────────────────

def _selective_bundle_adjust(
    triangulated_dir: Path, refined_dir: Path,
    connected_frames: list, safe_names: dict,
    min_observation_percentile: float = 50.0,
):
    """Refine the known-pose COLMAP model with a SELECTIVE bundle adjustment: camera
    poses are only allowed to move for images with a triangulated-observation count
    at or above `min_observation_percentile` of this job's own distribution — enough
    independent, verified 2D-3D correspondences to constrain a pose reliably. Every
    other camera's pose is locked exactly as given (matching the plain points-only
    bundle_adjuster call below), so this can't reproduce the "ghost duplicate" cloud
    that came from letting *all* cameras move (see the comment on that call — weakly-
    matched cameras in these low-texture scenes are barely constrained and drift
    almost freely under full refine_extrinsics=1). 3D points are always refined,
    same as before.

    Points that end up badly placed (e.g. a locked-only point resting on a near-
    degenerate pair of viewing rays) are not specially handled here - the existing
    bbox/track-length sanity filters applied to the triangulation output downstream
    already exist for exactly this failure mode and catch it the same way regardless
    of whether the point came from this path or the plain bundle_adjuster.

    Returns (used: bool, refined_poses: dict[file_path -> 4x4 nerf c2w list]).
    refined_poses is empty (and used=False) if pycolmap is unavailable, no image is
    registered, or nothing clears the observation bar.
    """
    if not HAS_PYCOLMAP:
        return False, {}

    recon = pycolmap.Reconstruction(str(triangulated_dir))
    reg_ids = recon.reg_image_ids()
    if not reg_ids:
        return False, {}

    counts = {iid: recon.image(iid).num_points3D for iid in reg_ids}
    vals = np.array(list(counts.values()))
    threshold = float(np.percentile(vals, min_observation_percentile))
    well_ids = {iid for iid in reg_ids if counts[iid] >= threshold and counts[iid] > 0}
    weak_ids = set(reg_ids) - well_ids
    if not well_ids:
        return False, {}
    if not weak_ids:
        # Keep at least one image locked as a gauge anchor so the whole
        # reconstruction can't translate/rotate freely with nothing to pin it.
        anchor = min(reg_ids, key=lambda i: counts[i])
        weak_ids = {anchor}
        well_ids.discard(anchor)

    config = pycolmap.BundleAdjustmentConfig()
    for iid in reg_ids:
        config.add_image(iid)
    for iid in weak_ids:
        config.set_constant_rig_from_world_pose(recon.image(iid).frame_id)
    for cam_id in recon.cameras:
        config.set_constant_cam_intrinsics(cam_id)

    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_sensor_from_rig = False
    options.print_summary = False

    adjuster = pycolmap.create_default_bundle_adjuster(options, config, recon)
    adjuster.solve()
    recon.write_binary(str(refined_dir))

    name_to_frame = {safe_names[fr["file_path"]]: fr for fr in connected_frames}
    refined_poses = {}
    for iid in well_ids:
        image = recon.image(iid)
        frame = name_to_frame.get(image.name)
        if frame is None:
            continue
        w2c = image.cam_from_world()
        R_c2w = w2c.rotation.matrix().T
        t_c2w = -R_c2w @ w2c.translation
        c2w = np.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = t_c2w
        refined_poses[frame["file_path"]] = _undo_nerf_flip(c2w).tolist()  # self-inverse

    logger.info(
        f"[KNOWN-POSE-COLMAP] Selective bundle adjustment: {len(well_ids)}/{len(reg_ids)} "
        f"camera poses refined (>= {threshold:.0f} observations), {len(weak_ids)} locked"
    )
    return True, refined_poses


# ── COLMAP known-pose model writer ───────────────────────────────────────────

def _make_safe_image_dir(frames: list, images_dir: Path, safe_dir: Path) -> dict:
    """COLMAP's own images.txt reader splits on whitespace for *all* fields,
    including NAME — a filename containing a space (e.g. "core propellers
    2_SensorFrame_00000.jpg", common here since several source folders have
    spaces in their names) gets silently truncated to its first word, and
    point_triangulator then fails with "Image with name ... does not exist in
    database". Sidestep this entirely by symlinking every image under a safe,
    space-free name and using that directory (not the original) for every
    COLMAP call in this pipeline. Returns {frame["file_path"]: safe_name} - keyed
    by the frame's own file_path (not position) so callers can freely filter/
    reorder the frame list afterwards without invalidating the mapping.
    """
    safe_dir.mkdir(parents=True, exist_ok=True)
    safe_names = {}
    for i, fr in enumerate(frames, start=1):
        src = images_dir / Path(fr["file_path"]).name
        ext = src.suffix
        safe_name = f"img_{i:05d}{ext}"
        dst = safe_dir / safe_name
        if not dst.exists():
            dst.symlink_to(src.resolve())
        safe_names[fr["file_path"]] = safe_name
    return safe_names


def _find_disconnected_image_names(database_path: Path) -> set:
    """Returns the safe-image filenames with zero verified matches to any other
    image. point_triangulator crashes (std::out_of_range in ObservationManager,
    observed on several low-texture datasets) when the known-pose model includes
    such a fully isolated image - COLMAP's own database loader logs it plainly
    ("Loading images... N in Ts (connected M)" with M < N) but point_triangulator
    doesn't handle it gracefully. Excluding these from the known-pose model avoids
    the crash; the image still trains normally later since Brush uses the dataset's
    own transforms.json/images, not this COLMAP-internal bootstrap model. Resolved
    by name (not raw image_id) to not depend on COLMAP's internal id assignment
    matching our own 1-based frame ordering."""
    import sqlite3
    conn = sqlite3.connect(str(database_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT image_id, name FROM images")
        id_to_name = dict(cur.fetchall())
        cur.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 0")
        connected_ids = set()
        for (pair_id,) in cur.fetchall():
            image_id2 = pair_id % 2147483647
            image_id1 = (pair_id - image_id2) // 2147483647
            connected_ids.add(image_id1)
            connected_ids.add(image_id2)
        disconnected_ids = set(id_to_name) - connected_ids
        return {id_to_name[i] for i in disconnected_ids}
    finally:
        conn.close()


def _write_known_pose_model(frames: list, safe_names: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_lines = ["# Camera list with one line of data per camera:\n",
                 "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n",
                 f"# Number of cameras: {len(frames)}\n"]
    img_lines = ["# Image list with two lines of data per image:\n",
                 "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
                 "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
                 f"# Number of images: {len(frames)}, mean observations per image: 0\n"]

    for i, fr in enumerate(frames, start=1):
        c2w_nerf = np.array(fr["transform_matrix"], dtype=np.float64)
        c2w = _undo_nerf_flip(c2w_nerf)
        R_w2c, t_w2c = _c2w_to_w2c(c2w)
        qw, qx, qy, qz = _rotmat_to_quat(R_w2c)

        fx, fy = fr["fl_x"], fr["fl_y"]
        cx, cy = fr["cx"], fr["cy"]
        w, h = int(fr["w"]), int(fr["h"])
        k1 = fr.get("k1", 0.0); k2 = fr.get("k2", 0.0)
        p1 = fr.get("p1", 0.0); p2 = fr.get("p2", 0.0)
        cam_lines.append(f"{i} OPENCV {w} {h} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2}\n")

        name = safe_names[fr["file_path"]]
        tx, ty, tz = t_w2c
        img_lines.append(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {name}\n\n")

    (out_dir / "cameras.txt").write_text("".join(cam_lines), encoding="ascii")
    (out_dir / "images.txt").write_text("".join(img_lines), encoding="ascii")
    (out_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0\n", encoding="ascii"
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

class KnownPoseTriangulator:
    def __init__(self, job_manager: "JobManager"):
        self._job_manager = job_manager
        self._config = get_config()
        self._colmap_exe = str(self._config.paths.colmap_exe)

    async def run(self, job: "Job", dataset_path: Path) -> bool:
        """Returns True on success (job left at COLMAP_DONE, dataset_path updated
        to point at the merged dataset). Returns False and marks the job failed
        on error."""
        job_id = job.job_id
        colmap_dir = job.colmap_dir
        colmap_dir.mkdir(parents=True, exist_ok=True)

        try:
            transforms_path = dataset_path / "transforms.json"
            if not transforms_path.exists():
                await self._job_manager.mark_failed(
                    job_id, f"hybrid_known_poses requires transforms.json in dataset_path: {dataset_path}")
                return False
            transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
            frames = transforms["frames"]

            images_dir = dataset_path / "images"
            orig_ply = dataset_path / transforms.get("ply_file_path", "points3d.ply")
            if not orig_ply.exists():
                orig_ply = dataset_path / "points3d.ply"

            database_path = colmap_dir / "known_pose.db"
            safe_images_dir = colmap_dir / "safe_images"
            known_pose_dir = colmap_dir / "known_pose_model"
            triangulated_dir = colmap_dir / "triangulated"
            refined_dir = colmap_dir / "refined"
            txt_export_dir = colmap_dir / "triangulated_txt"
            for d in (triangulated_dir, refined_dir, txt_export_dir):
                d.mkdir(parents=True, exist_ok=True)

            job_colmap_config = job.config.get("colmap", {})
            use_gpu = job_colmap_config.get("use_gpu", self._config.colmap.use_gpu)
            matcher = job_colmap_config.get("matcher_type", "exhaustive")
            max_image_size = job_colmap_config.get("max_image_size", self._config.colmap.max_image_size)

            await self._job_manager.update_job_progress(
                job_id, status=JobStatus.COLMAP_RUNNING,
                message="Known-pose triangulation: creating database...",
                overall_progress=5.0,
                colmap_progress=ColmapProgress(stage=ColmapStage.FEATURE_EXTRACTION, message="Starting"),
            )
            await self._run(["database_creator", "--database_path", str(database_path)])
            safe_names = _make_safe_image_dir(frames, images_dir, safe_images_dir)

            await self._job_manager.update_job_progress(
                job_id, message="Known-pose triangulation: extracting features...", overall_progress=8.0)
            await self._run([
                "feature_extractor",
                "--database_path", str(database_path),
                "--image_path", str(safe_images_dir),
                "--ImageReader.camera_model", "OPENCV",
                "--ImageReader.single_camera", "0",
                "--SiftExtraction.max_image_size", str(max_image_size),
                "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
                "--SiftExtraction.max_num_features", str(self._config.colmap.sift_max_features),
            ])

            await self._job_manager.update_job_progress(
                job_id, message=f"Known-pose triangulation: matching features ({matcher})...",
                overall_progress=15.0)
            await self._run([
                f"{matcher}_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1" if use_gpu else "0",
            ])

            disconnected = _find_disconnected_image_names(database_path)
            if disconnected:
                print(
                    f"[KNOWN-POSE-COLMAP] {len(disconnected)}/{len(frames)} images have zero "
                    f"verified matches to any other image (would crash point_triangulator) - "
                    f"excluding from known-pose model: {sorted(disconnected)}"
                )
                connected_frames = [fr for fr in frames if safe_names[fr["file_path"]] not in disconnected]
            else:
                connected_frames = frames

            await self._job_manager.update_job_progress(
                job_id, message="Known-pose triangulation: writing known-pose model...",
                overall_progress=35.0)
            _write_known_pose_model(connected_frames, safe_names, known_pose_dir)

            await self._job_manager.update_job_progress(
                job_id, message="Known-pose triangulation: triangulating points...",
                overall_progress=40.0)
            await self._run([
                "point_triangulator",
                "--database_path", str(database_path),
                "--image_path", str(safe_images_dir),
                "--input_path", str(known_pose_dir),
                "--output_path", str(triangulated_dir),
            ])

            # Bundle adjustment refine. Two modes:
            #
            # - selective (opt-in, requires pycolmap): only images with enough
            #   triangulated observations to reliably constrain a pose are allowed to
            #   move; every weakly-matched camera stays exactly as given. This can
            #   correct real drift in the source LiDAR/IMU poses where there's enough
            #   visual evidence to do so, without the failure mode below.
            #
            # - default (points only): COLMAP's bundle_adjuster defaults to
            #   refine_extrinsics=1 (also refines camera poses), which is exactly what
            #   we don't want here: the known poses are trusted (LiDAR/IMU tracking),
            #   and for the weakly-matched cameras in these low-texture scenes, letting
            #   BA move them is what produced a second, rotated "ghost" copy of the
            #   point cloud in testing (only cameras/points with few/noisy matches
            #   drift, since they're barely constrained — see
            #   Batch12_Scans_Training_Report_2026-08-13.md). Lock every camera
            #   parameter; only 3D point positions may move.
            final_model_dir = triangulated_dir
            refined_poses = {}
            used_selective = False
            if job_colmap_config.get("selective_pose_refine_enabled", False):
                min_percentile = job_colmap_config.get("selective_pose_refine_min_percentile", 50.0)
                try:
                    await self._job_manager.update_job_progress(
                        job_id, message="Known-pose triangulation: selective bundle adjustment "
                                         "(refining well-observed camera poses + points)...",
                        overall_progress=45.0)
                    used_selective, refined_poses = await asyncio.get_running_loop().run_in_executor(
                        None, _selective_bundle_adjust,
                        triangulated_dir, refined_dir, connected_frames, safe_names, min_percentile)
                    if used_selective:
                        final_model_dir = refined_dir
                    else:
                        logger.warning("Selective pose refine unavailable/skipped (pycolmap missing "
                                        "or no image met the observation bar) - falling back to "
                                        "points-only bundle_adjuster")
                except Exception as e:
                    logger.warning(f"Selective bundle adjustment failed, falling back to "
                                    f"points-only bundle_adjuster: {e}")

            if not used_selective:
                try:
                    await self._job_manager.update_job_progress(
                        job_id, message="Known-pose triangulation: bundle-adjusting (points only)...",
                        overall_progress=45.0)
                    await self._run([
                        "bundle_adjuster",
                        "--input_path", str(triangulated_dir),
                        "--output_path", str(refined_dir),
                        "--BundleAdjustment.refine_extrinsics", "0",
                        "--BundleAdjustment.refine_focal_length", "0",
                        "--BundleAdjustment.refine_principal_point", "0",
                        "--BundleAdjustment.refine_extra_params", "0",
                    ])
                    final_model_dir = refined_dir
                except Exception as e:
                    logger.warning(f"bundle_adjuster failed, using un-refined triangulation: {e}")

            await self._run([
                "model_converter",
                "--input_path", str(final_model_dir),
                "--output_path", str(txt_export_dir),
                "--output_type", "TXT",
            ])
            colmap_xyz, colmap_rgb, colmap_track_len = _parse_points3d_txt(txt_export_dir / "points3D.txt")
            lidar_xyz, lidar_rgb = _read_ply_xyzrgb(orig_ply)

            # Sanity-filter triangulated points: with these low-texture scenes, weak/
            # near-parallel viewing-ray matches produce "confidently wrong" triangulations
            # — points that fly off to absurd distances (thousands of units away) while
            # still showing a *low* reprojection error (a few pixels of angular error at
            # a near-degenerate baseline maps to a huge depth error, but a tiny pixel
            # error) — so filtering on COLMAP's own error field does not catch them.
            # Discard anything well outside the known scene volume instead.
            n_before = len(colmap_xyz)
            if n_before and len(lidar_xyz):
                lo = lidar_xyz.min(axis=0)
                hi = lidar_xyz.max(axis=0)
                margin = 0.5 * (hi - lo)
                keep = np.all((colmap_xyz >= lo - margin) & (colmap_xyz <= hi + margin), axis=1)
                colmap_xyz, colmap_rgb, colmap_track_len = colmap_xyz[keep], colmap_rgb[keep], colmap_track_len[keep]
                logger.info(f"Bbox sanity filter: kept {len(colmap_xyz):,}/{n_before:,} triangulated points")

            # Track-length filter: a track of 2 (the minimum) means the point rests on
            # just two rays — at the near-grazing viewing angles typical of silhouette
            # edges, that's a poorly-constrained triangulation even when reprojection
            # error looks small (see comment above). Half of this dataset's triangulated
            # points are track==2. Requiring a third independent observation removes the
            # weakest ones without the bbox filter's blind spot for small local errors.
            n_before_track = len(colmap_xyz)
            if n_before_track:
                min_track = job.config.get("hybrid_known_poses", {}).get("colmap_min_track_length", 3)
                keep = colmap_track_len >= min_track
                colmap_xyz, colmap_rgb = colmap_xyz[keep], colmap_rgb[keep]
                print(
                    f"[KNOWN-POSE-COLMAP] Track-length filter (>= {min_track}): kept "
                    f"{len(colmap_xyz):,}/{n_before_track:,} triangulated points"
                )

            await self._job_manager.update_job_progress(
                job_id, message=(f"Known-pose triangulation: denoising LiDAR cloud and merging "
                                  f"{len(colmap_xyz):,} COLMAP points (filtered from {n_before:,})..."),
                overall_progress=46.0)

            job_hybrid_config = job.config.get("hybrid_known_poses", {})
            lidar_xyz, lidar_rgb, denoise_stats = _denoise_and_prioritize_colmap(
                lidar_xyz, lidar_rgb, colmap_xyz,
                sor_k=job_hybrid_config.get("sor_k", 12),
                sor_std_ratio=job_hybrid_config.get("sor_std_ratio", 2.5),
                colmap_priority_radius=job_hybrid_config.get("colmap_priority_radius"),
                colmap_priority_radius_multiplier=job_hybrid_config.get("colmap_priority_radius_multiplier", 1.5),
            )
            print(
                f"[KNOWN-POSE-COLMAP] LiDAR cleanup: {denoise_stats['n_before']:,} -> "
                f"{denoise_stats['n_after_sor']:,} (after outlier removal) -> "
                f"{denoise_stats['n_after_priority']:,} (after ceding space to COLMAP points, "
                f"radius={denoise_stats['colmap_priority_radius']:.4f})"
            )

            merged_xyz = np.concatenate([lidar_xyz, colmap_xyz], axis=0) if len(colmap_xyz) else lidar_xyz
            merged_rgb = np.concatenate([lidar_rgb, colmap_rgb], axis=0) if len(colmap_rgb) else lidar_rgb

            scale_clamp_enabled = job_hybrid_config.get("init_scale_clamp_enabled", True)
            merged_log_scales = None
            if scale_clamp_enabled:
                clamp_percentile = job_hybrid_config.get("init_scale_clamp_percentile", 90.0)
                merged_log_scales = _compute_density_clamped_log_scales(merged_xyz, clamp_percentile)
                print(
                    f"[KNOWN-POSE-COLMAP] Density-clamped init scale (p{clamp_percentile}): "
                    f"ceiling={np.exp(merged_log_scales.max()):.5f}"
                )

            hybrid_dir = job.job_dir / "inputs" / "hybrid_dataset"
            hybrid_dir.mkdir(parents=True, exist_ok=True)
            # Write (not copy) transforms.json from the `transforms` dict already
            # loaded into memory above, patching in any selectively-refined poses -
            # this is also what makes writing safe on a retried job where
            # transforms_path was already rewritten to this same file by a previous
            # partial attempt (see JobManager's restart recovery): we're not doing a
            # copy of a file onto itself, just re-serializing data we already hold.
            if refined_poses:
                for fr in transforms["frames"]:
                    new_pose = refined_poses.get(fr["file_path"])
                    if new_pose is not None:
                        fr["transform_matrix"] = new_pose
                print(f"[KNOWN-POSE-COLMAP] Applied {len(refined_poses)} selectively-refined "
                      f"camera poses to transforms.json")
            (hybrid_dir / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")
            merged_ply_path = hybrid_dir / "points3d.ply"
            _write_ply_xyzrgb(merged_ply_path, merged_xyz, merged_rgb, merged_log_scales)

            images_link = hybrid_dir / "images"
            if images_link.exists() or images_link.is_symlink():
                if images_link.is_symlink() or images_link.is_file():
                    images_link.unlink()
                else:
                    shutil.rmtree(images_link)
            images_link.symlink_to(images_dir.resolve(), target_is_directory=True)

            # Optional depth supervision (see brush-train's --depth-loss-weight): if the
            # source dataset ships a sibling "depth/" directory (same auto-discovery
            # convention Brush already uses for "masks/"), carry it through unchanged.
            depth_dir = dataset_path / "depth"
            if depth_dir.is_dir():
                depth_link = hybrid_dir / "depth"
                if depth_link.exists() or depth_link.is_symlink():
                    if depth_link.is_symlink() or depth_link.is_file():
                        depth_link.unlink()
                    else:
                        shutil.rmtree(depth_link)
                depth_link.symlink_to(depth_dir.resolve(), target_is_directory=True)

            overexposure_enabled = job_hybrid_config.get("overexposure_mask_enabled", True)
            if overexposure_enabled:
                overexposure_threshold = job_hybrid_config.get("overexposure_threshold", 250)
                mask_stats = _generate_overexposure_masks(
                    images_dir, hybrid_dir / "masks", threshold=overexposure_threshold
                )
                print(
                    f"[KNOWN-POSE-COLMAP] Overexposure masks ({mask_stats['n_images']} images, "
                    f"threshold={overexposure_threshold}): mean blown={mask_stats['mean_blown_pct']}%, "
                    f"max blown={mask_stats['max_blown_pct']}%"
                )

            if job_hybrid_config.get("person_mask_enabled", False):
                dilate_px = int(job_hybrid_config.get("person_mask_dilate_px", 25))
                await self._job_manager.update_job_progress(
                    job_id, message="Generating person masks (DeepLabV3)...", overall_progress=48.0)
                loop = asyncio.get_running_loop()
                person_stats = await loop.run_in_executor(
                    None, lambda: _generate_person_masks(images_dir, hybrid_dir / "masks", dilate_px=dilate_px)
                )
                print(
                    f"[KNOWN-POSE-COLMAP] Person masks ({person_stats['n_images']} images, "
                    f"dilate={dilate_px}px): {person_stats['n_with_person']} contain people, "
                    f"mean masked={person_stats['mean_person_pct']}%, max={person_stats['max_person_pct']}%"
                )

            job.config["dataset_path"] = str(hybrid_dir)

            await self._job_manager.update_job_progress(
                job_id, status=JobStatus.COLMAP_DONE,
                message=(f"Known-pose triangulation done: {len(colmap_xyz):,} COLMAP points "
                         f"+ {len(lidar_xyz):,} LiDAR points = {len(merged_xyz):,} total"),
                overall_progress=50.0,
                colmap_progress=ColmapProgress(stage=ColmapStage.IMAGE_UNDISTORTION,
                                                stage_progress=100.0, message="Completed"),
            )
            return True

        except Exception as e:
            logger.exception("Known-pose triangulation failed")
            await self._job_manager.update_job_progress(
                job_id, status=JobStatus.COLMAP_FAILED,
                error=str(e), message=f"Known-pose triangulation error: {e}")
            return False

    async def _run(self, args: list):
        cmd = [self._colmap_exe] + args
        cmd_str = " ".join(str(c) for c in cmd)
        logger.info(f"Running COLMAP (known-pose): {cmd_str}")
        print(f"[KNOWN-POSE-COLMAP] Running: {cmd_str}")
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        output = []
        async for line in process.stdout:
            text = line.decode("utf-8", errors="ignore").rstrip()
            if text:
                output.append(text)
                print(f"[KNOWN-POSE-COLMAP] {text}")
        await process.wait()
        if process.returncode != 0:
            tail = "\n".join(output[-20:]) if output else "No output"
            raise RuntimeError(f"COLMAP {args[0]} failed with code {process.returncode}\nOutput:\n{tail}")
