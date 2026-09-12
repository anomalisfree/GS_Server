"""
On-demand video preview rendering for trained Gaussian Splatting models.

Not run automatically after training - only triggered by an explicit
POST /jobs/{job_id}/preview request (see app.py). Renders a smooth flythrough
along the same camera poses used to capture/train the scene and encodes it
to an mp4.

Renders with gsplat (github.com/nerfstudio-project/gsplat) - the same
tile-based CUDA rasterizer real-time 3DGS viewers use - whenever it's
importable and a GPU is available, using the full degree-3 SH coefficients
for view-dependent color. Falls back to a hand-rolled Python/PyTorch EWA
renderer (render_gpu() below) otherwise, e.g. on a host without a working
CUDA JIT toolchain for gsplat. See CLIENT_TRAINING_MODES_DEVELOPER_GUIDE.md.

PLY activation functions (see brush/crates/brush-serde/src/export.rs and
brush-render/src/gaussian_splats.rs):
  opacity = sigmoid(raw_opacity)
  scale   = exp(log_scale)
  color   = 0.5 + 0.28209479177387814 * f_dc   (SH DC term only)
  quat order (rot_0..3) = (w, x, y, z) - unused here (disc approx is isotropic)

Camera pose sourcing - THIS IS THE PART THAT DIFFERS BY TRAINING MODE, and got
it wrong the first time (see the render.py-based tooling in scratchpad/ that
preceded this module): a model trained in `colmap_images` (or
`hybrid_colmap_pointcloud`) mode lives in COLMAP's own, independently-solved
SfM coordinate frame - arbitrary rotation/translation/scale, unrelated to the
original scan's LiDAR/IMU world frame. Its camera poses must be read back from
COLMAP's own `images.bin`/`cameras.bin` (the exact ones Brush trained on, i.e.
the post-undistortion sparse model under `<job>/colmap/dense/sparse/`) - not
from the original dataset's transforms.json. `pointcloud_only` and
`hybrid_known_poses` jobs, by contrast, train directly on the prepared
dataset's own transforms.json (job.config["dataset_path"]), which IS in the
original world frame.

Additionally, COLMAP's own convention (as loaded by Brush - see
brush/crates/brush-dataset/src/formats/colmap.rs, which does NOT flip axes)
is OpenCV-style: X right, Y down, Z forward (camera looks down +Z). The
Nerfstudio transforms.json convention Brush loads for the other modes (see
formats/nerfstudio.rs) is OpenGL-style: Y up, Z backward (camera looks down
-Z) - Brush flips Y/Z on load to match its internal COLMAP-like convention,
which is why the two pose sources need different handling here. Rather than
carry two projection formulas, COLMAP poses are converted at load time into
the same "Y up / -Z forward" representation via the same axis flip (negating
the c2w rotation's Y and Z columns is its own inverse - same operation either
direction), so render_gpu() only ever has to deal with one convention.
"""
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import gsplat as _gsplat
except ImportError:
    _gsplat = None

SH_C0 = 0.28209479177387814

# COLMAP camera model id -> (name, num_params). Only the ones this pipeline
# actually produces need to resolve correctly (SIMPLE_RADIAL/OPENCV for raw
# feature_extractor output, PINHOLE for image_undistorter's undistorted
# output) - others fall back to a reasonable guess.
_COLMAP_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


class PreviewError(Exception):
    pass


# ---------------------------------------------------------------------------
# PLY loading (same format/activations as gs_server/floater_filter.py)
# ---------------------------------------------------------------------------
def load_ply_gpu(path: Path, device: str):
    with open(path, "rb") as f:
        data = f.read()

    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii", errors="ignore")

    n_vertex = None
    props = []
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):
            n_vertex = int(line.split()[-1])
        elif line.startswith("property") and n_vertex is not None:
            parts = line.split()
            ptype, pname = parts[1], parts[2]
            np_type = {
                "float": "f4", "float32": "f4", "double": "f8",
                "uchar": "u1", "uint8": "u1", "int": "i4", "int32": "i4",
            }.get(ptype, "f4")
            props.append((pname, np_type))

    dtype = np.dtype(props)
    arr = np.frombuffer(data, dtype=dtype, count=n_vertex, offset=header_end)

    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    scale = np.exp(np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]], axis=1).astype(np.float32))
    opacity = 1.0 / (1.0 + np.exp(-arr["opacity"].astype(np.float32)))
    dc = np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], axis=1).astype(np.float32)
    quat = np.stack([arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"]], axis=1).astype(np.float32)
    quat /= (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12)

    # Higher-order SH ("view-dependent color" - specular-ish highlights that
    # change with viewing angle, e.g. on the reflective metal surfaces common
    # in these scans). f_rest_0..44 = 15 non-DC coeffs per channel (R,G,B),
    # in [b1c0,b1c1,b1c2, b2c0..c4, b3c0..c6] order - see export.rs. Field
    # NAMES are looked up explicitly (not positional) because Brush's ply
    # writer emits these properties in lexicographic (f_rest_0, f_rest_1,
    # f_rest_10, f_rest_11, ...) rather than numeric order.
    field_names = set(arr.dtype.names)
    has_sh_rest = all(f"f_rest_{i}" in field_names for i in range(45))
    if has_sh_rest:
        sh_rest = np.stack(
            [np.stack([arr[f"f_rest_{c * 15 + i}"] for i in range(15)], axis=1) for c in range(3)],
            axis=1,
        ).astype(np.float32)  # (N, 3 channels, 15 coeffs)
    else:
        sh_rest = np.zeros((n_vertex, 3, 15), dtype=np.float32)

    return {
        "xyz": torch.from_numpy(xyz).to(device),
        "scale": torch.from_numpy(scale).to(device),
        "opacity": torch.from_numpy(opacity).to(device),
        "dc": torch.from_numpy(dc).to(device),
        "sh_rest": torch.from_numpy(sh_rest).to(device),
        "quat": torch.from_numpy(quat).to(device),
        "n": n_vertex,
    }


# ---------------------------------------------------------------------------
# COLMAP binary sparse-model reader (cameras.bin / images.bin)
# ---------------------------------------------------------------------------
def _read_next_bytes(f, num_bytes, fmt):
    data = f.read(num_bytes)
    return struct.unpack(fmt, data)


def read_colmap_cameras_bin(path: Path) -> dict:
    cameras = {}
    with open(path, "rb") as f:
        (num_cameras,) = _read_next_bytes(f, 8, "<Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_next_bytes(f, 24, "<iiQQ")
            model_name, num_params = _COLMAP_CAMERA_MODELS.get(model_id, ("UNKNOWN", 4))
            params = _read_next_bytes(f, 8 * num_params, "<" + "d" * num_params)
            cameras[camera_id] = {
                "model": model_name,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_colmap_images_bin(path: Path) -> list:
    images = []
    with open(path, "rb") as f:
        (num_images,) = _read_next_bytes(f, 8, "<Q")
        for _ in range(num_images):
            props = _read_next_bytes(f, 64, "<idddddddi")
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = props
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8", errors="ignore")
            (num_points2d,) = _read_next_bytes(f, 8, "<Q")
            f.read(24 * num_points2d)  # skip 2D point track data, unused here
            images.append({
                "name": name,
                "qvec": (qw, qx, qy, qz),
                "tvec": (tx, ty, tz),
                "camera_id": camera_id,
            })
    return images


def _get_intrinsics(camera: dict):
    model, w, h, p = camera["model"], camera["width"], camera["height"], camera["params"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    elif model in ("RADIAL", "RADIAL_FISHEYE", "FOV"):
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        fx = fy = max(w, h)
        cx, cy = w / 2.0, h / 2.0
    return float(fx), float(fy), float(cx), float(cy), int(w), int(h)


def _quat_to_mat(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _colmap_pose_to_c2w_nerf(qvec, tvec) -> np.ndarray:
    """COLMAP images.bin stores world-to-camera (qvec, tvec): X_cam = R@X_world + t.
    Convert to camera-to-world, then flip the rotation's Y/Z columns to match
    the same "Y up / -Z forward" convention used for the transforms.json path
    (see module docstring) - this does NOT change the actual camera position
    or viewing direction, only how the axes are labeled/signed."""
    R = _quat_to_mat(*qvec)
    t = np.array(tvec)
    R_c2w = R.T
    t_c2w = -R.T @ t
    c2w = np.eye(4)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = t_c2w
    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0
    return c2w


def load_colmap_camera_frames(sparse_dir: Path) -> list:
    """Load (c2w, fx, fy, cx, cy, w, h, image_name) for every registered image
    in a COLMAP sparse model, already converted to the render_gpu() convention."""
    cameras = read_colmap_cameras_bin(sparse_dir / "cameras.bin")
    images = read_colmap_images_bin(sparse_dir / "images.bin")
    images.sort(key=lambda im: im["name"])

    frames = []
    for im in images:
        camera = cameras.get(im["camera_id"])
        if camera is None:
            continue
        fx, fy, cx, cy, w, h = _get_intrinsics(camera)
        c2w = _colmap_pose_to_c2w_nerf(im["qvec"], im["tvec"])
        frames.append({"c2w": c2w, "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "w": w, "h": h,
                        "file_path": im["name"]})
    return frames


def load_nerfstudio_camera_frames(dataset_dir: Path) -> list:
    import json
    transforms_path = dataset_dir / "transforms.json"
    if not transforms_path.exists():
        raise PreviewError(f"transforms.json not found at {transforms_path}")
    with open(transforms_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    frames = []
    for fr in d["frames"]:
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        frames.append({
            "c2w": c2w,
            "fl_x": fr.get("fl_x", d.get("fl_x")),
            "fl_y": fr.get("fl_y", d.get("fl_y")),
            "cx": fr.get("cx", d.get("cx")),
            "cy": fr.get("cy", d.get("cy")),
            "w": fr.get("w", d.get("w")),
            "h": fr.get("h", d.get("h")),
            "file_path": fr["file_path"],
        })
    return frames


def load_job_camera_frames(job) -> list:
    """Pick the pose source matching how this specific job was actually
    trained (see module docstring for why this differs by training_mode)."""
    training_mode = job.config.get("training_mode", "colmap_images")

    if training_mode in ("pointcloud_only", "hybrid_known_poses"):
        dataset_path = job.config.get("dataset_path")
        if not dataset_path:
            raise PreviewError(f"Job has no dataset_path recorded for mode {training_mode}")
        return load_nerfstudio_camera_frames(Path(dataset_path))

    # colmap_images / hybrid_colmap_pointcloud: poses are COLMAP's own SfM
    # solve, read back from the exact (post-undistortion) sparse model Brush
    # trained on.
    sparse_dir = job.colmap_dir / "dense" / "sparse"
    if not (sparse_dir / "images.bin").exists():
        # Fall back to the pre-undistortion sparse model if dense/ is missing
        # for some reason (shouldn't normally happen for a completed job).
        candidates = [p for p in (job.colmap_dir / "sparse").glob("*") if (p / "images.bin").exists()]
        if not candidates:
            raise PreviewError(f"No COLMAP sparse model found under {job.colmap_dir}")
        sparse_dir = candidates[0]
    frames = load_colmap_camera_frames(sparse_dir)
    if not frames:
        raise PreviewError("COLMAP sparse model has no registered images")
    return frames


# ---------------------------------------------------------------------------
# Camera positions API - a cheap, synchronous alternative to rendering a full
# preview video: hands back a small, well-spread set of the actual capture
# camera poses so a human can quickly pick a good starting viewpoint in an
# external 3DGS viewer loading the same exported .ply, without scrolling
# through hundreds of near-duplicate training frames. No model loading or
# GPU rendering involved - just reads camera pose files.
# ---------------------------------------------------------------------------
def _farthest_point_sample_indices(positions: np.ndarray, count: int) -> list:
    """Greedy farthest-point sampling over camera positions - picks a set
    that's well spread out over the space the scan actually covers, rather
    than a naive evenly-spaced-by-index subsample (which bunches up wherever
    the operator lingered and under-samples the rest)."""
    n = positions.shape[0]
    count = min(count, n)
    if count <= 0:
        return []
    selected = [0]
    dist = np.linalg.norm(positions - positions[0], axis=1)
    for _ in range(count - 1):
        next_idx = int(np.argmax(dist))
        selected.append(next_idx)
        new_dist = np.linalg.norm(positions - positions[next_idx], axis=1)
        dist = np.minimum(dist, new_dist)
    return selected


def select_representative_cameras(frames: list, count: int) -> list:
    positions = np.stack([f["c2w"][:3, 3] for f in frames])
    indices = _farthest_point_sample_indices(positions, count)
    return [frames[i] for i in indices]


def camera_frame_to_json(frame: dict) -> dict:
    """Convert one camera frame (as returned by load_job_camera_frames) into
    a convention-agnostic description for external tools: `position`,
    `forward`, `up` and `right` are given directly in WORLD space - the same
    frame the exported .ply's splat positions live in - so a viewer never
    needs to know this pipeline's internal local-axis convention: just place
    a camera at `position`, aim it along `forward`, with `up` as its up
    vector. `quaternion_wxyz` (same local convention documented in this
    module's docstring - camera looks down its own -Z, +Y is up) is included
    too for tools that want to set a camera transform/quaternion directly."""
    c2w = frame["c2w"]
    R = c2w[:3, :3]
    position = c2w[:3, 3]
    forward = -R[:, 2]
    up = R[:, 1]
    right = R[:, 0]
    quat = _mat_to_quat(R)
    fov_x_deg = float(np.degrees(2.0 * np.arctan(frame["w"] / (2.0 * frame["fl_x"]))))
    fov_y_deg = float(np.degrees(2.0 * np.arctan(frame["h"] / (2.0 * frame["fl_y"]))))
    return {
        "position": position.tolist(),
        "forward": forward.tolist(),
        "up": up.tolist(),
        "right": right.tolist(),
        "quaternion_wxyz": quat.tolist(),
        "fov_x_deg": fov_x_deg,
        "fov_y_deg": fov_y_deg,
        "source_image": frame.get("file_path"),
    }


def get_camera_positions(job, count: int = 12) -> dict:
    """Entry point used by app.py. Picks `count` well-spread real capture
    camera poses for the job's actual training_mode (same pose-sourcing
    rules as render_preview_video - see load_job_camera_frames)."""
    frames = load_job_camera_frames(job)
    picked = select_representative_cameras(frames, count)
    return {
        "convention": (
            "Right-handed world space, same frame as the exported .ply's splat "
            "positions. `forward`/`up`/`right`/`position` are already expressed "
            "in world space - aim a camera at `position` along `forward` with "
            "`up` as its up vector, no extra convention knowledge needed. "
            "`quaternion_wxyz` (w,x,y,z) represents the same orientation for "
            "tools that want a quaternion/matrix instead, using a local axis "
            "convention where the camera looks down its own -Z with +Y up."
        ),
        "count": len(picked),
        "cameras": [camera_frame_to_json(f) for f in picked],
    }


def _quat_to_rotmat_components_brush(quat: torch.Tensor):
    """Batched (N,4) -> the 9 scalar components of Brush's own quat_to_mat
    (brush/crates/brush-render/src/shaders/helpers.wgsl - the OpenGL/songho.ca
    convention), as separate (N,) tensors rather than a (N,3,3) tensor.

    Deliberately avoids ever assembling a batched matrix: torch's batched
    matmul over millions of tiny 3x3/2x3 matrices turned out to be the
    dominant cost of this renderer (single-digit-minutes per video became
    ~40+ min) - plain elementwise ops on flat tensors are dramatically faster
    on GPU for matrices this small, so covariance propagation below is written
    out as explicit scalar formulas instead of `@`."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r00 = 1 - 2 * (y2 + z2)
    r10 = 2 * (xy + wz)
    r20 = 2 * (xz - wy)
    r01 = 2 * (xy - wz)
    r11 = 1 - 2 * (x2 + z2)
    r21 = 2 * (yz + wx)
    r02 = 2 * (xz + wy)
    r12 = 2 * (yz - wx)
    r22 = 1 - 2 * (x2 + y2)
    return r00, r01, r02, r10, r11, r12, r20, r21, r22


def _sh_to_color(viewdir: torch.Tensor, dc: torch.Tensor, sh_rest: torch.Tensor) -> torch.Tensor:
    """Full degree-3 spherical harmonics evaluation, matching Brush's own
    sh_coeffs_to_color exactly (brush/crates/brush-render/src/shaders/sh.wgsl,
    the Sloan JCGT 2013 formulas) - same magic constants, same coefficient
    order. This is what actually gives a real 3DGS viewer (e.g. superspl.at)
    its "shiny" look on reflective surfaces: color is view-dependent, not a
    flat per-splat RGB. Using only the DC term (as this renderer's first two
    versions did) discards that entirely and reads as dull/flat on the
    reflective metal surfaces common in these scans.

    viewdir: (N,3) unit vectors from camera to each splat, in WORLD space
    (Brush computes it the same way: normalize(mean - camera_position)).
    dc: (N,3). sh_rest: (N,3,15) ordered [b1c0,b1c1,b1c2,b2c0..c4,b3c0..c6]
    per channel (see load_ply_gpu)."""
    x, y, z = viewdir[:, 0:1], viewdir[:, 1:2], viewdir[:, 2:3]

    colors = SH_C0 * dc

    b1c0, b1c1, b1c2 = sh_rest[:, :, 0], sh_rest[:, :, 1], sh_rest[:, :, 2]
    f_tmp0a = 0.48860251190292
    colors = colors + f_tmp0a * (-y * b1c0 + z * b1c1 - x * b1c2)

    z2 = z * z
    f_tmp0b = -1.092548430592079 * z
    f_tmp1a = 0.5462742152960395
    f_c1 = x * x - y * y
    f_s1 = 2.0 * x * y
    p_sh6 = 0.9461746957575601 * z2 - 0.3153915652525201
    p_sh7 = f_tmp0b * x
    p_sh5 = f_tmp0b * y
    p_sh8 = f_tmp1a * f_c1
    p_sh4 = f_tmp1a * f_s1
    b2 = sh_rest[:, :, 3:8]
    colors = colors + p_sh4 * b2[:, :, 0] + p_sh5 * b2[:, :, 1] + p_sh6 * b2[:, :, 2] \
        + p_sh7 * b2[:, :, 3] + p_sh8 * b2[:, :, 4]

    f_tmp0c = -2.285228997322329 * z2 + 0.4570457994644658
    f_tmp1b = 1.445305721320277 * z
    f_tmp2a = -0.5900435899266435
    f_c2 = x * f_c1 - y * f_s1
    f_s2 = x * f_s1 + y * f_c1
    p_sh12 = z * (1.865881662950577 * z2 - 1.119528997770346)
    p_sh13 = f_tmp0c * x
    p_sh11 = f_tmp0c * y
    p_sh14 = f_tmp1b * f_c1
    p_sh10 = f_tmp1b * f_s1
    p_sh15 = f_tmp2a * f_c2
    p_sh9 = f_tmp2a * f_s2
    b3 = sh_rest[:, :, 8:15]
    colors = colors + p_sh9 * b3[:, :, 0] + p_sh10 * b3[:, :, 1] + p_sh11 * b3[:, :, 2] \
        + p_sh12 * b3[:, :, 3] + p_sh13 * b3[:, :, 4] + p_sh14 * b3[:, :, 5] + p_sh15 * b3[:, :, 6]

    return torch.clamp(colors + 0.5, 0.0, 1.0)


# ---------------------------------------------------------------------------
# gsplat-backed rendering - the real tile-based CUDA rasterizer (compiled
# extension, JIT-built on first use), used instead of the manual render_gpu()
# below whenever it's importable. render_gpu() does the same EWA math by hand
# in Python, launching ~300 small kernels per frame with CPU-GPU sync between
# each depth-chunk/radius-bucket; gsplat does one global depth sort + tile
# binning + per-tile compositing in a single fused kernel, which is why it
# renders a frame in under a millisecond instead of several seconds. Kept as
# a soft dependency (see the `_gsplat` import at the top of this module) since
# it needs a working CUDA toolchain to JIT-compile the first time - if that's
# unavailable, render_preview_video_sync() falls back to render_gpu().
# ---------------------------------------------------------------------------
def _c2w_nerf_to_viewmat_opencv(c2w: np.ndarray) -> np.ndarray:
    """Our c2w frames are stored in the "Y up / -Z forward" convention (see
    module docstring). gsplat, like COLMAP, expects a world-to-camera matrix
    in OpenCV convention (Y down, Z forward) - flip the rotation's Y/Z
    columns (self-inverse, same op as _colmap_pose_to_c2w_nerf) to get back
    to OpenCV, then invert camera-to-world into world-to-camera."""
    R = c2w[:3, :3].copy()
    t = c2w[:3, 3].copy()
    R[:, 1] *= -1.0
    R[:, 2] *= -1.0
    Rt = R.T
    viewmat = np.eye(4, dtype=np.float64)
    viewmat[:3, :3] = Rt
    viewmat[:3, 3] = -Rt @ t
    return viewmat


def render_gsplat_frame(splats, c2w: np.ndarray, K: torch.Tensor, out_w: int, out_h: int,
                         bg=(0.06, 0.06, 0.07), device: str = "cuda") -> np.ndarray:
    """Render a single camera pose via gsplat's rasterizer, using our own
    validated, clamped-to-[0,1] SH evaluation (_sh_to_color) rather than
    handing gsplat raw SH coefficients to evaluate internally. gsplat does
    real (unnormalized, alpha-under) compositing, so an unclamped color
    overshoot from a single poorly-constrained ("floater") gaussian - at a
    viewing angle nothing in training saw - lands straight in the image
    instead of being washed out by normalization, showing up as patches of
    the background flickering different colors as the camera moves. Colors
    are computed per-frame (view-dependent) since gsplat only accepts one
    shared color set per rasterization() call - see render_preview_video_sync,
    which therefore calls this once per frame rather than batching cameras."""
    viewmat = torch.tensor(_c2w_nerf_to_viewmat_opencv(c2w), dtype=torch.float32, device=device)
    cam_pos = torch.tensor(c2w[:3, 3], dtype=torch.float32, device=device)

    viewdir = splats["xyz"] - cam_pos.unsqueeze(0)
    viewdir = viewdir / (viewdir.norm(dim=1, keepdim=True) + 1e-12)
    color = _sh_to_color(viewdir, splats["dc"], splats["sh_rest"])

    bg_t = torch.tensor(bg, dtype=torch.float32, device=device)

    out_img, _, _ = _gsplat.rasterization(
        means=splats["xyz"],
        quats=splats["quat"],
        scales=splats["scale"],
        opacities=splats["opacity"],
        colors=color,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=out_w,
        height=out_h,
        backgrounds=bg_t,
        near_plane=0.02,
    )
    img = torch.clamp(out_img[0], 0.0, 1.0) * 255
    return img.byte().cpu().numpy()


# ---------------------------------------------------------------------------
# GPU EWA (elliptical weighted average) splatting - the same projection
# algorithm real-time 3DGS renderers use: each gaussian's actual 3D covariance
# (from its rotation quaternion + scale) is projected through the camera's
# perspective Jacobian into a 2D pixel-space covariance, giving a correctly
# oriented/stretched elliptical footprint per splat instead of an isotropic
# disc. An isotropic-disc approximation (this renderer's first version) always
# looks like a cloud of round blobs no matter how its radius is tuned, because
# it throws away the gaussian's orientation entirely - that's what this fixes.
#
# Fallback path only, used when gsplat isn't available (see render_gsplat_batch
# above) - kept for environments without a working CUDA JIT toolchain.
# ---------------------------------------------------------------------------
def render_gpu(splats, c2w, fl_x, fl_y, cx, cy, w, h,
               out_w=960, out_h=720, max_px_radius=64.0,
               num_chunks=32, bg=(0.06, 0.06, 0.07), device="cuda"):
    sx = out_w / w
    sy = out_h / h
    fl_x, fl_y, cx, cy = fl_x * sx, fl_y * sy, cx * sx, cy * sy

    c2w_R = torch.tensor(c2w[:3, :3], dtype=torch.float32, device=device)
    t = torch.tensor(c2w[:3, 3], dtype=torch.float32, device=device)
    bg_t = torch.tensor(bg, dtype=torch.float32, device=device)

    def bg_image():
        img = bg_t.view(1, 1, 3).expand(out_h, out_w, 3).clamp(0, 1) * 255
        return img.byte().cpu().numpy()

    xyz = splats["xyz"]
    Xc = (xyz - t) @ c2w_R
    depth = -Xc[:, 2]

    near = 0.02
    visible = depth > near
    if int(visible.sum()) == 0:
        return bg_image()

    xyz = xyz[visible]
    Xc = Xc[visible]
    depth = depth[visible]
    scale = splats["scale"][visible]
    opacity = splats["opacity"][visible]
    dc = splats["dc"][visible]
    sh_rest = splats["sh_rest"][visible]
    quat = splats["quat"][visible]

    inv_depth = 1.0 / depth
    u = cx + fl_x * Xc[:, 0] * inv_depth
    v = cy - fl_y * Xc[:, 1] * inv_depth

    margin = max_px_radius + 2
    in_frame = (u > -margin) & (u < out_w + margin) & (v > -margin) & (v < out_h + margin)
    if int(in_frame.sum()) == 0:
        return bg_image()

    xyz, Xc, u, v, depth = xyz[in_frame], Xc[in_frame], u[in_frame], v[in_frame], depth[in_frame]
    opacity, dc, sh_rest = opacity[in_frame], dc[in_frame], sh_rest[in_frame]
    scale, quat = scale[in_frame], quat[in_frame]

    # View-dependent color: same per-gaussian evaluation Brush itself uses
    # (viewdir in WORLD space, from camera to splat), not per-pixel - see
    # _sh_to_color's docstring.
    viewdir = xyz - t.unsqueeze(0)
    viewdir = viewdir / (viewdir.norm(dim=1, keepdim=True) + 1e-12)
    color = _sh_to_color(viewdir, dc, sh_rest)

    # 3D covariance in world space: Sigma = R * diag(scale^2) * R^T, expanded
    # into its 6 unique (symmetric) scalar components directly - see
    # _quat_to_rotmat_components_brush for why this avoids batched matmul.
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = _quat_to_rotmat_components_brush(quat)
    s0, s1, s2 = scale[:, 0] ** 2, scale[:, 1] ** 2, scale[:, 2] ** 2
    sw00 = r00 * r00 * s0 + r01 * r01 * s1 + r02 * r02 * s2
    sw01 = r00 * r10 * s0 + r01 * r11 * s1 + r02 * r12 * s2
    sw02 = r00 * r20 * s0 + r01 * r21 * s1 + r02 * r22 * s2
    sw11 = r10 * r10 * s0 + r11 * r11 * s1 + r12 * r12 * s2
    sw12 = r10 * r20 * s0 + r11 * r21 * s1 + r12 * r22 * s2
    sw22 = r20 * r20 * s0 + r21 * r21 * s1 + r22 * r22 * s2

    # -> camera space: Sigma_cam = W @ Sigma_world @ W^T, W = world-to-camera
    # rotation (constant this frame, so its 9 entries are plain Python floats
    # and each output component is a fixed linear combination of sw00..sw22).
    W = c2w[:3, :3].T
    def _sc(i, j):
        wi, wj = W[i], W[j]
        return (wi[0] * wj[0] * sw00 + wi[1] * wj[1] * sw11 + wi[2] * wj[2] * sw22
                + (wi[0] * wj[1] + wi[1] * wj[0]) * sw01
                + (wi[0] * wj[2] + wi[2] * wj[0]) * sw02
                + (wi[1] * wj[2] + wi[2] * wj[1]) * sw12)
    sc00, sc01, sc02 = _sc(0, 0), _sc(0, 1), _sc(0, 2)
    sc11, sc12, sc22 = _sc(1, 1), _sc(1, 2), _sc(2, 2)

    # Perspective Jacobian at each splat's own (x,y,z) - see module docstring
    # of this function's derivation: u = cx - fx*x/z, v = cy + fy*y/z (this
    # renderer's camera-looks--Z convention, z = Xc[:,2] which is negative
    # in front of the camera). J row0 = (j0x, 0, j0z), row1 = (0, j1y, j1z).
    z = Xc[:, 2]
    x = Xc[:, 0]
    y = Xc[:, 1]
    j0x = -fl_x / z
    j0z = fl_x * x / (z * z)
    j1y = fl_y / z
    j1z = -fl_y * y / (z * z)

    # Sigma_2d = J @ Sigma_cam @ J^T, expanded directly (2x2 symmetric).
    a = j0x * j0x * sc00 + 2 * j0x * j0z * sc02 + j0z * j0z * sc22
    c = j1y * j1y * sc11 + 2 * j1y * j1z * sc12 + j1z * j1z * sc22
    b = j0x * j1y * sc01 + j0x * j1z * sc02 + j0z * j1y * sc12 + j0z * j1z * sc22
    # Small dilation (low-pass filter) so near-degenerate (paper-thin) splats
    # don't vanish to sub-pixel width and flicker/alias.
    a = a + 0.3
    c = c + 0.3

    det = a * c - b * b
    det = torch.clamp(det, min=1e-8)
    inv_a = c / det
    inv_b = -b / det
    inv_c = a / det

    # 3-sigma bounding radius from the larger eigenvalue of the 2x2 covariance.
    mid = 0.5 * (a + c)
    disc = torch.sqrt(torch.clamp(mid * mid - det, min=0.0))
    max_eig = mid + disc
    px_radius = torch.clamp(3.0 * torch.sqrt(torch.clamp(max_eig, min=1e-6)), 0.6, max_px_radius)

    lo = torch.quantile(depth, 0.002).item()
    hi = torch.quantile(depth, 0.995).item()
    lo = max(lo, near)
    hi = max(hi, lo * 1.001)
    depth_c = torch.clamp(depth, lo, hi)
    chunk_edges = torch.tensor(np.geomspace(lo, hi, num_chunks + 1), dtype=torch.float32, device=device)
    chunk_id = torch.clamp(torch.searchsorted(chunk_edges, depth_c, right=True) - 1, 0, num_chunks - 1)

    out_color = bg_t.view(1, 1, 3).expand(out_h, out_w, 3).clone()

    radius_bins = torch.tensor(
        [0.6, 1.2, 2.0, 3.5, 6.0, 10.0, 18.0, 30.0, max_px_radius + 0.01], device=device
    )

    for ch in range(num_chunks - 1, -1, -1):
        m = chunk_id == ch
        if int(m.sum()) == 0:
            continue
        cu, cv = u[m], v[m]
        cop, ccol, crad = opacity[m], color[m], px_radius[m]
        cinv_a, cinv_b, cinv_c = inv_a[m], inv_b[m], inv_c[m]

        color_sum = torch.zeros((out_h * out_w, 3), dtype=torch.float32, device=device)
        weight_sum = torch.zeros(out_h * out_w, dtype=torch.float32, device=device)

        bin_idx = torch.searchsorted(radius_bins, crad, right=True)
        for b_i in range(radius_bins.shape[0]):
            bm = bin_idx == b_i
            if int(bm.sum()) == 0:
                continue
            R_int = max(int(np.ceil(radius_bins[min(b_i, radius_bins.shape[0] - 1)].item())), 1)
            rng = torch.arange(-R_int, R_int + 1, device=device)
            dy, dx = torch.meshgrid(rng, rng, indexing="ij")
            keep = (dx ** 2 + dy ** 2) <= (R_int + 0.5) ** 2
            dxs, dys = dx[keep].float(), dy[keep].float()
            dx2, dy2, dxdy = dxs * dxs, dys * dys, dxs * dys

            bu, bv, bop, bcol = cu[bm], cv[bm], cop[bm], ccol[bm]
            ba, bb, bc = cinv_a[bm], cinv_b[bm], cinv_c[bm]

            px = torch.round(bu).unsqueeze(1) + dxs.unsqueeze(0)
            py = torch.round(bv).unsqueeze(1) + dys.unsqueeze(0)
            valid = (px >= 0) & (px < out_w) & (py >= 0) & (py < out_h)

            mahalanobis = (ba.unsqueeze(1) * dx2.unsqueeze(0)
                           + 2 * bb.unsqueeze(1) * dxdy.unsqueeze(0)
                           + bc.unsqueeze(1) * dy2.unsqueeze(0))
            w_g = torch.exp(-0.5 * mahalanobis)
            weight = bop.unsqueeze(1) * w_g
            weight = torch.where(valid, weight, torch.zeros_like(weight))

            pxc = px.clamp(0, out_w - 1).long()
            pyc = py.clamp(0, out_h - 1).long()
            flat = (pyc * out_w + pxc).reshape(-1)
            wflat = weight.reshape(-1)

            weight_sum.index_add_(0, flat, wflat)
            colflat = (bcol.unsqueeze(1) * weight.unsqueeze(-1)).reshape(-1, 3)
            color_sum.index_add_(0, flat, colflat)

        weight_sum = weight_sum.view(out_h, out_w)
        color_sum = color_sum.view(out_h, out_w, 3)

        chunk_alpha = 1.0 - torch.exp(-weight_sum)
        safe_w = torch.where(weight_sum > 1e-8, weight_sum, torch.ones_like(weight_sum))
        chunk_color = color_sum / safe_w.unsqueeze(-1)

        al = chunk_alpha.unsqueeze(-1)
        out_color = out_color * (1 - al) + chunk_color * al

    img = torch.clamp(out_color, 0.0, 1.0) * 255
    return img.byte().cpu().numpy()


# ---------------------------------------------------------------------------
# Smooth camera-path interpolation (slerp rotation, lerp translation)
# ---------------------------------------------------------------------------
def _mat_to_quat(R):
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * S, (m21 - m12) / S, (m02 - m20) / S, (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w, x, y, z = (m21 - m12) / S, 0.25 * S, (m01 + m10) / S, (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w, x, y, z = (m02 - m20) / S, (m01 + m10) / S, 0.25 * S, (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w, x, y, z = (m10 - m01) / S, (m02 + m20) / S, (m12 + m21) / S, 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _quat_to_mat3(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _slerp(q0, q1, t):
    dot = np.dot(q0, q1)
    if dot < 0:
        q1, dot = -q1, -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta0 = np.arccos(dot)
    theta = theta0 * t
    q2 = q1 - q0 * dot
    q2 /= np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def interpolate_c2w_path(c2w_list, sub_steps):
    quats = [_mat_to_quat(c2w[:3, :3]) for c2w in c2w_list]
    trans = [c2w[:3, 3] for c2w in c2w_list]
    out = []
    for i in range(len(c2w_list) - 1):
        for s in range(sub_steps):
            t = s / sub_steps
            q = _slerp(quats[i], quats[i + 1], t)
            p = trans[i] * (1 - t) + trans[i + 1] * t
            c2w = np.eye(4)
            c2w[:3, :3] = _quat_to_mat3(q)
            c2w[:3, 3] = p
            out.append(c2w)
    out.append(c2w_list[-1].copy())
    return out


def _nearest_splat_distance(points: torch.Tensor, splats_xyz: torch.Tensor, chunk: int = 200_000) -> torch.Tensor:
    """Min distance from each of `points` (M,3) to the splat cloud (N,3),
    chunked over N so this doesn't blow up GPU memory for multi-million-splat
    models - M (camera waypoints) is always small (<=200), so a full (M,chunk)
    cdist per chunk is cheap."""
    dmin = torch.full((points.shape[0],), float("inf"), device=points.device)
    for s in range(0, splats_xyz.shape[0], chunk):
        d = torch.cdist(points, splats_xyz[s:s + chunk])
        dmin = torch.minimum(dmin, d.min(dim=1).values)
    return dmin


def _smooth_waypoint_path(c2w_list: list, splats_xyz: torch.Tensor, strength: float = 0.5,
                           window: int = 5, device: str = "cuda") -> list:
    """Round off sharp turns in the sampled capture path (real operator
    trajectories zig-zag - walking back and forth, turning to face different
    parts of the scene - which reads as jerky camera motion once the path is
    played back as a flythrough) via a Gaussian-weighted moving average over
    both position and orientation.

    Smoothing a path in isolation risks cutting corners straight through
    whatever the original path went around (a wall, a piece of equipment) -
    so each waypoint's smoothing is scaled down by how much CLOSER it would
    bring the camera to the actual splat cloud versus its own original
    clearance. A waypoint that was already grazing geometry gets little to no
    smoothing; one in open space gets the full effect. This only ever pulls
    a waypoint back toward its original (already-flown, so already safe)
    position - it never pushes it further out."""
    if strength <= 0.0 or len(c2w_list) < 3:
        return c2w_list

    n = len(c2w_list)
    orig_quats = [_mat_to_quat(c2w[:3, :3]) for c2w in c2w_list]
    orig_pos = np.stack([c2w[:3, 3] for c2w in c2w_list])

    half = window // 2
    offsets = np.arange(-half, half + 1)
    gauss = np.exp(-0.5 * (offsets / max(half / 2.0, 1e-6)) ** 2)

    smoothed_pos = np.zeros_like(orig_pos)
    smoothed_quats = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        w = gauss[lo - i + half: hi - i + half]
        w = w / w.sum()
        smoothed_pos[i] = (orig_pos[lo:hi] * w[:, None]).sum(axis=0)

        q_acc = orig_quats[lo].copy()
        w_acc = w[0]
        for k, j in enumerate(range(lo + 1, hi), start=1):
            qj = orig_quats[j]
            if np.dot(q_acc, qj) < 0:
                qj = -qj
            t = w[k] / (w_acc + w[k])
            q_acc = _slerp(q_acc, qj, t)
            w_acc += w[k]
        smoothed_quats.append(q_acc / np.linalg.norm(q_acc))

    orig_pos_t = torch.tensor(orig_pos, dtype=torch.float32, device=device)
    smoothed_pos_t = torch.tensor(smoothed_pos, dtype=torch.float32, device=device)
    orig_clearance = _nearest_splat_distance(orig_pos_t, splats_xyz)
    smoothed_clearance = _nearest_splat_distance(smoothed_pos_t, splats_xyz)
    clearance_ratio = torch.clamp(smoothed_clearance / (orig_clearance + 1e-6), max=1.0)
    per_point_strength = (strength * clearance_ratio).cpu().numpy()

    out = []
    for i, c2w in enumerate(c2w_list):
        s = per_point_strength[i]
        pos = orig_pos[i] * (1 - s) + smoothed_pos[i] * s
        q = _slerp(orig_quats[i], smoothed_quats[i], s)
        new_c2w = np.eye(4)
        new_c2w[:3, :3] = _quat_to_mat3(q)
        new_c2w[:3, 3] = pos
        out.append(new_c2w)
    return out


def _cull_floater_splats(splats: dict, scale_factor: float = 20.0, min_keep_frac: float = 0.9) -> dict:
    """Drop gaussians whose scale is a wild outlier relative to the rest of
    the model - a small fraction of badly-constrained ("floater") splats can
    end up with enormous, near-isotropic scale even after training-time
    floater filtering. render_gpu's naive compositing only capped their
    apparent SCREEN size (max_px_radius), which incidentally hid this; gsplat
    does true unnormalized alpha-under compositing, so a handful of these
    large, poorly-constrained gaussians - whose view-dependent SH color can
    overshoot wildly at viewing angles nothing in training saw - shows up as
    patches of the background flickering different colors as the camera
    moves. This is a rendering-only safety cull, not a change to the trained
    model file on disk."""
    scale = splats["scale"]
    max_axis = scale.max(dim=1).values
    median = torch.median(max_axis)
    keep = max_axis <= median * scale_factor
    n_total = max_axis.shape[0]
    n_keep = int(keep.sum().item())
    if n_keep < min_keep_frac * n_total:
        # Most of the model is "large" relative to its own median - not a
        # floater situation, leave the data alone rather than gutting it.
        return splats
    out = {}
    for k, v in splats.items():
        if torch.is_tensor(v) and v.shape[0] == n_total:
            out[k] = v[keep]
        else:
            out[k] = v
    out["n"] = n_keep
    return out


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def render_preview_video_sync(
    model_path: Path,
    frames: list,
    output_path: Path,
    num_waypoints: int = 40,
    sub_steps: int = 6,
    path_smoothing: float = 0.5,
    fps: int = 24,
    resolution: int = 960,
) -> dict:
    """Blocking - call via loop.run_in_executor. Returns stats dict."""
    import cv2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splats = load_ply_gpu(model_path, device)
    splats = _cull_floater_splats(splats)

    n = len(frames)
    if n < 2:
        raise PreviewError("Need at least 2 camera frames to build a flythrough")

    step = max(1, n // num_waypoints)
    waypoint_indices = list(range(0, n, step))
    waypoint_c2w = [frames[i]["c2w"] for i in waypoint_indices]
    if path_smoothing > 0.0 and device == "cuda":
        waypoint_c2w = _smooth_waypoint_path(waypoint_c2w, splats["xyz"], strength=path_smoothing, device=device)
    smooth_path = interpolate_c2w_path(waypoint_c2w, sub_steps)
    fr0 = frames[0]
    out_h = round(resolution * fr0["h"] / fr0["w"])

    use_gsplat = _gsplat is not None and device == "cuda"

    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = output_path.with_suffix(".tmp.mp4")
    try:
        if use_gsplat:
            sx = resolution / fr0["w"]
            sy = out_h / fr0["h"]
            fl_x, fl_y = fr0["fl_x"] * sx, fr0["fl_y"] * sy
            cx, cy = fr0["cx"] * sx, fr0["cy"] * sy
            K = torch.tensor([[fl_x, 0.0, cx], [0.0, fl_y, cy], [0.0, 0.0, 1.0]],
                              dtype=torch.float32, device=device)
            for c2w in smooth_path:
                img = render_gsplat_frame(splats, c2w, K, resolution, out_h, device=device)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if writer is None:
                    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (resolution, out_h))
                writer.write(img_bgr)
        else:
            # Splat footprints are computed from the actual projected 3D
            # covariance (see render_gpu), so they're correctly sized at any
            # resolution already - this is just a perf/safety ceiling on how
            # large a single splat's stamp can get, scaled with resolution so
            # it isn't relatively tighter at higher res.
            max_px_radius = 64.0 * (resolution / 960.0)
            for c2w in smooth_path:
                img = render_gpu(splats, c2w, fr0["fl_x"], fr0["fl_y"], fr0["cx"], fr0["cy"], fr0["w"], fr0["h"],
                                  out_w=resolution, out_h=out_h, max_px_radius=max_px_radius, device=device)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if writer is None:
                    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (resolution, out_h))
                writer.write(img_bgr)
    finally:
        if writer is not None:
            writer.release()
        if device == "cuda":
            torch.cuda.empty_cache()

    tmp_path.replace(output_path)
    return {
        "num_splats": splats["n"],
        "num_frames": len(smooth_path),
        "fps": fps,
        "duration_seconds": round(len(smooth_path) / fps, 1),
        "resolution": [resolution, out_h],
        "device": device,
        "renderer": "gsplat" if use_gsplat else "manual_ewa",
    }


def render_preview_video(job, output_path: Path, **kwargs) -> dict:
    """Entry point used by app.py. Picks the right pose source for this job's
    training_mode (see module docstring), then renders. Raises PreviewError
    on any user-facing failure (missing model, missing poses, etc)."""
    if not job.model_path or not Path(job.model_path).exists():
        raise PreviewError("Job has no trained model to render a preview from")

    frames = load_job_camera_frames(job)
    return render_preview_video_sync(Path(job.model_path), frames, output_path, **kwargs)
