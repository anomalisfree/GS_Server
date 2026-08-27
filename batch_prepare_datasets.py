"""
Batch-prepare Nerfstudio datasets (transforms.json + images + downsampled points3d.ply)
for every raw Leica scan folder in DATA_ROOT, auto-detecting whether the point cloud
needs a mm->m scale correction by comparing its bounding box to the camera pose bbox.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

DATA_ROOT = Path(r"D:\VSProjects\GS_Sever\data")
OUT_ROOT = Path(r"D:\VSProjects\GS_Sever\data_prepared")
TARGET_POINTS = 2_000_000

FOLDERS = [
    "C_Deck_FWD",
    "D_Deck_FWD_Section 1 (Front Tip)",
    "D_Deck_FWD_Section 2(behind section 1)",
    "SSUS_AFT_Engine_Room_ES22_iPhone_P3DGL",
    "SSUS_AFT_Engine_Room_P2.1_02_iPad_P3DNW",
    "TY_UpperDeckAFT_UseThisOne",
    "core machines next to long tube",
    "core propellers 2",
    "engine room b deck",
    "front tip full",
    "inside big towers",
    "long silver tube",
]


def parse_pose_bbox(sensor_frames_path: Path):
    xs, ys, zs = [], [], []
    with open(sensor_frames_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            idx = line.index(",")
            vals = list(map(float, line[idx + 1:].split()))
            if len(vals) != 16:
                continue
            m = np.array(vals).reshape(4, 4)
            xs.append(m[0, 3]); ys.append(m[1, 3]); zs.append(m[2, 3])
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    return np.array([xs.min(), ys.min(), zs.min()]), np.array([xs.max(), ys.max(), zs.max()])


def read_ply_header(path: Path):
    with path.open("rb") as f:
        header = bytearray()
        while True:
            line = f.readline()
            header.extend(line)
            if line.strip() == b"end_header":
                break
    text = header.decode("ascii", errors="strict")
    fields = []
    nverts = None
    in_vertex = False
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                nverts = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            fields.append((parts[2], parts[1]))
    return header, nverts, fields


_TYPE_MAP = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def sample_ply_bbox(ply_path: Path, header_len: int, nverts: int, fields, sample_stride: int):
    dtype = np.dtype([(name, f"<{_TYPE_MAP[t]}") for name, t in fields])
    data = np.memmap(ply_path, dtype=dtype, mode="r", offset=header_len, shape=(nverts,))
    sample = data[::sample_stride]
    lo = np.array([sample["x"].min(), sample["y"].min(), sample["z"].min()])
    hi = np.array([sample["x"].max(), sample["y"].max(), sample["z"].max()])
    return lo, hi


def best_scale(pose_lo, pose_hi, ply_lo, ply_hi):
    # Only use X/Y (horizontal) extent to judge scale. Z (vertical) is expected to
    # differ a lot even at the correct scale: LiDAR captures full floor-to-ceiling
    # structure while the camera trajectory stays at roughly one walking height, so
    # including Z would bias this toward spuriously "correcting" for a mismatch
    # that isn't a units bug at all (this fooled the very first version of this
    # heuristic into picking a wrong 10x-too-small scale for every ship-deck scan).
    pose_extent_xy = (pose_hi - pose_lo)[:2]
    candidates = [1.0, 0.001, 0.01, 0.1, 100.0, 1000.0]
    best, best_err = 1.0, float("inf")
    for c in candidates:
        ply_extent_xy = (ply_hi - ply_lo)[:2] * c
        denom = np.maximum(pose_extent_xy, 1e-6)
        err = np.mean(np.abs(ply_extent_xy - pose_extent_xy) / denom)
        if err < best_err:
            best_err, best = err, c
    return best, best_err


def main():
    OUT_ROOT.mkdir(exist_ok=True)
    results = []

    for folder in FOLDERS:
        data_dir = DATA_ROOT / folder
        out_dir = OUT_ROOT / folder
        print(f"\n{'='*70}\n{folder}\n{'='*70}")

        sensor_frames = next(data_dir.glob("*_SensorFrames.txt"), None)
        ply_files = [p for p in data_dir.glob("*.ply")]
        if sensor_frames is None or not ply_files:
            print(f"  [SKIP] missing SensorFrames or ply in {data_dir}")
            continue
        ply_path = ply_files[0]

        pose_lo, pose_hi = parse_pose_bbox(sensor_frames)
        header, nverts, fields = read_ply_header(ply_path)
        field_names = {n for n, _ in fields}
        if not {"x", "y", "z"} <= field_names:
            print(f"  [SKIP] ply missing xyz properties: {field_names}")
            continue
        stride = max(1, nverts // 500_000)
        ply_lo, ply_hi = sample_ply_bbox(ply_path, len(header), nverts, fields, stride)

        scale, err = best_scale(pose_lo, pose_hi, ply_lo, ply_hi)
        print(f"  pose bbox: lo={pose_lo} hi={pose_hi}")
        print(f"  ply  bbox (raw): lo={ply_lo} hi={ply_hi}")
        print(f"  chosen scale={scale}  (relative error={err:.3f})  nverts={nverts:,}")

        # 1. transforms.json + images via leica_to_nerfstudio.py
        out_dir.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [sys.executable, "leica_to_nerfstudio.py",
             "--data-dir", str(data_dir), "--out", str(out_dir), "--no-ply"],
            cwd=str(Path(__file__).parent), capture_output=True, text=True
        )
        if r.returncode != 0:
            print("  [ERROR] leica_to_nerfstudio.py failed:")
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            continue
        else:
            print("  transforms.json + images OK")

        # 2. downsample + scale-correct point cloud
        out_ply = out_dir / "points3d.ply"
        r = subprocess.run(
            [sys.executable, "downsample_ply_memmap.py",
             "--src", str(ply_path), "--dst", str(out_ply),
             "--target", str(TARGET_POINTS), "--scale", str(scale)],
            cwd=str(Path(__file__).parent), capture_output=True, text=True
        )
        if r.returncode != 0:
            print("  [ERROR] downsample failed:")
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            continue
        print("  " + r.stdout.strip().replace("\n", "\n  "))

        # 3. add ply_file_path to transforms.json
        tpath = out_dir / "transforms.json"
        d = json.load(open(tpath))
        d["ply_file_path"] = "points3d.ply"
        json.dump(d, open(tpath, "w"), indent=2)

        n_images = len(list((out_dir / "images").glob("*.jpg")))
        results.append({
            "folder": folder,
            "out_dir": str(out_dir),
            "n_images": n_images,
            "scale": scale,
            "scale_error": round(err, 3),
            "n_points": nverts,
        })

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for r in results:
        print(json.dumps(r))

    with open(OUT_ROOT / "_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
