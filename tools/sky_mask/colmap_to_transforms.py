"""Run ON the server (python3.11).
Write transforms.json (NeRF convention, per-frame intrinsics) into a COLMAP-format dataset dir, using the
exact conversion gs_server.preview_renderer applies to COLMAP jobs.

Why: the preview/camera_positions code takes poses from transforms.json for pointcloud_only and
hybrid_known_poses jobs.  A pointcloud_only job trained on a COLMAP-format dataset (images/ + sparse/) has
none, and POST /jobs/{id}/preview fails with "transforms.json not found at <dataset>/transforms.json".

  python3.11 colmap_to_transforms.py /data/datasets/<name>

GS_SERVER_ROOT (default /home/anoma/gs_server) must be the directory that CONTAINS the real package folder
gs_server/.  On the production VM /home/anoma/gs_server is itself a folder with an __init__.py, so it has to be
put FIRST on sys.path (done below) or `from gs_server import preview_renderer` resolves to the wrong package.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("GS_SERVER_ROOT", "/home/anoma/gs_server"))
from gs_server import preview_renderer as pr  # noqa: E402

ds = Path(sys.argv[1])
frames = pr.load_colmap_camera_frames(ds / "sparse")
out = {"camera_model": "PINHOLE", "frames": []}
for f in frames:
    out["frames"].append({
        "file_path": "images/" + f["file_path"],
        "fl_x": float(f["fl_x"]), "fl_y": float(f["fl_y"]), "cx": float(f["cx"]), "cy": float(f["cy"]),
        "w": int(f["w"]), "h": int(f["h"]),
        "transform_matrix": f["c2w"].tolist(),
    })
(ds / "transforms.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("wrote", ds / "transforms.json", "frames:", len(frames))
