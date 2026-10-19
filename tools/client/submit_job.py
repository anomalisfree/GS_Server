#!/usr/bin/env python
"""Create a training job on the GS server and (for photo modes) upload the images.

Server address: --server or env GS_SERVER (e.g. http://<EXTERNAL_IP>:8080; the IP comes from
`gcloud compute instances describe gs-training-server ...`, see CLAUDE.md).

Examples
  # photos -> COLMAP -> Brush (one job from several frame folders)
  python tools/client/submit_job.py --name my_scene --mode colmap_images \
      --frames frames8/ --colmap-config '{"single_camera": true}'

  # lighter/faster COLMAP for big frame sets (see the CameraSet report for the trade-offs)
  python tools/client/submit_job.py --name my_scene --mode colmap_images --frames frames8/ \
      --colmap-config '{"single_camera": true, "matcher_type": "sequential", "sequential_overlap": 20,
                        "sequential_loop_detection": true, "max_image_size": 1280}'

  # dataset that already lives ON THE SERVER (no upload)
  python tools/client/submit_job.py --name scene --mode pointcloud_only \
      --dataset-path /data/datasets/CameraSet_sky8

  python tools/client/submit_job.py ... --dry-run     # print the request, touch nothing

NOTE: config blocks (colmap/brush/hybrid) are TOP-LEVEL fields of POST /jobs - a nested {"config":{...}}
is silently ignored by the server. This script sends them correctly and prints what the server stored.
Needs: requests.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import requests

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect(dirs):
    files = []
    for d in dirs:
        files += [p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXT]
    return sorted(files, key=lambda p: p.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=os.environ.get("GS_SERVER"), help="http://host:8080 (or env GS_SERVER)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="colmap_images",
                    choices=["colmap_images", "pointcloud_only", "hybrid_colmap_pointcloud", "hybrid_known_poses"])
    ap.add_argument("--frames", action="append", default=[], help="image directory to upload (repeatable)")
    ap.add_argument("--dataset-path", help="server-side dataset path (pointcloud_only / hybrid_known_poses)")
    ap.add_argument("--pointcloud", help="local points3d.ply to upload (hybrid_colmap_pointcloud)")
    ap.add_argument("--colmap-config", default="{}", help="JSON object")
    ap.add_argument("--brush-config", default='{"total_steps": 30000, "max_resolution": 1920}', help="JSON object")
    ap.add_argument("--hybrid-config", default="{}", help="JSON object (hybrid_known_poses options)")
    ap.add_argument("--batch", type=int, default=25, help="files per upload request (default 25)")
    ap.add_argument("--no-auto-start", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    body = {"name": a.name, "training_mode": a.mode, "auto_start": not a.no_auto_start}
    for key, raw in (("colmap_config", a.colmap_config), ("brush_config", a.brush_config),
                     ("hybrid_config", a.hybrid_config)):
        cfg = json.loads(raw)
        if cfg:
            body[key] = cfg
    if a.dataset_path:
        body["dataset_path"] = a.dataset_path
    files = collect(a.frames)
    print("request body:", json.dumps(body, indent=1))
    print(f"images to upload: {len(files)}" + (f"  (+ point cloud {a.pointcloud})" if a.pointcloud else ""))
    if a.dry_run:
        return
    if not a.server:
        sys.exit("no server: pass --server or set GS_SERVER")
    base = a.server.rstrip("/")

    r = requests.post(f"{base}/jobs", json=body, timeout=60)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print("created job", job_id)
    cfg = requests.get(f"{base}/jobs/{job_id}", timeout=60).json()["config"]
    print("server stored config:", json.dumps({k: cfg.get(k) for k in
          ("training_mode", "colmap", "brush", "hybrid_known_poses", "dataset_path")}))

    if a.pointcloud:   # must arrive BEFORE upload/complete
        with open(a.pointcloud, "rb") as fh:
            requests.post(f"{base}/jobs/{job_id}/pointcloud", files={"file": fh}, timeout=1800).raise_for_status()
        print("point cloud uploaded")

    for i in range(0, len(files), a.batch):
        chunk = files[i:i + a.batch]
        handles = [open(p, "rb") for p in chunk]
        try:
            multipart = [("files", (p.name, h, "image/jpeg")) for p, h in zip(chunk, handles)]
            resp = requests.post(f"{base}/jobs/{job_id}/upload", files=multipart, timeout=900)
            resp.raise_for_status()
        finally:
            for h in handles:
                h.close()
        print(f"  uploaded {min(i + a.batch, len(files))}/{len(files)}  {resp.json()}")
    if files:
        requests.post(f"{base}/jobs/{job_id}/upload/complete", timeout=120).raise_for_status()
        print("upload/complete sent")
    elif a.no_auto_start:
        print("no images: start it with POST /jobs/<id>/start")
    print(f"JOB_ID={job_id}")


if __name__ == "__main__":
    main()
