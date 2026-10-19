#!/usr/bin/env python
"""Poll a job until it finishes; optionally render the flythrough video and download video + model.

  python tools/client/wait_for_job.py <JOB_ID> --server http://<IP>:8080 \
      --preview --out results --label Scan02

--preview requests POST /jobs/{id}/preview after the job completed (needs a pose source: COLMAP jobs and
datasets with transforms.json work; a pointcloud_only job on a COLMAP-format dataset needs a
transforms.json next to the images, see tools/sky_mask/colmap_to_transforms.py).
Exit code: 0 completed, 1 failed/cancelled, 2 timeout.  Needs: requests.

Typical timing seen on the L4 server: COLMAP 20-120 min for 200-700 images, Brush ~17-25 min for
30k steps, preview 1-4 min.
"""
import argparse
import os
import sys
import time

import requests

TERMINAL = {"completed", "failed", "colmap_failed", "training_failed", "cancelled"}


def get(base, path):
    for _ in range(5):
        try:
            return requests.get(f"{base}{path}", timeout=60).json()
        except Exception:
            time.sleep(10)
    raise SystemExit("server not reachable")


def download(url, dest):
    with requests.get(url, stream=True, timeout=900) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"saved {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_id")
    ap.add_argument("--server", default=os.environ.get("GS_SERVER"))
    ap.add_argument("--poll", type=int, default=60, help="seconds between polls")
    ap.add_argument("--timeout-h", type=float, default=16)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--waypoints", type=int, default=40)
    ap.add_argument("--sub-steps", type=int, default=24)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--resolution", type=int, default=960)
    ap.add_argument("--out", help="download directory for <label>_model.ply and <label>_preview.mp4")
    ap.add_argument("--label", help="file-name label (default: first 8 chars of the job id)")
    a = ap.parse_args()
    if not a.server:
        sys.exit("pass --server or set GS_SERVER")
    base, jid = a.server.rstrip("/"), a.job_id
    label = a.label or jid[:8]

    last, deadline = None, time.time() + a.timeout_h * 3600
    while time.time() < deadline:
        p = get(base, f"/jobs/{jid}/progress")
        c, t = p.get("colmap") or {}, p.get("training") or {}
        key = (p["status"], c.get("stage"), t.get("current_step", 0) // 5000)
        if key != last:
            extra = f"step {t['current_step']}/{t['total_steps']} splats {t.get('splat_count')}" if t else (c.get("stage") or "")
            print(time.strftime("%H:%M:%S"), p["status"], f"{p['overall_progress']:.1f}%", extra, flush=True)
            last = key
        if p["status"] in TERMINAL:
            break
        time.sleep(a.poll)
    else:
        print("timeout"); sys.exit(2)
    if p["status"] != "completed":
        print("ended with", p["status"], "-", str(p.get("error"))[:400]); sys.exit(1)

    if a.preview:
        body = {"num_waypoints": a.waypoints, "sub_steps": a.sub_steps, "fps": a.fps, "resolution": a.resolution}
        r = requests.post(f"{base}/jobs/{jid}/preview", json=body, timeout=60)
        if r.status_code not in (200, 409):
            r.raise_for_status()
        while True:
            q = get(base, f"/jobs/{jid}/preview")
            if q["status"] in ("completed", "failed"):
                break
            time.sleep(15)
        print("preview:", q["status"], q.get("error") or q.get("stats"))
        if q["status"] == "completed" and a.out:
            os.makedirs(a.out, exist_ok=True)
            download(q["download_url"], os.path.join(a.out, f"{label}_preview.mp4"))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        download(f"{base}/jobs/{jid}/model", os.path.join(a.out, f"{label}_model.ply"))
    print("done")


if __name__ == "__main__":
    main()
