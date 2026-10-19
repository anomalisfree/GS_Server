"""Run ON the server (python3.11 - the system python3 has no OpenCV).
Copy a finished COLMAP job's undistorted set into a NEW dataset dir and add sky masks, ready for a
`pointcloud_only` job (Brush auto-discovers masks/<stem>.png).

  sudo -u anoma python3.11 make_sky_dataset.py /data/jobs/<job_id>/colmap/dense /data/datasets/<new_name>
  python3.11 colmap_to_transforms.py /data/datasets/<new_name>        # needed for the preview video
  python tools/client/submit_job.py --name X --mode pointcloud_only --dataset-path /data/datasets/<new_name>

Masks are computed on the already-undistorted images, so they line up with what Brush trains on.
Refuses to overwrite an existing destination.  The copy is ~0.3-1 GB (images + sparse only).
"""
import os
import shutil
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sky_mask import sky_mask  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
if os.path.exists(dst):
    sys.exit(f"refusing to overwrite existing {dst}")
t0 = time.time()
os.makedirs(dst)
for sub in ("images", "sparse"):
    shutil.copytree(os.path.join(src, sub), os.path.join(dst, sub))
print(f"copied images+sparse in {time.time() - t0:.0f}s", flush=True)

img_dir, mask_dir = os.path.join(dst, "images"), os.path.join(dst, "masks")
os.makedirs(mask_dir)
names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
frac = []
for i, n in enumerate(names):
    m = sky_mask(cv2.imread(os.path.join(img_dir, n)))
    cv2.imwrite(os.path.join(mask_dir, os.path.splitext(n)[0] + ".png"), m)
    frac.append(100.0 * (m == 0).mean())
    if (i + 1) % 100 == 0:
        print(f"  masks {i + 1}/{len(names)}", flush=True)
fr = np.array(frac)
print(f"images={len(names)} masks={len(os.listdir(mask_dir))}")
print(f"sky % per image: mean {fr.mean():.1f} median {np.median(fr):.1f} max {fr.max():.1f}; "
      f">5%: {(fr > 5).sum()}  >20%: {(fr > 20).sum()}  0%: {(fr == 0).sum()}")
print(f"total {time.time() - t0:.0f}s; dataset at {dst}")
