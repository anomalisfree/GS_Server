"""
Render sample images from E57 scans to see actual quality
"""
import numpy as np
from PIL import Image
import pye57
import os

E57_PATH = r"d:\VSProjects\GS_Sever\data\2026-06-23 e57 NicoOffice 1.51min 5m TD 5m Threshold export 1.e57"
OUT_DIR  = r"d:\VSProjects\GS_Sever\data\sample_frames"

os.makedirs(OUT_DIR, exist_ok=True)

e57 = pye57.E57(E57_PATH)
total = e57.scan_count
print(f"Total scans: {total}")

# Sample scans spread across the dataset
indices = [0, total//4, total//2, 3*total//4, total-1]

for idx in indices:
    print(f"Rendering scan {idx}...", end=" ", flush=True)
    raw = e57.read_scan_raw(idx)

    row = np.array(raw['rowIndex'], dtype=np.int32)
    col = np.array(raw['columnIndex'], dtype=np.int32)
    r   = np.array(raw['colorRed'],   dtype=np.uint8)
    g   = np.array(raw['colorGreen'], dtype=np.uint8)
    b   = np.array(raw['colorBlue'],  dtype=np.uint8)

    H = int(row.max()) + 1
    W = int(col.max()) + 1

    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[row, col] = np.stack([r, g, b], axis=1)

    out_path = os.path.join(OUT_DIR, f"scan_{idx:04d}.jpg")
    Image.fromarray(img).save(out_path, quality=95)
    print(f"saved {W}x{H} -> {out_path}")

e57.close()
print("Done.")
