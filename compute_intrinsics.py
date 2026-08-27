"""
Compute camera intrinsics (focal length) from E57 scan data.

Each scan in E57 has:
 - cartesianX/Y/Z in the scanner's LOCAL sensor frame
 - rowIndex/columnIndex: pixel coordinates in the image
 - Scanner pose: R (rotation from sensor to world), T (translation in world)

Assuming perspective projection: col = fx*(X/Z)+cx, row = fy*(Y/Z)+cy
(or with sign flips depending on sensor orientation).
"""

import numpy as np
import pye57

E57_PATH = r"d:\VSProjects\GS_Sever\data\2026-06-23 e57 NicoOffice 1.51min 5m TD 5m Threshold export 1.e57"

e57 = pye57.E57(E57_PATH)

# Use first few scans for robust intrinsic estimation
all_fx, all_fy, all_cx, all_cy = [], [], [], []

for scan_idx in range(min(10, e57.scan_count)):
    raw = e57.read_scan_raw(scan_idx)
    
    x = np.array(raw['cartesianX'], dtype=np.float64)
    y = np.array(raw['cartesianY'], dtype=np.float64)
    z = np.array(raw['cartesianZ'], dtype=np.float64)
    row = np.array(raw['rowIndex'], dtype=np.float64)
    col = np.array(raw['columnIndex'], dtype=np.float64)
    
    H = int(row.max()) + 1
    W = int(col.max()) + 1
    
    # Filter valid points: z > 0 (in front of camera), not NaN, reasonable range
    valid = (z > 0.1) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    
    x, y, z = x[valid], y[valid], z[valid]
    row, col = row[valid], col[valid]
    
    if len(x) < 100:
        continue
    
    xz = x / z  # horizontal direction angle
    yz = y / z  # vertical direction angle
    
    # Perspective: col = fx * xz + cx, row = fy * yz + cy
    # Fit cx via: col = fx*xz + cx → solve via least squares with bias term
    # [xz  1] @ [fx, cx]^T = col
    A_col = np.stack([xz, np.ones_like(xz)], axis=1)
    fx_cx, _, _, _ = np.linalg.lstsq(A_col, col, rcond=None)
    
    A_row = np.stack([yz, np.ones_like(yz)], axis=1)
    fy_cy, _, _, _ = np.linalg.lstsq(A_row, row, rcond=None)
    
    fx, cx_fit = fx_cx
    fy, cy_fit = fy_cy
    
    all_fx.append(fx)
    all_fy.append(fy)
    all_cx.append(cx_fit)
    all_cy.append(cy_fit)
    
    print(f"Scan {scan_idx:3d}: W={W} H={H}  fx={fx:6.1f}  fy={fy:6.1f}  "
          f"cx={cx_fit:5.1f}/{W/2:.1f}  cy={cy_fit:5.1f}/{H/2:.1f}")

print("\n=== Aggregate intrinsics ===")
fx_med = float(np.median(all_fx))
fy_med = float(np.median(all_fy))
cx_med = float(np.median(all_cx))
cy_med = float(np.median(all_cy))
print(f"  fx = {fx_med:.2f}")
print(f"  fy = {fy_med:.2f}")
print(f"  cx = {cx_med:.2f}  (W/2 = {W/2:.1f})")
print(f"  cy = {cy_med:.2f}  (H/2 = {H/2:.1f})")

import math
W_nom, H_nom = 639, 479
hfov = math.degrees(2 * math.atan(W_nom / 2 / abs(fx_med)))
vfov = math.degrees(2 * math.atan(H_nom / 2 / abs(fy_med)))
print(f"  HFOV = {hfov:.1f}°")
print(f"  VFOV = {vfov:.1f}°")

# Check residuals on scan 0
raw = e57.read_scan_raw(0)
x = np.array(raw['cartesianX'], dtype=np.float64)
y = np.array(raw['cartesianY'], dtype=np.float64)
z = np.array(raw['cartesianZ'], dtype=np.float64)
row0 = np.array(raw['rowIndex'], dtype=np.float64)
col0 = np.array(raw['columnIndex'], dtype=np.float64)
valid = (z > 0.1) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
x, y, z, row0, col0 = x[valid], y[valid], z[valid], row0[valid], col0[valid]
col_pred = fx_med * (x/z) + cx_med
row_pred = fy_med * (y/z) + cy_med
col_err = np.abs(col_pred - col0)
row_err = np.abs(row_pred - row0)
print(f"\nResiduals on scan 0:")
print(f"  col MAE = {col_err.mean():.3f} px (median {np.median(col_err):.3f})")
print(f"  row MAE = {row_err.mean():.3f} px (median {np.median(row_err):.3f})")

e57.close()
