"""
Deep inspect first scan: determine grid resolution, projection type, FOV
"""
import numpy as np
import pye57

E57_PATH = r"d:\VSProjects\GS_Sever\data\2026-06-23 e57 NicoOffice 1.51min 5m TD 5m Threshold export 1.e57"

e57 = pye57.E57(E57_PATH)
print(f"Total scans: {e57.scan_count}")

# Inspect scan 0
raw = e57.read_scan_raw(0)
header = e57.get_header(0)

row = np.array(raw['rowIndex'], dtype=np.int32)
col = np.array(raw['columnIndex'], dtype=np.int32)
x = np.array(raw['cartesianX'], dtype=np.float64)
y = np.array(raw['cartesianY'], dtype=np.float64)
z = np.array(raw['cartesianZ'], dtype=np.float64)
r = np.array(raw['colorRed'])
g = np.array(raw['colorGreen'])
b = np.array(raw['colorBlue'])

print(f"\n=== Scan 0 grid ===")
print(f"  Row range:    [{row.min()}, {row.max()}]  => {row.max()+1} rows")
print(f"  Col range:    [{col.min()}, {col.max()}]  => {col.max()+1} cols")
print(f"  Point count:  {len(x)}")
print(f"  Grid coverage: {len(x) / ((row.max()+1)*(col.max()+1)) * 100:.1f}%")

print(f"\n=== Point cloud extent (scan-local) ===")
print(f"  X: [{x.min():.3f}, {x.max():.3f}]")
print(f"  Y: [{y.min():.3f}, {y.max():.3f}]")
print(f"  Z: [{z.min():.3f}, {z.max():.3f}]")

# Compute range (distance)
rng = np.sqrt(x**2 + y**2 + z**2)
print(f"  Range: [{rng.min():.3f}, {rng.max():.3f}] m")

# Compute normalized directions
eps = 1e-9
nx = x / (rng + eps)
ny = y / (rng + eps)
nz = z / (rng + eps)

# Azimuth and elevation in scan-local frame (assuming Z = up in sensor frame)
azimuth   = np.degrees(np.arctan2(ny, nx))    # [-180, 180]
elevation = np.degrees(np.arcsin(np.clip(nz, -1, 1)))  # [-90, 90]

print(f"\n=== Angular extents ===")
print(f"  Azimuth:   [{azimuth.min():.1f}°, {azimuth.max():.1f}°] => span {azimuth.max()-azimuth.min():.1f}°")
print(f"  Elevation: [{elevation.min():.1f}°, {elevation.max():.1f}°] => span {elevation.max()-elevation.min():.1f}°")

# Correlate row/col with azimuth/elevation
# Sort by row, check if azimuth or elevation corresponds to row
sample_idx = np.argsort(row)[:1000]
print(f"\n=== Row vs Azimuth/Elevation (sample) ===")
print(f"  row corr with azimuth:   {np.corrcoef(row[sample_idx], azimuth[sample_idx])[0,1]:.4f}")
print(f"  row corr with elevation: {np.corrcoef(row[sample_idx], elevation[sample_idx])[0,1]:.4f}")
sample_idx2 = np.argsort(col)[:1000]
print(f"  col corr with azimuth:   {np.corrcoef(col[sample_idx2], azimuth[sample_idx2])[0,1]:.4f}")
print(f"  col corr with elevation: {np.corrcoef(col[sample_idx2], elevation[sample_idx2])[0,1]:.4f}")

# Check color range
print(f"\n=== Color range ===")
print(f"  R: [{r.min()}, {r.max()}]  dtype: {r.dtype}")
print(f"  G: [{g.min()}, {g.max()}]  dtype: {g.dtype}")
print(f"  B: [{b.min()}, {b.max()}]  dtype: {b.dtype}")

# Scanner pose
print(f"\n=== Scanner pose (scan 0) ===")
print(f"  Translation: {np.round(header.translation, 6)}")
print(f"  Rotation:    {np.round(header.rotation, 6)}")

e57.close()
print("\nDone.")
