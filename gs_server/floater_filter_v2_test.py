import sys
sys.path.insert(0, '/home/anoma/gs_server/gs_server')
from floater_filter import _read_ply, _write_ply
import numpy as np
from scipy.spatial import cKDTree

MODEL = '/data/jobs/210ca694-26fa-49a5-968f-c1835367e2e6/output/model_30000.ply'
CLOUD = '/data/jobs/210ca694-26fa-49a5-968f-c1835367e2e6/inputs/hybrid_dataset/points3d.ply'
OUT_DIR = '/data/jobs/210ca694-26fa-49a5-968f-c1835367e2e6/output/'

_, cloud = _read_ply(CLOUD)
cloud_xyz = np.stack([cloud['x'], cloud['y'], cloud['z']], axis=1)
tree = cKDTree(cloud_xyz)
d2, _ = tree.query(cloud_xyz[:20000], k=2, workers=-1)
mean_spacing = d2[:, 1].mean()

header, data = _read_ply(MODEL)
xyz = np.stack([data['x'], data['y'], data['z']], axis=1)
scales = np.exp(np.stack([data['scale_0'], data['scale_1'], data['scale_2']], axis=1))
maxdim = scales.max(axis=1)
mindim = scales.min(axis=1)
aniso = maxdim / np.maximum(mindim, 1e-12)
dist, _ = tree.query(xyz, k=1, workers=-1)

BIG_BLOB = maxdim > 0.1

combos = [
    ('combo_conservative', 50, 4),
    ('combo_medium', 10, 4),
    ('combo_aggressive', 10, 2),
]

for name, a_t, mult in combos:
    thr = mult * mean_spacing
    needle = (aniso > a_t) & (dist > thr)
    remove = BIG_BLOB | needle
    filtered = data[~remove]
    out_path = OUT_DIR + f'model_30000_{name}.ply'
    _write_ply(out_path, header, filtered)
    print(f'{name}: aniso>{a_t} dist>{mult}xspacing({thr:.4f})  removed={remove.sum()} ({100*remove.sum()/len(data):.3f}%)  -> {out_path}')
