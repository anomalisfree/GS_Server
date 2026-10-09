"""
Post-training removal of floater Gaussians from an exported PLY.

Two independent filter passes, chained in brush_runner.py (needle filter
first, then cluster filter - each writes to the same "_filtered.ply" output):

- filter_needle_floaters: removes Gaussians that are both highly elongated
  (max/min scale ratio) and large in absolute scene terms - a splat's own
  shape, judged in isolation. A calibration sweep of Brush's
  --opac-loss-weight / --scale-loss-weight showed those loss terms don't
  target anisotropy at all: they just prune splat count indiscriminately,
  and the survivors bloat in size to cover the resulting gaps (see config.py
  BrushConfig.opac_decay comment) - this filters the export directly instead.

- filter_isolated_clusters: removes small clusters of Gaussians that are
  spatially disconnected from the main scanned structure - scattered debris
  around the edge of the scan volume. The needle filter has no notion of
  spatial isolation (a splat can be perfectly ordinary in shape/size and
  still be junk if it's sitting alone in empty space far from everything
  else), so this population survives it entirely.

- filter_photometric_inconsistency: a third, independent pass added after
  investigating the AU_Las_Vegas "floating tile" artifact, which survived
  BOTH filters above plus two training-time regularizers (position-anchor,
  aniso-reg) - see that investigation report. Diagnosis found the offending
  splats were only moderately anisotropic and moderately larger than normal
  (not the extreme outliers either filter above targets), but consistently
  had a much higher live-training photometric residual (see brush-train's
  photo_consistency_enabled) than their surroundings - i.e. they never
  converge to a stable color/position because they're being asked to
  explain inconsistent observations from different viewing angles. This is
  the first signal in this investigation that actually separates the known
  floaters cleanly (median 99.5th percentile of the model's own distribution,
  vs no separation at all for every geometric signal tried). Only meaningful
  for models trained with photo_consistency_enabled (no-ops otherwise, since
  the PLY won't have the required properties).
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

_TYPE_MAP = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "int": "<i4", "int32": "<i4", "short": "<i2", "int16": "<i2",
    "ushort": "<u2", "uint16": "<u2", "uint": "<u4", "uint32": "<u4",
}


def _read_ply(path):
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: unexpected EOF before end_header")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        data_offset = f.tell()
        fmt = None
        props = []
        nverts = None
        in_vertex = False
        for line in header_lines:
            s = line.decode("ascii", errors="ignore").strip()
            if s.startswith("format"):
                fmt = s.split()[1]
            elif s.startswith("element"):
                parts = s.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    nverts = int(parts[2])
            elif s.startswith("property") and in_vertex:
                parts = s.split()
                props.append((parts[2], parts[1]))
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: unsupported PLY format {fmt!r}")
        dtype = np.dtype([(name, _TYPE_MAP[t]) for name, t in props])
        f.seek(data_offset)
        data = np.fromfile(f, dtype=dtype, count=nverts)
    return header_lines, data


def _write_ply(path, header_lines, data):
    header_lines = list(header_lines)
    for i, line in enumerate(header_lines):
        if line.strip().startswith(b"element vertex"):
            header_lines[i] = f"element vertex {len(data)}\n".encode("ascii")
            break
    with open(path, "wb") as f:
        f.write(b"".join(header_lines))
        data.tofile(f)


def filter_needle_floaters(in_path, out_path, aniso_threshold=20000.0, size_multiplier=15.0):
    """Remove Gaussians with (max/min scale > aniso_threshold) AND (max scale > K
    times this model's own median splat size).

    Size is relative, not absolute: a fixed absolute cutoff (e.g. 10cm) was
    validated on a densely-initialized (LiDAR) model where the median splat is
    ~0.007 units, so 10cm was a clear outlier tail. Fed a sparsely-initialized
    model (e.g. colmap_images on a fragmented reconstruction, median splat size
    ~0.13 units for one observed case), the same absolute cutoff sat *below* the
    median and threw away 59% of the model - splats that are perfectly normal
    for that model's own point density. Scaling the threshold off the model's own
    median size self-calibrates regardless of reconstruction density or scene
    scale, and reproduces the same ~0.5% removal rate on the reference model that
    the original 10cm cutoff was tuned against (size_multiplier=15 ~= 10cm /
    that model's median of ~0.0068).

    Returns a stats dict. If the PLY has no scale_0/1/2 properties (not a Gaussian
    splat export), copies the file through unchanged and returns skipped=True.
    """
    header_lines, data = _read_ply(in_path)

    if not {"scale_0", "scale_1", "scale_2"} <= set(data.dtype.names):
        _write_ply(out_path, header_lines, data)
        return {"skipped": True, "input_count": int(len(data))}

    scales = np.exp(np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1))
    maxdim = scales.max(axis=1)
    mindim = scales.min(axis=1)
    aniso = maxdim / np.maximum(mindim, 1e-12)

    min_size = size_multiplier * float(np.median(maxdim))
    remove = (aniso > aniso_threshold) & (maxdim > min_size)
    filtered = data[~remove]

    _write_ply(out_path, header_lines, filtered)

    n = len(data)
    return {
        "skipped": False,
        "input_count": int(n),
        "output_count": int(len(filtered)),
        "removed_count": int(remove.sum()),
        "removed_pct": round(100.0 * remove.sum() / n, 3) if n else 0.0,
        "aniso_threshold": aniso_threshold,
        "size_multiplier": size_multiplier,
        "median_maxdim": float(np.median(maxdim)),
        "effective_min_size": min_size,
    }


def filter_isolated_clusters(in_path, out_path, k=8, radius_multiplier=5.0, bbox_padding=None):
    """Remove Gaussians that sit in small clusters spatially disconnected from
    the main scanned structure - scattered debris around the edge of the scan
    volume that filter_needle_floaters doesn't catch, since it only looks at
    a splat's own shape (anisotropy/size), never whether it's isolated in
    space.

    Diagnosed against a real job (colmap_images, ~310k splats): a naive
    "keep only the largest connected component" is WRONG - a real, coherent
    ~19% chunk of that model came back as a *separate* connected component
    purely because of a locally sparse/thin bridging region, not because it's
    debris (confirmed: it sat 100% inside the main component's own bounding
    box). So this only removes a non-main component if it lies entirely
    OUTSIDE the main component's (padded) bounding box - i.e. genuinely
    beyond the scanned volume, not just disconnected by a local density gap
    within it. On that same job this correctly isolated ~1% of the model
    (~1000 tiny clusters scattered just past the scan boundary) while leaving
    the large internal chunk untouched.

    `radius_multiplier` sets the connectivity radius as a multiple of the
    model's own median nearest-neighbor spacing (self-calibrating across
    scene scale/density, same rationale as filter_needle_floaters' size
    cutoff). `bbox_padding` defaults to 2x that radius - a small safety
    margin so real structure just past the sampled bounding box edge isn't
    clipped.

    Returns a stats dict. If the PLY has no scale_0/1/2 properties, or has
    too few points to build a meaningful connectivity graph, copies the file
    through unchanged and returns skipped=True.
    """
    header_lines, data = _read_ply(in_path)

    if not {"scale_0", "scale_1", "scale_2"} <= set(data.dtype.names):
        _write_ply(out_path, header_lines, data)
        return {"skipped": True, "input_count": int(len(data)), "reason": "not a gaussian splat export"}

    n = len(data)
    if n < 1000:
        _write_ply(out_path, header_lines, data)
        return {"skipped": True, "input_count": n, "reason": "too few points"}

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)

    tree = cKDTree(xyz)
    k_query = min(k + 1, n)
    dists, idxs = tree.query(xyz, k=k_query, workers=-1)
    knn_dist = dists[:, 1:].mean(axis=1) if k_query > 1 else np.zeros(n)
    median_nn = float(np.median(knn_dist))
    radius = median_nn * radius_multiplier
    if bbox_padding is None:
        bbox_padding = radius * 2.0

    # Connectivity graph built directly from the k-NN query above (at most
    # n*k edges, regardless of local point density) rather than a radius-ball
    # query_pairs search - the latter's edge count is unbounded and blew up
    # memory on a real multi-million-splat dense model (a single dense pocket
    # can contribute tens of millions of pairs on its own). Edges beyond the
    # connectivity radius are dropped so a sparse/isolated point doesn't get
    # spuriously linked to its k-th nearest neighbor just because it exists,
    # even when that neighbor is actually far away.
    if k_query <= 1:
        _write_ply(out_path, header_lines, data)
        return {"skipped": True, "input_count": n, "reason": "too few points for k-NN graph"}
    src = np.repeat(np.arange(n), k_query - 1)
    dst = idxs[:, 1:].ravel()
    edge_dist = dists[:, 1:].ravel()
    keep_edge = edge_dist <= radius
    src, dst = src[keep_edge], dst[keep_edge]
    if src.size == 0:
        _write_ply(out_path, header_lines, data)
        return {"skipped": True, "input_count": n, "reason": "degenerate connectivity graph"}

    graph = csr_matrix((np.ones(src.size), (src, dst)), shape=(n, n))
    ncomp, labels = connected_components(graph, directed=False)
    comp_sizes = np.bincount(labels)
    main_label = int(np.argmax(comp_sizes))

    main_pts = xyz[labels == main_label]
    bbmin = main_pts.min(axis=0) - bbox_padding
    bbmax = main_pts.max(axis=0) + bbox_padding
    inside_main_bbox = np.all((xyz >= bbmin) & (xyz <= bbmax), axis=1)

    remove = (labels != main_label) & ~inside_main_bbox
    filtered = data[~remove]

    _write_ply(out_path, header_lines, filtered)

    return {
        "skipped": False,
        "input_count": n,
        "output_count": int(len(filtered)),
        "removed_count": int(remove.sum()),
        "removed_pct": round(100.0 * remove.sum() / n, 3) if n else 0.0,
        "n_components": int(ncomp),
        "main_component_frac": round(float(comp_sizes[main_label]) / n, 4),
        "radius": radius,
        "bbox_padding": bbox_padding,
    }


def filter_photometric_inconsistency(in_path, out_path, percentile=97.5, min_observations=30):
    """Remove Gaussians whose live-training photo-consistency residual
    (photo_resid_avg, accumulated by Brush from photo_consistency_warmup_steps
    onward - see photo_consistency_enabled) sits in the extreme tail of THIS
    model's own distribution.

    The threshold is a percentile of this model's own photo_resid_avg
    distribution rather than an absolute value - same self-calibrating
    rationale as filter_needle_floaters' size cutoff (residual magnitude
    depends on scene content/texture density, an absolute cutoff tuned on
    one model would misfire on another). Validated against the AU_Las_Vegas
    floating-tile artifact: known floaters landed at a median 99.5th
    percentile here, vs an even spread across the whole distribution for
    filter_needle_floaters'/filter_isolated_clusters' geometric signals -
    the cleanest separation found in that investigation, though still not a
    clean 100% cut (~60-80% recall depending on percentile chosen), so this
    is a genuine probabilistic filter, not a hard rule.

    `min_observations` guards against noisy stats from splats only sampled a
    handful of times (e.g. created very late in training, or rarely
    visible) - their average residual isn't trustworthy either way, so they
    are left untouched regardless of their nominal value.

    Returns a stats dict. If the PLY has no photo_resid_avg/photo_resid_count
    properties (the model wasn't trained with photo_consistency_enabled), or
    too few splats have enough observations to calibrate a threshold, copies
    the file through unchanged and returns skipped=True.
    """
    header_lines, data = _read_ply(in_path)

    if not {"photo_resid_avg", "photo_resid_count"} <= set(data.dtype.names):
        _write_ply(out_path, header_lines, data)
        return {
            "skipped": True,
            "input_count": int(len(data)),
            "reason": "no photo_resid_avg/photo_resid_count properties (photo_consistency_enabled was off)",
        }

    n = len(data)
    count = data["photo_resid_count"]
    avg = data["photo_resid_avg"]
    has_enough = count >= min_observations

    if has_enough.sum() < 100:
        _write_ply(out_path, header_lines, data)
        return {"skipped": True, "input_count": n, "reason": "too few splats with enough observations to calibrate"}

    threshold = float(np.percentile(avg[has_enough], percentile))
    remove = has_enough & (avg > threshold)
    filtered = data[~remove]

    _write_ply(out_path, header_lines, filtered)

    return {
        "skipped": False,
        "input_count": n,
        "output_count": int(len(filtered)),
        "removed_count": int(remove.sum()),
        "removed_pct": round(100.0 * remove.sum() / n, 3) if n else 0.0,
        "percentile": percentile,
        "threshold": threshold,
        "min_observations": min_observations,
        "splats_with_enough_observations": int(has_enough.sum()),
    }


if __name__ == "__main__":
    import sys
    import json

    in_path, out_path = sys.argv[1], sys.argv[2]
    kwargs = {}
    if len(sys.argv) > 3:
        kwargs["aniso_threshold"] = float(sys.argv[3])
    if len(sys.argv) > 4:
        kwargs["size_multiplier"] = float(sys.argv[4])
    stats = filter_needle_floaters(in_path, out_path, **kwargs)
    print(json.dumps(stats, indent=2))
