"""
Post-training removal of "needle" floater Gaussians at object silhouette edges.

A calibration sweep of Brush's --opac-loss-weight / --scale-loss-weight showed
those loss terms don't target anisotropy at all: they just prune splat count
indiscriminately, and the survivors bloat in size to cover the resulting gaps
(see config.py BrushConfig.opac_decay comment). This module instead filters the
exported PLY directly after training, removing only Gaussians that are both
highly elongated (max/min scale ratio) AND large in absolute scene terms —
exactly the signature of a needle artifact sticking out into empty space.
Small anisotropic Gaussians (e.g. thin real edges) are left alone.
"""

import numpy as np

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
