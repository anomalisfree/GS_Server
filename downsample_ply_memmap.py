import argparse
from pathlib import Path

import numpy as np


# PLY property type -> numpy dtype char
_PLY_TYPE_MAP = {
    'char': 'i1', 'int8': 'i1',
    'uchar': 'u1', 'uint8': 'u1',
    'short': 'i2', 'int16': 'i2',
    'ushort': 'u2', 'uint16': 'u2',
    'int': 'i4', 'int32': 'i4',
    'uint': 'u4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4',
    'double': 'f8', 'float64': 'f8',
}


def read_ply_header(path: Path):
    """Parse a binary_little_endian PLY header, returning (raw_header, nverts, dtype).

    dtype is built dynamically from the 'property <type> <name>' lines for the
    vertex element, so files with extra per-point properties (e.g. an
    scalar_Intensity float from CloudCompare/scanner exports) are read correctly
    instead of assuming a fixed xyz+rgb layout.
    """
    with path.open('rb') as f:
        header = bytearray()
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError('Unexpected EOF while reading PLY header')
            header.extend(line)
            if line.strip() == b'end_header':
                break
    text = header.decode('ascii', errors='strict')
    lines = text.splitlines()
    fmt = None
    nverts = None
    in_vertex_element = False
    fields = []  # list of (name, numpy_type_str)
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'format':
            fmt = parts[1]
        elif parts[0] == 'element':
            in_vertex_element = (parts[1] == 'vertex')
            if in_vertex_element:
                nverts = int(parts[2])
        elif parts[0] == 'property' and in_vertex_element:
            ply_type, name = parts[1], parts[2]
            np_type = _PLY_TYPE_MAP.get(ply_type)
            if np_type is None:
                raise RuntimeError(f'Unsupported PLY property type: {ply_type}')
            fields.append((name, np_type))
    if fmt != 'binary_little_endian':
        raise RuntimeError(f'Unsupported format: {fmt}')
    if nverts is None:
        raise RuntimeError('No vertex count in header')
    if not fields:
        raise RuntimeError('No vertex properties found in header')
    dtype = np.dtype([(name, f'<{t}') for name, t in fields])
    return header, nverts, dtype


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--dst', required=True)
    p.add_argument('--target', type=int, default=8_000_000)
    p.add_argument('--scale', type=float, default=1.0,
                    help='Uniform scale applied to x,y,z (e.g. 0.001 to convert mm to m)')
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    header, nverts, src_dtype = read_ply_header(src)
    src_fields = set(src_dtype.names)
    required = {'x', 'y', 'z', 'red', 'green', 'blue'}
    missing = required - src_fields
    if missing:
        raise RuntimeError(f'Source PLY is missing required properties: {sorted(missing)}')

    if nverts <= args.target:
        step = 1
    else:
        step = int(np.ceil(nverts / args.target))

    data = np.memmap(src, dtype=src_dtype, mode='r', offset=len(header), shape=(nverts,))
    sampled = data[::step]
    sampled_count = sampled.shape[0]

    # Output only xyz+rgb (Brush/Nerfstudio init doesn't need extra scanner
    # properties like intensity), cast to the canonical output dtype.
    out_dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    out = np.empty(sampled_count, dtype=out_dtype)
    for name in ('x', 'y', 'z'):
        out[name] = sampled[name].astype(np.float32) * np.float32(args.scale)
    for name in ('red', 'green', 'blue'):
        out[name] = sampled[name]

    new_header = (
        b'ply\n'
        b'format binary_little_endian 1.0\n'
        + f'element vertex {sampled_count}\n'.encode('ascii')
        + b'property float x\n'
        + b'property float y\n'
        + b'property float z\n'
        + b'property uchar red\n'
        + b'property uchar green\n'
        + b'property uchar blue\n'
        + b'end_header\n'
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('wb') as f:
        f.write(new_header)
        f.write(out.tobytes())

    print(f'source_vertices={nverts}')
    print(f'step={step}')
    print(f'sampled_vertices={sampled_count}')
    print(f'output={dst}')
    print(f'output_size_mb={dst.stat().st_size/1e6:.1f}')


if __name__ == '__main__':
    main()
