"""
Inspect E57 file structure - check scans, colors, embedded images, scanner poses
"""

import sys
import numpy as np
import pye57

E57_PATH = r"d:\VSProjects\GS_Sever\data\2026-06-23 e57 NicoOffice 1.51min 5m TD 5m Threshold export 1.e57"

def inspect_e57(path: str):
    print(f"\n{'='*60}")
    print(f"Inspecting: {path}")
    print(f"{'='*60}\n")

    e57 = pye57.E57(path)
    print(f"Total scans: {e57.scan_count}")

    for i in range(e57.scan_count):
        print(f"\n--- Scan {i} ---")
        header = e57.get_header(i)

        # Basic info
        print(f"  Point count: {header.point_count}")

        # Available fields
        fields = header.point_fields
        print(f"  Fields: {fields}")

        # Check for color
        has_color = any(f in fields for f in ('colorRed', 'colorGreen', 'colorBlue'))
        has_intensity = 'intensity' in fields
        has_xyz = all(f in fields for f in ('cartesianX', 'cartesianY', 'cartesianZ'))
        has_spherical = all(f in fields for f in ('sphericalRange', 'sphericalAzimuth', 'sphericalElevation'))
        has_row_col = 'rowIndex' in fields and 'columnIndex' in fields

        print(f"  Has XYZ:         {has_xyz}")
        print(f"  Has spherical:   {has_spherical}")
        print(f"  Has RGB color:   {has_color}")
        print(f"  Has intensity:   {has_intensity}")
        print(f"  Has row/col idx: {has_row_col}")

        # Scanner pose (rotation + translation)
        try:
            rotation = header.rotation
            translation = header.translation
            print(f"  Scanner pose:")
            print(f"    Translation: {np.round(translation, 4)}")
            print(f"    Rotation (quat): {np.round(rotation, 6)}")
        except Exception as ex:
            print(f"  Scanner pose: unavailable ({ex})")

        # Read small sample
        if has_xyz and header.point_count > 0:
            try:
                raw = e57.read_scan_raw(i)
                x = raw.get('cartesianX', np.array([]))
                print(f"  Sample X range: [{np.nanmin(x):.3f}, {np.nanmax(x):.3f}]")
            except Exception as ex:
                print(f"  Could not read sample: {ex}")

    # Check for embedded images (Images2D)
    print(f"\n--- Embedded Images ---")
    try:
        xml = e57.root.to_dict()
        images2d = xml.get('images2D', None)
        if images2d:
            print(f"  Found Images2D section: {images2d}")
        else:
            print("  No embedded Images2D found")
    except Exception as ex:
        print(f"  Could not check images: {ex}")

    e57.close()
    print(f"\n{'='*60}")
    print("Inspection complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else E57_PATH
    inspect_e57(path)
