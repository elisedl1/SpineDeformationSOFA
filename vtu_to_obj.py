import meshio
import numpy as np
import os
import argparse
import glob
import sys


def vtuToObj(file_path):
    """
    Transform vtu file resulted from sofa framework to obj file
    :param file_path:
    :return:
    """
    mesh_vtu = meshio.read(file_path)

    mesh = meshio.Mesh(
       # mesh_vtu.points * 1e3,
        mesh_vtu.points,
        mesh_vtu.cells,
        # Optionally provide extra data on points, cells, etc.
        mesh_vtu.point_data,
        # Each item in cell data must match the cells array
        mesh_vtu.cell_data,
    )

    dst_filename = file_path.replace(".vtu", ".obj")
    mesh.write(dst_filename)
    return dst_filename


def process_directory(dir_path):
    """
    Process all .vtu files in a given directory.
    :param dir_path: Path to directory containing .vtu files
    """
    vtu_files = glob.glob(os.path.join(dir_path, "*.vtu"))

    if not vtu_files:
        print(f"No .vtu files found in: {dir_path}")
        return

    print(f"Found {len(vtu_files)} .vtu file(s) in: {dir_path}\n")

    success, failed = 0, 0
    for file_path in sorted(vtu_files):
        try:
            dst = vtuToObj(file_path)
            print(f"  ✓ {os.path.basename(file_path)} → {os.path.basename(dst)}")
            success += 1
        except Exception as e:
            print(f"  ✗ {os.path.basename(file_path)} — ERROR: {e}")
            failed += 1

    print(f"\nDone. {success} converted, {failed} failed.")


# File path
dir_path = "/Users/elise/elisedonszelmann-lund/Masters_Utils/Pig_Data/pig2/Registration/CT_segmentations/sofa_1"

process_directory(dir_path)