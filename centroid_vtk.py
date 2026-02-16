"""
Compute centroids of vertebral body VTK meshes (L1-L4).
"""

import vtk
import os

DATA_DIR = r"C:\Users\elise\Documents\SpineSimulation\vert_data"

def compute_centroid(vtk_path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_path)
    reader.Update()

    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        # Try unstructured grid reader as fallback
        reader2 = vtk.vtkUnstructuredGridReader()
        reader2.SetFileName(vtk_path)
        reader2.Update()
        data = reader2.GetOutput()
    else:
        data = polydata

    com = vtk.vtkCenterOfMass()
    com.SetInputData(data)
    com.SetUseScalarsAsWeights(False)
    com.Update()

    return com.GetCenter()


if __name__ == "__main__":
    for level in ["L1", "L2", "L3", "L4"]:
        filename = f"{level}_body.vtk"
        filepath = os.path.join(DATA_DIR, filename)

        if not os.path.exists(filepath):
            print(f"[WARNING] {filepath} not found, skipping.")
            continue

        cx, cy, cz = compute_centroid(filepath)
        print(f"{level}: centroid = ({cx:.4f}, {cy:.4f}, {cz:.4f})")