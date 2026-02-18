"""
Compute per-vertebra rigid transforms from
original to deformed OBJ meshes and save as ITK .tfm files for Slicer.

How to run:
    python compute_rigid_transform.py \
        --original_dir /path/to/original_sofa \
        --deformed_dir /path/to/sofa_1 \
        --output_dir /path/to/output_transforms
"""

import numpy as np
import trimesh
import os
import argparse
import glob


def rigid_transform_3d(A, B):
    """
    Compute rigid transform (R, t) such that B ≈ R @ A + t.
    Uses SVD-based Procrustes alignment.
    """
    assert A.shape == B.shape

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = A_centered.T @ B_centered

    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det = +1, not reflection)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    t = centroid_B - R @ centroid_A

    residuals = (R @ A.T).T + t - B
    rmse = np.sqrt((residuals ** 2).sum(axis=1).mean())

    return R, t, rmse


def make_4x4(R, t):
    """Combine R (3x3) and t (3,) into a 4x4 homogeneous matrix."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def save_itk_tfm(filepath, R, t):
    """
    Save rigid transform as ITK .tfm for 3D Slicer.

    ITK stores the transform FROM output space TO input space (i.e. the inverse
    of the forward original->deformed transform), so we invert R and t before writing.
    """
    R_itk = R.T
    t_itk = -R.T @ t

    with open(filepath, 'w') as f:
        f.write('#Insight Transform File V1.0\n')
        f.write('#Transform 0\n')
        f.write('Transform: AffineTransform_double_3_3\n')
        params = list(R_itk.flatten()) + list(t_itk)
        f.write('Parameters: ' + ' '.join(f'{v:.10f}' for v in params) + '\n')
        f.write('FixedParameters: 0 0 0\n')


def compute_rigid_transform(original_obj, deformed_obj, output_path):
    orig_mesh = trimesh.load(original_obj, process=False, force='mesh')
    def_mesh  = trimesh.load(deformed_obj,  process=False, force='mesh')

    orig_pts = np.array(orig_mesh.vertices)
    def_pts  = np.array(def_mesh.vertices)

    if orig_pts.shape != def_pts.shape:
        raise ValueError(
            f"Vertex count mismatch: original has {orig_pts.shape[0]}, "
            f"deformed has {def_pts.shape[0]}. "
            f"Meshes must have the same topology (paired vertices)."
        )

    R, t, rmse = rigid_transform_3d(orig_pts, def_pts)
    T = make_4x4(R, t)

    if not output_path.endswith('.tfm'):
        output_path = output_path + '.tfm'

    save_itk_tfm(output_path, R, t)

    return T, rmse


def process_all(original_dir, deformed_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    deformed_files = sorted(glob.glob(os.path.join(deformed_dir, '*.obj')))

    results = {}
    for def_file in deformed_files:
        def_name = os.path.basename(def_file)
        label = def_name.split('_')[0]

        orig_candidates = glob.glob(os.path.join(original_dir, f'{label}_*.obj'))
        if not orig_candidates:
            print(f"WARNING: No original mesh found for {label}, skipping.")
            continue

        orig_file = orig_candidates[0]
        output_path = os.path.join(output_dir, f'{label}_rigid_transform')

        print(f"\n{'='*60}")
        print(f"Vertebra: {label}")
        print(f"  Original: {orig_file}")
        print(f"  Deformed: {def_file}")

        try:
            T, rmse = compute_rigid_transform(orig_file, def_file, output_path)
            print(f"  RMSE:   {rmse:.6f} mm")
            print(f"  Saved:  {output_path}.tfm")
            print(f"  4x4:\n{T}")
            results[label] = {'T': T, 'rmse': rmse}
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for label, res in sorted(results.items()):
        print(f"  {label}: RMSE = {res['rmse']:.6f}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute per-vertebra rigid transforms and save as .tfm for 3D Slicer"
    )
    parser.add_argument('--original_dir', required=True)
    parser.add_argument('--deformed_dir',  required=True)
    parser.add_argument('--output_dir',    required=True)

    args = parser.parse_args()
    process_all(args.original_dir, args.deformed_dir, args.output_dir)