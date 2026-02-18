"""
Beam-based lumbar spine simulation using SOFA.

Disc spacing is enforced by:
1. Beam FEM stiffness (resists compression/bending continuously)
2. RestShapeSprings on vertebra beam nodes (elastic floor toward CT rest pose)
3. Collision between endplate meshes with per-vertebra collision groups
   (group=v_idx means same-vertebra parts don't self-collide,
    but adjacent vertebrae DO collide with each other)

Coordinate axes:
  AP_AXIS  — anterior-posterior (hardcoded, normalized)
  SI_AXIS  — superior-inferior (computed from beam direction)
  LM_AXIS  — lateral-medial   (computed as AP x SI)

Per-vertebra forces are configured in VERTEBRA_FORCE_CONFIG below.
Each entry is a dict with keys:
  'magnitude' : float   — force magnitude per mesh vertex
  'axis'      : str     — 'AP', 'SI', or 'LM'
  'anterior'  : bool    — True = positive axis direction, False = negative
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation
from typing import List

# ============================================================
# CONFIGURATION
# ============================================================

MESH_DIR = r'C:\Users\elise\Documents\SpineSimulation\spine_folder'

MESH_FILES = {
    0: os.path.join(MESH_DIR, 'original_L1', 'L1_decimated.obj'),
    1: os.path.join(MESH_DIR, 'original_L2', 'L2_decimated.obj'),
    2: os.path.join(MESH_DIR, 'original_L3', 'L3_decimated.obj'),
    3: os.path.join(MESH_DIR, 'original_L4', 'L4_decimated.obj'),
}

FACET_MESH_FILES = {
    0: os.path.join(MESH_DIR, 'original_L1', 'L1_facet_decimated.obj'),
    1: os.path.join(MESH_DIR, 'original_L2', 'L2_facet_decimated.obj'),
    2: os.path.join(MESH_DIR, 'original_L3', 'L3_facet_decimated.obj'),
    3: os.path.join(MESH_DIR, 'original_L4', 'L4_facet_decimated.obj'),
}

# Endplate meshes — the disc-facing surfaces that should maintain spacing
ENDPLATE_MESH_FILES = {
    0: os.path.join(MESH_DIR, 'original_L1', 'L1_ends_decimated.obj'),
    1: os.path.join(MESH_DIR, 'original_L2', 'L2_ends_decimated.obj'),
    2: os.path.join(MESH_DIR, 'original_L3', 'L3_ends_decimated.obj'),
    3: os.path.join(MESH_DIR, 'original_L4', 'L4_ends_decimated.obj'),
}

VERTEBRA_CENTERS = [
    [-26.4200, 226.9736, -334.1898],  # L1 (v0)
    [-23.6073, 203.9596, -310.3250],  # L2 (v1)
    [-22.1764, 179.0056, -285.8145],  # L3 (v2)
    [-20.7015, 153.0243, -262.7382],  # L4 (v3)
]

# ============================================================
# ANATOMICAL AXES
# AP_AXIS is hardcoded; SI and LM are derived from it.
#
#   SI_AXIS: superior-inferior — computed from the overall beam direction
#            (L1 center -> L4 center), then orthogonalised against AP_AXIS
#   LM_AXIS: lateral-medial   — computed as cross(AP_AXIS, SI_AXIS)
#
# All three axes are unit vectors and mutually orthogonal.
# ============================================================

AP_AXIS_RAW = np.array([-0.07840471, 0.68894254, 0.72056289])


def _compute_anatomical_axes(vertebra_centers, ap_axis_raw):
    """Return (AP, SI, LM) unit vectors derived from the hardcoded AP axis."""
    ap = ap_axis_raw / np.linalg.norm(ap_axis_raw)

    # SI: overall cranio-caudal direction of the spine (L1 -> L4),
    # projected out of the AP direction so the three axes are orthogonal.
    spine_vec = np.array(vertebra_centers[-1]) - np.array(vertebra_centers[0])
    si_raw = spine_vec - np.dot(spine_vec, ap) * ap
    si = si_raw / np.linalg.norm(si_raw)

    # LM: right-hand cross product
    lm = np.cross(ap, si)
    lm = lm / np.linalg.norm(lm)

    return ap, si, lm


AP_AXIS, SI_AXIS, LM_AXIS = _compute_anatomical_axes(VERTEBRA_CENTERS, AP_AXIS_RAW)

print("Anatomical axes (unit vectors):")
print("  AP: %s" % np.round(AP_AXIS, 6).tolist())
print("  SI: %s" % np.round(SI_AXIS, 6).tolist())
print("  LM: %s" % np.round(LM_AXIS, 6).tolist())

# ============================================================
# PER-VERTEBRA FORCE CONFIGURATION
#
# Each vertebra entry has:
#   'magnitude' : force per mesh vertex (float)
#   'axis'      : which anatomical axis to push along
#                 'AP'  — anterior-posterior
#                 'SI'  — superior-inferior
#                 'LM'  — lateral-medial
#   'positive'  : True  = force in the positive axis direction
#                 False = force in the negative axis direction
#
# Examples:
#   Anterior push on L2:  axis='AP', positive=True
#   Caudal compression:   axis='SI', positive=False
#   Left lateral bend:    axis='LM', positive=True  (or False for right)
# ============================================================

VERTEBRA_FORCE_CONFIG = {
    0: {'magnitude': 0.5, 'axis': 'LM', 'positive': False},   # L1
    1: {'magnitude': 1.2, 'axis': 'LM', 'positive': False},   # L2
    2: {'magnitude': 1.2, 'axis': 'LM', 'positive': False},   # L3
    3: {'magnitude': 0.5, 'axis': 'LM', 'positive': False}   # L4
}

# ============================================================
# BEAM PARAMETERS
# ============================================================
BEAM_YOUNG_MODULUS = 5e3
BEAM_POISSON_RATIO = 0.45
BEAM_RADIUS = 15.0
BEAM_RADIUS_INNER = 0.0

# Anchor extension scale: 1.0 = one vertebra-spacing, 1.5 = 1.5x, etc.
ANCHOR_SCALE = 1.5

# RestShapeSprings
REST_SHAPE_STIFFNESS = 500.0
REST_SHAPE_ANGULAR_STIFFNESS = 500.0

# ============================================================
# VTK EXPORT — set to True to save deformed meshes
# ============================================================
SAVE_VTK = True
VTK_OUTPUT_DIR = r'C:\Users\elise\Documents\SpineSimulation\spine_folder\deformed'
VTK_EXPORT_EVERY_N_STEPS = 20

# ============================================================
# DISC SPACING — per-vertebra collision proximity
# ============================================================
VERTEBRA_PROXIMITY = {
    0: 1.175,
    1: 1.395,
    2: 1.405,
    3: 1.185,
}

# ============================================================
# GEOMETRY UTILITIES
# ============================================================

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def make_force_vec3(cfg):
    """
    Build a 3-element force vector from a vertebra force config dict.

    cfg keys:
        'magnitude' : float
        'axis'      : 'AP' | 'SI' | 'LM'
        'positive'  : bool
    """
    axis_map = {'AP': AP_AXIS, 'SI': SI_AXIS, 'LM': LM_AXIS}
    axis_name = cfg.get('axis', 'AP').upper()
    if axis_name not in axis_map:
        raise ValueError("Unknown axis '%s'. Choose from AP, SI, LM." % axis_name)

    direction = 1.0 if cfg.get('positive', True) else -1.0
    fv = direction * cfg['magnitude'] * axis_map[axis_name]
    return [float(fv[0]), float(fv[1]), float(fv[2])]


def compute_quaternions_for_path(points):
    points = np.array(points)
    n = len(points)
    quaternions = np.zeros((n, 4))
    world_basis = np.eye(3)
    prev_up = None

    for i in range(n):
        if i < n - 1:
            x_axis = normalize(points[i + 1] - points[i])
        else:
            quaternions[i] = quaternions[i - 1]
            continue

        if i == 0:
            dots = np.abs([np.dot(x_axis, b) for b in world_basis])
            candidate_up = world_basis[np.argmin(dots)].copy()
        else:
            candidate_up = prev_up

        y_axis = normalize(candidate_up - np.dot(candidate_up, x_axis) * x_axis)
        z_axis = normalize(np.cross(x_axis, y_axis))

        rot_mat = np.column_stack([x_axis, y_axis, z_axis])
        rot = Rotation.from_matrix(rot_mat)
        quaternions[i] = rot.as_quat(scalar_first=False)
        prev_up = y_axis

    return quaternions


def build_beam_topology(centers):
    centers = [np.array(c) for c in centers]
    anchor_start = centers[0]  + ANCHOR_SCALE * (centers[0]  - centers[1])
    anchor_end   = centers[-1] + ANCHOR_SCALE * (centers[-1] - centers[-2])
    extended = [anchor_start] + centers + [anchor_end]

    refined = []
    for i in range(len(extended) - 1):
        refined.append(extended[i].tolist())
        mid = ((extended[i] + extended[i + 1]) / 2.0).tolist()
        refined.append(mid)
    refined.append(extended[-1].tolist())

    edges = []
    for i in range(len(refined) - 1):
        edges.extend([i, i + 1])

    quats = compute_quaternions_for_path(refined)
    positions = []
    for i in range(len(refined)):
        positions.append([*refined[i], *quats[i]])

    vertebra_node_indices = {}
    for v_idx in range(len(centers)):
        vertebra_node_indices[v_idx] = 2 * (v_idx + 1)

    return positions, edges, vertebra_node_indices, refined


# ============================================================
# SCENE CREATION
# ============================================================

def createScene(root):
    root.name = "root"
    root.dt = 0.01
    root.gravity = [0, 0, 0]

    # --- Plugins ---
    plugins = [
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.Collision.Detection.Algorithm",
        "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Geometry",
        "Sofa.Component.Collision.Response.Contact",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Constraint.Lagrangian.Solver",
        "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.IO.Mesh",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.Mapping.Linear",
        "Sofa.Component.Mass",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Visual",
    ]
    for p in plugins:
        root.addObject('RequiredPlugin', name=p)

    root.addObject('VisualStyle', displayFlags=(
        "showBehaviorModels showCollisionModels showVisual "
        "showInteractionForceFields"
    ))

    # --- Animation + Constraints ---
    root.addObject('FreeMotionAnimationLoop')
    root.addObject('GenericConstraintSolver',
                   maxIterations=1000, tolerance=1.0e-6)

    # --- Collision pipeline ---
    root.addObject('CollisionPipeline', depth=6, verbose=0, draw=0)
    root.addObject('BruteForceBroadPhase')
    root.addObject('BVHNarrowPhase')
    root.addObject('MinProximityIntersection', name="Proximity",
                   alarmDistance=5.0, contactDistance=3.0)
    root.addObject('CollisionResponse', name="Response",
                   response="FrictionContactConstraint",
                   responseParams="mu=0.4")

    # --- VTK output directory ---
    if SAVE_VTK:
        os.makedirs(VTK_OUTPUT_DIR, exist_ok=True)
        print("VTK export ENABLED -> " + VTK_OUTPUT_DIR)
    else:
        print("VTK export DISABLED (set SAVE_VTK = True to enable)")

    # ==========================================================
    # BEAM NODE
    # ==========================================================
    beam_node = root.addChild('SpineBeam')

    beam_node.addObject('EulerImplicitSolver',
                        rayleighStiffness=0.1,
                        rayleighMass=0.1,
                        printLog=False)
    beam_node.addObject('BTDLinearSolver',
                        template="BTDMatrix6d",
                        printLog=False, verbose=False)

    # --- Build beam ---
    positions, edges, vert_node_map, refined_points = build_beam_topology(
        VERTEBRA_CENTERS
    )
    n_beam_nodes = len(positions)
    print("Beam chain: %d nodes" % n_beam_nodes)
    print("Vertebra -> beam node mapping: %s" % str(vert_node_map))

    beam_node.addObject('MeshTopology', name="BeamTopo", lines=edges)
    beam_node.addObject('MechanicalObject',
                        template="Rigid3", name="DOFs",
                        position=positions)
    beam_node.addObject('UniformMass',
                        vertexMass="1 1 0.01 0 0 0 0.1 0 0 0 0.1",
                        printLog=False)

    # --- Beam FEM ---
    beam_node.addObject('BeamFEMForceField',
                        name="DiscBeamFEM",
                        radius=BEAM_RADIUS,
                        radiusInner=BEAM_RADIUS_INNER,
                        youngModulus=BEAM_YOUNG_MODULUS,
                        poissonRatio=BEAM_POISSON_RATIO)

    # --- RestShapeSprings on vertebra nodes ---
    vertebra_beam_indices = [vert_node_map[i] for i in range(4)]
    beam_node.addObject('RestShapeSpringsForceField',
                        name="DiscSpacing",
                        points=vertebra_beam_indices,
                        stiffness=REST_SHAPE_STIFFNESS,
                        angularStiffness=REST_SHAPE_ANGULAR_STIFFNESS)

    # --- Fix anchor endpoints only ---
    fixed_indices = [0, n_beam_nodes - 1]
    beam_node.addObject('FixedProjectiveConstraint',
                        name="FixAnchors",
                        indices=fixed_indices)

    # --- Constraint correction ---
    beam_node.addObject('LinearSolverConstraintCorrection')

    # ==========================================================
    # VERTEBRAE
    # ==========================================================
    colors = [
        [0.2, 0.4, 0.8, 1.0],
        [0.2, 0.7, 0.4, 1.0],
        [0.8, 0.6, 0.2, 1.0],
        [0.7, 0.2, 0.2, 1.0],
    ]

    for v_idx in range(4):
        beam_idx = vert_node_map[v_idx]
        force_cfg = VERTEBRA_FORCE_CONFIG[v_idx]
        fc = make_force_vec3(force_cfg)
        prox = VERTEBRA_PROXIMITY[v_idx]

        # --- Mechanical mesh (forces applied here) ---
        mecha_node = beam_node.addChild("Vertebra_%d" % v_idx)
        mecha_node.addObject('MeshOBJLoader', name="loader",
                             filename=MESH_FILES[v_idx],
                             printLog=False, flipNormals=0)
        mecha_node.addObject('MeshTopology', name="topology",
                             src="@loader")
        mecha_node.addObject('MechanicalObject', name="dofs",
                             template="Vec3d", src="@loader")

        n_verts = len(mecha_node.dofs.position.value)
        forces_vec = np.zeros((n_verts, 3))
        forces_vec[:, :] = fc[:3]

        mecha_node.addObject('ConstantForceField',
                             name='CFF',
                             forces=forces_vec.tolist())

        # RigidMapping to beam
        mecha_node.addObject('RigidMapping',
                             index=beam_idx,
                             globalToLocalCoords=True,
                             input='@..', output='@.')

        # --- Visual ---
        vis_node = mecha_node.addChild('Visual')
        vis_node.addObject('OglModel', name="ogl",
                           src="@../loader", color=colors[v_idx])
        vis_node.addObject('IdentityMapping')

        # --- VTK Export ---
        # Press Ctrl+E in the SOFA GUI to save the current deformed state.
        # Files saved as: L1_deformed_0.vtu, L2_deformed_0.vtu, etc.
        # Open in ParaView: Extract Surface filter to get a polygonal mesh.
        if SAVE_VTK:
            vtk_path = os.path.join(VTK_OUTPUT_DIR, "L%d_deformed_" % (v_idx + 1))
            mecha_node.addObject('VTKExporter',
                                 filename=vtk_path,
                                 listening=True,
                                 edges=0,
                                 triangles=1,
                                 quads=0,
                                 tetras=0,
                                 pointsDataFields='dofs.position',
                                 exportAtEnd=True,
                                 exportEveryNumberOfSteps=0)

        # --- Endplate collision ---
        ep_file = ENDPLATE_MESH_FILES.get(v_idx)
        if ep_file and os.path.exists(ep_file):
            ep_node = beam_node.addChild("Endplate_%d" % v_idx)
            ep_node.addObject('MeshOBJLoader', name="loader",
                              filename=ep_file)
            ep_node.addObject('MeshTopology', name="Topo",
                              src="@loader")
            ep_node.addObject('MechanicalObject', name="CollDofs",
                              template="Vec3d", src="@loader")
            ep_node.addObject('TriangleCollisionModel',
                              name="Triangles",
                              proximity=prox,
                              group=v_idx,
                              contactStiffness=5000)
            ep_node.addObject('LineCollisionModel',
                              name="Lines",
                              proximity=prox,
                              group=v_idx)
            ep_node.addObject('PointCollisionModel',
                              name="Points",
                              proximity=prox,
                              group=v_idx)
            ep_node.addObject('RigidMapping',
                              index=beam_idx,
                              globalToLocalCoords=True)

        # --- Facet joint collision ---
        fj_file = FACET_MESH_FILES.get(v_idx)
        if fj_file and os.path.exists(fj_file):
            fj_node = beam_node.addChild("FacetJoint_%d" % v_idx)
            fj_node.addObject('MeshOBJLoader', name="loader",
                              filename=fj_file)
            fj_node.addObject('MeshTopology', name="Topo",
                              src="@loader")
            fj_node.addObject('MechanicalObject', name="CollDofs",
                              template="Vec3d", src="@loader")
            fj_node.addObject('TriangleCollisionModel',
                              name="Triangles",
                              proximity=0.3,
                              group=v_idx,
                              contactStiffness=5000)
            fj_node.addObject('LineCollisionModel',
                              name="Lines",
                              proximity=0.3,
                              group=v_idx)
            fj_node.addObject('PointCollisionModel',
                              name="Points",
                              proximity=0.3,
                              group=v_idx)
            fj_node.addObject('RigidMapping',
                              index=beam_idx,
                              globalToLocalCoords=True)

        print("  v%d: beam=%d, verts=%d, force=%.3f along %s (%s)" % (
            v_idx, beam_idx, n_verts,
            force_cfg['magnitude'],
            force_cfg['axis'],
            '+' if force_cfg.get('positive', True) else '-'
        ))

    # ==========================================================
    print("\n--- Scene Summary ---")
    print("  Beam: E=%s, R=%s, nu=%s" % (BEAM_YOUNG_MODULUS, BEAM_RADIUS, BEAM_POISSON_RATIO))
    print("  RestShapeSprings: k=%s, k_ang=%s" % (REST_SHAPE_STIFFNESS, REST_SHAPE_ANGULAR_STIFFNESS))
    print("  Fixed anchors: %s (scale=%.2f)" % (str(fixed_indices), ANCHOR_SCALE))
    print("  Vertebra beam nodes: %s" % str(vertebra_beam_indices))
    if SAVE_VTK:
        print("  VTK saving to: %s (every %d steps)" % (VTK_OUTPUT_DIR, VTK_EXPORT_EVERY_N_STEPS))

    return root


# Run with: runSofa -l SofaPython3 deform_beam.py