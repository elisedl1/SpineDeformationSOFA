import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Tuple
import matplotlib.pyplot as plt

cmap = plt.cm.viridis




def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def compute_quaternions_for_path(points: List[List[float]]) -> np.ndarray:
    """
    Compute quaternions for each point in a 3D path.

    The algorithm:
    1. For the first point: compute X-axis from point direction, then find the world
       basis vector with smallest scalar product to X, use it as initial up vector
    2. For subsequent points: use the previous up vector, orthonormalize it against
       the new X-axis to maintain consistency
    3. Compute Z-axis (right) via cross product of X and Y

    Args:
        points: List of 3D points as tuples (x, y, z)

    Returns:
        Array of quaternions with shape (n, 4) where each quaternion is [x, y, z, w]
    """
    points = np.array(points)
    n_points = len(points)
    quaternions = np.zeros((n_points, 4))

    # World basis vectors
    world_basis = np.array([
        [1.0, 0.0, 0.0],  # X
        [0.0, 1.0, 0.0],  # Y
        [0.0, 0.0, 1.0]   # Z
    ])

    previous_up_vector = None  # Will be computed for first point, then propagated

    for i in range(n_points):
        if i < n_points - 1:
            # X-axis (forward direction): from current point to next point
            x_axis = points[i + 1] - points[i]
            x_axis = normalize(x_axis)
        else:
            # Last point: copy the previous quaternion
            quaternions[i] = quaternions[i - 1]
            continue

        if i == 0:
            # First point: find world basis vector with smallest scalar product to X
            dot_products = np.abs([np.dot(x_axis, basis) for basis in world_basis])
            min_idx = np.argmin(dot_products)
            candidate_up_vector = world_basis[min_idx].copy()
        else:
            # Subsequent points: use the previous up vector
            candidate_up_vector = previous_up_vector

        # Orthonormalize candidate_up_vector with respect to x_axis
        # Remove the component of candidate_up_vector that's parallel to x_axis
        orthogonal_component = candidate_up_vector - np.dot(candidate_up_vector, x_axis) * x_axis
        y_axis = normalize(orthogonal_component)

        # Z-axis (right): cross product of X and Y
        z_axis = np.cross(x_axis, y_axis)
        z_axis = normalize(z_axis)

        # Build rotation matrix with columns [X, Y, Z]
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

        # Convert rotation matrix to quaternion using scipy
        rot = Rotation.from_matrix(rotation_matrix)
        quaternions[i] = rot.as_quat(scalar_first=False)  # Returns [x, y, z, w]

        # Save the computed y_axis as the up vector for the next iteration
        previous_up_vector = y_axis

    return quaternions



def createScene(root_node):
    root_node.name = "root"
    root_node.dt = 0.01
    root_node.gravity = [0, 0, 0]

    root_node.addObject('RequiredPlugin', name="Sofa.Component.Collision.Detection.Algorithm")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.Collision.Detection.Intersection")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.Collision.Geometry")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.Collision.Response.Contact")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.Constraint.Projective")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.IO.Mesh")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.LinearSolver.Direct")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.LinearSolver.Iterative")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.Mapping.Linear")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.Mass")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.ODESolver.Backward")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.SolidMechanics.FEM.Elastic")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.StateContainer")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.Topology.Container.Constant")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.Visual")


    root_node.addObject('VisualStyle', displayFlags="showBehaviorModels hideForceFields showCollisionModels showVisual showInteractionForceFields")

    root_node.addObject('FreeMotionAnimationLoop')
    root_node.addObject('CollisionPipeline', depth="6", verbose="0", draw="0")
    root_node.addObject('BruteForceBroadPhase')
    root_node.addObject('BVHNarrowPhase')
    root_node.addObject('MinProximityIntersection', name="Proximity", alarmDistance="0.3", contactDistance="0.2")
    root_node.addObject('CollisionResponse', name="Response", response="FrictionContactConstraint", responseParams="mu=0.4")
    beam_with_triangulated_cube_collision = root_node.addChild('beam-withTriangulatedCubeCollision')
    # root_node.addObject('BlockGaussSeidelConstraintSolver', maxIterations=1000, tolerance=1.0e-6)
    root_node.addObject('GenericConstraintSolver', maxIterations=1000, tolerance=1.0e-6)



    beam_with_triangulated_cube_collision.addObject('EulerImplicitSolver', rayleighStiffness="0.1", printLog="false", rayleighMass="0.1")
    beam_with_triangulated_cube_collision.addObject('BTDLinearSolver', template="BTDMatrix6d", printLog="false", verbose="false")


 
    ##Add point before the first point to attach an outside point
    FirstPoint = [0, 0, 0]
    for i in range(3):
        FirstPoint[i] = 2*LCenters[0][i] - LCenters[1][i]
    LCenters = [FirstPoint, *LCenters]

    FinalCenters = []
    ##Adding a point in the center of each edge
    for i in range(len(LCenters) -1):
        middlePoint = [0, 0, 0]
        for j in range(3):
            middlePoint[j] = (LCenters[i][j] + LCenters[i+1][j])/2.0
        FinalCenters.append(LCenters[i])
        FinalCenters.append(middlePoint)

    FinalCenters.append(LCenters[-1])

    edges = [ ]
    for i in range(len(FinalCenters) -1):
        edges = [*edges, i, i+1]

    ##Compute the quaternions to align them with the models
    quats = compute_quaternions_for_path(FinalCenters)
    BeamPosition = []
    for id in range(len(FinalCenters)):
        BeamPosition.append([*FinalCenters[id], *quats[id]])
    dofPerVertebra=[2, 4, 6, 8]

    beam_with_triangulated_cube_collision.addObject('MeshTopology', name="lines", lines=edges)

    beam_with_triangulated_cube_collision.addObject('MechanicalObject', template="Rigid3", name="DOFs", position=BeamPosition)
    beam_with_triangulated_cube_collision.addObject('UniformMass', vertexMass="1 1 0.01 0 0 0 0.1 0 0 0 0.1", printLog="false")
    beam_with_triangulated_cube_collision.addObject('BeamFEMForceField', name="FEM", radius=10, radiusInner=0, youngModulus=2e4, poissonRatio=0.49)


    # beam_with_triangulated_cube_collision.addObject('ConstantForceField',
    #                                                  indices=dofPerVertebra, 
    #                                                  forces=[[100000,0,0,0,0,0],[0,0,0,0,0,0],[0,10000,0,0,0,0],[-8000,0,0,0,0,0]])

    beam_with_triangulated_cube_collision.addObject('ConstantForceField',
                                                     indices=dofPerVertebra, 
                                                     forces=[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,10000,0,0,0,0]])


    beam_with_triangulated_cube_collision.addObject('FixedProjectiveConstraint', name="FixedProjectiveConstraint", indices=0)
    beam_with_triangulated_cube_collision.addObject("LinearSolverConstraintCorrection")

    ##Lookup table to avoid placing a vertebra on fixed point

    # vertebra bodies
    for vertebraID in range(len(dofPerVertebra)):
        Vertebra = beam_with_triangulated_cube_collision.addChild(f"Vertebra{vertebraID+1}")
        Vertebra.addObject("MeshOBJLoader", name="MeshObjLoader", scale3d= [1]*3, filename=f"vertebra/L{vertebraID+1}_decimated.obj" )
        color = cmap(vertebraID/(len(LCenters)-1))
        Vertebra.addObject("OglModel", name="Visual", src="@MeshObjLoader", color=color)
        Vertebra.addObject("RigidMapping", index=dofPerVertebra[vertebraID], globalToLocalCoords=True)

    # vertebra sides    
    for vertebraID in range(len(dofPerVertebra)):
        Vertebra = beam_with_triangulated_cube_collision.addChild(f"Vertebra_coll{vertebraID+1}")
        Vertebra.addObject("MeshOBJLoader", name="MeshObjLoader", scale3d= [1]*3, filename=f"vertebra/L{vertebraID+1}_ends_decimated.obj" )
        color = cmap(vertebraID/(len(LCenters)-1))
        Vertebra.addObject("MeshTopology", name="Topo", src="@MeshObjLoader")

        Vertebra.addObject("MechanicalObject", name="Coll", src="@MeshObjLoader")
        # Vertebra.addObject("LineCollisionModel", name="Lines", contactDistance=0.1)
        Vertebra.addObject("LineCollisionModel", name="Lines", proximity=0.8)

        Vertebra.addObject("RigidMapping", index=dofPerVertebra[vertebraID], globalToLocalCoords=True)

    # facet joints
    for vertebraID in range(len(dofPerVertebra)):
        Vertebra = beam_with_triangulated_cube_collision.addChild(f"Vertebra_coll{vertebraID+1}")
        Vertebra.addObject("MeshOBJLoader", name="MeshObjLoader", scale3d= [1]*3, filename=f"vertebra/L{vertebraID+1}_facet_decimated.obj" )
        color = cmap(vertebraID/(len(LCenters)-1))
        Vertebra.addObject("MeshTopology", name="Topo", src="@MeshObjLoader")

        Vertebra.addObject("MechanicalObject", name="Coll", src="@MeshObjLoader")
        # Vertebra.addObject("LineCollisionModel", name="Lines", contactDistance=0.1)
        Vertebra.addObject("LineCollisionModel", name="Lines", proximity=0.1)
# 
        Vertebra.addObject("RigidMapping", index=dofPerVertebra[vertebraID], globalToLocalCoords=True)

# cd C:\Users\elise\Documents\SpineSimulation
# runSofa -l SofaPython3 BeamSpine.py