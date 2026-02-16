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
    """
    points = np.array(points)
    n_points = len(points)
    quaternions = np.zeros((n_points, 4))

    world_basis = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    previous_up_vector = None

    for i in range(n_points):
        if i < n_points - 1:
            x_axis = points[i + 1] - points[i]
            x_axis = normalize(x_axis)
        else:
            quaternions[i] = quaternions[i - 1]
            continue

        if i == 0:
            dot_products = np.abs([np.dot(x_axis, basis) for basis in world_basis])
            min_idx = np.argmin(dot_products)
            candidate_up_vector = world_basis[min_idx].copy()
        else:
            candidate_up_vector = previous_up_vector

        orthogonal_component = candidate_up_vector - np.dot(candidate_up_vector, x_axis) * x_axis
        y_axis = normalize(orthogonal_component)

        z_axis = np.cross(x_axis, y_axis)
        z_axis = normalize(z_axis)

        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

        rot = Rotation.from_matrix(rotation_matrix)
        quaternions[i] = rot.as_quat(scalar_first=False)

        previous_up_vector = y_axis

    return quaternions


def createScene(root_node):
    root_node.name = "root"
    root_node.dt = 0.01
    root_node.gravity = [0, 0, -9810]  # Prone position

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
    root_node.addObject('RequiredPlugin', name="Sofa.Component.SolidMechanics.Spring")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.StateContainer")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.Topology.Container.Constant")
    root_node.addObject('RequiredPlugin', name="Sofa.Component.Visual")

    root_node.addObject('VisualStyle', displayFlags="showBehaviorModels hideForceFields showCollisionModels showVisual")

    root_node.addObject('FreeMotionAnimationLoop')
    root_node.addObject('CollisionPipeline', depth="6", verbose="0", draw="0")
    root_node.addObject('BruteForceBroadPhase')
    root_node.addObject('BVHNarrowPhase')
    root_node.addObject('MinProximityIntersection', name="Proximity", alarmDistance="0.3", contactDistance="0.2")
    root_node.addObject('CollisionResponse', name="Response", response="FrictionContactConstraint", responseParams="mu=0.4")
    root_node.addObject('GenericConstraintSolver', maxIterations=1000, tolerance=1.0e-6)

    # Vertebra centers
    LCenters = [[-27.46778638, 231.49188166, -327.82943955], 
                [-25.0427509, 208.15541483, -304.45299873], 
                [-23.19301023, 183.54069994, -279.33333233], 
                [-21.36679076, 157.00524955, -256.02454536]]

    # Add attachment point before L1
    FirstPoint = [0, 0, 0]
    for i in range(3):
        FirstPoint[i] = 2*LCenters[0][i] - LCenters[1][i]
    LCenters = [FirstPoint, *LCenters]

    FinalCenters = []
    # Adding a point in the center of each edge
    for i in range(len(LCenters) - 1):
        middlePoint = [0, 0, 0]
        for j in range(3):
            middlePoint[j] = (LCenters[i][j] + LCenters[i+1][j])/2.0
        FinalCenters.append(LCenters[i])
        FinalCenters.append(middlePoint)

    FinalCenters.append(LCenters[-1])

    edges = []
    for i in range(len(FinalCenters) - 1):
        edges = [*edges, i, i+1]

    # Compute the quaternions to align them with the models
    quats = compute_quaternions_for_path(FinalCenters)
    BeamPosition = []
    for id in range(len(FinalCenters)):
        BeamPosition.append([*FinalCenters[id], *quats[id]])
    
    dofPerVertebra = [2, 4, 6, 8]

    # Parent node for spine
    spine = root_node.addChild('SpineWithSprings')
    spine.addObject('EulerImplicitSolver', rayleighStiffness="0.1", printLog="false", rayleighMass="0.1")
    spine.addObject('BTDLinearSolver', template="BTDMatrix6d", printLog="false", verbose="false")

    # Main beam structure
    spine.addObject('MeshTopology', name="lines", lines=edges)
    spine.addObject('MechanicalObject', template="Rigid3", name="DOFs", position=BeamPosition)
    spine.addObject('UniformMass', vertexMass="1 1 0.01 0 0 0 0.1 0 0 0 0.1", printLog="false")
    spine.addObject('BeamFEMForceField', name="FEM", radius=10, radiusInner=0, youngModulus=2e4, poissonRatio=0.49)

    # Apply forces
    spine.addObject('ConstantForceField',
                   indices=dofPerVertebra, 
                   forces=[[0,0,0,0,0,0],      # L1
                           [0,0,8000,0,0,0],   # L2
                           [0,0,10000,0,0,0],  # L3
                           [0,0,0,0,0,0]])     # L4

    spine.addObject('FixedProjectiveConstraint', name="FixedProjectiveConstraint", indices=0)
    spine.addObject("LinearSolverConstraintCorrection")

    # Add fixed anchor points for L1 and L4 (like paper's approach)
    # L1 anchor (simulates connection to T12)
    fixed_L1 = spine.addChild('fixed_anchor_L1')
    # Position above L1
    anchor_L1_pos = [
        -27.46778638,
        231.49188166 + 25,  # 25mm above L1
        -327.82943955
    ]
    fixed_L1.addObject('MechanicalObject',
                      name='AnchorPoint_L1',
                      template='Vec3d',
                      position=[anchor_L1_pos])
    fixed_L1.addObject('UniformMass', name='Mass', vertexMass='1')
    fixed_L1.addObject('FixedConstraint',
                      template='Vec3d',
                      name='fixedConstraint_L1',
                      indices='0')

    # L4 anchor (simulates connection to L5/S1)
    fixed_L4 = spine.addChild('fixed_anchor_L4')
    # Position below L4
    anchor_L4_pos = [
        -21.36679076,
        157.00524955 - 25,  # 25mm below L4
        -256.02454536
    ]
    fixed_L4.addObject('MechanicalObject',
                      name='AnchorPoint_L4',
                      template='Vec3d',
                      position=[anchor_L4_pos])
    fixed_L4.addObject('UniformMass', name='Mass', vertexMass='1')
    fixed_L4.addObject('FixedConstraint',
                      template='Vec3d',
                      name='fixedConstraint_L4',
                      indices='0')

    # Connect L1 ends to anchor with strong springs
    spine.addObject('StiffSpringForceField',
                   name='L1_anchor_springs',
                   template='Vec3d',
                   object1='@fixed_anchor_L1/AnchorPoint_L1',
                   object2='@Vertebra_ends_L1/EndsPoints',
                   stiffness=10000,  # Very stiff to restrict movement
                   damping=100)

    # Connect L4 ends to anchor with strong springs
    spine.addObject('StiffSpringForceField',
                   name='L4_anchor_springs',
                   template='Vec3d',
                   object1='@fixed_anchor_L4/AnchorPoint_L4',
                   object2='@Vertebra_ends_L4/EndsPoints',
                   stiffness=10000,  # Very stiff to restrict movement
                   damping=100)

    # Add vertebra bodies with collision meshes
    for vertebraID in range(len(dofPerVertebra)):
        # Main vertebra visual
        Vertebra_visual = spine.addChild(f"Vertebra_visual_L{vertebraID+1}")
        Vertebra_visual.addObject("MeshOBJLoader", 
                                 name="VisualLoader", 
                                 scale3d=[1]*3, 
                                 filename=f"vertebra/L{vertebraID+1}_decimated.obj")
        color = cmap(vertebraID/len(dofPerVertebra))
        Vertebra_visual.addObject("OglModel", name="Visual", src="@VisualLoader", color=color)
        Vertebra_visual.addObject("RigidMapping", index=dofPerVertebra[vertebraID], globalToLocalCoords=True)

        # Vertebra ends (for intervertebral disc springs)
        Vertebra_ends = spine.addChild(f"Vertebra_ends_L{vertebraID+1}")
        Vertebra_ends.addObject("MeshOBJLoader", 
                               name="EndsLoader", 
                               scale3d=[1]*3, 
                               filename=f"vertebra/L{vertebraID+1}_ends_decimated.obj")
        Vertebra_ends.addObject("MeshTopology", name="EndsTopo", src="@EndsLoader")
        Vertebra_ends.addObject("MechanicalObject", 
                               name="EndsPoints", 
                               template="Vec3d",
                               src="@EndsLoader")
        Vertebra_ends.addObject("RigidMapping", 
                               input="@../DOFs",
                               index=dofPerVertebra[vertebraID], 
                               globalToLocalCoords=True)

        # Facet joints (for facet springs)
        Vertebra_facets = spine.addChild(f"Vertebra_facets_L{vertebraID+1}")
        Vertebra_facets.addObject("MeshOBJLoader", 
                                 name="FacetLoader", 
                                 scale3d=[1]*3, 
                                 filename=f"vertebra/L{vertebraID+1}_facet_decimated.obj")
        Vertebra_facets.addObject("MeshTopology", name="FacetTopo", src="@FacetLoader")
        Vertebra_facets.addObject("MechanicalObject", 
                                 name="FacetPoints", 
                                 template="Vec3d",
                                 src="@FacetLoader")
        Vertebra_facets.addObject("RigidMapping", 
                                 input="@../DOFs",
                                 index=dofPerVertebra[vertebraID], 
                                 globalToLocalCoords=True)

    # Add springs between adjacent vertebrae
    # Connect vertebra ends (intervertebral disc)
    for vertebraID in range(len(dofPerVertebra) - 1):
        curr_idx = vertebraID + 1
        next_idx = vertebraID + 2
        
        # Intervertebral disc springs (body to body)
        # Paper uses: 500 N/m stiffness, 3 N/s damping, 400-800 springs total
        spine.addObject('StiffSpringForceField',
                       name=f'DiscSprings_L{curr_idx}_L{next_idx}',
                       template='Vec3d',
                       object1=f'@Vertebra_ends_L{curr_idx}/EndsPoints',
                       object2=f'@Vertebra_ends_L{next_idx}/EndsPoints',
                       stiffness=500,      # N/m per spring
                       damping=3,          # N/s per spring
                       # SOFA will automatically create springs between nearby points
                       # or you can specify which points to connect
                       )
        
        # Facet joint springs
        # Paper uses: 8000 N/m stiffness, 500 N/s damping, 200-500 springs total
        spine.addObject('StiffSpringForceField',
                       name=f'FacetSprings_L{curr_idx}_L{next_idx}',
                       template='Vec3d',
                       object1=f'@Vertebra_facets_L{curr_idx}/FacetPoints',
                       object2=f'@Vertebra_facets_L{next_idx}/FacetPoints',
                       stiffness=8000,     # N/m per spring
                       damping=500,        # N/s per spring
                       )

    return root_node


if __name__ == "__main__":
    print("=" * 70)
    print("SIMPLIFIED SPRING-BASED SPINE SIMULATION")
    print("=" * 70)
    print()
    print("Based on the paper's approach but simplified:")
    print()
    print("Structure:")
    print("  - BeamFEMForceField for overall spine structure")
    print("  - Each vertebra has 'ends' mesh (superior/inferior surfaces)")
    print("  - Each vertebra has 'facet' mesh (facet joint surfaces)")
    print()
    print("Springs:")
    print("  - Intervertebral disc: StiffSpringForceField between 'ends' meshes")
    print("    • Stiffness: 500 N/m per spring")
    print("    • Damping: 3 N/s per spring")
    print("  - Facet joints: StiffSpringForceField between 'facet' meshes")
    print("    • Stiffness: 8000 N/m per spring")
    print("    • Damping: 500 N/s per spring")
    print()
    print("Forces:")
    print("  - L2: 8000N upward")
    print("  - L3: 10000N upward")
    print()
    print("Note: SOFA's StiffSpringForceField will automatically create springs")
    print("between nearby points in the two objects, or you can manually specify")
    print("which points to connect using the 'spring' parameter.")
    print("=" * 70)