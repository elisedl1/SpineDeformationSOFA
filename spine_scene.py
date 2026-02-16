"""
SOFA scene file for spine deformation - Direct loading with runSofa

Usage:
    runSofa -l SofaPython3 spine_scene.py

Edit the configuration section below to set your paths.
"""

import os
import json
import random
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================
SPINE_ID = "original"
ROOT_PATH_VERTEBRAE = Path(r"C:/Users/elise/Documents/SpineSimulation/spine_folder")
PARAMETERS_FILE = Path(r"C:/Users/elise/Documents/SpineSimulation/parameters.json")

FORCE_ID = 0
USE_ANTERIOR_LOADING = False  # True = flexion, False = extension

# ============================================================================
# DO NOT EDIT BELOW THIS LINE
# ============================================================================

import Sofa
import Sofa.Core
import SofaRuntime

SofaRuntime.importPlugin("Sofa.Component.StateContainer")
SofaRuntime.importPlugin("Sofa.GL.Component.Rendering3D")
SofaRuntime.importPlugin("Sofa.Component.Collision.Detection.Algorithm")
SofaRuntime.importPlugin("Sofa.Component.Collision.Detection.Intersection")
SofaRuntime.importPlugin("Sofa.Component.Collision.Geometry")
SofaRuntime.importPlugin("Sofa.Component.Collision.Response.Contact")
SofaRuntime.importPlugin("Sofa.Component.Visual")
SofaRuntime.importPlugin("Sofa.Component.IO.Mesh")
SofaRuntime.importPlugin("Sofa.Component.Topology.Container.Constant")
SofaRuntime.importPlugin("Sofa.Component.MechanicalLoad")
SofaRuntime.importPlugin("Sofa.Component.Mapping.Linear")
SofaRuntime.importPlugin("Sofa.Component.Mapping.NonLinear")
SofaRuntime.importPlugin("Sofa.Component.ODESolver.Backward")
SofaRuntime.importPlugin("Sofa.Component.LinearSolver.Iterative")
SofaRuntime.importPlugin("Sofa.Component.Mass")
SofaRuntime.importPlugin("Sofa.Component.SolidMechanics.Spring")
SofaRuntime.importPlugin("Sofa.Component.Constraint.Projective")


def get_vertebra_mesh_paths(root_path, spine_id, vert_number):
    """Get paths to all three mesh components for a vertebra."""
    folder_name = os.path.join(root_path, f"{spine_id}_L{vert_number}")
    
    if not os.path.exists(folder_name):
        raise FileNotFoundError(f"Vertebra folder not found: {folder_name}")
    
    paths = {
        'body': os.path.join(folder_name, f"L{vert_number}_decimated.obj"),
        'endplates': os.path.join(folder_name, f"L{vert_number}_ends_decimated.obj"),
        'facets': os.path.join(folder_name, f"L{vert_number}_facet_decimated.obj"),
        'folder': folder_name
    }
    
    # Verify all files exist
    for key, path in paths.items():
        if key != 'folder' and not os.path.exists(path):
            raise FileNotFoundError(f"Missing mesh file: {path}")
    
    return paths


def add_collision_model(root):
    """Add collision detection components to the scene."""
    root.addObject('CollisionPipeline', verbose='0')
    root.addObject('BruteForceBroadPhase')
    root.addObject('BVHNarrowPhase')
    root.addObject('CollisionResponse', response='PenalityContactForceField')
    root.addObject('LocalMinDistance', 
                   name='Proximity', 
                   alarmDistance='0.0005', 
                   contactDistance='0.000001',
                   angleCone='0.0')
    root.addObject('DiscreteIntersection')


def add_vertebra_node(parent_node, vert_number, mesh_paths, force_field, spine_id, force_id):
    """
    Create a vertebra with rigid body dynamics.
    
    Structure:
    vertX/
      ├── center_mass (Rigid3d - this is what moves, the main DOF)
      ├── body/ (visible mesh, RigidMapped to center_mass)
      │   └── visual/
      ├── endplates/ (invisible, RigidMapped to center_mass, for disc springs)
      └── facets/ (invisible, RigidMapped to center_mass, for facet springs)
    
    All meshes are RigidMapped to the center_mass, so when center_mass moves,
    all meshes move with it as a rigid body.
    """
    vert_label = f"vert{vert_number}"
    vert_node = parent_node.addChild(vert_label)

    # -----------------------------
    # RIGID CENTER OF MASS (the main mechanical DOF)
    # This Rigid3d object is what actually has degrees of freedom
    # Everything else is mapped to it
    # -----------------------------
    vert_node.addObject(
        "MechanicalObject",
        name="center_mass",
        template="Rigid3d"
    )
    # Add mass so vertebra can move under forces (not RestShapeSprings!)
    vert_node.addObject(
        "UniformMass",
        totalMass="1.0"
    )

    # -----------------------------
    # BODY MESH (visible, with collision, mapped to rigid center)
    # -----------------------------
    body_node = vert_node.addChild("body")
    body_node.addObject(
        "MeshOBJLoader",
        name="loader",
        filename=str(mesh_paths["body"]),
        printLog='false',
        flipNormals=False
    )
    body_node.addObject("MeshTopology", name="topology", src="@loader")
    body_node.addObject(
        "MechanicalObject", 
        name="points", 
        template="Vec3d", 
        src="@topology"
    )
    body_node.addObject("TriangleCollisionModel")
    
    # Visual model (child of body)
    visual_node = body_node.addChild("visual")
    visual_node.addObject(
        "OglModel",
        name="Visual",
        src="@../loader",
        color="white"
    )
    visual_node.addObject("IdentityMapping")
    
    # CRITICAL: Map body mesh to rigid center BEFORE any other operations
    # input="@.." means parent node (vert_node/center_mass)
    # output="@." means this node (body_node/points)
    body_node.addObject("RigidMapping", input="@..", output="@.")

    # -----------------------------
    # Apply force to the RIGID CENTER (not the mapped body)
    # The force should be applied to the Rigid3d DOF, not the mapped Vec3d
    # -----------------------------
    # Parse the force string and apply to rigid center
    force_values = force_field[vert_label].split()
    vert_node.addObject(
        "ConstantForceField",
        name="external_force",
        template="Rigid3d",
        forces=f"{force_values[0]} {force_values[1]} {force_values[2]} 0 0 0",  # Translation + rotation
        indices="0"
    )

    # -----------------------------
    # ENDPLATES MESH (invisible, mapped to rigid center, for disc springs)
    # -----------------------------
    endplates_node = vert_node.addChild("endplates")
    endplates_node.addObject(
        "MeshOBJLoader",
        name="loader",
        filename=str(mesh_paths["endplates"]),
        printLog='false'
    )
    endplates_node.addObject("MeshTopology", name="topology", src="@loader")
    endplates_node.addObject(
        "MechanicalObject",
        name="points",
        template="Vec3d",
        src="@topology"
    )
    # Map endplates to rigid center
    endplates_node.addObject("RigidMapping", input="@..", output="@.")

    # -----------------------------
    # FACETS MESH (invisible, mapped to rigid center, for facet springs)
    # -----------------------------
    facets_node = vert_node.addChild("facets")
    facets_node.addObject(
        "MeshOBJLoader",
        name="loader",
        filename=str(mesh_paths["facets"]),
        printLog='false'
    )
    facets_node.addObject("MeshTopology", name="topology", src="@loader")
    facets_node.addObject(
        "MechanicalObject",
        name="points",
        template="Vec3d",
        src="@topology"
    )
    # Map facets to rigid center
    facets_node.addObject("RigidMapping", input="@..", output="@.")

    return vert_node


def add_intervertebral_coupling(parent_node, vert1_num, vert2_num, parameters):
    """
    Add spring force fields between mapped meshes of adjacent vertebrae.
    
    IMPORTANT: Springs connect the MAPPED endplate/facet meshes, not the rigid centers!
    The mapped meshes move with their rigid centers, so springs between them
    create forces on the rigid bodies.
    """
    vert1_label = f"vert{vert1_num}"
    vert2_label = f"vert{vert2_num}"
    
    # Disc coupling (endplate to endplate)
    # This simulates the intervertebral disc
    parent_node.addObject('StiffSpringForceField',
                         template='Vec3d',
                         name=f'disc_{vert1_num}_{vert2_num}',
                         object1=f'@{vert1_label}/endplates/points',
                         object2=f'@{vert2_label}/endplates/points',
                         stiffness=parameters['disc']['stiffness'],
                         damping=parameters['disc']['damping'])
    
    # Facet joint coupling
    # This simulates the facet joints
    parent_node.addObject('StiffSpringForceField',
                         template='Vec3d',
                         name=f'facets_{vert1_num}_{vert2_num}',
                         object1=f'@{vert1_label}/facets/points',
                         object2=f'@{vert2_label}/facets/points',
                         stiffness=parameters['facets']['stiffness'],
                         damping=parameters['facets']['damping'])


def add_boundary_conditions(together_node, parameters):
    """
    Fix L1 and L4 rigid centers to simulate boundary conditions.
    
    IMPORTANT: The FixedConstraint must be added to the node that CONTAINS
    the MechanicalObject, not a parent node!
    """
    # Fix L1 (superior boundary) - add constraint to vert1 node
    vert1_node = together_node.getChild('vert1')
    vert1_node.addObject(
        'FixedConstraint',
        template='Rigid3d',
        name='fix_L1'
    )
    
    # Fix L4 (inferior boundary) - add constraint to vert4 node
    vert4_node = together_node.getChild('vert4')
    vert4_node.addObject(
        'FixedConstraint',
        template='Rigid3d',
        name='fix_L4'
    )


def createScene(rootNode):
    """
    Create the SOFA scene.
    This function is automatically called by SOFA when loading the scene.
    """
    # Required plugins
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Algorithm')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Intersection')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Geometry')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Response.Contact')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Visual')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.IO.Mesh')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.StateContainer')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Constant')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.MechanicalLoad')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mapping.Linear')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mapping.NonLinear')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.ODESolver.Backward')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Iterative')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mass')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.Spring')
    rootNode.addObject('RequiredPlugin', name='Sofa.GL.Component.Rendering3D')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Projective')

    # Load parameters
    with open(PARAMETERS_FILE) as f:
        parameters = json.load(f)
    
    # Define force field - matching paper forces (up to 0.7 N)
    # Distribution: stronger on middle vertebrae (L2, L3) to induce bending
    if USE_ANTERIOR_LOADING:
        force_field = {
            'vert1': '0.0 0.2 0.0',    # 0.2 N upward (anterior)
            'vert2': '0.0 0.5 0.0',    # 0.5 N upward
            'vert3': '0.0 0.7 0.0',    # 0.7 N upward (maximum from paper)
            'vert4': '0.0 0.3 0.0'     # 0.3 N upward
        }
    else:
        force_field = {
            'vert1': '0.0 -0.2 0.0',   # 0.2 N downward (posterior)
            'vert2': '0.0 -0.5 0.0',   # 0.5 N downward
            'vert3': '0.0 -0.7 0.0',   # 0.7 N downward (maximum from paper)
            'vert4': '0.0 -0.3 0.0'    # 0.3 N downward
        }
    
    # Scene setup
    rootNode.addObject('DefaultVisualManagerLoop')
    rootNode.addObject('DefaultAnimationLoop')
    rootNode.addObject('VisualStyle', 
                       displayFlags='showVisual showBehaviorModels showInteractionForceFields showForceFields')
    
    # No gravity - paper uses only applied forces for flexion/extension testing
    rootNode.gravity = [0, 0, 0]
    
    add_collision_model(rootNode)
    
    # Create parent node for all vertebrae
    together = rootNode.addChild('Together')
    together.addObject('EulerImplicitSolver',
                      name='odesolver',
                      printLog='false',
                      rayleighStiffness=0.1,    # Increased damping
                      rayleighMass=0.1)
    together.addObject('CGLinearSolver',
                      name='linear_solver',
                      iterations=50,             # Reduced for faster computation
                      tolerance=1e-06,           # Relaxed tolerance
                      threshold='1e-09')
    
    # Add all vertebrae (L1-L4)
    print(f"Loading spine: {SPINE_ID}")
    for vert_num in range(1, 5):
        print(f"  Loading L{vert_num}...")
        mesh_paths = get_vertebra_mesh_paths(ROOT_PATH_VERTEBRAE, SPINE_ID, vert_num)
        add_vertebra_node(together, vert_num, mesh_paths, force_field, SPINE_ID, FORCE_ID)
    
    # Add intervertebral coupling (discs + facets)
    print("Adding intervertebral coupling...")
    for vert_num in range(1, 4):  # L1-L2, L2-L3, L3-L4
        add_intervertebral_coupling(together, vert_num, vert_num + 1, parameters)
    
    # Add boundary conditions (fix L1 and L4 rigid centers)
    print("Adding boundary conditions...")
    add_boundary_conditions(together, parameters)
    
    print("Scene created successfully!")
    print("\nPress 'Animate' button in SOFA to see the spine deform!")
    print(f"Loading direction: {'Anterior (flexion)' if USE_ANTERIOR_LOADING else 'Posterior (extension)'}")
    
    return rootNode