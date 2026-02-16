import os
import argparse
import json
import glob
import sys
import random
import numpy as np

# Set these in terminal before running:
os.environ['SOFA_ROOT'] = '/home/farid/SOFA/v22.12.00/'
existing_value = os.environ.get('LD_LIBRARY_PATH', '')
new_value = "/home/farid/SOFA/v22.12.00/lib/:/home/farid/SOFA/v22.12.00/collections/SofaSimulation/lib/:" + existing_value
os.environ['LD_LIBRARY_PATH'] = new_value

import Sofa
import Sofa.Core
import SofaRuntime
import Sofa.Gui

SofaRuntime.importPlugin("Sofa.Component.StateContainer")
SofaRuntime.importPlugin("SofaOpenglVisual")
SofaRuntime.importPlugin("Sofa.Component.Collision.Detection.Algorithm")
SofaRuntime.importPlugin("Sofa.Component.Collision.Detection.Intersection")
SofaRuntime.importPlugin('SofaComponentAll')


def get_force_field(anterior=True):
    """
    Generate random force fields for spine deformation.
    Simulates loading patterns (flexion/extension).
    """
    if anterior:
        x_axis_force = random.randint(-10, 9) * 0.001
        y_axis_force_v1_v4 = random.randint(10, 15) * 0.001
        y_axis_force_v2_v3 = random.randint(15, 25) * 0.001
    else:
        x_axis_force = random.randint(-10, 9) * 0.001
        y_axis_force_v1_v4 = random.randint(-15, -10) * 0.001
        y_axis_force_v2_v3 = random.randint(-25, -15) * 0.001

    force_fields = {
        'vert1': f'0.0 {y_axis_force_v1_v4} 0.0',
        'vert2': f'{x_axis_force} {y_axis_force_v2_v3} 0.0',
        'vert3': f'{x_axis_force} {y_axis_force_v2_v3} 0.0',
        'vert4': f'0.0 {y_axis_force_v1_v4} 0.0'
    }

    return force_fields


def get_vertebra_mesh_paths(root_path, spine_id, vert_number):
    """
    Get paths to all three mesh components for a vertebra.
    
    Args:
        root_path: Root directory containing vertebra folders
        spine_id: Spine identifier (e.g., 'sub-verse500')
        vert_number: Vertebra number (1-4 for L1-L4)
    
    Returns:
        Dictionary with paths to body, endplate, and facet meshes
    """
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
    root.addObject('DefaultPipeline', verbose='0', name='CollisionPipeline')
    root.addObject('BruteForceBroadPhase')
    root.addObject('BVHNarrowPhase')
    root.addObject('DefaultContactManager', response='PenalityContactForceField')
    root.addObject('LocalMinDistance', 
                   name='Proximity', 
                   alarmDistance='0.0005', 
                   contactDistance='0.000001',
                   angleCone='0.0')
    root.addObject('DiscreteIntersection')


def add_vertebra_node(parent_node, vert_number, mesh_paths, force_field, spine_id, force_id):
    """
    Create a vertebra node with multiple anatomical meshes.
    
    This creates a hierarchical structure:
    vertX/
      ├── body/      (vertebral body mesh)
      ├── endplates/ (superior/inferior endplates)
      └── facets/    (facet joint surfaces)
    
    Each component has its own mechanical properties and can be
    coupled independently to adjacent vertebrae.
    """
    vert_label = f"vert{vert_number}"
    vert_node = parent_node.addChild(vert_label)
    
    # Create sub-nodes for each anatomical component
    components = ['body', 'endplates', 'facets']
    
    for component in components:
        comp_node = vert_node.addChild(component)
        
        # Load mesh
        comp_node.addObject('MeshObjLoader',
                           name=f'loader_{component}',
                           filename=mesh_paths[component],
                           printLog='false',
                           flipNormals='0')
        
        # Create topology
        comp_node.addObject('MeshTopology',
                           name='topology',
                           src=f'@loader_{component}')
        
        # Create mechanical object
        comp_node.addObject('MechanicalObject',
                           name='points',
                           template='Vec3d',
                           src='@topology')
        
        # Add collision model (only for body to avoid over-constraining)
        if component == 'body':
            comp_node.addObject('TriangleCollisionModel')
            
            # Add visual model as child of body
            visual_node = comp_node.addChild('visual')
            visual_node.addObject('OglModel',
                                 name='Visual',
                                 src=f'@../loader_{component}',
                                 color='white')
            visual_node.addObject('IdentityMapping')
        
        # Apply external forces (only to body)
        if component == 'body':
            comp_node.addObject('ConstantForceField',
                               force=force_field[vert_label])
            
            # Add VTK exporter for body mesh
            output_filename = os.path.join(
                mesh_paths['folder'],
                f"{spine_id}_L{vert_number}_force{force_id}_deformed_"
            )
            comp_node.addObject('VTKExporter',
                               filename=output_filename,
                               listening='true',
                               edges='0',
                               triangles='1',
                               quads='0',
                               tetras='0',
                               pointsDataFields='points.position',
                               exportEveryNumberOfSteps='20')
    
    return vert_node


def add_intervertebral_coupling(parent_node, vert1_num, vert2_num, parameters):
    """
    Add spring force fields to couple adjacent vertebrae.
    
    This creates three types of connections:
    1. Disc coupling: endplate-to-endplate (soft, allows compression)
    2. Facet coupling: facet-to-facet (stiffer, guides motion)
    3. Body-to-body: optional additional stabilization
    
    SOFA automatically creates springs between corresponding points
    in the two meshes - no manual indexing required.
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
    # This simulates the facet joints - typically stiffer than discs
    parent_node.addObject('StiffSpringForceField',
                         template='Vec3d',
                         name=f'facets_{vert1_num}_{vert2_num}',
                         object1=f'@{vert1_label}/facets/points',
                         object2=f'@{vert2_label}/facets/points',
                         stiffness=parameters['facets']['stiffness'],
                         damping=parameters['facets']['damping'])


def add_boundary_conditions(parent_node, parameters):
    """
    Fix L1 and L4 to simulate boundary conditions.
    
    In a typical biomechanical test or surgical scenario:
    - L1 (top): fixed to simulate sacral attachment
    - L4 (bottom): fixed to simulate thoracic attachment
    
    You can also fix only specific components (e.g., just endplates)
    to allow some vertebral body deformation.
    """
    # Fix L1 (superior boundary)
    for component in ['body', 'endplates', 'facets']:
        parent_node.addObject('FixedConstraint',
                             template='Vec3d',
                             name=f'fix_L1_{component}',
                             object=f'@vert1/{component}/points')
    
    # Fix L4 (inferior boundary)
    for component in ['body', 'endplates', 'facets']:
        parent_node.addObject('FixedConstraint',
                             template='Vec3d',
                             name=f'fix_L4_{component}',
                             object=f'@vert4/{component}/points')


def createScene(root, spine_id, parameters, root_path_vertebrae, force_field, force_id):
    """
    Create the complete SOFA scene for spine deformation.
    
    Scene hierarchy:
    root/
      └── together/
          ├── vert1/ (L1)
          │   ├── body/
          │   ├── endplates/
          │   └── facets/
          ├── vert2/ (L2)
          ├── vert3/ (L3)
          ├── vert4/ (L4)
          ├── [coupling force fields]
          └── [boundary conditions]
    """
    root.addObject('DefaultVisualManagerLoop')
    root.addObject('DefaultAnimationLoop')
    root.addObject('VisualStyle', 
                   displayFlags='showVisual showBehaviorModels showInteractionForceFields')
    
    add_collision_model(root)
    
    # Create parent node for all vertebrae
    together = root.addChild('Together')
    together.addObject('EulerImplicitSolver',
                      name='odesolver',
                      printLog='0',
                      rayleighStiffness=parameters['solver']['rayleigh_stiffness'],
                      rayleighMass=parameters['solver']['rayleigh_mass'])
    together.addObject('CGLinearSolver',
                      name='linear_solver',
                      iterations=parameters['solver']['iterations'],
                      tolerance=parameters['solver']['tolerance'],
                      threshold='1e-15')
    
    # Add all vertebrae (L1-L4)
    for vert_num in range(1, 5):
        mesh_paths = get_vertebra_mesh_paths(root_path_vertebrae, spine_id, vert_num)
        add_vertebra_node(together, vert_num, mesh_paths, force_field, spine_id, force_id)
    
    # Add intervertebral coupling (discs + facets)
    for vert_num in range(1, 4):  # L1-L2, L2-L3, L3-L4
        add_intervertebral_coupling(together, vert_num, vert_num + 1, parameters)
    
    # Add boundary conditions (fix L1 and L4)
    add_boundary_conditions(together, parameters)
    
    return root


def deform_one_spine(spine_id, parameters, root_path_vertebrae, force_field, force_id, use_gui=True):
    """
    Run a single spine deformation simulation.
    
    Args:
        spine_id: Identifier for the spine
        parameters: Dictionary with biomechanical parameters
        root_path_vertebrae: Path to vertebra mesh folders
        force_field: Dictionary of force vectors per vertebra
        force_id: Identifier for this deformation case
        use_gui: Whether to show GUI (False for batch processing)
    """
    print(f"Deforming {spine_id} (case {force_id})")
    
    root = Sofa.Core.Node('root')
    createScene(root, spine_id, parameters, root_path_vertebrae, force_field, force_id)
    Sofa.Simulation.init(root)
    
    if not use_gui:
        # Batch mode: run for specified number of iterations
        for iteration in range(20):
            print(f"  Iteration: {iteration + 1}/20")
            Sofa.Simulation.animate(root, root.dt.value)
    else:
        # Interactive mode: launch GUI
        print("Supported GUIs: " + Sofa.Gui.GUIManager.ListSupportedGUI(","))
        Sofa.Gui.GUIManager.Init("spine_deformation", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
        Sofa.Gui.GUIManager.MainLoop(root)
        Sofa.Gui.GUIManager.closeGUI()
        print("GUI closed")
    
    print("Simulation complete.")


def deform_all_spines(txt_file, parameters_file, vertebrae_root_folder, 
                      nr_deformations_per_spine, forces_folder):
    """
    Batch process multiple spines with multiple deformation cases each.
    
    For each spine:
    - Generates random force fields (anterior/posterior loading)
    - Saves force field to text file
    - Runs simulation
    - Exports deformed meshes as VTU files
    """
    # Load biomechanical parameters
    with open(parameters_file) as f:
        parameters = json.load(f)
    
    # Read list of spine IDs
    with open(txt_file) as f:
        spine_ids = [line.strip() for line in f]
    
    # Create output folder for force fields
    os.makedirs(forces_folder, exist_ok=True)
    
    print(f"Processing {len(spine_ids)} spines with {nr_deformations_per_spine} deformations each")
    
    for spine_id in spine_ids:
        print(f"\n{'='*60}")
        print(f"Spine: {spine_id}")
        print(f"{'='*60}")
        
        for deform_id in range(nr_deformations_per_spine):
            # Randomly choose anterior or posterior loading
            anterior = np.random.uniform() > 0.5
            force_field = get_force_field(anterior=anterior)
            
            # Save force field
            force_file = os.path.join(forces_folder, f"{spine_id}_{deform_id}.txt")
            with open(force_file, 'w') as f:
                f.write(f"# {'Anterior' if anterior else 'Posterior'} loading\n")
                for vert, force in force_field.items():
                    f.write(f"{vert}: {force}\n")
            
            # Run simulation
            try:
                deform_one_spine(
                    spine_id=spine_id,
                    parameters=parameters,
                    root_path_vertebrae=vertebrae_root_folder,
                    force_field=force_field,
                    force_id=deform_id,
                    use_gui=False
                )
            except Exception as e:
                print(f"ERROR processing {spine_id} case {deform_id}: {e}", file=sys.stderr)
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="SOFA-based lumbar spine deformation simulation (simplified multi-mesh version)"
    )
    
    parser.add_argument(
        "--root_path_vertebrae",
        required=True,
        help="Root path to vertebra folders (each containing L1-L4 meshes)"
    )
    
    parser.add_argument(
        "--parameters_file",
        required=True,
        help="JSON file with biomechanical parameters"
    )
    
    parser.add_argument(
        "--list_file_names",
        help="Text file with list of spine IDs (one per line)"
    )
    
    parser.add_argument(
        "--deform_all",
        action="store_true",
        help="Process all spines in batch mode (no GUI)"
    )
    
    parser.add_argument(
        "--forces_folder",
        default="./forces",
        help="Folder to save force field text files"
    )
    
    parser.add_argument(
        "--nr_deform_per_spine",
        type=int,
        default=1,
        help="Number of deformation cases per spine"
    )
    
    parser.add_argument(
        "--spine_id",
        help="Single spine ID to process (for interactive mode)"
    )
    
    parser.add_argument(
        "--use_gui",
        action="store_true",
        help="Show GUI for single spine (requires --spine_id)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SOFA Spine Deformation - Multi-Mesh Architecture")
    print("="*60)
    
    if args.deform_all:
        # Batch processing mode
        if not args.list_file_names:
            print("ERROR: --list_file_names required for batch mode", file=sys.stderr)
            sys.exit(1)
        
        deform_all_spines(
            txt_file=args.list_file_names,
            parameters_file=args.parameters_file,
            vertebrae_root_folder=args.root_path_vertebrae,
            nr_deformations_per_spine=args.nr_deform_per_spine,
            forces_folder=args.forces_folder
        )
    
    elif args.spine_id:
        # Single spine mode
        with open(args.parameters_file) as f:
            parameters = json.load(f)
        
        # Use constant forces for interactive visualization
        force_field = {
            'vert1': '0.0 0.01 0.0',
            'vert2': '0.0 0.02 0.0',
            'vert3': '0.0 0.02 0.0',
            'vert4': '0.0 0.01 0.0'
        }
        
        deform_one_spine(
            spine_id=args.spine_id,
            parameters=parameters,
            root_path_vertebrae=args.root_path_vertebrae,
            force_field=force_field,
            force_id=0,
            use_gui=args.use_gui
        )
    
    else:
        print("ERROR: Specify either --deform_all or --spine_id", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    print("\nDone!")


    '''

    runSofa -l SofaPython3 sofa_spine_deformation.py \
    --root_path_vertebrae \Users\elise\Documents\SpineSimulation\spine_folder \
    --parameters_file ./parameters.json \
    --spine_id original \
    --use_gui 


    python sofa_spine_deformation.py \
    --root_path_vertebrae /path/to/vertebrae \
    --parameters_file ./parameters.json \
    --spine_id sub-verse500 \
    --use_gui



    # batch
    python sofa_spine_deformation.py \
    --root_path_vertebrae ./spine_folder  \
    --parameters_file ./parameters.json \
    --list_file_names spine_list.txt \
    --deform_all \
    --nr_deform_per_spine 2 \
    --forces_folder ./forces
    
    '''