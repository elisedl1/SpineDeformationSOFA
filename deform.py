import os
import json
import random
import sys
import numpy as np
sys.path.append(r"C:\\Users\\elise\SOFA\\v25.06.00\\lib\\python3\site-packages")








# ---------- SETUP SOFA ENVIRONMENT ----------
os.environ['SOFA_ROOT'] = r'C:\\Users\\elise\SOFA\\v25.06.00\\'
existing_value = os.environ.get('PATH', '')
new_value = os.environ['SOFA_ROOT'] + r'\bin;' + existing_value
os.environ['PATH'] = new_value








import Sofa.Core
import SofaRuntime
import Sofa.Gui
















SofaRuntime.importPlugin("Sofa.Component.StateContainer")
SofaRuntime.importPlugin("Sofa.Component.Mapping")
SofaRuntime.importPlugin("Sofa.Component.MechanicalLoad")
SofaRuntime.importPlugin("Sofa.Component.SolidMechanics.Spring")
SofaRuntime.importPlugin("Sofa.Component.Collision.Geometry")
SofaRuntime.importPlugin("Sofa.Component.Collision.Detection.Algorithm")
SofaRuntime.importPlugin("Sofa.Component.Collision.Detection.Intersection")
SofaRuntime.importPlugin("Sofa.Component.Collision.Response.Contact")
SofaRuntime.importPlugin("Sofa.Component.Topology.Container.Dynamic")
SofaRuntime.importPlugin("Sofa.Component.Visual")
SofaRuntime.importPlugin("Sofa.GL.Component.Rendering3D")
SofaRuntime.importPlugin("Sofa.Component.Mass")
SofaRuntime.importPlugin("Sofa.Component.Constraint.Projective")
SofaRuntime.importPlugin("Sofa.Component.IO.Mesh")
SofaRuntime.importPlugin("Sofa.Component.ODESolver.Backward")
SofaRuntime.importPlugin("Sofa.Component.LinearSolver.Iterative")




# ---------- FORCE FIELDS ----------
constant_force_fields_on_anterior = {
    'vert0': '0 0.01 0.0',
    'vert1': '0 0.07 0.0',
    'vert2': '0 0.07 0.0',
    'vert3': '0 0.01 0.0',
}




constant_force_fields_on_posterior = {
    'vert0': '0 -0.01 0.0',
    'vert1': '-0.01 -0.05 0.0',
    'vert2': '-0.01 -0.05 0.0',
    'vert3': '0 -0.01 0.0',
}




# ---------- FUNCTIONS ----------
def add_collision_function(root):
    root.addObject('CollisionPipeline', verbose='0', name='CollisionPipeline')
    root.addObject('BruteForceBroadPhase')
    root.addObject('BVHNarrowPhase')
    root.addObject('CollisionResponse', response='PenalityContactForceField', name='collision response')
    root.addObject('LocalMinDistance', name='Proximity', alarmDistance='0.0005', contactDistance='0.000001', angleCone='0.0')
    root.addObject('DiscreteIntersection')




# ---------- AP AXIS FROM CT (hardcoded) ----------
AP_AXIS = np.array([-0.07840471, 0.68894254, 0.72056289])
AP_AXIS = AP_AXIS / np.linalg.norm(AP_AXIS)




# ---------- FORCE FIELDS ----------
def make_force_fields(anterior=True):
    direction = 1.0 if anterior else -1.0
    total_forces_N = {
        'vert0': 0.5,
        'vert1': 1,
        'vert2': 1,
        'vert3': 0.5,
    }
    force_fields = {}
    for vert, force_N in total_forces_N.items():
        fv = direction * force_N * AP_AXIS
        force_fields[vert] = f'{fv[0]} {fv[1]} {fv[2]}'
    return force_fields





def find_nearest_vertex_indices(mesh_file, fixed_positions):
    """Find the nearest mesh vertex index for each fixed point position."""
    # Read vertices from OBJ
    vertices = []
    for line in open(mesh_file):
        if line.startswith('v '):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    vertices = np.array(vertices)
   
    nearest_indices = []
    for fp in fixed_positions:
        distances = np.linalg.norm(vertices - np.array(fp), axis=1)
        nearest_indices.append(int(np.argmin(distances)))
   
    return nearest_indices




def build_fixed_springs(fixed_positions, mesh_vertex_indices, stiffness=1000, damping=50):
    springs = []
    for i, mesh_idx in enumerate(mesh_vertex_indices):
        springs.append(f"{i} {mesh_idx} {stiffness} {damping} 0.0001")
    return ' '.join(springs)


SPRING_SCALE = 1.0

MESH_CENTERS = {
    0: [-27.46778638, 231.49188166, -327.82943955],
    1: [-25.0427509,  208.15541483, -304.45299873],
    2: [-23.19301023, 183.54069994, -279.33333233],
    3: [-21.36679076, 157.00524955, -256.02454536],
}



def add_vertebra_node(parent_node, nr_vertebra, mesh_file, force_field, is_boundary=False):
    node_name = f'vert{nr_vertebra}'
    curr_vert = parent_node.addChild(node_name)
    curr_vert.addObject('MechanicalObject', name='rigid', template='Rigid3d')


    # ALL vertebrae get strong anchors — matches original code
    curr_vert.addObject('RestShapeSpringsForceField', name = 'fixedPoints',
                        stiffness='0', angularStiffness='0')
   
    # # add rigid center fixedconstraint to v0 and v3
    # if is_boundary:
    #     curr_vert.addObject('FixedConstraint', indices='0', template='Rigid3d')


    fc = [float(x) for x in force_field[f'vert{nr_vertebra}'].split()]
    rigid_force = f'{fc[0]} {fc[1]} {fc[2]} 0 0 0'
    print(f"vert{nr_vertebra}: total rigid force = {rigid_force}")


    # curr_vert.addObject('ConstantForceField', forces=rigid_force)


    # mechanical model node as child from current vertebra node
    mecha = curr_vert.addChild('mecha')
    mecha.addObject('MeshOBJLoader', name='loader', filename=mesh_file,
                     printLog='true', flipNormals='0')
    mecha.addObject('MeshTopology', name='topology', src='@loader')
    mecha.addObject('MechanicalObject', name='dofs', template='Vec3d', src='@topology')


    mesh_points_count = len(mecha.dofs.position.value)
    forces_vec = np.zeros((mesh_points_count, 3))


    if not is_boundary:
        forces_vec[:, :] = fc[:3]


    mecha.addObject('ConstantForceField',
                name='CFF',
                forces=forces_vec.tolist())
   
    mecha.addObject('TriangleCollisionModel')
    mecha.addObject('RigidMapping', input='@..', output='@.')


    visual = mecha.addChild('Visual')
    visual.addObject('OglModel', name='ogl', src='@../loader', color='white')
    visual.addObject('IdentityMapping')
    return curr_vert


def format_springs_for_sofa(spring_list, fraction=1.0, stiffness_scale=SPRING_SCALE):
    if fraction < 1.0:
        n_keep = max(1, int(len(spring_list) * fraction))
        spring_list = random.sample(spring_list, n_keep)
   
    formatted = []
    for s in spring_list:
        k = s['stiffness'] * stiffness_scale
        formatted.append(f"{s['i']} {s['j']} {k} {s['damping']} {s['d0']}")
    return ' '.join(formatted)



def add_springs_between_vertebrae(parent_node, springs_data, idx_first, idx_second):
    """
    Add StiffSpringForceField between the *mapped mesh DOFs* of two vertebrae.
    Paths: @vert{i}/mecha/dofs  <-->  @vert{j}/mecha/dofs
    """
    pair_name = f'v{idx_first}v{idx_second}'
    print(f"Adding springs for: {pair_name}")




    obj1 = f'@vert{idx_first}/mecha/dofs'
    obj2 = f'@vert{idx_second}/mecha/dofs'



    pair_data = springs_data['springs'][pair_name]





    # Body (disc) springs
    if 'body' in pair_data:
        body_springs = format_springs_for_sofa(pair_data['body'], 1.00)
        parent_node.addObject('StiffSpringForceField',
                              template='Vec3d',
                              name=f'body_{pair_name}',
                              object1=obj1,
                              object2=obj2,
                              spring=body_springs)
        print(f"  -> Added {len(pair_data['body'])} body springs")

    import random
    # Facet left springs
    if 'facet_left' in pair_data:
        all_fl_springs = pair_data['facet_left']
        sample_size = max(1, int(len(all_fl_springs) * 0.05))  # 30% of springs, at least 1
        fl_springs_sample = random.sample(all_fl_springs, sample_size)

        fl_springs = format_springs_for_sofa(fl_springs_sample, 1.00)
        parent_node.addObject('StiffSpringForceField',
                            template='Vec3d',
                            name=f'facet_left_{pair_name}',
                            object1=obj1,
                            object2=obj2,
                            spring=fl_springs)
        print(f"  -> Added {len(fl_springs_sample)} facet_left springs (30% sampled)")






def add_fixed_points(parent_node, nr_vertebra, springs_data):
    """Add fixed anchor points for boundary vertebrae (e.g. v0 and v3)."""
    key = f'v{nr_vertebra}'
    if key not in springs_data.get('fixed_points_positions', {}):
        print(f"No fixed points for {key}, skipping")
        return

    fp_node = parent_node.addChild(f'fixed_points_{nr_vertebra}')
    fp_node.addObject('MechanicalObject',
                      name='particles',
                      template='Vec3d',
                      position=springs_data['fixed_points_positions'][key])
    fp_node.addObject('MeshTopology', name='topology',
                      hexas=springs_data['fixed_points_indices'][key])
    fp_node.addObject('UniformMass', name='mass', vertexMass='1')
    fp_node.addObject('FixedConstraint',
                      template='Vec3d',
                      name='fix',
                      indices=springs_data['fixed_points_indices'][key])




#     return root
def create_scene(root, mesh_files, json_file, force_field):
    nr_vertebrae = len(mesh_files)



    # ---------- Scene setup ----------
    root.addObject('RequiredPlugin', name='Sofa.Component.Visual')
    root.addObject('DefaultVisualManagerLoop')
    root.addObject('DefaultAnimationLoop')
    root.addObject('VisualStyle',
                   displayFlags='showVisual showBehaviorModels showInteractionForceFields')




    add_collision_function(root)


    # ---------- Solver node (all vertebrae share one solver) ----------
    root.dt.value = 0.001
    solver_node = root.addChild('Spine')
    solver_node.addObject('EulerImplicitSolver', name='odesolver', printLog='0',
                          rayleighStiffness='0.09', rayleighMass='0.1')
    solver_node.addObject('CGLinearSolver', name='linsolver',
                          iterations='50', tolerance='1e-06', threshold='1e-15')



    # ---------- Add vertebrae (0-indexed) ----------
    for i, mesh in enumerate(mesh_files):
            is_boundary = (i == 0 or i == nr_vertebrae - 1)
            add_vertebra_node(solver_node, i, mesh, force_field, is_boundary=is_boundary)



    # ---------- Load springs JSON ----------
    with open(json_file, 'r') as f:
        springs_data = json.load(f)



    # ---------- Add springs between consecutive vertebrae ----------
    for i in range(nr_vertebrae - 1):
        pair_name = f'v{i}v{i+1}'
        if pair_name in springs_data.get('springs', {}):
            add_springs_between_vertebrae(solver_node, springs_data, i, i + 1)
        else:
            print(f"WARNING: Missing spring pair {pair_name}, skipping")



    # ---------- Fixed points for v0 ----------
    add_fixed_points(solver_node, 0, springs_data)
    v0_nearest = find_nearest_vertex_indices(mesh_files[0],
                    springs_data['fixed_points_positions']['v0'])
    v0_fixed_springs = build_fixed_springs(
                    springs_data['fixed_points_positions']['v0'], v0_nearest)
    solver_node.addObject('StiffSpringForceField',
                          name='anchor_v0',
                          object1='@fixed_points_0/particles',
                          object2='@vert0/mecha/dofs',
                          spring=v0_fixed_springs)

    # ---------- Fixed points for v3 ----------
    add_fixed_points(solver_node, 3, springs_data)
    v3_nearest = find_nearest_vertex_indices(mesh_files[3],
                    springs_data['fixed_points_positions']['v3'])
    v3_fixed_springs = build_fixed_springs(
                    springs_data['fixed_points_positions']['v3'], v3_nearest)
    solver_node.addObject('StiffSpringForceField',
                          name='anchor_v3',
                          object1='@fixed_points_3/particles',
                          object2='@vert3/mecha/dofs',
                          spring=v3_fixed_springs)



    return root







# ---------- RUN DEFORMATION ----------
def deform_one_spine(mesh_files, json_file, force_field, use_gui=True):
    root = Sofa.Core.Node('root')
    create_scene(root, mesh_files, json_file, force_field)
    Sofa.Simulation.init(root)








    if use_gui:
        Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
        Sofa.Gui.GUIManager.MainLoop(root)
        Sofa.Gui.GUIManager.closeGUI()
        print("GUI closed.")
    else:
        for iteration in range(20):
            print("Iteration:" + str(iteration))
            Sofa.Simulation.animate(root, root.dt.value)








    print("Simulation done.")








def createScene(root):
    mesh_files = [
        r'C:\Users\elise\Documents\SpineSimulation\spine_folder\original_L1\L1_decimated.obj',
        r'C:\Users\elise\Documents\SpineSimulation\spine_folder\original_L2\L2_decimated.obj',
        r'C:\Users\elise\Documents\SpineSimulation\spine_folder\original_L3\L3_decimated.obj',
        r'C:\Users\elise\Documents\SpineSimulation\spine_folder\original_L4\L4_decimated.obj',
    ]
    json_file = r'C:\Users\elise\Documents\SpineSimulation\spine_folder\sofa_springs_indices.json'
   
    force_field = make_force_fields(anterior=True)
   
    print("AP-aligned forces:", force_field)
   
    return create_scene(root, mesh_files, json_file, force_field)



















