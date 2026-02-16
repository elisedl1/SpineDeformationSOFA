import json
import pyvista as pv
import numpy as np

# ---------- SETTINGS ----------
mesh_files = [
    r"C:\Users\elise\Documents\SpineSimulation\spine_folder\original_L1\L1_decimated.obj",
    r"C:\Users\elise\Documents\SpineSimulation\spine_folder\original_L2\L2_decimated.obj",
    r"C:\Users\elise\Documents\SpineSimulation\spine_folder\original_L3\L3_decimated.obj",
    r"C:\Users\elise\Documents\SpineSimulation\spine_folder\original_L4\L4_decimated.obj"
]

input_json_file = r"C:\Users\elise\Documents\SpineSimulation\spine_folder\sofa_springs.json"
output_json_file = r"C:\Users\elise\Documents\SpineSimulation\spine_folder\sofa_springs_indices.json"


# ---------- HELPER FUNCTIONS ----------
def load_mesh_vertices(mesh_file):
    mesh = pv.read(mesh_file)
    return np.array(mesh.points)

def find_closest_vertex(mesh_vertices, point):
    """
    Find the vertex index in mesh_vertices closest to the given 3D point.
    """
    distances = np.linalg.norm(mesh_vertices - point, axis=1)
    return int(np.argmin(distances))

# ---------- LOAD MESHES ----------
meshes_vertices = [load_mesh_vertices(f) for f in mesh_files]

# ---------- LOAD SPRINGS JSON ----------
with open(input_json_file, 'r') as f:
    springs_data = json.load(f)

# ---------- CONVERT SPRINGS ----------
converted_springs = {}

for pair_name, spring_types in springs_data['springs'].items():
    converted_springs[pair_name] = {}
    
    # Determine which vertebra indices the spring is between
    v0 = int(pair_name[1])
    v1 = int(pair_name[3])
    verts0 = meshes_vertices[v0]
    verts1 = meshes_vertices[v1]
    
    for spring_type, spring_list in spring_types.items():
        converted_list = []
        for spring in spring_list:
            i_index = find_closest_vertex(verts0, np.array(spring['i']))
            j_index = find_closest_vertex(verts1, np.array(spring['j']))
            
            converted_spring = {
                "i": i_index,
                "j": j_index,
                "d0": spring["d0"],
                "stiffness": spring["stiffness"],
                "damping": spring["damping"]
            }
            converted_list.append(converted_spring)
        
        converted_springs[pair_name][spring_type] = converted_list

# ---------- HANDLE FIXED POINTS ----------
# Simply copy them over if they are already indices
converted_fixed_points_positions = springs_data.get('fixed_points_positions', {})
converted_fixed_points_indices = springs_data.get('fixed_points_indices', {})

# ---------- SAVE NEW JSON ----------
converted_data = {
    "springs": converted_springs,
    "fixed_points_positions": converted_fixed_points_positions,
    "fixed_points_indices": converted_fixed_points_indices
}

with open(output_json_file, 'w') as f:
    json.dump(converted_data, f, indent=4)

print(f"Conversion done! Saved to {output_json_file}")
