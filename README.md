# SOFA Spine Deformation Simulation

A biomechanically-constrained lumbar spine deformation framework using SOFA (Simulation Open Framework Architecture) to generate realistic spinal curvature changes for CT-to-ultrasound registration validation.

## Overview

This project implements a finite element-based spine simulation to generate anatomically plausible spinal deformations representing postural changes (e.g., supine to prone positioning). The deformations are used to validate groupwise CT-US registration algorithms for intraoperative spinal navigation.

**Based on:** "Fast Robust Groupwise CT-US Registration for Intraoperative Spinal Interventions" (see `Real_time_Robust_Groupwise_CT_US_Registration.pdf`)

## Key Features

- **Anatomically realistic deformations**: Simulates inter-vertebral disc spacing, facet joint kinematics, and collision avoidance
- **Beam-based FEM model**: Efficient representation of overall spinal biomechanics
- **Multiple constraint mechanisms**: 
  - RestShapeSprings for disc spacing maintenance
  - Collision detection between vertebral endplates
  - Facet joint contact modeling
- **Automated mesh processing**: Converts SOFA VTU outputs to OBJ format for registration pipelines
- **Flexible force configurations**: Supports anterior/posterior loading, lateral bending, and custom force patterns

## Architecture

### Simulation Components

The spine model consists of:

1. **Beam FEM structure**: Continuous elastic beam connecting vertebral centers
2. **Vertebral bodies**: Individual rigid meshes for L1-L4 with:
   - Body mesh (vertebra)
   - Endplate meshes (superior/inferior surfaces for disc contact)
   - Facet meshes (articular processes for facet joint contact)
3. **Biomechanical constraints**:
   - Inter-vertebral disc spacing via RestShapeSprings
   - Collision models on endplates and facets
   - Fixed boundary conditions at L1 (superior) and L4 (inferior)

### Coordinate System

The simulation uses anatomically-oriented axes:

- **AP (Anterior-Posterior)**: Hardcoded from CT imaging coordinate system
- **SI (Superior-Inferior)**: Computed from L1→L4 direction, orthogonalized to AP
- **LM (Lateral-Medial)**: Right-hand cross product of AP × SI

## Installation

### Prerequisites

- **SOFA Framework**: v22.12.00 or later ([download](https://www.sofa-framework.org/download/))
- **Python**: 3.8+ with SofaPython3 plugin
- **Python packages**:
  ```bash
  pip install numpy scipy trimesh meshio pyvista vtk
  ```

### SOFA Plugins Required

Ensure these SOFA plugins are enabled:
- Sofa.Component.Collision.Detection.Algorithm
- Sofa.Component.Collision.Geometry
- Sofa.Component.SolidMechanics.FEM.Elastic
- Sofa.Component.SolidMechanics.Spring
- Sofa.Component.Mapping.Linear
- Sofa.GL.Component.Rendering3D

## Usage

### 1. Interactive Simulation (GUI)

Run a single spine deformation with visual feedback:

```bash
runSofa -l SofaPython3 deform_beam.py
```

**Controls in SOFA GUI**:
- **Animate**: Start/stop simulation
- **Step**: Advance one time step
- **Reset**: Return to initial configuration
- **Ctrl+E**: Export current deformed meshes to VTU

### 2. Batch Processing

Generate multiple deformations for a dataset:

```bash
python sofa_spine_deformation.py \
    --root_path_vertebrae ./spine_folder \
    --parameters_file ./parameters.json \
    --list_file_names spine_list.txt \
    --deform_all \
    --nr_deform_per_spine 3 \
    --forces_folder ./forces
```

### 3. Custom Force Configuration

Edit `VERTEBRA_FORCE_CONFIG` in `deform_beam.py`:

```python
VERTEBRA_FORCE_CONFIG = {
    0: {'magnitude': 0.5, 'axis': 'AP', 'positive': True},   # L1: anterior push
    1: {'magnitude': 1.2, 'axis': 'AP', 'positive': True},   # L2: stronger anterior
    2: {'magnitude': 1.2, 'axis': 'AP', 'positive': True},   # L3: stronger anterior
    3: {'magnitude': 0.5, 'axis': 'AP', 'positive': True}    # L4: anterior push
}
```

**Axis options**: `'AP'` (anterior-posterior), `'SI'` (superior-inferior), `'LM'` (lateral-medial)  
**Direction**: `positive: True` = positive axis direction, `False` = negative

## Parameters

### Biomechanical Parameters

| Parameter | Value | Description | Source |
|-----------|-------|-------------|--------|
| **Beam Young's Modulus** | 5,000 Pa | Overall spine stiffness | Tuned for clinical curvature range |
| **Beam Poisson Ratio** | 0.45 | Beam lateral contraction | Standard soft tissue value |
| **Beam Radius** | 15 mm | Beam cross-section size | Approximate vertebral body width |
| **RestShape Stiffness** | 500 N/m | Disc spacing enforcement | Balances flexibility/stability |
| **RestShape Angular Stiffness** | 500 N·m/rad | Rotational disc constraint | Prevents excessive rotation |

### Collision Parameters (Per-Vertebra Proximity)

| Vertebra | Proximity (mm) | Notes |
|----------|----------------|-------|
| L1 | 1.175 | Superior endplate spacing |
| L2 | 1.395 | Mid-lumbar disc height |
| L3 | 1.405 | Mid-lumbar disc height |
| L4 | 1.185 | Inferior endplate spacing |

These values represent the minimum allowed inter-vertebral distance before collision forces activate.

### Solver Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Time Step (dt) | 0.01 s | Integration time step |
| Rayleigh Stiffness | 0.1 | Numerical damping on stiffness |
| Rayleigh Mass | 0.1 | Numerical damping on mass |
| Max Iterations | 1000 | Constraint solver iterations |
| Tolerance | 1×10⁻⁶ | Convergence threshold |

### Force Configurations (Examples)

#### Anterior Flexion (Supine → Prone)
```python
L1: 0.5 N anterior
L2: 1.2 N anterior
L3: 1.2 N anterior
L4: 0.5 N anterior
```
**Result**: 3-7° inter-vertebral angle change, 4-7 mm TRE

#### Lateral Bending (Right)
```python
L1: 0.5 N lateral (right)
L2: 1.2 N lateral (right)
L3: 1.2 N lateral (right)
L4: 0.5 N lateral (right)
```
**Result**: 4-7° lateral curvature, 5-8 mm TRE

## File Structure

```
project/
├── deform_beam.py              # Main beam-based simulation (recommended)
├── deform.py                   # Spring-based simulation (legacy)
├── NewSpine.py                 # Alternative beam implementation
├── sofa_spine_deformation.py   # Batch processing script
├── spine_scene.py              # SOFA scene template
├── parameters.json             # Biomechanical parameters
├── compute_rigid_transform.py  # Extract transforms from deformed meshes
├── vtu_to_obj.py              # Convert SOFA output to OBJ
├── centroid_vtk.py            # Compute vertebral centroids
└── README.md                  # This file

spine_folder/
├── original_L1/
│   ├── L1_decimated.obj       # Vertebral body mesh
│   ├── L1_ends_decimated.obj  # Endplate surfaces
│   └── L1_facet_decimated.obj # Facet joint surfaces
├── original_L2/
├── original_L3/
└── original_L4/
```

## Workflow

### 1. Prepare Input Meshes

Ensure each vertebra folder contains three decimated meshes:
- `L{N}_decimated.obj`: Main vertebral body
- `L{N}_ends_decimated.obj`: Superior/inferior endplates
- `L{N}_facet_decimated.obj`: Facet joint surfaces

**Decimation target**: ~5,000-10,000 vertices per mesh for efficiency

### 2. Configure Simulation

Edit parameters in `deform_beam.py`:
```python
MESH_DIR = r'C:\path\to\spine_folder'
VERTEBRA_CENTERS = [
    [-27.46, 231.49, -327.83],  # L1
    [-25.04, 208.16, -304.45],  # L2
    [-23.19, 183.54, -279.33],  # L3
    [-21.37, 157.01, -256.02]   # L4
]
```

Compute centroids using:
```bash
python centroid_vtk.py
```

### 3. Run Simulation

```bash
# Interactive (GUI)
runSofa -l SofaPython3 deform_beam.py

# Headless (batch)
python deform_beam.py --no-gui --iterations 200
```

### 4. Extract Results

Deformed meshes are saved as VTU files in `VTK_OUTPUT_DIR`:
```
deformed/
├── L1_deformed_0.vtu
├── L2_deformed_0.vtu
├── L3_deformed_0.vtu
└── L4_deformed_0.vtu
```

Convert to OBJ:
```bash
python vtu_to_obj.py --input deformed/
```

### 5. Compute Rigid Transforms

Extract per-vertebra rigid transformations for registration:
```bash
python compute_rigid_transform.py \
    --original_dir spine_folder/original_sofa \
    --deformed_dir spine_folder/deformed \
    --output_dir transforms
```

Output: ITK `.tfm` files compatible with 3D Slicer

## Validation Results

From paper validation using porcine cadaver data:

| Metric | Value | Description |
|--------|-------|-------------|
| **Mean TRE** | 1.65 ± 0.42 mm | Target Registration Error across all conditions |
| **IVD Angle Change** | 3-8° | Inter-vertebral disc angle variation |
| **Initial Misalignment** | 5-10 mm | Starting TRE before registration |
| **Clinically Acceptable** | < 2 mm | Success threshold for spinal surgery |

### Deformation Scenarios Tested

1. **Mild anterior curvature**: Forces [0.5, 2.0, 2.0, 0.5] N → ΔTRE = 4-7 mm
2. **Moderate anterior curvature**: Forces [0.5, 1.8, 1.8, 0.5] N → ΔTRE = 5-9 mm
3. **Moderate left curvature**: Forces [0.5, 1.2, 1.2, 0.5] N → ΔTRE = 5-8 mm

All scenarios produced physiologically plausible inter-vertebral kinematics matching literature ranges.

## Troubleshooting

### Issue: Simulation Instability

**Symptoms**: Divergence, NaN values, explosion

**Solutions**:
1. Reduce time step: `root.dt = 0.005` (instead of 0.01)
2. Increase Rayleigh damping: `rayleighStiffness=0.2, rayleighMass=0.2`
3. Tighten solver tolerance: `tolerance=1e-9`
4. Reduce force magnitudes by 50%

### Issue: VTK Export Not Working

**Symptoms**: No `.vtu` files generated

**Solutions**:
1. Verify `SAVE_VTK = True` in `deform_beam.py`
2. Check output directory exists and is writable
3. Ensure `Sofa.Component.IO.Mesh` plugin is loaded
4. Try manual export with Ctrl+E in SOFA GUI

## Advanced Usage

### Creating Custom Deformation Scenarios

1. **Define anatomical axes** based on your CT coordinate system:
```python
AP_AXIS_RAW = np.array([...])  # From CT metadata
```

2. **Configure force patterns** for specific clinical scenarios:
```python
# Example: Simulate surgical positioning
VERTEBRA_FORCE_CONFIG = {
    0: {'magnitude': 0.3, 'axis': 'SI', 'positive': False},  # Caudal compression
    1: {'magnitude': 0.8, 'axis': 'AP', 'positive': True},   # Anterior push
    2: {'magnitude': 0.8, 'axis': 'AP', 'positive': True},
    3: {'magnitude': 0.3, 'axis': 'SI', 'positive': False}
}
```

3. **Adjust biomechanical parameters** for patient-specific properties:
```python
# Stiffer spine 
BEAM_YOUNG_MODULUS = 8000
REST_SHAPE_STIFFNESS = 800

# More flexible spine 
BEAM_YOUNG_MODULUS = 3000
REST_SHAPE_STIFFNESS = 300
```

### Integrating with Registration Pipeline

After generating deformations:

1. Use deformed meshes as "intraoperative" data (i.e. ultrasound)
2. Apply `.tfm` transforms to original CT volumes
3. Validate registration algorithms against ground truth transforms
4. Compute TRE using anatomical landmarks


## References

1. SOFA Framework: https://www.sofa-framework.org/
