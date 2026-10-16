# Asimov reference livery

Geometry: Menlo Research, https://github.com/menloresearch/asimov-1
Source revision: 732cc60dcb8f2b4fd26c3d7346b35f9b89c3cd47
Source paths: sim-model/assets/meshes/*.STL
License: see SOFTWARE-LICENSE.txt (upstream GPL-2.0).

Modifications: Blender materials, per-face palette UVs, smooth shading normals,
OBJ export and assembled Blender scene, based on the user's supplied photograph.
STL import can remove coincident duplicate triangles. Vertex positions and
articulation are preserved; physics still uses the original collision model.

The .blend holds separate editable materials. palette.png and the OBJ UVs
reproduce their base colors in MuJoCo/Viser, without Blender's full metal shader.
The result is an approximation, not an official texture or a new robot model.
