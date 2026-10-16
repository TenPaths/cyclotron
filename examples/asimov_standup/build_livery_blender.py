"""Run with Blender --background --python build_livery_blender.py.

Creates editable materials, per-face palette UVs, OBJ assets for MuJoCo, and
a studio preview. The palette follows the user's red/black/bronze photo.
Only appearance changes: no subdivision, remeshing, or shape modifications.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parent
SOURCE = Path(
    os.environ.get(
        "ASIMOV_1_MODEL_DIR",
        ROOT.parents[1] / "third_party" / "asimov-1" / "sim-model",
    )
).expanduser()
OUT = ROOT / "assets" / "reference_livery"
OUT.mkdir(parents=True, exist_ok=True)

# Swatches are stored as sRGB in the PNG; Blender shader inputs are linear.
COLORS = [(22, 27, 30), (235, 79, 14), (174, 128, 77), (172, 180, 184)]
NAMES = ["Charcoal polymer", "Vermilion shell", "Bronze joint covers", "Brushed metal"]


def linear(c):
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def face_color(name, p, n):
    x, y, z = p
    if name == "NECK_PITCH":
        # Keep the forward faceplate black, with a colored perimeter and rear shell.
        visor = x > 0.025 and abs(y) < 0.052 and -0.025 < z < 0.115
        return 0 if visor or z < -0.025 else 1
    if name == "WAIST_YAW":
        # Front reference: orange breastplate with a black, arched belly cutout.
        # The bottom edge rises centrally and extends lower at the side lobes.
        lower_edge = 0.10 + 0.075 * max(0.0, 1.0 - (abs(y) / 0.075) ** 2)
        return 1 if lower_edge < z < 0.315 else 0
    if name == "ANKLE_B_NEW":
        return 1 if z < -0.0244 else 0
    if "SHOULDER_PITCH" in name:
        return 1 if abs(x) > 0.032 and abs(n.x) > 0.6 else 0
    if "ELBOW" in name:
        return 2 if abs(y) > 0.0275 and abs(n.y) > 0.65 else 0
    if "KNEE" in name:
        cover = abs(y) > 0.032 and abs(x) < 0.041 and -0.107 < z < 0.041
        return 2 if cover and abs(n.y) > 0.6 else 0
    if "ANKLE_A" in name:
        return 3 if abs(y) > 0.025 else 0
    return 0


def cover_components(mesh, name):
    """Color complete disconnected metal covers instead of slicing their edges."""
    if not any(part in name for part in ("KNEE", "ELBOW")):
        return None
    parents = list(range(len(mesh.vertices)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for edge in mesh.edges:
        a, b = (root(i) for i in edge.vertices)
        parents[a] = b
    bounds = {}
    for vertex in mesh.vertices:
        r = root(vertex.index)
        if r not in bounds:
            bounds[r] = [vertex.co.copy(), vertex.co.copy()]
        lo, hi = bounds[r]
        for axis in range(3):
            lo[axis] = min(lo[axis], vertex.co[axis])
            hi[axis] = max(hi[axis], vertex.co[axis])
    colors = {}
    for r, (lo, hi) in bounds.items():
        if "SHOULDER_PITCH" in name:
            gold = lo.x > 0.032 or hi.x < -0.032
        else:
            limit = 0.032 if "KNEE" in name else 0.027
            gold = (lo.y > limit or hi.y < -limit) and hi.z > 0.03
        colors[r] = 2 if gold else 0
    return [colors[root(poly.vertices[0])] for poly in mesh.polygons]


bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
materials = []
for i, (name, rgb) in enumerate(zip(NAMES, COLORS)):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*[linear(c) for c in rgb], 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = mat.diffuse_color
    bsdf.inputs["Metallic"].default_value = 0.65 if i >= 2 else 0.0
    bsdf.inputs["Roughness"].default_value = 0.34 if i >= 2 else 0.3
    materials.append(mat)

atlas = bpy.data.images.new("Asimov palette atlas", width=256, height=64)
atlas.colorspace_settings.name = "sRGB"
atlas.pixels = [
    v
    for _y in range(64)
    for x in range(256)
    for v in (*[c / 255.0 for c in COLORS[x // 64]], 1.0)
]
atlas.filepath_raw = str(OUT / "palette.png")
atlas.file_format = "PNG"
atlas.save()
atlas.pack()

# A planar torso UV texture makes the curved breastplate edge independent of
# the coarse STL triangles. Palette-per-face painting would give a jagged edge.
chest = bpy.data.images.new("Breastplate paint", width=512, height=1024)
chest.colorspace_settings.name = "sRGB"
pixels = []
for row in range(1024):
    z = -0.01 + (row + 0.5) / 1024 * 0.40
    for col in range(512):
        y = -0.13 + (col + 0.5) / 512 * 0.26
        lower = 0.10 + 0.075 * max(0.0, 1.0 - (abs(y) / 0.075) ** 2)
        color = COLORS[1 if lower < z < 0.315 else 0]
        pixels.extend([*[v / 255 for v in color], 1.0])
chest.pixels = pixels
chest.filepath_raw = str(OUT / "breastplate.png")
chest.file_format = "PNG"
chest.save()
chest.pack()
chest_material = materials[0].copy()
chest_material.name = "Orange breastplate with curved belly cutout"
node = chest_material.node_tree.nodes.new("ShaderNodeTexImage")
node.image = chest
chest_material.node_tree.links.new(
    node.outputs["Color"],
    chest_material.node_tree.nodes.get("Principled BSDF").inputs["Base Color"],
)

tree = ET.parse(SOURCE / "xmls" / "asimov_1.xml")
mesh_files = {m.attrib["name"]: m.attrib["file"] for m in tree.findall("./asset/mesh")}
objects = {}
for mesh_name, file in mesh_files.items():
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.wm.stl_import(filepath=str(SOURCE / "assets" / "meshes" / file))
    obj = bpy.context.object
    obj.name = mesh_name
    bpy.ops.object.shade_smooth_by_angle()
    for mat in materials:
        obj.data.materials.append(mat)
    uv = obj.data.uv_layers.new(name="PaletteUV")
    covers = cover_components(obj.data, Path(file).stem)
    for poly in obj.data.polygons:
        color = face_color(Path(file).stem, poly.center, poly.normal)
        if covers is not None and covers[poly.index] == 2:
            color = 2
        poly.material_index = color
        for loop in poly.loop_indices:
            uv.data[loop].uv = ((color + 0.5) / 4.0, 0.5)
    if Path(file).stem == "WAIST_YAW":
        obj.data.materials.append(chest_material)
        for poly in obj.data.polygons:
            poly.material_index = len(obj.data.materials) - 1
            for loop in poly.loop_indices:
                v = obj.data.vertices[obj.data.loops[loop].vertex_index].co
                uv.data[loop].uv = ((v.y + 0.13) / 0.26, (v.z + 0.01) / 0.40)
    # Export in original link coordinates; MuJoCo still owns all articulation.
    bpy.ops.wm.obj_export(
        filepath=str(OUT / f"{Path(file).stem}.obj"),
        export_selected_objects=True,
        forward_axis="Y",
        up_axis="Z",
        export_materials=False,
        export_uv=True,
        export_normals=True,
        apply_modifiers=False,
    )
    objects[mesh_name] = obj


def transform(element):
    pos = [float(v) for v in element.get("pos", "0 0 0").split()]
    quat = [float(v) for v in element.get("quat", "1 0 0 0").split()]
    return Matrix.Translation(Vector(pos)) @ Quaternion(quat).to_matrix().to_4x4()


def place(body, parent):
    world = parent @ transform(body)
    for geom in body.findall("geom"):
        if geom.get("class") != "visual":
            continue
        source = objects[geom.attrib["mesh"]]
        obj = source if not source.get("placed") else source.copy()
        if obj is not source:
            bpy.context.collection.objects.link(obj)
        obj.matrix_world = world @ transform(geom)
        obj["placed"] = True
    for child in body.findall("body"):
        place(child, world)


for body in tree.findall("./worldbody/body"):
    place(body, Matrix.Identity(4))

bpy.ops.mesh.primitive_plane_add(size=200)
floor = bpy.context.object
floor.name = "Studio floor (not exported)"
floor_mat = bpy.data.materials.new("Studio floor")
floor_mat.diffuse_color = (0.065, 0.075, 0.09, 1)
floor.data.materials.append(floor_mat)
for pos, power, size in [
    ((2, -3, 4), 450, 4),
    ((-2, 1, 3), 600, 3),
    ((1, 3, 2), 200, 3),
]:
    bpy.ops.object.light_add(type="AREA", location=pos)
    lamp = bpy.context.object
    lamp.data.energy = power
    lamp.data.shape = "DISK"
    lamp.data.size = size
    lamp.rotation_euler = (
        (Vector((0, 0, 0.7)) - lamp.location).to_track_quat("-Z", "Y").to_euler()
    )
bpy.ops.object.camera_add(location=(3.0, -2.5, 1.65))
camera = bpy.context.object
camera.rotation_euler = (
    (Vector((0, 0, 0.68)) - camera.location).to_track_quat("-Z", "Y").to_euler()
)
camera.data.type = "ORTHO"
camera.data.ortho_scale = 1.6
scene = bpy.context.scene
scene.camera = camera
scene.render.engine = "CYCLES"
scene.cycles.samples = 24
scene.render.resolution_x = 900
scene.render.resolution_y = 1000
scene.render.resolution_percentage = 100
scene.world.color = (0.3, 0.3, 0.3)
scene.view_settings.view_transform = "AgX"
scene.render.filepath = str(OUT / "preview.png")
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "asimov_reference.blend"))
bpy.ops.render.render(write_still=True)
print("Reference livery exported:", OUT)
