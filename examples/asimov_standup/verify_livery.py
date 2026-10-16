"""Check exported livery geometry, physics, Viser UVs and MuJoCo rendering."""

import os

os.environ["MUJOCO_GL"] = "egl"
import mujoco
import numpy as np
from asimov_standup import asset
from PIL import Image

spec = asset.get_spec()
model = spec.compile()
from mjviser.conversions import mujoco_mesh_to_trimesh

for i in range(model.ngeom):
    if model.geom_group[i] == 2:
        mesh = mujoco_mesh_to_trimesh(model, i)
        assert mesh.visual.kind == "texture"
        assert mesh.visual.material.baseColorTexture is not None
print("PASS: all visual parts retain textures in the Viser conversion")
asset.apply_livery = lambda spec: None
baseline = asset.get_spec().compile()
for field in (
    "body_mass",
    "body_inertia",
    "body_pos",
    "geom_friction",
    "geom_contype",
    "geom_conaffinity",
    "jnt_range",
    "dof_damping",
    "qpos0",
):
    np.testing.assert_array_equal(getattr(model, field), getattr(baseline, field))
collision = baseline.geom_group == 3
for field in ("geom_size", "geom_pos", "geom_quat"):
    np.testing.assert_array_equal(
        getattr(model, field)[collision], getattr(baseline, field)[collision]
    )
# Compare actual world-space visual bounds (OBJ export must not rotate/scale).
for current in (model, baseline):
    d = mujoco.MjData(current)
    mujoco.mj_forward(current, d)
    bounds = []
    for i in range(current.ngeom):
        if current.geom_group[i] != 2:
            continue
        mid = current.geom_dataid[i]
        start = current.mesh_vertadr[mid]
        vertices = current.mesh_vert[start : start + current.mesh_vertnum[mid]]
        world = vertices @ d.geom_xmat[i].reshape(3, 3).T + d.geom_xpos[i]
        bounds.append([world.min(0), world.max(0)])
    if current is model:
        colored_bounds = np.array(bounds)
    else:
        np.testing.assert_allclose(colored_bounds, bounds, atol=2e-5)
print("PASS: dynamics/contact properties and visual geometry bounds unchanged")
spec.worldbody.add_light(pos=[2, -3, 4], dir=[-2, 3, -4], diffuse=[0.8] * 3)
spec.worldbody.add_light(pos=[-2, 1, 3], dir=[2, -1, -3], diffuse=[0.5] * 3)
spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.18, 0.20, 0.23, 1]
)
model = spec.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
camera = mujoco.MjvCamera()
camera.lookat[:] = [0, 0, 0.65]
camera.distance = 2.2
camera.azimuth = 135
camera.elevation = -12
option = mujoco.MjvOption()
option.geomgroup[3] = 0
with mujoco.Renderer(model, height=900, width=900) as renderer:
    renderer.update_scene(data, camera=camera, scene_option=option)
    Image.fromarray(renderer.render()).save(asset.LIVERY_DIR / "mujoco-preview.png")
print("PASS: textured MuJoCo render saved")
