"""Asimov 1 MuJoCo asset and actuator configuration."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


def _find_model_dir() -> Path:
    """Find the Cyclotron submodule without assuming a user's home path."""
    override = os.environ.get("ASIMOV_1_MODEL_DIR")
    if override:
        return Path(override).expanduser().resolve()

    anchors = (Path(__file__).resolve(), Path.cwd().resolve())
    for anchor in anchors:
        for parent in (anchor, *anchor.parents):
            candidate = parent / "third_party" / "asimov-1" / "sim-model"
            if (candidate / "xmls" / "asimov_1.xml").is_file():
                return candidate
    return Path.home() / "cyclotron" / "third_party" / "asimov-1" / "sim-model"


MODEL_DIR = _find_model_dir()
ASIMOV_XML = MODEL_DIR / "xmls" / "asimov_1.xml"

LIVERY_DIR = Path(__file__).resolve().parents[1] / "assets" / "reference_livery"


def apply_livery(spec: mujoco.MjSpec) -> None:
    """Use Blender-exported UVs and palette on visual meshes only."""
    spec.add_texture(
        name="asimov_palette",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=str(LIVERY_DIR / "palette.png"),
    )
    textures = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
    textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "asimov_palette"
    spec.add_material(
        name="asimov_reference",
        textures=textures,
        rgba=[1, 1, 1, 1],
        specular=0.25,
        shininess=0.3,
    )
    spec.add_texture(
        name="asimov_breastplate",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=str(LIVERY_DIR / "breastplate.png"),
    )
    chest_textures = textures.copy()
    chest_textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "asimov_breastplate"
    spec.add_material(
        name="asimov_chest",
        textures=chest_textures,
        rgba=[1, 1, 1, 1],
        specular=0.25,
        shininess=0.3,
    )
    original_meshes = {mesh.name: mesh for mesh in spec.meshes}
    colored_meshes = {}
    for geom in spec.geoms:
        if geom.group == 2 and geom.name.endswith("_visual"):
            if geom.meshname not in colored_meshes:
                original = original_meshes[geom.meshname]
                path = LIVERY_DIR / (Path(original.file).stem + ".obj")
                if not path.exists():
                    raise FileNotFoundError(f"Missing Blender livery mesh: {path}")
                name = "livery_" + geom.meshname
                spec.add_mesh(name=name, file=str(path), scale=original.scale)
                colored_meshes[geom.meshname] = name
            geom.meshname = colored_meshes[geom.meshname]
            geom.material = (
                "asimov_chest"
                if geom.name == "waist_yaw_link_visual"
                else "asimov_reference"
            )
            geom.rgba = [0.5, 0.5, 0.5, 1.0]


JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
)


def get_spec() -> mujoco.MjSpec:  # pyright: ignore[reportAttributeAccessIssue]
    """Load the official model and remove its standalone floor and lights."""
    if not ASIMOV_XML.exists():
        raise FileNotFoundError(
            f"Asimov MJCF was not found at {ASIMOV_XML}. Set ASIMOV_1_MODEL_DIR."
        )
    spec = mujoco.MjSpec.from_file(  # pyright: ignore[reportAttributeAccessIssue]
        str(ASIMOV_XML)
    )
    apply_livery(spec)
    for geom in tuple(spec.geoms):
        if geom.name == "floor":
            spec.delete(geom)
    for light in tuple(spec.lights):
        spec.delete(light)
    defaults = mujoco.MjSpec().option  # pyright: ignore[reportAttributeAccessIssue]
    spec.option.timestep = defaults.timestep
    spec.option.iterations = defaults.iterations
    spec.option.ls_iterations = defaults.ls_iterations
    spec.option.noslip_iterations = defaults.noslip_iterations
    spec.option.impratio = defaults.impratio
    spec.option.integrator = defaults.integrator
    spec.option.cone = defaults.cone
    return spec


ACTUATORS = (
    BuiltinPositionActuatorCfg(
        target_names_expr=(".*_hip_pitch_joint",),
        stiffness=150.0,
        damping=5.0,
        effort_limit=45.0,
        armature=0.0698,
        frictionloss=0.70,
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=(".*_hip_roll_joint",),
        stiffness=150.0,
        damping=5.0,
        effort_limit=45.0,
        armature=0.1400,
        frictionloss=0.20,
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=(".*_hip_yaw_joint",),
        stiffness=150.0,
        damping=5.0,
        effort_limit=28.0,
        armature=0.0687,
        frictionloss=0.70,
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=(".*_knee_joint",),
        stiffness=150.0,
        damping=5.0,
        effort_limit=45.0,
        armature=0.0330,
        frictionloss=0.70,
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=(".*_ankle_pitch_joint",),
        stiffness=110.0,
        damping=5.0,
        effort_limit=40.0,
        armature=0.0484,
        frictionloss=0.40,
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=(".*_ankle_roll_joint",),
        stiffness=110.0,
        damping=5.0,
        effort_limit=17.0,
        armature=0.0484,
        frictionloss=0.40,
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=("waist_yaw_joint",),
        stiffness=65.0,
        damping=5.0,
        effort_limit=40.0,
        armature=0.0698,
        frictionloss=0.70,
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=(".*_shoulder_pitch_joint",),
        stiffness=57.0,
        damping=5.0,
        effort_limit=30.0,
        armature=0.1400,
        frictionloss=0.20,
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=(".*_shoulder_roll_joint",),
        stiffness=86.0,
        damping=5.0,
        effort_limit=25.0,
        armature=0.0330,
        frictionloss=0.70,
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=(".*_shoulder_yaw_joint",),
        stiffness=96.0,
        damping=5.0,
        effort_limit=20.0,
        armature=0.0687,
        frictionloss=0.70,
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=(".*_elbow_joint", ".*_wrist_yaw_joint"),
        stiffness=40.0,
        damping=2.0,
        effort_limit=12.0,
        armature=0.0242,
        frictionloss=0.40,
    ),
)

STANDING_JOINT_POS = {
    "left_hip_pitch_joint": -0.15,
    "right_hip_pitch_joint": 0.15,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.45,
    "right_knee_joint": -0.45,
    "left_ankle_pitch_joint": -0.30,
    "right_ankle_pitch_joint": 0.30,
    ".*_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": -0.25,
    "right_shoulder_pitch_joint": 0.25,
    "left_shoulder_roll_joint": -0.05,
    "right_shoulder_roll_joint": 0.05,
    ".*_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.40,
    "right_elbow_joint": -0.40,
    ".*_wrist_yaw_joint": 0.0,
}

ARTICULATION = EntityArticulationInfoCfg(
    actuators=ACTUATORS,
    soft_joint_pos_limit_factor=0.95,
)


def get_robot_cfg() -> EntityCfg:
    """Return a fresh Asimov configuration with Menlo's standing joint pose."""
    return EntityCfg(
        spec_fn=get_spec,
        articulation=ARTICULATION,
        sort_actuators=True,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.63),
            joint_pos=STANDING_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
    )
