# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov-1 configurations."""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
ASIMOV_1_MODEL_DIR = str(_REPOSITORY_ROOT / "third_party" / "asimov-1" / "sim-model")
ASIMOV_1_URDF_PATH = str(
    Path(os.environ.get("ASIMOV_1_MODEL_DIR", ASIMOV_1_MODEL_DIR)).expanduser() / "urdf" / "asimov_1.urdf"
)
ASIMOV_1_USD_PATH = str(Path(__file__).with_name("asimov_1_kitless.usdc"))


DELAY_MIN_LAG = 0
DELAY_MAX_LAG = 5
ASIMOV_1_ACTION_SCALE = 0.25


ASIMOV_1_JOINT_NAMES = [
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
]


ASIMOV_1_ACTUATORS = {
    "hip_pitch": DelayedPDActuatorCfg(
        joint_names_expr=[".*_hip_pitch_joint"],
        stiffness=150.0,
        damping=5.0,
        effort_limit=45.0,
        armature=0.0698,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "hip_roll": DelayedPDActuatorCfg(
        joint_names_expr=[".*_hip_roll_joint"],
        stiffness=150.0,
        damping=5.0,
        effort_limit=45.0,
        armature=0.1400,
        friction=0.20,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "hip_yaw": DelayedPDActuatorCfg(
        joint_names_expr=[".*_hip_yaw_joint"],
        stiffness=150.0,
        damping=5.0,
        effort_limit=28.0,
        armature=0.0687,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "knee": DelayedPDActuatorCfg(
        joint_names_expr=[".*_knee_joint"],
        stiffness=150.0,
        damping=5.0,
        effort_limit=45.0,
        armature=0.0330,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "ankle_pitch": DelayedPDActuatorCfg(
        joint_names_expr=[".*_ankle_pitch_joint"],
        stiffness=110.0,
        damping=5.0,
        effort_limit=40.0,
        armature=0.0484,
        friction=0.40,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "ankle_roll": DelayedPDActuatorCfg(
        joint_names_expr=[".*_ankle_roll_joint"],
        stiffness=110.0,
        damping=5.0,
        effort_limit=17.0,
        armature=0.0484,
        friction=0.40,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "waist": DelayedPDActuatorCfg(
        joint_names_expr=["waist_yaw_joint"],
        stiffness=65.0,
        damping=5.0,
        effort_limit=40.0,
        armature=0.0698,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "shoulder_pitch": DelayedPDActuatorCfg(
        joint_names_expr=[".*_shoulder_pitch_joint"],
        stiffness=57.0,
        damping=5.0,
        effort_limit=30.0,
        armature=0.1400,
        friction=0.20,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "shoulder_roll": DelayedPDActuatorCfg(
        joint_names_expr=[".*_shoulder_roll_joint"],
        stiffness=86.0,
        damping=5.0,
        effort_limit=25.0,
        armature=0.0330,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "shoulder_yaw": DelayedPDActuatorCfg(
        joint_names_expr=[".*_shoulder_yaw_joint"],
        stiffness=96.0,
        damping=5.0,
        effort_limit=20.0,
        armature=0.0687,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "elbow_wrist": DelayedPDActuatorCfg(
        joint_names_expr=[".*_elbow_joint", ".*_wrist_yaw_joint"],
        stiffness=40.0,
        damping=2.0,
        effort_limit=12.0,
        armature=0.0242,
        friction=0.40,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
}


def _implicit_actuator(joint_names_expr, stiffness, damping, effort, armature, friction):
    """Create a solver-side PD drive for standalone Isaac Lab/Newton.

    ``effort_limit_sim`` is intentional: in the installed standalone Isaac Lab
    API it is the solver-drive clamp, whereas ``effort_limit`` is the
    actuator-model-side field used by explicit models.
    """
    return ImplicitActuatorCfg(
        joint_names_expr=joint_names_expr,
        stiffness=stiffness,
        damping=damping,
        effort_limit_sim=effort,
        armature=armature,
        friction=friction,
    )


# Start from the official Cyclotron measured/model-derived actuator settings,
# adapted below for Newton's implicit solver response.
# Cyclotron's locomotion policy uses conservative continuous clamps (45 Nm at
# the hip/knee).  Single support roughly doubles the leg load, so use the
# published asimov-v0 peak limits for the shared leg hardware.  Cyclotron's
# armature and friction values remain unchanged.
ASIMOV_1_IMPLICIT_ACTUATORS = {
    # Newton's implicit drives need more damping than Cyclotron's delayed
    # explicit PD loop.  The sagittal 200/10 gains are also in line with
    # Isaac Lab's H1 implicit configuration, while the hardware peak-torque
    # clamps below remain unchanged.
    "hip_pitch": _implicit_actuator([".*_hip_pitch_joint"], 200.0, 10.0, 120.0, 0.0698, 0.70),
    "hip_roll": _implicit_actuator([".*_hip_roll_joint"], 150.0, 10.0, 90.0, 0.1400, 0.20),
    "hip_yaw": _implicit_actuator([".*_hip_yaw_joint"], 150.0, 8.0, 60.0, 0.0687, 0.70),
    "knee": _implicit_actuator([".*_knee_joint"], 200.0, 10.0, 75.0, 0.0330, 0.70),
    "ankle_pitch": _implicit_actuator([".*_ankle_pitch_joint"], 110.0, 5.0, 36.0, 0.0484, 0.40),
    "ankle_roll": _implicit_actuator([".*_ankle_roll_joint"], 110.0, 5.0, 36.0, 0.0484, 0.40),
    "waist": _implicit_actuator(["waist_yaw_joint"], 65.0, 5.0, 40.0, 0.0698, 0.70),
    "shoulder_pitch": _implicit_actuator([".*_shoulder_pitch_joint"], 57.0, 5.0, 30.0, 0.1400, 0.20),
    "shoulder_roll": _implicit_actuator([".*_shoulder_roll_joint"], 86.0, 5.0, 25.0, 0.0330, 0.70),
    "shoulder_yaw": _implicit_actuator([".*_shoulder_yaw_joint"], 96.0, 5.0, 20.0, 0.0687, 0.70),
    "elbow_wrist": _implicit_actuator(
        [".*_elbow_joint", ".*_wrist_yaw_joint"], 40.0, 2.0, 12.0, 0.0242, 0.40
    ),
}


ASIMOV_1_STANDING_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.639),
    joint_pos={
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
    },
    joint_vel={".*": 0.0},
)


# Start the progressive single-support curriculum from Cyclotron's own
# actuator-matched double-support pose.  The task then shifts load to the left
# foot and raises the right foot instead of spawning in an unstable end pose.
ASIMOV_1_ONE_LEG_INIT_STATE = ArticulationCfg.InitialStateCfg(
    # The lightweight kitless USD uses primitive foot collisions rather than
    # Cyclotron's original imported colliders.  At z=0.639 those colliders
    # begin about 3 cm above the floor, so the first touchdown injects enough
    # energy to buckle the knees before the policy can react.  This measured
    # height starts the same nominal pose at contact instead.
    pos=(0.0, 0.0, 0.610),
    rot=(0.0, 0.0, 0.0, 1.0),
    # Start the balance skill with the right leg already folded.  Asking PPO
    # to discover foot lift from a stable double-support local optimum proved
    # unnecessarily sparse; the transfer/transition policy can be trained
    # after the single-support controller itself is reliable.
    joint_pos={
        **{
            name: value
            for name, value in ASIMOV_1_STANDING_INIT_STATE.joint_pos.items()
            if name not in (".*_hip_roll_joint", ".*_ankle_roll_joint")
        },
        # Translate the pelvis over the support sole through the loaded leg
        # while keeping the torso and foot approximately upright.
        "left_hip_roll_joint": -0.095,
        # This ankle axis is sign-reversed relative to the hip in the URDF.
        "left_ankle_roll_joint": -0.095,
        "right_hip_roll_joint": 0.0,
        "right_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": 0.65,
        "right_knee_joint": -1.30,
        "right_ankle_pitch_joint": 0.34,
    },
    joint_vel={".*": 0.0},
)


ASIMOV_1_IMPLICIT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ASIMOV_1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ASIMOV_1_ONE_LEG_INIT_STATE,
    soft_joint_pos_limit_factor=0.92,
    actuators=ASIMOV_1_IMPLICIT_ACTUATORS,
)


ASIMOV_1_DELAYED_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=ASIMOV_1_URDF_PATH,
        fix_base=False,
        merge_fixed_joints=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ASIMOV_1_STANDING_INIT_STATE,
    soft_joint_pos_limit_factor=0.9,
    actuators=ASIMOV_1_ACTUATORS,
)
