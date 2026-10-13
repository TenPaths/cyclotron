"""Standalone Newton task: balance Asimov-1 on its left leg."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

from cyclotron.assets.robots.asimov_1 import ASIMOV_1_IMPLICIT_CFG, ASIMOV_1_JOINT_NAMES

from . import mdp
from .velocity_env_cfg import ObservationsCfg

GROUND_USD_PATH = str(Path(__file__).resolve().parents[2] / "assets" / "ground_box.usdc")

SUPPORT_FOOT = "left_ankle_roll_link"
SWING_FOOT = "right_ankle_roll_link"
TORSO_BODY = "waist_yaw_link"

SUPPORT_LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
]
SWING_LEG_JOINTS = [
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
ARM_JOINTS = [
    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
    ".*_elbow_joint", ".*_wrist_yaw_joint",
]


def _body(name: str) -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=[name], preserve_order=True)


@configclass
class Asimov1OneLegSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=GROUND_USD_PATH,
        env_spacing=2.0,
        collision_group=-1,
        debug_vis=False,
    )

    robot: ArticulationCfg = ASIMOV_1_IMPLICIT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    feet_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class CommandsCfg:
    # Keep Cyclotron's 3-command / 78-observation interface.  Zero velocity is
    # the command for this stationary skill, which keeps future policy transfer simple.
    twist = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ASIMOV_1_JOINT_NAMES),
        preserve_order=True,
        scale={
            # Keep the loaded support leg conservative, but give the swing
            # leg enough residual range to raise its foot 14 cm.  The old
            # symmetric 0.20/0.22 rad hip/knee limits made that target
            # kinematically unreachable from the standing offset.
            "left_hip_pitch_joint": 0.20,
            "left_hip_roll_joint": 0.14,
            "left_hip_yaw_joint": 0.12,
            "left_knee_joint": 0.22,
            "left_ankle_pitch_joint": 0.10,
            "left_ankle_roll_joint": 0.06,
            "right_hip_pitch_joint": 0.45,
            "right_hip_roll_joint": 0.28,
            "right_hip_yaw_joint": 0.22,
            "right_knee_joint": 0.70,
            "right_ankle_pitch_joint": 0.35,
            "right_ankle_roll_joint": 0.18,
            "waist_yaw_joint": 0.20,
            ".*_shoulder_.*_joint": 0.25,
            ".*_elbow_joint": 0.22,
            ".*_wrist_yaw_joint": 0.15,
        },
        use_default_offset=True,
    )


@configclass
class EventCfg:
    # Learn the transition from ordinary double support; no external wrench
    # holds the pelvis or unloads the robot during training.
    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[SUPPORT_FOOT, SWING_FOOT], preserve_order=True
            ),
            "static_friction_range": (0.85, 1.15),
            "dynamic_friction_range": (0.80, 1.10),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.01, 0.01), "y": (-0.01, 0.01), "yaw": (-0.03, 0.03),
                "pitch": (-0.025, 0.025), "roll": (-0.025, 0.025),
            },
            "velocity_range": {
                "x": (-0.04, 0.04), "y": (-0.04, 0.04), "z": (-0.02, 0.02),
                "roll": (-0.08, 0.08), "pitch": (-0.08, 0.08), "yaw": (-0.05, 0.05),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.025, 0.025),
            "velocity_range": (-0.10, 0.10),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )
    pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.98, 1.02),
            "damping_distribution_params": (0.98, 1.02),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=2.0)
    com_target = RewTerm(
        func=mdp.progressive_com_target,
        weight=10.0,
        params={
            # Broad enough to retain a useful gradient across the measured
            # ~0.21 m lateral foot separation.
            "std": 0.150,
            "start_step": 0,
            "end_step": 0,
            "asset_cfg": SceneEntityCfg("robot"),
            "support_foot_cfg": _body(SUPPORT_FOOT),
            "swing_foot_cfg": _body(SWING_FOOT),
        },
    )
    swing_height = RewTerm(
        func=mdp.progressive_swing_foot_height,
        weight=10.0,
        params={
            # The old 8 cm target was below the roughly 12 cm reset height,
            # so the policy was actually paid to lower the free foot.  Keep
            # it clearly raised while retaining room for balance corrections.
            "start_height": 0.140,
            "end_height": 0.140,
            # Keep the height signal informative even before the policy has
            # learned to unload the foot; a narrow kernel vanishes at ground.
            "std": 0.055,
            "start_step": 0,
            "end_step": 0,
            "asset_cfg": _body(SWING_FOOT),
        },
    )
    support_contact = RewTerm(
        func=mdp.contact_state,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("feet_contact", body_names=[SUPPORT_FOOT]),
            "desired_contact": True,
        },
    )
    swing_air = RewTerm(
        func=mdp.progressive_swing_contact,
        weight=8.0,
        params={
            "sensor_cfg": SceneEntityCfg("feet_contact", body_names=[SWING_FOOT]),
            "start_step": 0,
            "end_step": 0,
        },
    )
    swing_contact_penalty = RewTerm(
        func=mdp.progressive_swing_contact_penalty,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("feet_contact", body_names=[SWING_FOOT]),
            "start_step": 0,
            "end_step": 0,
        },
    )
    swing_height_error = RewTerm(
        func=mdp.progressive_swing_height_error,
        weight=-100.0,
        params={
            "start_height": 0.140,
            "end_height": 0.140,
            "start_step": 0,
            "end_step": 0,
            "asset_cfg": _body(SWING_FOOT),
        },
    )
    torso_upright = RewTerm(
        func=mdp.body_upright,
        weight=5.0,
        params={"std": 0.20, "asset_cfg": _body(TORSO_BODY)},
    )
    support_foot_flat = RewTerm(
        func=mdp.foot_upright,
        weight=2.0,
        params={"std": 0.16, "asset_cfg": _body(SUPPORT_FOOT)},
    )
    low_root_motion = RewTerm(
        func=mdp.root_motion,
        weight=3.0,
        params={"linear_std": 0.35, "angular_std": 0.55},
    )
    pelvis_height = RewTerm(
        func=mdp.root_height,
        weight=1.0,
        params={"target_height": 0.570, "std": 0.07},
    )
    support_pose = RewTerm(
        func=mdp.pose_tracking,
        weight=1.0,
        params={
            "std": 0.16,
            "asset_cfg": SceneEntityCfg("robot", joint_names=SUPPORT_LEG_JOINTS),
        },
    )
    free_limb_pose = RewTerm(
        func=mdp.pose_tracking,
        weight=1.5,
        params={
            # This is a soft anchor, not a lock: the policy can still move
            # the arms and folded leg when that improves balance.
            "std": 0.35,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=SWING_LEG_JOINTS + ARM_JOINTS
            ),
        },
    )
    support_slip = RewTerm(
        func=mdp.support_foot_slip,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("feet_contact", body_names=[SUPPORT_FOOT]),
            "asset_cfg": _body(SUPPORT_FOOT),
        },
    )
    arm_motion = RewTerm(
        func=mdp.joint_motion_l2,
        weight=-0.04,
        params={
            "metric_name": "arm_joint_speed",
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
        },
    )
    swing_leg_motion = RewTerm(
        func=mdp.joint_motion_l2,
        weight=-0.06,
        params={
            "metric_name": "swing_leg_joint_speed",
            "asset_cfg": SceneEntityCfg("robot", joint_names=SWING_LEG_JOINTS),
        },
    )
    torso_motion = RewTerm(
        func=mdp.body_angular_velocity_penalty,
        weight=-0.08,
        params={"asset_cfg": _body(TORSO_BODY)},
    )
    angular_momentum = RewTerm(func=mdp.angular_momentum_penalty, weight=-0.03)
    # Half the standard walking task's coefficient: suppress rapid command
    # reversals while preserving deliberate balance corrections.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.25)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.002)
    joint_acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    torque = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.5)
    # Reward terms are integrated by step_dt (0.02 s).  -100 therefore gives
    # a meaningful -2.0 impulse on a fall instead of the ineffective -0.1
    # produced by the locomotion task's old -5 weight.
    termination = RewTerm(func=mdp.is_terminated, weight=-100.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    excessive_tilt = DoneTerm(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(75.0)}
    )
    pelvis_too_low = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.22}
    )
    # Keep touchdown non-terminal during training so PPO receives complete
    # balance/recovery trajectories; dense contact and height costs still
    # penalize it.  PLAY forces this term active with a strict one-second gate.
    swing_contact = DoneTerm(
        func=mdp.swing_contact_after_grace,
        params={
            "sensor_cfg": SceneEntityCfg("feet_contact", body_names=[SWING_FOOT]),
            "activate_step": 1_000_000_000,
            "grace_steps": 150,
        },
    )


@configclass
class Asimov1OneLegEnvCfg(ManagerBasedRLEnvCfg):
    scene: Asimov1OneLegSceneCfg = Asimov1OneLegSceneCfg(num_envs=2048, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=160,
                nconmax=48,
                iterations=20,
                ls_iterations=20,
                integrator="implicitfast",
                impratio=10.0,
                cone="elliptic",
                use_mujoco_contacts=True,
            ),
            num_substeps=2,
            debug_mode=False,
        )
        # Newton must consume Isaac Lab's implicit actuator targets directly;
        # the generic target-forwarding path only updates telemetry for this
        # kitless backend and leaves the USD's baked drives in control.
        self.sim.use_newton_actuators = True


@configclass
class Asimov1OneLegEnvCfg_PLAY(Asimov1OneLegEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 60.0
        self.observations.policy.enable_corruption = False
        self.events.foot_friction = None
        self.events.pd_gains = None
        self.events.reset_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"] = {
            key: (0.0, 0.0) for key in self.events.reset_base.params["pose_range"]
        }
        self.events.reset_base.params["velocity_range"] = {
            key: (0.0, 0.0) for key in self.events.reset_base.params["velocity_range"]
        }
        # Playback evaluates the final one-leg objective immediately instead
        # of replaying the early double-support curriculum phase.
        self.rewards.com_target.params["fixed_progress"] = 1.0
        self.rewards.swing_height.params["fixed_progress"] = 1.0
        self.rewards.swing_air.params["fixed_progress"] = 1.0
        self.rewards.swing_contact_penalty.params["fixed_progress"] = 1.0
        self.rewards.swing_height_error.params["fixed_progress"] = 1.0
        self.terminations.swing_contact.params["fixed_active"] = True
        # Evaluation remains strict even though training uses longer
        # trajectories to learn recovery from an early touchdown.
        self.terminations.swing_contact.params["grace_steps"] = 50
