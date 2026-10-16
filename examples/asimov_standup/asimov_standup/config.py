"""Environment and PPO configuration for Asimov 1 stand-up learning."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

from . import mdp
from .asset import JOINT_NAMES, get_robot_cfg

ROBOT = SceneEntityCfg("robot")
PELVIS = SceneEntityCfg("robot", body_names=("pelvis_link",))
HEAD = SceneEntityCfg("robot", body_names=("neck_pitch_link",))
ALL_JOINTS = SceneEntityCfg("robot", joint_names=JOINT_NAMES)


def apply_refinement_preset(cfg: ManagerBasedRlEnvCfg) -> None:
    """Retarget a learned recovery policy toward a straight, quiet stance."""
    cfg.rewards["righting"].weight = 6.0
    cfg.rewards["standing"].weight = 10.0
    cfg.rewards["standing_posture"].func = mdp.refined_standing_posture
    cfg.rewards["standing_posture"].weight = 4.0
    cfg.rewards["standing_posture"].params.update(
        {"std": 0.25, "activation_height": 0.42}
    )
    cfg.rewards["base_motion"].weight = -0.05
    cfg.rewards["joint_torque"].weight = -5.0e-6
    cfg.rewards["joint_velocity"].weight = -1.0e-4
    cfg.rewards["action_rate"].weight = -0.01
    cfg.rewards["joint_limits"].weight = -1.0


def standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the contact-rich supine/prone stand-up task."""
    actor_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=base_mdp.base_lin_vel,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=base_mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        ),
        "projected_gravity": ObservationTermCfg(
            func=base_mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02),
        ),
        "root_height": ObservationTermCfg(func=mdp.root_height),
        "joint_pos": ObservationTermCfg(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": ALL_JOINTS},
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": ALL_JOINTS},
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            scale=0.1,
        ),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
    }
    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms={
                **actor_terms,
                "base_lin_vel": ObservationTermCfg(func=base_mdp.base_lin_vel),
                "base_ang_vel": ObservationTermCfg(func=base_mdp.base_ang_vel),
                "projected_gravity": ObservationTermCfg(
                    func=base_mdp.projected_gravity
                ),
                "joint_pos": ObservationTermCfg(
                    func=base_mdp.joint_pos_rel,
                    params={"asset_cfg": ALL_JOINTS},
                ),
                "joint_vel": ObservationTermCfg(
                    func=base_mdp.joint_vel_rel,
                    params={"asset_cfg": ALL_JOINTS},
                    scale=0.1,
                ),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=JOINT_NAMES,
            scale=0.25,
            use_default_offset=True,
        )
    }

    events = {
        "reset_lying": EventTermCfg(
            func=mdp.reset_lying_pose,
            mode="reset",
            params={
                "mode": "supine",
                "standing_probability": 0.0 if play else 0.25,
                "asset_cfg": ROBOT,
            },
        ),
    }
    if not play:
        events["height_assist"] = EventTermCfg(
            func=mdp.height_assist,
            mode="step",
            params={
                "max_force": 180.0,
                "decay_steps": 48_000,
                "target_height": 0.60,
                "asset_cfg": PELVIS,
            },
        )

    rewards = {
        "righting": RewardTermCfg(
            func=mdp.upright_progress,
            weight=4.0,
            params={"floor": 0.06, "target_height": 0.60},
        ),
        "pelvis_rising": RewardTermCfg(
            func=mdp.height_progress,
            weight=2.0,
            params={"floor": 0.06, "target": 0.60},
        ),
        "head_rising": RewardTermCfg(
            func=mdp.body_height_progress,
            weight=8.0,
            params={"floor": 0.08, "target": 1.05, "asset_cfg": HEAD},
        ),
        "standing": RewardTermCfg(
            func=mdp.standing_bonus,
            weight=8.0,
            params={"activation_height": 0.28, "transition_width": 0.05},
        ),
        "standing_posture": RewardTermCfg(
            func=mdp.standing_posture,
            weight=1.0,
            params={
                "std": 0.35,
                "activation_height": 0.45,
                "asset_cfg": ALL_JOINTS,
            },
        ),
        "base_motion": RewardTermCfg(func=mdp.base_motion_l2, weight=-0.01),
        "joint_torque": RewardTermCfg(
            func=base_mdp.joint_torques_l2,
            weight=-2.0e-6,
        ),
        "joint_velocity": RewardTermCfg(
            func=base_mdp.joint_vel_l2,
            weight=-2.0e-5,
        ),
        "action_rate": RewardTermCfg(
            func=base_mdp.action_rate_l2,
            weight=-0.001,
        ),
        "joint_limits": RewardTermCfg(
            func=base_mdp.joint_pos_limits,
            weight=-0.5,
        ),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_robot_cfg()},
            num_envs=512,
            env_spacing=2.0,
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations={
            "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        },
        metrics={
            "stand_success": MetricsTermCfg(
                func=mdp.success_metric,
                params={"minimum_height": 0.56},
            ),
            "pelvis_height": MetricsTermCfg(func=mdp.mean_root_height),
            "head_height": MetricsTermCfg(
                func=mdp.mean_body_height,
                params={"asset_cfg": HEAD},
            ),
            "upright": MetricsTermCfg(func=mdp.mean_upright),
            "posture_error": MetricsTermCfg(
                func=mdp.mean_joint_posture_error,
                params={"asset_cfg": ALL_JOINTS},
            ),
            "base_speed": MetricsTermCfg(func=mdp.mean_base_speed),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="pelvis_link",
            distance=2.2,
            elevation=-10.0,
            azimuth=130.0,
        ),
        sim=SimulationCfg(
            nconmax=120,
            njmax=700,
            mujoco=MujocoCfg(
                timestep=0.005,
                integrator="implicitfast",
                solver="newton",
                iterations=20,
                ls_iterations=20,
                cone="elliptic",
                impratio=10.0,
            ),
        ),
        decimation=4,
        episode_length_s=8.0,
    )
    return cfg


def ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO defaults sized for the 23-DoF stand-up policy."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.8,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="asimov1_standup",
        logger="tensorboard",
        upload_model=False,
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=10_000,
    )
