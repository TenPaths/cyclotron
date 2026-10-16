"""MDP terms for learning Asimov 1 fall recovery."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")
RESET_MODES = ("supine", "prone", "left", "right", "mixed", "random")


def _floor_pose_correction(env, env_ids, root_pose):
    """Place collision geometry just above the flat floor using FK.

    A private CPU data object avoids forwarding or disturbing other GPU worlds
    during a partial reset.
    """
    model = env.sim.mj_model
    cache = getattr(env, "_standup_reset_fk", None)
    if cache is None:
        ids = np.flatnonzero((model.geom_group == 3) & (model.geom_bodyid != 0))
        cache = (mujoco.MjData(model), ids)
        env._standup_reset_fk = cache
    data, ids = cache
    qposes = env.sim.data.qpos[env_ids].detach().cpu().numpy()
    floors = env.scene.env_origins[env_ids, 2].detach().cpu().numpy()
    offsets = []
    for qpos, floor in zip(qposes, floors):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        bottom = collision_bottom(model, data, ids)
        offsets.append(floor + 0.012 - bottom)
    root_pose[:, 2] += torch.as_tensor(
        offsets, device=env.device, dtype=root_pose.dtype
    )


def collision_bottom(model, data, ids):
    """Lowest point of Asimov's capsule/sphere/box/cylinder collision shapes."""
    lows = []
    for i in ids:
        kind = model.geom_type[i]
        size = model.geom_size[i]
        z_axes = data.geom_xmat[i].reshape(3, 3)[2]
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = size[0] + size[1] * abs(z_axes[2])
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            extent = size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.abs(z_axes) @ size
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent = size[1] * abs(z_axes[2]) + size[0] * np.sqrt(
                max(0, 1 - z_axes[2] ** 2)
            )
        else:
            raise ValueError(f"Unsupported collision shape: {kind}")
        lows.append(data.geom_xpos[i, 2] - extent)
    return min(lows)


def root_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Pelvis height as a one-dimensional observation."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2:3]


def mean_root_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Pelvis height as a scalar metric per environment."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2]


def body_height_progress(
    env: ManagerBasedRlEnv,
    floor: float,
    target: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense normalized height of a selected body (normally the head)."""
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    assert not isinstance(body_ids, slice)
    height = asset.data.body_link_pos_w[:, body_ids[0], 2]
    return torch.clamp((height - floor) / (target - floor), 0.0, 1.0)


def mean_body_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Selected body height as a metric per environment."""
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    assert not isinstance(body_ids, slice)
    return asset.data.body_link_pos_w[:, body_ids[0], 2]


def reset_lying_pose(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    root_height_range: tuple[float, float] = (0.13, 0.18),
    joint_noise: float = 0.08,
    mode: str = "supine",
    standing_probability: float = 0.10,
) -> None:
    """Reset on the ground, plus a few goal states for reverse curriculum."""
    env_ids = resolve_env_ids(env, env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    count = len(env_ids)

    if mode not in RESET_MODES:
        raise ValueError(f"Unknown reset mode: {mode}")
    if not math.isfinite(joint_noise) or not 0 <= joint_noise <= 1.0:
        raise ValueError("joint_noise must be finite and between 0 and 1 radian")
    if not 0 <= standing_probability <= 1:
        raise ValueError("standing_probability must be between 0 and 1")
    if count == 0:
        return
    if mode == "mixed":
        side = torch.where(
            torch.rand(count, device=env.device) < 0.5,
            torch.full((count,), -1.0, device=env.device),
            torch.ones(count, device=env.device),
        )
    else:
        value = -1.0 if mode == "supine" else 1.0
        side = torch.full((count,), value, device=env.device)
    standing = torch.rand(count, device=env.device) < standing_probability
    roll = sample_uniform(-0.08, 0.08, (count,), env.device)
    pitch = side * (math.pi / 2) + sample_uniform(-0.10, 0.10, (count,), env.device)
    yaw = sample_uniform(-math.pi, math.pi, (count,), env.device)
    if mode in {"left", "right", "random"}:
        pose = (
            torch.randint(0, 4, (count,), device=env.device)
            if mode == "random"
            else torch.full((count,), 2 if mode == "left" else 3, device=env.device)
        )
        # Supine, prone and either side, with modest tilt and arbitrary heading.
        rolls = torch.tensor([0, 0, math.pi / 2, -math.pi / 2], device=env.device)
        pitches = torch.tensor([-math.pi / 2, math.pi / 2, 0, 0], device=env.device)
        roll = rolls[pose] + sample_uniform(-0.15, 0.15, (count,), env.device)
        pitch = pitches[pose] + sample_uniform(-0.15, 0.15, (count,), env.device)
    standing_count = int(standing.sum())
    roll[standing] = sample_uniform(-0.03, 0.03, (standing_count,), env.device)
    pitch[standing] = sample_uniform(-0.03, 0.03, (standing_count,), env.device)

    root_pose = torch.zeros((count, 7), device=env.device)
    root_pose[:, :3] = env.scene.env_origins[env_ids]
    root_pose[:, :2] += sample_uniform(-0.08, 0.08, (count, 2), env.device)
    root_pose[:, 2] += sample_uniform(*root_height_range, (count,), env.device)
    root_pose[standing, 2] = env.scene.env_origins[env_ids[standing], 2] + 0.63
    root_pose[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
    asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

    root_velocity = sample_uniform(-0.05, 0.05, (count, 6), env.device)
    asset.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)

    default_pos = asset.data.default_joint_pos
    default_vel = asset.data.default_joint_vel
    limits = asset.data.soft_joint_pos_limits
    assert default_pos is not None and default_vel is not None and limits is not None
    joint_pos = default_pos[env_ids].clone()
    joint_pos += sample_uniform(-joint_noise, joint_noise, joint_pos.shape, env.device)
    if standing.any():
        standing_shape = joint_pos[standing].shape
        joint_pos[standing] = default_pos[env_ids[standing]] + sample_uniform(
            -0.025, 0.025, standing_shape, env.device
        )
    joint_pos.clamp_(limits[env_ids, :, 0], limits[env_ids, :, 1])
    joint_vel = default_vel[env_ids].clone()
    joint_vel += sample_uniform(-0.05, 0.05, joint_vel.shape, env.device)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    # Preserve legacy small-noise supine/prone resets for old evaluations.
    if mode in {"left", "right", "random"} or joint_noise > 0.08:
        _floor_pose_correction(env, env_ids, root_pose)
        asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)


class height_assist:
    """Apply upward curriculum force that vanishes over training."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        self._body_ids = cfg.params["asset_cfg"].body_ids
        self._num_bodies = len(self._body_ids)
        self._device = env.device

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        max_force: float,
        decay_steps: int,
        target_height: float,
        asset_cfg: SceneEntityCfg,
    ) -> None:
        del env_ids, asset_cfg
        progress = min(env.common_step_counter / max(decay_steps, 1), 1.0)
        height = self._asset.data.root_link_pos_w[:, 2]
        height_error = torch.clamp(1.0 - height / target_height, 0.0, 1.0)
        force_z = max_force * (1.0 - progress) * height_error
        forces = torch.zeros((env.num_envs, self._num_bodies, 3), device=self._device)
        torques = torch.zeros_like(forces)
        forces[:, :, 2] = force_z[:, None]
        self._asset.write_external_wrench_to_sim(
            forces, torques, body_ids=self._body_ids
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids


def _upright_score(asset: Entity) -> torch.Tensor:
    return torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)


def height_progress(
    env: ManagerBasedRlEnv,
    floor: float,
    target: float,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Dense progress from floor height to standing pelvis height."""
    asset: Entity = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2]
    return torch.clamp((height - floor) / (target - floor), 0.0, 1.0)


def upright_progress(
    env: ManagerBasedRlEnv,
    floor: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Reward pelvis alignment only in proportion to achieved height."""
    asset: Entity = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2]
    height_gate = torch.clamp((height - floor) / (target_height - floor), 0.0, 1.0)
    return _upright_score(asset) * height_gate


def standing_bonus(
    env: ManagerBasedRlEnv,
    activation_height: float,
    transition_width: float,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Smooth coupled objective for completing and holding the stand."""
    asset: Entity = env.scene[asset_cfg.name]
    height_score = torch.sigmoid(
        (asset.data.root_link_pos_w[:, 2] - activation_height) / transition_width
    )
    upright = _upright_score(asset)
    lin_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)
    ang_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    still = torch.exp(-0.5 * lin_speed) * torch.exp(-0.15 * ang_speed)
    return height_score * torch.square(upright) * still


def standing_posture(
    env: ManagerBasedRlEnv,
    std: float,
    activation_height: float,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Track the nominal pose only after the robot has substantially risen."""
    asset: Entity = env.scene[asset_cfg.name]
    default_pos = asset.data.default_joint_pos
    assert default_pos is not None
    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - default_pos[:, asset_cfg.joint_ids]
    )
    pose_score = torch.exp(-torch.mean(torch.square(error), dim=1) / std**2)
    height_gate = torch.sigmoid(
        20.0 * (asset.data.root_link_pos_w[:, 2] - activation_height)
    )
    return pose_score * height_gate


def refined_standing_posture(
    env: ManagerBasedRlEnv,
    std: float,
    activation_height: float,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Reward the nominal pose only when it is upright and nearly stationary.

    Coupling the terms prevents a tilted or thrashing robot from collecting a
    large posture reward merely because its joints happen to be near nominal.
    """
    asset: Entity = env.scene[asset_cfg.name]
    default_pos = asset.data.default_joint_pos
    assert default_pos is not None
    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - default_pos[:, asset_cfg.joint_ids]
    )
    pose_score = torch.exp(-torch.mean(torch.square(error), dim=1) / std**2)
    height_gate = torch.sigmoid(
        20.0 * (asset.data.root_link_pos_w[:, 2] - activation_height)
    )
    upright = torch.square(_upright_score(asset))
    lin_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)
    ang_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    still = torch.exp(-0.75 * lin_speed) * torch.exp(-0.20 * ang_speed)
    return pose_score * height_gate * upright * still


def base_motion_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Penalize residual pelvis motion, strongest near an upright stance."""
    asset: Entity = env.scene[asset_cfg.name]
    motion = torch.sum(torch.square(asset.data.root_link_lin_vel_w), dim=1)
    motion += 0.25 * torch.sum(torch.square(asset.data.root_link_ang_vel_w), dim=1)
    return motion * _upright_score(asset)


def success_metric(
    env: ManagerBasedRlEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Per-step stand success indicator for logs."""
    asset: Entity = env.scene[asset_cfg.name]
    height_ok = asset.data.root_link_pos_w[:, 2] >= minimum_height
    upright_ok = _upright_score(asset) >= 0.9
    return (height_ok & upright_ok).float()


def mean_upright(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Pelvis upright score for detecting leaning policies in logs."""
    asset: Entity = env.scene[asset_cfg.name]
    return _upright_score(asset)


def mean_joint_posture_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """RMS joint error from the nominal standing pose, in radians."""
    asset: Entity = env.scene[asset_cfg.name]
    default_pos = asset.data.default_joint_pos
    assert default_pos is not None
    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - default_pos[:, asset_cfg.joint_ids]
    )
    return torch.sqrt(torch.mean(torch.square(error), dim=1))


def mean_base_speed(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Combined pelvis linear/angular speed used to expose residual motion."""
    asset: Entity = env.scene[asset_cfg.name]
    linear = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)
    angular = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    return linear + 0.25 * angular
