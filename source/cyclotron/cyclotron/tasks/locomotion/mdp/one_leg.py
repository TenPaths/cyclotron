"""Task terms for quasi-static single-leg balance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor

from .observations import foot_pos_w, foot_vel_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _linear_progress(
    env: ManagerBasedRLEnv,
    start_step: int,
    end_step: int,
    step_offset: int = 0,
    fixed_progress: float | None = None,
) -> float:
    """Return a global, deterministic curriculum fraction in ``[0, 1]``."""
    if fixed_progress is not None:
        return max(0.0, min(1.0, fixed_progress))
    step = env.common_step_counter + step_offset
    if end_step <= start_step:
        return float(step >= end_step)
    return max(0.0, min(1.0, (step - start_step) / (end_step - start_step)))


class progressive_com_target(ManagerTermBase):
    """Move the desired whole-body COM from between the feet to the support foot."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._masses = asset.data.body_mass.clone()
        self._total_mass = self._masses.sum(dim=1)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        start_step: int,
        end_step: int,
        asset_cfg: SceneEntityCfg,
        support_foot_cfg: SceneEntityCfg,
        swing_foot_cfg: SceneEntityCfg,
        step_offset: int = 0,
        fixed_progress: float | None = None,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        com = (asset.data.body_com_pos_w * self._masses.unsqueeze(-1)).sum(dim=1)
        com = com / self._total_mass.unsqueeze(-1)
        support = foot_pos_w(env, support_foot_cfg, offset=(0.0, 0.0, -0.025)).squeeze(1)
        swing = foot_pos_w(env, swing_foot_cfg, offset=(0.0, 0.0, -0.025)).squeeze(1)
        progress = _linear_progress(env, start_step, end_step, step_offset, fixed_progress)
        target = (1.0 - progress) * 0.5 * (support + swing) + progress * support
        error = torch.sum(torch.square(com[:, :2] - target[:, :2]), dim=1)
        log = env.extras.setdefault("log", {})
        log["Curriculum/weight_shift"] = progress
        log["Metrics/com_target_error"] = torch.sqrt(error).mean()
        return torch.exp(-error / std**2)


def progressive_swing_foot_height(
    env: ManagerBasedRLEnv,
    start_height: float,
    end_height: float,
    std: float,
    start_step: int,
    end_step: int,
    asset_cfg: SceneEntityCfg,
    step_offset: int = 0,
    fixed_progress: float | None = None,
) -> torch.Tensor:
    progress = _linear_progress(env, start_step, end_step, step_offset, fixed_progress)
    target_height = start_height + progress * (end_height - start_height)
    height = foot_pos_w(env, asset_cfg).squeeze(1)[:, 2]
    error = torch.square(height - target_height)
    log = env.extras.setdefault("log", {})
    log["Curriculum/foot_lift"] = progress
    log["Curriculum/swing_height_target"] = target_height
    log["Metrics/swing_foot_height"] = height.mean()
    return torch.exp(-error / std**2)


def contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    desired_contact: bool,
    force_threshold: float = 2.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = torch.any(
        sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > force_threshold, dim=1
    )
    return (contact if desired_contact else ~contact).float()


def progressive_swing_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    start_step: int,
    end_step: int,
    force_threshold: float = 2.0,
    step_offset: int = 0,
    fixed_progress: float | None = None,
) -> torch.Tensor:
    """Progressively reward swing-foot air without ever rewarding its contact.

    Double-support stability is learned from the other task rewards before
    ``start_step``.  Paying for swing-foot contact here created a local optimum
    where the policy could remain on two feet until late in the curriculum.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = torch.any(
        sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > force_threshold, dim=1
    ).float()
    progress = _linear_progress(env, start_step, end_step, step_offset, fixed_progress)
    return progress * (1.0 - contact)


def progressive_swing_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    start_step: int,
    end_step: int,
    force_threshold: float = 2.0,
    step_offset: int = 0,
    fixed_progress: float | None = None,
) -> torch.Tensor:
    """Penalize continued swing-foot support as the lift phase advances."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = torch.any(
        sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > force_threshold, dim=1
    ).float()
    progress = _linear_progress(env, start_step, end_step, step_offset, fixed_progress)
    env.extras.setdefault("log", {})["Metrics/swing_contact_rate"] = contact.mean()
    return progress * contact


def progressive_swing_height_error(
    env: ManagerBasedRLEnv,
    start_height: float,
    end_height: float,
    start_step: int,
    end_step: int,
    asset_cfg: SceneEntityCfg,
    step_offset: int = 0,
    fixed_progress: float | None = None,
) -> torch.Tensor:
    """Dense squared height error that cannot vanish far from the target."""
    progress = _linear_progress(env, start_step, end_step, step_offset, fixed_progress)
    target_height = start_height + progress * (end_height - start_height)
    height = foot_pos_w(env, asset_cfg).squeeze(1)[:, 2]
    return progress * torch.square(height - target_height)


def swing_contact_after_grace(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    activate_step: int,
    grace_steps: int,
    force_threshold: float = 2.0,
    step_offset: int = 0,
    fixed_active: bool = False,
) -> torch.Tensor:
    """Fail final-task episodes that keep using the nominal swing foot."""
    active = fixed_active or env.common_step_counter + step_offset >= activate_step
    if not active:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = torch.any(
        sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > force_threshold, dim=1
    )
    return contact & (env.episode_length_buf > grace_steps)


def support_foot_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 2.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = torch.any(
        sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > force_threshold, dim=1
    ).float()
    velocity = foot_vel_w(env, asset_cfg).squeeze(1)[:, :2]
    return torch.sum(torch.square(velocity), dim=1) * contact


def joint_motion_l2(
    env: ManagerBasedRLEnv,
    metric_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize selected joint motion and expose an interpretable RMS-speed metric."""
    asset: Articulation = env.scene[asset_cfg.name]
    velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    squared = torch.square(velocity)
    env.extras.setdefault("log", {})[f"Metrics/{metric_name}_rms_rad_s"] = torch.sqrt(
        torch.mean(squared)
    )
    return torch.sum(squared, dim=1)


def body_upright(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids].squeeze(1)
    gravity = asset.data.GRAVITY_VEC_W.expand(env.num_envs, -1)
    projected = math_utils.quat_apply_inverse(quat, gravity)
    tilt_sq = torch.sum(torch.square(projected[:, :2]), dim=1)
    return torch.exp(-tilt_sq / std**2)


def foot_upright(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return body_upright(env, std, asset_cfg)


def root_motion(
    env: ManagerBasedRLEnv,
    linear_std: float,
    angular_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    lin_sq = torch.sum(torch.square(asset.data.root_lin_vel_b), dim=1)
    ang_sq = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    return torch.exp(-lin_sq / linear_std**2 - ang_sq / angular_std**2)


def root_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-error / std**2)


def pose_tracking(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.exp(-torch.mean(torch.square(error), dim=1) / std**2)
