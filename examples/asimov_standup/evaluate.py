"""Evaluate a checkpoint from lying poses without training-time assistance."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from asimov_standup import TASK_ID
from asimov_standup.mdp import RESET_MODES
from play import latest_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--reset-mode", choices=RESET_MODES, default="supine")
    parser.add_argument("--joint-noise", type=float, default=0.08)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.5,
        help="Continuous stable-standing time required for held_success_rate.",
    )
    args = parser.parse_args()
    if not 0 <= args.joint_noise <= 1:
        parser.error("--joint-noise must be between 0 and 1 radian")

    if args.hold_seconds <= 0:
        parser.error("--hold-seconds must be greater than zero")
    torch.manual_seed(args.seed)
    checkpoint = args.checkpoint or latest_checkpoint(args.log_root)
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.events["reset_lying"].params["mode"] = args.reset_mode
    env_cfg.events["reset_lying"].params["joint_noise"] = args.joint_noise
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    agent_cfg = load_rl_cfg(TASK_ID)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=args.device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)

    obs, _ = env.reset()
    robot = raw_env.scene["robot"]
    head_id = robot.find_bodies("neck_pitch_link")[0][0]
    max_height = torch.zeros(args.num_envs, device=args.device)
    max_head_height = torch.zeros(args.num_envs, device=args.device)
    reached_stand = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
    held_stand = torch.zeros_like(reached_stand)
    standing_steps = torch.zeros(args.num_envs, device=args.device)
    consecutive_steps = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
    first_stand_step = torch.full(
        (args.num_envs,), -1, dtype=torch.long, device=args.device
    )
    posture_error_sum = torch.zeros(args.num_envs, device=args.device)
    base_speed_sum = torch.zeros(args.num_envs, device=args.device)
    stable_sample_count = torch.zeros(args.num_envs, device=args.device)
    required_hold_steps = round(args.hold_seconds / raw_env.step_dt)
    default_joint_pos = robot.data.default_joint_pos
    assert default_joint_pos is not None
    with torch.inference_mode():
        for step in range(raw_env.max_episode_length):
            obs, _, _, _ = env.step(policy(obs))
            height = robot.data.root_link_pos_w[:, 2]
            head_height = robot.data.body_link_pos_w[:, head_id, 2]
            upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 1.0)
            max_height = torch.maximum(max_height, height)
            max_head_height = torch.maximum(max_head_height, head_height)
            standing = (height >= 0.56) & (upright >= 0.9)
            newly_standing = standing & (first_stand_step < 0)
            first_stand_step[newly_standing] = step
            reached_stand |= standing
            standing_steps += standing.float()
            consecutive_steps = torch.where(
                standing, consecutive_steps + 1, torch.zeros_like(consecutive_steps)
            )
            held_stand |= consecutive_steps >= required_hold_steps

            joint_error = torch.sqrt(
                torch.mean(torch.square(robot.data.joint_pos - default_joint_pos), dim=1)
            )
            lin_speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w, dim=1)
            ang_speed = torch.linalg.vector_norm(robot.data.root_link_ang_vel_w, dim=1)
            base_speed = lin_speed + 0.25 * ang_speed
            posture_error_sum += torch.where(standing, joint_error, 0.0)
            base_speed_sum += torch.where(standing, base_speed, 0.0)
            stable_sample_count += standing.float()

    final_height = raw_env.scene["robot"].data.root_link_pos_w[:, 2]
    print(f"checkpoint={checkpoint}")
    print(f"reset_mode={args.reset_mode} envs={args.num_envs} assist=off")
    print(f"success_rate={reached_stand.float().mean().item():.4f}")
    print(f"held_success_rate={held_stand.float().mean().item():.4f}")
    print(
        f"mean_standing_fraction="
        f"{(standing_steps / raw_env.max_episode_length).mean().item():.4f}"
    )
    if reached_stand.any():
        mean_time = (
            first_stand_step[reached_stand].float().mean().item() * raw_env.step_dt
        )
        print(f"mean_time_to_first_stand_s={mean_time:.4f}")
    valid = stable_sample_count > 0
    if valid.any():
        print(
            f"standing_posture_error_rad="
            f"{(posture_error_sum[valid] / stable_sample_count[valid]).mean().item():.4f}"
        )
        print(
            f"standing_base_speed="
            f"{(base_speed_sum[valid] / stable_sample_count[valid]).mean().item():.4f}"
        )
    print(f"mean_max_pelvis_height={max_height.mean().item():.4f}")
    print(f"mean_max_head_height={max_head_height.mean().item():.4f}")
    print(f"mean_final_pelvis_height={final_height.mean().item():.4f}")
    env.close()


if __name__ == "__main__":
    main()
