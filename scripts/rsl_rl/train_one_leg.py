#!/usr/bin/env python3
"""Train Cyclotron Asimov-1 to balance quietly on its left leg.

This entry point uses standalone Isaac Lab's kitless Newton/MJWarp backend.
It does not launch Isaac Sim or PhysX.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "cyclotron"))

parser = argparse.ArgumentParser(
    description="Train Asimov-1 single-leg balance with Newton/MJWarp."
)
parser.add_argument("--task", default="Asimov1-OneLeg-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video", action="store_true", help="Record periodic training clips.")
parser.add_argument(
    "--video_interval", type=int, default=None, help="Recording interval in environment steps."
)
parser.add_argument(
    "--video_length", type=int, default=None, help="Clip length in environment steps."
)
parser.add_argument(
    "--resume_checkpoint",
    type=Path,
    default=None,
    help="Resume policy and optimizer from an RSL-RL model_*.pt checkpoint.",
)
parser.add_argument(
    "--curriculum_step_offset",
    type=int,
    default=None,
    help="Override the resumed curriculum step (normally inferred from the checkpoint name).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

import cyclotron.tasks  # noqa: F401 -- register Cyclotron tasks
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_tasks.utils.sim_launcher import launch_simulation


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed

    checkpoint = args_cli.resume_checkpoint.resolve() if args_cli.resume_checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    curriculum_step_offset = args_cli.curriculum_step_offset
    if curriculum_step_offset is None and checkpoint is not None:
        match = re.fullmatch(r"model_(\d+)", checkpoint.stem)
        if match is None:
            raise ValueError(
                "Cannot infer curriculum position from checkpoint name; "
                "pass --curriculum_step_offset explicitly."
            )
        curriculum_step_offset = (int(match.group(1)) + 1) * agent_cfg.num_steps_per_env
    curriculum_step_offset = curriculum_step_offset or 0
    for term_name in (
        "com_target",
        "swing_height",
        "swing_air",
        "swing_contact_penalty",
        "swing_height_error",
    ):
        getattr(env_cfg.rewards, term_name).params["step_offset"] = curriculum_step_offset
    env_cfg.terminations.swing_contact.params["step_offset"] = curriculum_step_offset

    step_dt = env_cfg.sim.dt * env_cfg.decimation
    video_interval = args_cli.video_interval or 500
    video_length = args_cli.video_length or int(5.0 / step_dt)

    log_root = REPOSITORY_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name
    log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(log_dir)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video else None,
        )
        if args_cli.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(log_dir / "videos"),
                step_trigger=lambda step: step % video_interval == 0,
                video_length=video_length,
                fps=30,
                disable_logger=True,
            )
            print(f"[INFO] Videos: {log_dir / 'videos'}")

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        env.unwrapped.render_enabled = bool(args_cli.video or args_cli.visualizer)
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device
        )
        if checkpoint is not None:
            runner.load(str(checkpoint), map_location=agent_cfg.device)
            print(
                f"[INFO] Resumed: {checkpoint} "
                f"(curriculum step {curriculum_step_offset})"
            )
        else:
            # Begin around the explicit one-leg reference instead of a large
            # random residual action that immediately destroys the pose.
            with __import__("torch").no_grad():
                actor_head = runner.alg.actor.mlp[-1]
                actor_head.weight.zero_()
                actor_head.bias.zero_()

        runner.add_git_repo_to_log(__file__)
        print(f"[INFO] Logs: {log_dir}")
        print(
            f"[INFO] Starting one-leg PPO: {args_cli.num_envs} envs, "
            f"{agent_cfg.max_iterations} iterations, device={args_cli.device}"
        )
        start = time.time()
        try:
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=False,
            )
        finally:
            print(f"[INFO] Runtime: {time.time() - start:.1f} s")
            env.close()


if __name__ == "__main__":
    main()
