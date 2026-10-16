"""Play a trained Asimov stand-up policy in Viser on a non-Jenkins port."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import ViserPlayViewer

from asimov_standup import TASK_ID
from asimov_standup.mdp import RESET_MODES


def latest_checkpoint(log_root: Path) -> Path:
    checkpoints = list((log_root / "asimov1_standup").glob("*/model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoint found. Pass --checkpoint or run train.py first."
        )
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    parser.add_argument(
        "--agent", choices=("trained", "zero", "random"), default="trained"
    )
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--reset-mode", choices=RESET_MODES, default="random")
    parser.add_argument(
        "--joint-noise",
        type=float,
        default=0.4,
        help="Independent joint offsets in radians, from 0 to 1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=8.0,
        help="Simulated seconds before the robot resets to the ground.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        choices=(1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0),
        default=1.0,
        help="Simulation-time multiplier; rendering remains at --fps.",
    )
    args = parser.parse_args()
    if not 0 <= args.joint_noise <= 1:
        parser.error("--joint-noise must be between 0 and 1 radian")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.events["reset_lying"].params.update(
        mode=args.reset_mode,
        joint_noise=args.joint_noise,
    )
    if args.episode_seconds <= 0:
        parser.error("--episode-seconds must be greater than zero")
    env_cfg.episode_length_s = args.episode_seconds
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    agent_cfg = load_rl_cfg(TASK_ID)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args.agent == "trained":
        checkpoint = args.checkpoint or latest_checkpoint(args.log_root)
        print(f"[INFO] Loading checkpoint: {checkpoint}")
        runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=args.device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=args.device)
    elif args.agent == "zero":
        policy = lambda obs: torch.zeros(
            env.unwrapped.action_space.shape,
            device=env.unwrapped.device,
        )
    else:
        policy = lambda obs: (
            2.0
            * torch.rand(
                env.unwrapped.action_space.shape,
                device=env.unwrapped.device,
            )
            - 1.0
        )

    server = viser.ViserServer(port=args.port, label="Asimov 1 Stand-up")
    try:
        viewer = ViserPlayViewer(
            env,
            policy,
            frame_rate=args.fps,
            viser_server=server,
        )
        normal_index = viewer.SPEED_MULTIPLIERS.index(1.0)
        target_index = viewer.SPEED_MULTIPLIERS.index(args.speed)
        for _ in range(normal_index - target_index):
            viewer.decrease_speed()
        for _ in range(target_index - normal_index):
            viewer.increase_speed()
        viewer.run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
