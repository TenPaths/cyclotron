"""Train Asimov 1 to recover from randomized supine and prone poses."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from asimov_standup import TASK_ID
from asimov_standup.config import apply_refinement_preset
from asimov_standup.mdp import RESET_MODES
from asimov_standup.runner import WARM_START_ENV, WARM_START_STD_ENV
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument(
        "--phase",
        choices=("discovery", "refinement"),
        default="discovery",
        help="Use refinement after recovery is learned to straighten and settle it.",
    )
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument(
        "--assist-decay-iterations",
        type=int,
        default=4_000,
        help="Linearly remove the upward assist over this many PPO iterations.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="supine_prone")
    parser.add_argument(
        "--reset-mode",
        choices=RESET_MODES,
        default="random",
        help="Lying directions to start from; supine is the easier staged run.",
    )
    parser.add_argument("--log-root", default="logs/rsl_rl")
    parser.add_argument("--joint-noise", type=float, default=0.4)
    parser.add_argument(
        "--warm-start",
        type=Path,
        help="Load only the actor from a checkpoint; critic and optimizer stay fresh.",
    )
    parser.add_argument(
        "--warm-start-std",
        type=float,
        default=0.3,
        help="Reset exploration standard deviation after loading the actor.",
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--entropy-coef", type=float)
    parser.add_argument(
        "--max-assist-force",
        type=float,
        default=180.0,
        help="Maximum upward curriculum force in newtons; use 0 for fine-tuning.",
    )
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.joint_noise <= 1:
        parser.error("--joint-noise must be between 0 and 1 radian")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than zero")
    if args.entropy_coef is not None and args.entropy_coef < 0:
        parser.error("--entropy-coef must not be negative")
    if args.max_assist_force < 0:
        parser.error("--max-assist-force must not be negative")
    if args.warm_start_std <= 0:
        parser.error("--warm-start-std must be greater than zero")
    if args.warm_start is not None:
        checkpoint = args.warm_start.expanduser().resolve()
        if not checkpoint.is_file():
            parser.error(f"--warm-start checkpoint does not exist: {checkpoint}")
        os.environ[WARM_START_ENV] = str(checkpoint)
        os.environ[WARM_START_STD_ENV] = str(args.warm_start_std)
    else:
        os.environ.pop(WARM_START_ENV, None)
        os.environ.pop(WARM_START_STD_ENV, None)

    env_cfg = load_env_cfg(TASK_ID)
    if args.phase == "refinement":
        apply_refinement_preset(env_cfg)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.events["reset_lying"].params["mode"] = args.reset_mode
    env_cfg.events["reset_lying"].params["joint_noise"] = args.joint_noise
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.events["height_assist"].params["max_force"] = args.max_assist_force
    env_cfg.events["height_assist"].params["decay_steps"] = (
        args.assist_decay_iterations * agent_cfg.num_steps_per_env
    )
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.seed = args.seed
    agent_cfg.run_name = args.run_name
    if args.learning_rate is not None:
        agent_cfg.algorithm.learning_rate = args.learning_rate
    if args.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = args.entropy_coef

    cfg = TrainConfig(
        env=env_cfg,
        agent=agent_cfg,
        video=args.video,
        log_root=args.log_root,
    )
    launch_training(TASK_ID, cfg)


if __name__ == "__main__":
    main()
