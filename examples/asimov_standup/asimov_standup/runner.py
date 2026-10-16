"""Training runner with actor-only initialization for task transfer."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from mjlab.rl import MjlabOnPolicyRunner

WARM_START_ENV = "ASIMOV_STANDUP_WARM_START"
WARM_START_STD_ENV = "ASIMOV_STANDUP_WARM_START_STD"


class StandupRunner(MjlabOnPolicyRunner):
    """Optionally initialize the actor while keeping new PPO training state."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        checkpoint_value = os.environ.get(WARM_START_ENV)
        if not checkpoint_value:
            return

        checkpoint = Path(checkpoint_value)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Warm-start checkpoint does not exist: {checkpoint}"
            )

        self.load(
            str(checkpoint),
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            strict=True,
            map_location=self.device,
        )
        initial_std = float(os.environ.get(WARM_START_STD_ENV, "0.3"))
        distribution = self.alg.get_policy().distribution
        with torch.no_grad():
            if hasattr(distribution, "std_param"):
                distribution.std_param.fill_(initial_std)
            elif hasattr(distribution, "log_std_param"):
                distribution.log_std_param.fill_(math.log(initial_std))
            else:
                raise TypeError("Warm start requires a Gaussian actor distribution")
        # Mjlab's load extension restores curriculum state independently of
        # load_cfg. A transfer run needs its own counter and iteration numbering.
        self.env.unwrapped.common_step_counter = 0
        self.current_learning_iteration = 0
        print(f"[INFO] Warm-started actor only from: {checkpoint}")
        print(f"[INFO] Reset actor exploration std to: {initial_std}")
        print("[INFO] Critic, optimizer, iteration and curriculum counter are fresh")
