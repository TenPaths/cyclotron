"""Build and step the Asimov stand-up environment without training."""

import torch
from mjlab.envs import ManagerBasedRlEnv

from asimov_standup.config import standup_env_cfg


def main() -> None:
    cfg = standup_env_cfg(play=True)
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    observations, _ = env.reset()
    actor_obs = observations["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    assert actor_obs.shape[0] == 4
    for _ in range(8):
        action = torch.zeros(env.action_space.shape, device=env.device)
        observations, rewards, terminated, time_out, _ = env.step(action)
        actor_obs = observations["actor"]
        assert isinstance(actor_obs, torch.Tensor)
        assert torch.isfinite(actor_obs).all()
        assert torch.isfinite(rewards).all()
        assert not terminated.any()
        assert not time_out.any()
    height = env.scene["robot"].data.root_link_pos_w[:, 2]
    print(f"smoke test passed; pelvis heights={height.tolist()}")
    env.close()


if __name__ == "__main__":
    main()
