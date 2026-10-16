"""Exercise diverse resets, floor clearance, limits and partial reset isolation."""

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from asimov_standup.config import standup_env_cfg
from asimov_standup.mdp import RESET_MODES, collision_bottom, reset_lying_pose

cfg = standup_env_cfg(play=True)
cfg.scene.num_envs = 32
cfg.events["reset_lying"].params.update(mode="random", joint_noise=0.4)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
try:
    env.reset()
    robot = env.scene["robot"]
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    ids = np.flatnonzero((model.geom_group == 3) & (model.geom_bodyid != 0))
    orientations = set()
    for mode in RESET_MODES:
        reset_lying_pose(env, None, mode=mode, joint_noise=0.4, standing_probability=0)
        qposes = env.sim.data.qpos.detach().cpu().numpy().copy()
        joint_pos = robot.data.joint_pos
        limits = robot.data.soft_joint_pos_limits
        assert (joint_pos >= limits[..., 0] - 1e-6).all()
        assert (joint_pos <= limits[..., 1] + 1e-6).all()
        for qpos in qposes:
            data.qpos[:] = qpos
            mujoco.mj_kinematics(model, data)
            bottom = collision_bottom(model, data, ids)
            assert 0.010 < bottom < 0.014, (mode, bottom)
        assert np.std(qposes[:, 7:], axis=0).mean() > 0.08
        if mode == "random":
            # Use the free joint rotation directly to classify lying direction.
            for qpos in qposes:
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, qpos[3:7])
                gravity = -rotation.reshape(3, 3)[2]
                axis = int(np.argmax(np.abs(gravity)))
                orientations.add((axis, int(np.sign(gravity[axis]))))
        before = env.sim.data.qpos.clone()
        reset_lying_pose(
            env,
            torch.tensor([1, 7], device=env.device),
            mode=mode,
            joint_noise=0.4,
            standing_probability=0,
        )
        untouched = torch.ones(32, device=env.device, dtype=torch.bool)
        untouched[[1, 7]] = False
        assert torch.equal(before[untouched], env.sim.data.qpos[untouched])
        env.sim.forward()
        for _ in range(12):
            obs, rewards, *_ = env.step(
                torch.zeros(env.action_space.shape, device=env.device)
            )
            assert torch.isfinite(obs["actor"]).all() and torch.isfinite(rewards).all()
        print(f"PASS {mode}: clearance, diversity, partial reset and simulation")
    assert orientations == {(0, -1), (0, 1), (1, -1), (1, 1)}, orientations
    print("PASS: all four random lying directions sampled")
    torch.manual_seed(123)
    reset_lying_pose(env, None, mode="random", joint_noise=0.4, standing_probability=0)
    first = env.sim.data.qpos.clone()
    torch.manual_seed(123)
    reset_lying_pose(env, None, mode="random", joint_noise=0.4, standing_probability=0)
    assert torch.equal(first, env.sim.data.qpos)
    print("PASS: reproducible poses with fixed seed")
finally:
    env.close()
