"""Exercise diverse resets, floor clearance, limits and partial reset isolation."""

import mujoco
import numpy as np
import torch
from asimov_standup.config import standup_env_cfg
from asimov_standup.mdp import (
    RESET_MODES,
    collision_bottom,
    reset_lying_pose,
    success_metric,
)
from mjlab.envs import ManagerBasedRlEnv

cfg = standup_env_cfg(play=True)
cfg.scene.num_envs = 32
cfg.events["reset_lying"].params.update(mode="random", joint_noise=0.4)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
try:
    env.reset()
    robot = env.scene["robot"]
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    robot_ids = robot.indexing.geom_ids.detach().cpu().numpy()
    ids = robot_ids[model.geom_group[robot_ids] == 3]
    assert all(model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE for i in ids)
    orientations = set()
    # Regression: this old default path used to skip floor correction.
    reset_lying_pose(
        env, None, mode="supine", joint_noise=0.08, standing_probability=0
    )
    for qpos in env.sim.data.qpos.detach().cpu().numpy():
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        bottom = collision_bottom(model, data, ids)
        assert 0.010 < bottom < 0.014, ("default supine", bottom)
    print("PASS: default supine reset has floor clearance")
    for mode in RESET_MODES:
        reset_lying_pose(env, None, mode=mode, joint_noise=0.4, standing_probability=0)
        qposes = env.sim.data.qpos.detach().cpu().numpy().copy()
        side_gravity = []
        joint_pos = robot.data.joint_pos
        limits = robot.data.soft_joint_pos_limits
        assert (joint_pos >= limits[..., 0] - 1e-6).all()
        assert (joint_pos <= limits[..., 1] + 1e-6).all()
        for qpos in qposes:
            data.qpos[:] = qpos
            mujoco.mj_kinematics(model, data)
            bottom = collision_bottom(model, data, ids)
            assert 0.010 < bottom < 0.014, (mode, bottom)
            mujoco.mj_fwdPosition(model, data)
            contact = data.contact
            robot_pair = np.isin(contact.geom1, ids) & np.isin(contact.geom2, ids)
            depth = -contact.dist[robot_pair].min(initial=0.0)
            assert depth <= 0.001, (mode, "self-penetration", depth)
            if mode in {"left", "right"}:
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, qpos[3:7])
                side_gravity.append(-rotation.reshape(3, 3)[2, 1])
        assert np.std(qposes[:, 7:], axis=0).mean() > 0.08
        if mode == "left":
            assert min(side_gravity) > 0, side_gravity
        elif mode == "right":
            assert max(side_gravity) < 0, side_gravity
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
    reset_lying_pose(env, None, standing_probability=1.0)
    assert not env._standup_recovery_eligible.any()
    assert not success_metric(env, minimum_height=0.0).any()
    reset_lying_pose(env, None, standing_probability=0.0)
    assert env._standup_recovery_eligible.all()
    print("PASS: standing resets are excluded from recovery success")
    feet_term = env.reward_manager.get_term_cfg("feet_flat")
    reset_lying_pose(env, None, standing_probability=1.0)
    env.sim.forward()
    flat = feet_term.func(env, **feet_term.params)
    assert flat.mean() > 0.7, ("nominal stance should have flat feet", flat)
    reset_lying_pose(env, None, standing_probability=0.0)
    env.sim.forward()
    flat = feet_term.func(env, **feet_term.params)
    assert flat.max() < 0.05, ("lying poses must be gated out", flat)
    print("PASS: feet_flat rewards a flat stance and ignores lying poses")
    stop_term = env.reward_manager.get_term_cfg("ankle_roll_stop")
    reset_lying_pose(env, None, standing_probability=1.0)
    env.sim.forward()
    assert stop_term.func(env, **stop_term.params).max() < 0.01, "nominal ankles are free"
    ankle_ids = stop_term.params["asset_cfg"].joint_ids
    joint_pos = robot.data.joint_pos.clone()
    joint_pos[:, ankle_ids] = robot.data.soft_joint_pos_limits[:, ankle_ids, 1]
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    env.sim.forward()
    at_stop = stop_term.func(env, **stop_term.params)
    assert torch.allclose(at_stop, torch.full_like(at_stop, 2.0)), at_stop
    print("PASS: ankle_roll_stop is zero at the nominal pose and 1 per joint at its limit")
finally:
    env.close()
