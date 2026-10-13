# One-leg stand training

This branch adds a standalone Isaac Lab/Newton task that trains Asimov-1 to
stand quietly on its left leg. The 23-action policy may use its waist, arms,
and folded right leg for balance, but velocity, angular-momentum, and
action-rate costs discourage continuous thrashing. The right-foot target is
0.14 m and right-foot contact is penalized.

## Train

From the repository root:

```bash
/home/yoll/isaaclab/env_isaaclab/bin/python scripts/rsl_rl/train_one_leg.py \
  --num_envs 1024 \
  --max_iterations 1500 \
  --video \
  --video_interval 500 \
  --video_length 250 \
  --headless
```

At the 50 Hz policy rate, 250 steps produces a five-second training clip.
Checkpoints and videos are written under `logs/rsl_rl/asimov1_one_leg/`.

Resume an existing checkpoint with:

```bash
/home/yoll/isaaclab/env_isaaclab/bin/python scripts/rsl_rl/train_one_leg.py \
  --num_envs 1024 \
  --max_iterations 1500 \
  --resume_checkpoint /absolute/path/to/model_600.pt \
  --video \
  --video_interval 500 \
  --video_length 250 \
  --headless
```

Useful smoothness signals in TensorBoard are:

- `Metrics/arm_joint_speed_rms_rad_s`
- `Metrics/swing_leg_joint_speed_rms_rad_s`
- `Metrics/angular_momentum_mean`
- `Metrics/swing_foot_height`
- `Metrics/swing_contact_rate`

The actor observation and action sizes remain 78 and 23, respectively, so
checkpoints from the earlier one-leg task remain shape-compatible.
