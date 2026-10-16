# Asimov 1 stand-up learning in mjlab

This extension loads Menlo Research's official Asimov 1 MJCF from the Cyclotron
checkout and trains a 23-DoF position-control policy with PPO. The default
discovery run starts from randomized supine poses on flat ground. Prone and mixed
resets are available explicitly; learning both directions at once is much harder
and is not the recommended first run.

The reward is deliberately staged: dense height gain from the actual floor pose,
height-gated pelvis righting, dense head height, a broad coupled stable-standing
objective, and nominal-posture tracking that activates only after the robot has
risen. Twenty-five percent
of training resets start near the standing goal (reverse curriculum), so PPO sees
successful states before it can reach them from the floor. Discovery-phase
regularizers are intentionally weak. During training, an upward force of at most 180 N
(approximately 57% of robot weight) helps early exploration and linearly decays to
zero over 2,000 PPO iterations by default.

## Commands

Run these from a Cyclotron checkout inside Ubuntu/WSL. This example uses
[mjlab](https://github.com/mujocolab/mjlab) as its training environment, so
install mjlab separately and adjust `--project ~/mjlab` if it lives elsewhere.

```bash
cd examples/asimov_standup

# Validate model loading, contacts, resets, observations, rewards, and CUDA stepping.
uv run --project ~/mjlab python smoke_test.py

# RTX 4060 starting point. Reduce to 128 if GPU memory is tight.
uv run --project ~/mjlab python train.py --num-envs 512 --max-iterations 10000 \
  --reset-mode supine --assist-decay-iterations 2000 --run-name supine_v4

# Live metrics at http://localhost:6006. The setuptools overlay supplies
# pkg_resources, which TensorBoard 2.20 still imports.
uv run --project ~/mjlab --with setuptools==80.9.0 tensorboard \
  --logdir logs/rsl_rl --port 6006

# Preview the latest checkpoint with randomized poses at http://localhost:8081.
uv run --project ~/mjlab python play.py --port 8081

# Back, belly, either side; random heading and per-joint offsets (radians).
uv run --project ~/mjlab python play.py --reset-mode random --joint-noise 0.4 \
  --speed 0.25 --fps 60 --episode-seconds 2 --port 8081

# Original reset distribution for the old supine checkpoint.
uv run --project ~/mjlab python play.py --reset-mode supine --joint-noise 0.08

# Train diverse recovery in a separate run; old policies may fail these poses.
uv run --project ~/mjlab python train.py --reset-mode random --joint-noise 0.4 \
  --run-name random_recovery --num-envs 512

# Recommended actor-only fine-tune from the smoother supine checkpoint.
# Critic, optimizer, PPO iteration and curriculum counter start fresh.
uv run --project ~/mjlab python train.py \
  --warm-start logs/rsl_rl/asimov1_standup/<supine-run>/model_3500.pt \
  --warm-start-std 0.3 --reset-mode random --joint-noise 0.4 \
  --max-assist-force 0 \
  --learning-rate 0.0001 --entropy-coef 0.003 \
  --num-envs 512 --max-iterations 3000 --run-name random_from_3500

# Phase 2: keep the learned randomized recovery, then optimize a straighter,
# quieter final stance. This starts a fresh critic because the reward changed.
uv run --project ~/mjlab python train.py --phase refinement \
  --warm-start logs/rsl_rl/asimov1_standup/<random-recovery-run>/model_2900.pt \
  --warm-start-std 0.15 --reset-mode random --joint-noise 0.4 \
  --max-assist-force 0 \
  --learning-rate 0.00005 --entropy-coef 0.001 \
  --num-envs 512 --max-iterations 2000 --run-name refinement_from_2900

# Quarter-speed physics while keeping the browser stream at 60 FPS.
uv run --project ~/mjlab python play.py --speed 0.25 --fps 60 --port 8081

# Slow motion with a short 2-second simulated episode (8 real seconds at 0.25x).
uv run --project ~/mjlab python play.py --speed 0.25 --fps 60 \
  --episode-seconds 2 --port 8081

# Objective batch evaluation from the floor, with training assistance disabled.
uv run --project ~/mjlab python evaluate.py --reset-mode supine

# Inspect the untrained lying reset without needing a checkpoint.
uv run --project ~/mjlab python play.py --agent zero --port 8081
```

Training from scratch is a long-running optimization job. Monitor
`Metrics/stand_success` and `Metrics/pelvis_height` in the training output. Do not
resume the old `supine_prone` checkpoint: its collapsed policy and exploration state
are part of the bad local optimum. A short
smoke run checks the complete PPO path but is not expected to learn the behavior:

```bash
uv run --project ~/mjlab python train.py --num-envs 32 --max-iterations 2 \
  --run-name smoke
```

The model is discovered from Cyclotron's `third_party/asimov-1/sim-model`
submodule. Set `ASIMOV_1_MODEL_DIR` if it is stored elsewhere.

Playback defaults to `--reset-mode random --joint-noise 0.4 --seed 42`.
`mixed` retains its original meaning (back/belly only); `random` also includes
both sides. `left` and `right` isolate side starts. Joint noise is clamped to
soft joint limits. Diverse poses are positioned 12 mm above the flat floor using
collision-shape forward kinematics, then settle under physics. This does not
guarantee every randomized posture is recoverable by an old checkpoint.
Training and evaluation retain their original supine/0.08 defaults unless
explicitly overridden. Run `python test_random_resets.py` in the mjlab environment
to check reset directions, limits, floor clearance and partial reset isolation.

## Third-party assets

The robot uses a Blender-authored livery based on the supplied photograph:
orange head shell, breastplate with a black arched belly cutout, shoulder caps
and sole edges; black face/limbs and bronze joint covers.
`assets/reference_livery/asimov_reference.blend` contains the
editable materials and assembled model; `preview.png` is a Blender render.
`build_livery_blender.py` reproduces the assets using Blender 4.3 on Windows
and the WSL source model (adjust `SOURCE` if relocated).
OBJ UVs, `palette.png`, and the smooth `breastplate.png` mask carry the colors
into MuJoCo and Viser. Viser's current
texture conversion does not reproduce Blender's metallic/roughness settings.
This is a photo-inspired approximation on the supplied geometry, not an official
texture. Only visual meshes/materials change; collisions and inertial properties
remain original, so existing checkpoints need no retraining.
Restart `play.py` to see the new appearance in Viser (port 8081).

The physics model remains in the official `menloresearch/cyclotron` checkout.
Derived visual meshes and the Blender scene are stored in `assets/reference_livery`
alongside source attribution and the upstream software license.
Cyclotron is BSD-3-Clause; the Asimov software
and simulation model checkout declares GPL-2.0, while its hardware sources declare
CERN-OHL-S-2.0. Keep those upstream license files with any redistributed assets.

## Design basis

- Cyclotron provides the official Asimov model, joint ordering, standing pose, PD
  gains, effort limits, armatures, and friction values.
- HoST uses PPO, staged stand-up rewards, diverse initial postures, smoothness
  regularization, and a decaying vertical pulling-force curriculum.
- Recent real-world get-up work similarly separates trajectory discovery from later
  smoothness and robustness refinement. This baseline implements the discovery phase;
  deployable hardware control still requires a second refinement/domain-randomization
  phase and safety validation.
