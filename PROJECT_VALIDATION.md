# Project Validation Report

Date: 2026-07-21

This build combines the uploaded project with the waist-enabled Pink-Abs imitation pipeline and the simplified three-stage task contract.

```text
Quest/OpenXR Pink-Abs demonstration recording
  -> 26-D HDF5 validation
  -> physical replay validation
  -> Isaac Lab Mimic annotation
  -> successful Mimic data generation
  -> robomimic BC-RNN training
  -> policy rollout evaluation
```

## Learning contract

- Default demonstration/Mimic/BC action space: **26-D Pink-Abs**
- Action order: left wrist pose 7, right wrist pose 7, right hand 6, left hand 6
- Pink solves both wrist targets over both arms plus `torso_Joint` and `Yaw_Joint`
- The legacy 24-D IK-Rel environments remain registered, but action layouts must not be mixed
- Recording timeout is disabled in the recording config and the official recording launcher path
- Object-dropping termination remains enabled

## Public success sequence

```text
right_grasped
  -> handover_completed
  -> placed
```

Mean finger flexion and individual finger-joint positions are not read by the phase tracker.

1. `right_grasped`: the right EEF is near the cuboid while its center is lifted at least `GRASP_LIFT_HEIGHT` above the initial center height.
2. `handover_completed`: after pickup, the cuboid is inside the `HANDOVER_POS` region, the left EEF is near it, and the right EEF is clear.
3. `placed`: after transfer, the cuboid is stably at `CUBOID_GOAL_POS`; the first valid placement frame has the left EEF near and the right EEF clear.

A final both-hands-clear phase is intentionally not required. Demo recording still requires the live stable-placement condition for the configured number of consecutive success steps.

## Main files changed

- `source/.../g1d_bimanual/g1d_handover_phase.py`
- `source/.../g1d_bimanual/g1d_bimanual_env_cfg.py`
- `source/.../g1d_bimanual/g1d_bimanual_mimic_env.py`
- `source/.../g1d_bimanual/g1d_bimanual_mimic_env_cfg.py`
- `scripts/validate_dataset.py`
- `scripts/smoke_test_environment.py`
- `start_record_demos_xr.sh`
- `.env.teleop`
- `.env.teleop.example`
- `start_teleop_xr.sh`
- `README.md`
- `PROJECT_VALIDATION.md`

## Static validation performed

- All Python files parsed and compiled successfully
- All shell scripts passed `bash -n`
- JSON and TOML files parsed successfully
- No success/Mimic code references hand-closure thresholds or active finger-joint IDs
- The public phase-signal references are consistently three-stage
- Synthetic 26-D raw, annotated, and generated HDF5 files passed the dataset validator
- Incorrect signal order was rejected by the dataset validator
- 24-D data was rejected when `pink-abs` was requested
- Pink action-layout registration and Mimic classes were retained
- A wheel was built successfully
- `bc_handover.json` was confirmed in wheel package data
- The output ZIP passed integrity and path-safety checks

## Runtime validation required on the target workstation

```bash
cd /workspace/my_isaac_project/g1d_bimanual_cuboid_handover_lab

/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e \
  source/g1d_bimanual_cuboid_handover

/workspace/isaaclab/_isaac_sim/python.sh \
  scripts/smoke_test_environment.py \
  --task G1D-Bimanual-CuboidHandover-Pink-Abs-Record-v0 \
  --enable_pinocchio \
  --device cuda:0 \
  --headless
```

Do not begin large-scale collection until the smoke test reports:

```text
G1-D cuboid-handover environment smoke test PASSED
```
