# Cortex internal notes

What this fork adds on top of upstream openpi (base: `215abfb`). Model and training code are otherwise unchanged.

## Dependencies

- LeRobot v3.0 datasets: `lerobot>=0.4.0,<0.5.0` (was a pinned git commit). `data_loader.py` uses the new import path
  and v3.0 task table.
- `rerun-sdk==0.22.1` is pinned because newer versions require numpy>=2.

## YAM robot support

All YAM setups are bimanual: 2x (6 joints + gripper), cameras `top` + left/right wrist.

| Action space | Policy | Data config | Train configs |
|---|---|---|---|
| ABC EEF (xyz + 6D rot + gripper per arm, 20-dim) | `yam_abc_eef_policy.py` | `LeRobotYamAbcEefDataConfig` | `pi05_yam_abc`, `pi05_yam_abc_lora`, `pi05_yam_abc_sim_lora` |
| ABC joint (absolute, 14-dim) | `yam_abc_joint_policy.py` | `LeRobotYamAbcJointDataConfig` | `pi05_yam_abc_joint` |
| Ma2 joint (14-dim) | `yam_ma2_policy.py` | `LeRobotYamMa2DataConfig` | `pi05_yam_ma2` |

- **EEF actions are chunk-relative**: training converts each action pose to the current EEF frame
  (`inv(T_state) @ T_action`, `RelativeEefActions`); inference converts back (`AbsoluteEefActions`).
- **Joint actions are absolute**, no delta transform.
- Full fine-tune recipe (`pi05_yam_abc`, `pi05_yam_abc_joint`): from `pi05_base`, action horizon 30, batch 256,
  cosine LR 5e-5 → 2.5e-6 with 1k warmup, EMA 0.99, 15k–30k steps.

## State dropout (Pi05)

`StateDropout` in `transforms.py` hides the state from the Pi05 prompt for a fraction of training samples
(`state_dropout_prob`, default 0.3 for ABC EEF), so the policy does not over-rely on it. At serve time,
`serve_policy.py --drop-state` hides it for every request; this is in-distribution thanks to the training dropout.

## Norm stats

- Committed in the repo (`assets/` is no longer ignored), at `assets/<robot>/<dataset>/norm_stats.json`.
- `scripts/yam_abc_eef_norm_stats.py`: from the dataset's precomputed stats in `meta/` (`observation.eef` and the
  chunk-relative action stats).
- `scripts/yam_abc_joint_norm_stats.py`: from the parquet files only (no video decoding), matching the loader's action
  chunks.
- Train on another dataset without a new config: `--data.repo-id <id> --data.assets.asset-id <robot>/<dataset>`.
  `serve_policy.py` then picks up the norm stats saved in the checkpoint automatically (override with
  `--policy.asset-id`).

## Training and data

- `DataConfig.episodes`: train on a subset of episodes.
- `train.py` also logs `learning_rate`, `time_per_step`, `samples_per_sec` and `data_time_per_step` (close to
  `time_per_step` means data loading is the bottleneck).
- Keep datasets on local disk; network filesystems slow down video decoding.

## Cloud workflow

1. Machine setup, once per machine: `scripts/setup_train_nebius.sh` (Nebius) or `scripts/setup_train.sh`
   (Lambda / AWS, can sync the dataset from S3).
2. Per dataset: `scripts/prepare_dataset.sh <hf_repo_id> --exp-name <name>` downloads the dataset, computes norm
   stats, runs a self-check and prints the train command.
3. Back up checkpoints next to training: `scripts/sync_checkpoints.sh push <ckpt_dir> <dest> --pid <train_pid>`,
   where `<dest>` is a local dir, `s3://...` or `hf://<user>/<repo>/...`. On a new machine, `pull` then train with
   `--resume`. Only finalized checkpoints are copied, so a restore never picks up a partial step.
