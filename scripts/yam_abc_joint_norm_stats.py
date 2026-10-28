"""Builds openpi norm stats for `pi05_yam_abc_joint` from a LeRobot dataset's parquet files (no video decoding).

State stats are over all `observation.state` frames. Action stats are over the action chunks the training loader
builds: `action[t : t + horizon]` within each episode, with the last frame repeated past the episode end.

Usage:
    uv run scripts/yam_abc_joint_norm_stats.py --dataset-dir <HF_LEROBOT_HOME>/<repo_id> --output-dir <assets dir>
"""

import pathlib

import numpy as np
import pandas as pd
import tyro

import openpi.shared.normalize as normalize


def _norm_stats(x: np.ndarray) -> normalize.NormStats:
    return normalize.NormStats(
        mean=x.mean(0).astype(np.float32),
        std=x.std(0).astype(np.float32),
        q01=np.quantile(x, 0.01, axis=0).astype(np.float32),
        q99=np.quantile(x, 0.99, axis=0).astype(np.float32),
    )


def main(dataset_dir: pathlib.Path, output_dir: pathlib.Path, horizon: int = 30) -> None:
    files = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {dataset_dir / 'data'}")
    columns = ["episode_index", "frame_index", "observation.state", "action"]
    frames = pd.concat([pd.read_parquet(f, columns=columns) for f in files])
    frames = frames.sort_values(["episode_index", "frame_index"], ignore_index=True)

    state = np.stack(frames["observation.state"].to_numpy()).astype(np.float64)
    action = np.stack(frames["action"].to_numpy()).astype(np.float64)
    chunks = []
    for idx in frames.groupby("episode_index").indices.values():
        a = action[idx]
        t = np.minimum(np.arange(len(a))[:, None] + np.arange(horizon)[None], len(a) - 1)
        chunks.append(a[t].reshape(-1, a.shape[1]))

    normalize.save(output_dir, {"state": _norm_stats(state), "actions": _norm_stats(np.concatenate(chunks))})
    print(f"{len(state)} frames, {frames['episode_index'].nunique()} episodes -> {output_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    tyro.cli(main)
