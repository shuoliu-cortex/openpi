"""Builds openpi norm stats for `pi05_yam_abc` from the YamAbc dataset's precomputed stats.

State stats come from `observation.eef` in `meta/stats.json`. Action stats come from the chunk-relative EEF stats
(`abc_chunk_stats_H{horizon}_shared.json`), which use the same representation as `RelativeEefActions`: per arm
inv(T(observation.eef[t])) @ T(action.eef[t + k]), pooled over all k.

Usage:
    uv run scripts/yam_abc_norm_stats.py --meta-dir meta --horizon 30
"""

import json
import pathlib

import numpy as np
import tyro

import openpi.shared.normalize as normalize


def _norm_stats(stats: dict) -> normalize.NormStats:
    return normalize.NormStats(
        mean=np.asarray(stats["mean"], dtype=np.float32),
        std=np.asarray(stats["std"], dtype=np.float32),
        q01=np.asarray(stats["q01"], dtype=np.float32),
        q99=np.asarray(stats["q99"], dtype=np.float32),
    )


def main(
    meta_dir: pathlib.Path = pathlib.Path("meta"),
    horizon: int = 30,
    output_dir: pathlib.Path = pathlib.Path("assets/pi05_yam_abc/abc130k-eef"),
) -> None:
    dataset_stats = json.loads((meta_dir / "stats.json").read_text())
    chunk_stats = json.loads((meta_dir / f"abc_chunk_stats_H{horizon}_shared.json").read_text())
    if chunk_stats["horizon"] != horizon:
        raise ValueError(f"Chunk stats horizon {chunk_stats['horizon']} does not match {horizon}.")
    if chunk_stats["anchor"] != "observation.eef at the chunk start t":
        raise ValueError(f"Unexpected chunk stats anchor: {chunk_stats['anchor']}")

    norm_stats = {
        "state": _norm_stats(dataset_stats["observation.eef"]),
        "actions": _norm_stats(chunk_stats["stats"]),
    }
    normalize.save(output_dir, norm_stats)
    print(f"Wrote {output_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    tyro.cli(main)
