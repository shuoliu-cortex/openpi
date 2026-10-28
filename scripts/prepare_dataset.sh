#!/usr/bin/env bash
# Prepares a YamAbc joint-space LeRobot dataset for training `pi05_yam_abc_joint`, then prints the train command.
# Run scripts/setup_train_nebius.sh once per machine first. Safe to re-run: finished steps are skipped.
#
# Usage:
#   scripts/prepare_dataset.sh <repo_id> [options]
#
#   scripts/prepare_dataset.sh cortexairobot/jdsample-wipe-plates-v3 --exp-name jd_wipe_v1
#   scripts/prepare_dataset.sh cortexairobot/jdsample-wipe-plates-v3 --asset-id jd_sample/jdsample-wipe-plates-v3
#
# <repo_id> is the Hugging Face dataset repo, downloaded to $HF_LEROBOT_HOME/<repo_id>. A dataset you put there
# yourself (or any local/<name> repo id) is used as is; pass --skip-download to not contact the Hub.
#
# Options:
#   --asset-id ID       Where the norm stats live: assets/pi05_yam_abc_joint/<ID>/norm_stats.json
#                       (default <repo_id>). Computed from the parquet files if missing.
#   --revision REV      Branch, tag or commit of the dataset repo (default: main).
#   --data-root DIR     Use DIR as HF_LEROBOT_HOME.
#   --exp-name NAME     Fill in the experiment name in the printed commands.
#   --skip-download     Do not download; use the dataset already at $HF_LEROBOT_HOME/<repo_id>.
#   --skip-check        Do not run the self-check (loads a sample, decodes its videos, runs the transforms).
#
# The wrist cameras are detected from meta/info.json: observation.images.{left_wrist,right_wrist} (the config
# default) or observation.images.{left,right}.

set -euo pipefail

CONFIG_NAME="pi05_yam_abc_joint"

[[ $# -ge 1 && "$1" != -* ]] || { sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; exit 1; }
REPO_ID=$1
shift
ASSET_ID=""
REVISION=""
DATA_ROOT=""
EXP_NAME="<name>"
SKIP_DOWNLOAD=0
SKIP_CHECK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --asset-id) ASSET_ID="$2"; shift 2 ;;
        --revision) REVISION="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --exp-name) EXP_NAME="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        --skip-check) SKIP_CHECK=1; shift ;;
        -h|--help) sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."
if [[ -n "$DATA_ROOT" ]]; then export HF_LEROBOT_HOME="$DATA_ROOT"; fi
# Same default as LeRobot: $HF_LEROBOT_HOME, else $HF_HOME/lerobot, else ~/.cache/huggingface/lerobot.
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME:-$HOME/.cache/huggingface}/lerobot}"
export PATH="$HOME/.local/bin:$PATH"
DATA_DIR="$HF_LEROBOT_HOME/$REPO_ID"
ASSET_ID="${ASSET_ID:-$REPO_ID}"
NORM_STATS="assets/$CONFIG_NAME/$ASSET_ID/norm_stats.json"

step() { echo; echo "==> $*"; }
ok() { echo "    ok: $*"; }
warn() { echo "    WARNING: $*" >&2; }
die() { echo "    ERROR: $*" >&2; exit 1; }

# 1. Dataset.
step "Dataset $REPO_ID at $DATA_DIR"
if [[ $SKIP_DOWNLOAD -eq 1 || "$REPO_ID" == local/* ]]; then
    ok "download skipped"
else
    # Private repos need a token: `uv run hf auth login` or HF_TOKEN.
    uv run hf auth whoami >/dev/null 2>&1 || warn "not logged in to Hugging Face; a private dataset will fail to download."
    mkdir -p "$DATA_DIR"
    # Downloads into the LeRobot layout directly; files already present are skipped, so re-runs resume.
    uv run hf download "$REPO_ID" --repo-type dataset ${REVISION:+--revision "$REVISION"} \
        --local-dir "$DATA_DIR" --max-workers 16 --quiet >/dev/null
    ok "downloaded $REPO_ID${REVISION:+@$REVISION}"
fi
for d in meta data videos; do
    [[ -d "$DATA_DIR/$d" ]] || die "missing $DATA_DIR/$d"
done
ok "$(find "$DATA_DIR/data" -name '*.parquet' | wc -l) parquet files, $(find "$DATA_DIR/videos" -name '*.mp4' | wc -l) videos"

# 2. Features: 14-dim joint state/action and the camera names, which decide the config overrides.
step "Features"
WRIST_CAMS=$(uv run python - "$DATA_DIR/meta/info.json" <<'EOF'
import json
import sys

features = json.load(open(sys.argv[1]))["features"]
for key in ("observation.state", "action"):
    assert key in features and features[key]["shape"] == [14], f"{key}: expected shape [14], got {features.get(key)}"
cams = {k.removeprefix("observation.images.") for k in features if k.startswith("observation.images.")}
for wrist in (("left_wrist", "right_wrist"), ("left", "right")):
    if {"top", *wrist} <= cams:
        print(*wrist)
        break
else:
    sys.exit(f"expected cameras top + left_wrist/right_wrist or left/right, got {sorted(cams)}")
EOF
) || die "unsupported dataset (see the error above)."
ok "joint state/action [14], cameras top + ${WRIST_CAMS/ //}"

OVERRIDES=(--data.repo-id "$REPO_ID")
if [[ "$ASSET_ID" != "$REPO_ID" ]]; then OVERRIDES+=(--data.assets.asset-id "$ASSET_ID"); fi
if [[ "$WRIST_CAMS" != "left_wrist right_wrist" ]]; then
    read -r -a cams <<<"$WRIST_CAMS"
    OVERRIDES+=(--data.wrist-camera-keys "${cams[@]}")
fi

# 3. Norm stats.
step "Norm stats"
if [[ -f "$NORM_STATS" ]]; then
    ok "$NORM_STATS"
else
    uv run scripts/yam_abc_joint_norm_stats.py --dataset-dir "$DATA_DIR" --output-dir "$(dirname "$NORM_STATS")"
    ok "computed $NORM_STATS (commit it so other machines reuse it)"
fi

# 4. Self-check with the exact config the train command will use.
step "Self-check"
if [[ $SKIP_CHECK -eq 1 ]]; then
    ok "skipped (--skip-check)"
else
    uv run python - "$CONFIG_NAME" "${OVERRIDES[@]}" <<'EOF'
import sys

import openpi.policies.yam_abc_joint_policy as _joint
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

# Parse the config exactly as scripts/train.py does, overrides included.
sys.argv = ["train.py", sys.argv[1], "--exp-name", "check", *sys.argv[2:]]
cfg = _config.cli()
dc = cfg.data.create(cfg.assets_dirs, cfg.model)
assert dc.norm_stats is not None, "norm stats not loaded"

raw = _data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
x = raw[len(raw) // 2]
assert tuple(x["action"].shape) == (cfg.model.action_horizon, _joint.ABC_JOINT_DIM), x["action"].shape
for key in dc.repack_transforms.inputs[0].structure["images"].values():
    assert x[key].ndim == 3, key
print(f"    frames={len(raw)} prompt={x['prompt']!r}")

s = _data_loader.transform_dataset(raw, dc)[len(raw) // 2]
assert tuple(s["actions"].shape) == (cfg.model.action_horizon, cfg.model.action_dim), s["actions"].shape
print(f"    model input ok: actions {s['actions'].shape}, prompt tokens {int(s['tokenized_prompt_mask'].sum())}")
EOF
    ok "sample loads, videos decode, transforms run"
fi

CKPT_DIR="checkpoints/$CONFIG_NAME/$EXP_NAME"
step "Done"
cat <<EOF
    Train (inside tmux; HF_LEROBOT_HOME=$HF_LEROBOT_HOME must be set):
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $CONFIG_NAME --exp-name $EXP_NAME \\
            ${OVERRIDES[*]}
    Back up checkpoints to a Hugging Face model repo (second tmux window):
        scripts/sync_checkpoints.sh push $CKPT_DIR hf://<user>/<repo>/$CKPT_DIR \\
            --pid \$(pgrep -f "exp-name $EXP_NAME" | head -1) --prune
    Resume on a new machine: rerun this script, then
        scripts/sync_checkpoints.sh pull $CKPT_DIR hf://<user>/<repo>/$CKPT_DIR
        and add --resume to the train command.
EOF
