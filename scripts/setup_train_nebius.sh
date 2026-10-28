#!/usr/bin/env bash
# Sets up a fresh Nebius AI Cloud GPU VM (Ubuntu image with NVIDIA drivers + CUDA) to train `pi05_yam_abc_joint`.
# Safe to re-run: every step checks first and skips work that is already done.
#
# Usage:
#   scripts/setup_train_nebius.sh                                          # environment only
#   scripts/setup_train_nebius.sh --data-root /mnt/data --hf-repo <user/dataset>
#
# Options:
#   --data-root DIR       Use DIR as HF_LEROBOT_HOME (e.g. a mounted Nebius secondary disk). Keep the dataset on
#                         local/block disk; network filesystems slow down video decoding.
#   --hf-repo REPO        Download the dataset from this Hugging Face dataset repo (must contain meta/, data/,
#                         videos/). For a private repo, log in first: uv run hf auth login (or set HF_TOKEN).
#   --hf-revision REV     Branch, tag or commit of the dataset repo (default: main).
#   --norm-max-frames N   Frames sampled to compute joint norm stats (default 200000; 0 = whole dataset).
#   --skip-apt            Do not install system packages.
#   --skip-check          Do not run the dataset self-check.
#
# The dataset is expected at $HF_LEROBOT_HOME/local/abc130k-eef
# (HF_LEROBOT_HOME defaults to $HF_HOME/lerobot, or ~/.cache/huggingface/lerobot without HF_HOME).

set -euo pipefail

CONFIG_NAME="pi05_yam_abc_joint"
REPO_ID="local/abc130k-eef"
NORM_STATS="assets/$CONFIG_NAME/abc130k-joint/norm_stats.json"
MIN_GPU_MEM_MIB=70000

DATA_ROOT=""
HF_REPO=""
HF_REVISION=""
NORM_MAX_FRAMES=200000
SKIP_APT=0
SKIP_CHECK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --hf-repo) HF_REPO="$2"; shift 2 ;;
        --hf-revision) HF_REVISION="$2"; shift 2 ;;
        --norm-max-frames) NORM_MAX_FRAMES="$2"; shift 2 ;;
        --skip-apt) SKIP_APT=1; shift ;;
        --skip-check) SKIP_CHECK=1; shift ;;
        -h|--help) sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."
if [[ -n "$DATA_ROOT" ]]; then export HF_LEROBOT_HOME="$DATA_ROOT"; fi
# Same default as LeRobot: $HF_LEROBOT_HOME, else $HF_HOME/lerobot, else ~/.cache/huggingface/lerobot.
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME:-$HOME/.cache/huggingface}/lerobot}"
DATA_DIR="$HF_LEROBOT_HOME/$REPO_ID"
export PATH="$HOME/.local/bin:$PATH"

step() { echo; echo "==> $*"; }
ok() { echo "    ok: $*"; }
warn() { echo "    WARNING: $*" >&2; }
die() { echo "    ERROR: $*" >&2; exit 1; }

SUDO=""
if [[ $EUID -ne 0 ]]; then SUDO="sudo"; fi

# 1. GPU.
step "Checking GPUs"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found. Create the VM from a Nebius image with NVIDIA drivers (Ubuntu ... CUDA)."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /'
min_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | awk 'NR==1')
if (( min_mem < MIN_GPU_MEM_MIB )); then
    warn "Smallest GPU has ${min_mem} MiB. Full pi05 fine-tuning needs ~80 GB GPUs;"
    warn "use --fsdp-devices <num_gpus> or a smaller --batch-size."
else
    ok "GPU memory ${min_mem} MiB"
fi

# 2. Data disk.
step "Data disk"
mkdir -p "$HF_LEROBOT_HOME"
ok "HF_LEROBOT_HOME=$HF_LEROBOT_HOME ($(df -h --output=avail "$HF_LEROBOT_HOME" | tail -1 | tr -d ' ') free)"

# 3. System packages: FFmpeg (torchcodec needs the system libraries; videos are AV1), git-lfs, build tools.
step "System packages"
# Capture the decoder list first: `grep -q` exiting early would SIGPIPE ffmpeg and fail under pipefail.
has_av1() { command -v ffmpeg >/dev/null && [[ "$(ffmpeg -hide_banner -decoders 2>/dev/null)" =~ libdav1d|libaom-av1 ]]; }
if [[ $SKIP_APT -eq 1 ]]; then
    ok "skipped (--skip-apt)"
elif has_av1 && command -v git-lfs >/dev/null && command -v gcc >/dev/null \
    && command -v rsync >/dev/null && command -v tmux >/dev/null; then
    ok "ffmpeg, git-lfs, rsync, tmux and build tools already installed"
else
    $SUDO apt-get update -y
    $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg git git-lfs build-essential curl rsync tmux
fi
has_av1 || die "ffmpeg is missing or has no AV1 decoder (libdav1d / libaom-av1)."
ffmpeg_major=$(ffmpeg -version | awk 'NR==1' | sed -E 's/^ffmpeg version n?([0-9]+).*/\1/')
if [[ "$ffmpeg_major" =~ ^[0-9]+$ ]] && (( ffmpeg_major < 4 || ffmpeg_major > 7 )); then
    warn "ffmpeg major version $ffmpeg_major; torchcodec supports FFmpeg 4-7."
else
    ok "ffmpeg $ffmpeg_major with AV1 decoder"
fi

# 4. uv + Python environment.
step "Python environment (uv)"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
ok "$(uv --version)"
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv run python -c "import torchcodec.decoders" 2>/dev/null || die "torchcodec cannot load the system FFmpeg libraries."
ok "environment synced, torchcodec loads"

# 5. JAX on the GPUs: catches driver / CUDA mismatches before a long training run starts.
step "JAX GPU check"
uv run python - <<'EOF' || die "JAX cannot run on the GPUs (see the error above)."
import jax
import jax.numpy as jnp

devices = jax.devices()
assert devices[0].platform == "gpu", f"JAX is not using the GPU: {devices}"
x = jnp.ones((1024, 1024), dtype=jnp.bfloat16)
float((x @ x).sum().block_until_ready())
print(f"    {len(devices)} x {devices[0].device_kind}, jax {jax.__version__}")
EOF
ok "JAX runs on the GPUs"

# 6. Dataset.
step "Dataset at $DATA_DIR"
if [[ -n "$HF_REPO" ]]; then
    # Private repos need a token: `uv run hf auth login` or HF_TOKEN.
    uv run hf auth whoami >/dev/null 2>&1 || warn "not logged in to Hugging Face; a private dataset will fail to download."
    mkdir -p "$DATA_DIR"
    # Downloads into the LeRobot layout directly; files already present are skipped, so re-runs resume.
    uv run hf download "$HF_REPO" --repo-type dataset ${HF_REVISION:+--revision "$HF_REVISION"} \
        --local-dir "$DATA_DIR" --max-workers 16 --quiet >/dev/null
    ok "downloaded $HF_REPO${HF_REVISION:+@$HF_REVISION}"
fi
data_ok=1
for d in meta data videos; do
    if [[ ! -d "$DATA_DIR/$d" ]]; then
        warn "missing $DATA_DIR/$d"
        data_ok=0
    fi
done
if [[ $data_ok -eq 1 ]]; then
    ok "$(find "$DATA_DIR/data" -name '*.parquet' | wc -l) parquet files, $(find "$DATA_DIR/videos" -name '*.mp4' | wc -l) videos"
else
    warn "dataset incomplete. Download it with: $0 --hf-repo <user/dataset>"
fi

# 7. Norm stats: joint stats are computed from the data (there are no precomputed ones in meta/).
step "Norm stats"
if [[ -f "$NORM_STATS" ]]; then
    ok "$NORM_STATS"
elif [[ $data_ok -eq 0 ]]; then
    die "$NORM_STATS not found and the dataset is incomplete, cannot compute it."
else
    max_frames_arg=()
    if (( NORM_MAX_FRAMES > 0 )); then max_frames_arg=(--max-frames "$NORM_MAX_FRAMES"); fi
    uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME" "${max_frames_arg[@]}"
    # compute_norm_stats.py writes under the repo id, but the config loads stats from its asset id.
    mkdir -p "$(dirname "$NORM_STATS")"
    mv "assets/$CONFIG_NAME/$REPO_ID/norm_stats.json" "$NORM_STATS"
    rm -rf "assets/$CONFIG_NAME/${REPO_ID%%/*}"
    ok "computed $NORM_STATS"
fi

# 8. Self-check: load a sample, decode its videos and run the full transform chain.
step "Self-check"
if [[ $SKIP_CHECK -eq 1 ]]; then
    ok "skipped (--skip-check)"
elif [[ $data_ok -eq 0 ]]; then
    warn "skipped, dataset incomplete"
else
    uv run python - "$CONFIG_NAME" <<'EOF'
import sys

import openpi.policies.yam_abc_joint_policy as _joint
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

cfg = _config.get_config(sys.argv[1])
dc = cfg.data.create(cfg.assets_dirs, cfg.model)
assert dc.norm_stats is not None, "norm stats not loaded"

raw = _data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
x = raw[len(raw) // 2]
assert tuple(x["action"].shape) == (cfg.model.action_horizon, _joint.ABC_JOINT_DIM), x["action"].shape
assert tuple(x["observation.state"].shape) == (_joint.ABC_JOINT_DIM,), x["observation.state"].shape
for cam in ("top", "left_wrist", "right_wrist"):
    assert x[f"observation.images.{cam}"].ndim == 3, cam
print(f"    frames={len(raw)} prompt={x['prompt']!r}")

s = _data_loader.transform_dataset(raw, dc)[len(raw) // 2]
assert tuple(s["actions"].shape) == (cfg.model.action_horizon, cfg.model.action_dim), s["actions"].shape
print(f"    model input ok: actions {s['actions'].shape}, prompt tokens {int(s['tokenized_prompt_mask'].sum())}")
EOF
    ok "sample loads, videos decode, transforms run"
fi

step "Done"
cat <<EOF
    Keep HF_LEROBOT_HOME set in the training shell:
        export HF_LEROBOT_HOME=$HF_LEROBOT_HOME
    Log in to wandb (or pass --no-wandb-enabled):
        uv run wandb login
    Train (run inside tmux):
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $CONFIG_NAME --exp-name <name>
    Back up checkpoints to a Hugging Face model repo in a second window (needs a write token):
        scripts/sync_checkpoints.sh push checkpoints/$CONFIG_NAME/<name> hf://<user>/<repo>/$CONFIG_NAME/<name> \\
            --pid \$(pgrep -f "train.py $CONFIG_NAME" | head -1) --prune
EOF
