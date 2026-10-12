#!/usr/bin/env bash
# Sets up a fresh Ubuntu GPU machine (e.g. Lambda Cloud, AWS EC2 Deep Learning AMI) to train `pi05_yam_abc`.
# Safe to re-run: every step checks first and skips work that is already done.
#
# Usage:
#   scripts/setup_train.sh                                   # environment only
#   scripts/setup_train.sh --data-s3 s3://bucket/abc130k-eef # also sync the dataset from S3
#
# Options:
#   --data-s3 URI     Sync the dataset from this S3 prefix (must contain meta/, data/, videos/).
#                     Installs the aws CLI if needed. Without an instance IAM role (e.g. on Lambda), log in first:
#                       aws configure sso --use-device-code   (once), then
#                       aws sso login --use-device-code --profile <profile> && export AWS_PROFILE=<profile>
#   --skip-apt        Do not install system packages.
#   --skip-check      Do not run the dataset self-check.
#
# The dataset is expected at $HF_LEROBOT_HOME/local/abc130k-eef
# (HF_LEROBOT_HOME defaults to $HF_HOME/lerobot, or ~/.cache/huggingface/lerobot without HF_HOME).
# Keep it on fast local disk; network filesystems slow down video decoding.

set -euo pipefail

CONFIG_NAME="pi05_yam_abc"
REPO_ID="local/abc130k-eef"
NORM_STATS="assets/pi05_yam_abc/abc130k-eef/norm_stats.json"
MIN_GPU_MEM_MIB=70000

DATA_S3=""
SKIP_APT=0
SKIP_CHECK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-s3) DATA_S3="$2"; shift 2 ;;
        --skip-apt) SKIP_APT=1; shift ;;
        --skip-check) SKIP_CHECK=1; shift ;;
        -h|--help) sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."
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
command -v nvidia-smi >/dev/null || die "nvidia-smi not found. Use an image with NVIDIA drivers (e.g. Lambda Stack, AWS Deep Learning AMI)."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /'
min_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | awk 'NR==1')
if (( min_mem < MIN_GPU_MEM_MIB )); then
    warn "Smallest GPU has ${min_mem} MiB. Full pi05 fine-tuning needs ~80 GB GPUs;"
    warn "use --fsdp-devices <num_gpus>, a smaller --batch-size, or the pi05_yam_abc_lora config."
else
    ok "GPU memory ${min_mem} MiB"
fi

# 2. System packages: FFmpeg (torchcodec needs the system libraries; videos are AV1), git-lfs, build tools.
step "System packages"
# Capture the decoder list first: `grep -q` exiting early would SIGPIPE ffmpeg and fail under pipefail.
has_av1() { command -v ffmpeg >/dev/null && [[ "$(ffmpeg -hide_banner -decoders 2>/dev/null)" =~ libdav1d|libaom-av1 ]]; }
if [[ $SKIP_APT -eq 1 ]]; then
    ok "skipped (--skip-apt)"
elif has_av1 && command -v git-lfs >/dev/null && command -v gcc >/dev/null && command -v unzip >/dev/null \
    && command -v rsync >/dev/null; then
    ok "ffmpeg, git-lfs, rsync and build tools already installed"
else
    $SUDO apt-get update -y
    $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg git git-lfs build-essential curl unzip rsync
fi
has_av1 || die "ffmpeg is missing or has no AV1 decoder (libdav1d / libaom-av1)."
ffmpeg_major=$(ffmpeg -version | awk 'NR==1' | sed -E 's/^ffmpeg version n?([0-9]+).*/\1/')
if [[ "$ffmpeg_major" =~ ^[0-9]+$ ]] && (( ffmpeg_major < 4 || ffmpeg_major > 7 )); then
    warn "ffmpeg major version $ffmpeg_major; torchcodec supports FFmpeg 4-7."
else
    ok "ffmpeg $ffmpeg_major with AV1 decoder"
fi

# 3. uv + Python environment.
step "Python environment (uv)"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
ok "$(uv --version)"
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv run python -c "import torchcodec.decoders" 2>/dev/null || die "torchcodec cannot load the system FFmpeg libraries."
ok "environment synced, torchcodec loads"

# 4. JAX on the GPUs: catches driver / CUDA mismatches before a long training run starts.
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

# 5. Dataset.
step "Dataset at $DATA_DIR"
if [[ -n "$DATA_S3" ]]; then
    if ! command -v aws >/dev/null; then
        tmp=$(mktemp -d)
        curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "$tmp/awscliv2.zip"
        unzip -q "$tmp/awscliv2.zip" -d "$tmp"
        "$tmp/aws/install" -i "$HOME/.local/aws-cli" -b "$HOME/.local/bin" --update
        rm -rf "$tmp"
        ok "installed $(aws --version)"
    fi
    aws sts get-caller-identity >/dev/null 2>&1 \
        || die "No valid AWS credentials. Log in with: aws sso login --use-device-code --profile <profile> && export AWS_PROFILE=<profile>"
    mkdir -p "$DATA_DIR"
    aws s3 sync "$DATA_S3" "$DATA_DIR" --only-show-errors
    ok "synced from $DATA_S3"
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
    warn "dataset incomplete. Sync it with: $0 --data-s3 <s3 prefix>"
fi

# 6. Norm stats.
step "Norm stats"
[[ -f "$NORM_STATS" ]] || die "$NORM_STATS not found. Build it with: uv run scripts/yam_abc_norm_stats.py --meta-dir $DATA_DIR/meta --horizon 30"
ok "$NORM_STATS"

# 7. Self-check: load a sample, decode its videos and run the full transform chain.
step "Self-check"
if [[ $SKIP_CHECK -eq 1 ]]; then
    ok "skipped (--skip-check)"
elif [[ $data_ok -eq 0 ]]; then
    warn "skipped, dataset incomplete"
else
    uv run python - "$CONFIG_NAME" <<'EOF'
import sys

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

cfg = _config.get_config(sys.argv[1])
dc = cfg.data.create(cfg.assets_dirs, cfg.model)
assert dc.norm_stats is not None, "norm stats not loaded"

raw = _data_loader.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
x = raw[len(raw) // 2]
assert tuple(x["action.eef"].shape) == (cfg.model.action_horizon, 20), x["action.eef"].shape
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
    Train (run inside tmux/screen):
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $CONFIG_NAME --exp-name <name>
    Back up checkpoints in a second window (e.g. to a Lambda persistent filesystem):
        scripts/sync_checkpoints.sh push checkpoints/$CONFIG_NAME/<name> <backup dir>/$CONFIG_NAME/<name> \\
            --pid \$(pgrep -f "train.py $CONFIG_NAME" | head -1) --prune
EOF
