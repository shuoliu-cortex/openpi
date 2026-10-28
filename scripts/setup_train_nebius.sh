#!/usr/bin/env bash
# Sets up a fresh Nebius AI Cloud GPU VM (Ubuntu image with NVIDIA drivers + CUDA) for openpi training: system
# packages, the uv environment, and a JAX GPU check. Run once per machine; then prepare each dataset with
# scripts/prepare_dataset.sh. Safe to re-run: every step checks first and skips work that is already done.
#
# Usage:
#   scripts/setup_train_nebius.sh [--data-root /mnt/data]
#
# Options:
#   --data-root DIR   Directory for datasets (HF_LEROBOT_HOME), e.g. a mounted Nebius secondary disk. Keep datasets
#                     on local/block disk; network filesystems slow down video decoding.
#   --skip-apt        Do not install system packages.

set -euo pipefail

MIN_GPU_MEM_MIB=70000

DATA_ROOT=""
SKIP_APT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --skip-apt) SKIP_APT=1; shift ;;
        -h|--help) sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."
if [[ -n "$DATA_ROOT" ]]; then export HF_LEROBOT_HOME="$DATA_ROOT"; fi
# Same default as LeRobot: $HF_LEROBOT_HOME, else $HF_HOME/lerobot, else ~/.cache/huggingface/lerobot.
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME:-$HOME/.cache/huggingface}/lerobot}"
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

step "Done"
cat <<EOF
    Keep HF_LEROBOT_HOME set in every shell (e.g. add it to ~/.bashrc):
        export HF_LEROBOT_HOME=$HF_LEROBOT_HOME
    Log in to Hugging Face (datasets, checkpoint backup) and wandb (or pass --no-wandb-enabled to train.py):
        uv run hf auth login
        uv run wandb login
    Then prepare a dataset, which prints the train command:
        scripts/prepare_dataset.sh <user/dataset>
EOF
