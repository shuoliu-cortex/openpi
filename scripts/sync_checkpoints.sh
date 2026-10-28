#!/usr/bin/env bash
# Periodically backs up openpi training checkpoints, and restores them on a new machine.
# The backup destination is a local directory (e.g. a Lambda persistent filesystem), an S3 prefix, or a path in a
# Hugging Face model repo (hf://<user>/<repo>[/<path>], created private if missing; needs a write token).
#
# Back up (run next to training, e.g. in a second tmux window):
#   scripts/sync_checkpoints.sh push checkpoints/pi05_yam_abc/<exp> /lambda/nfs/<fs>/checkpoints/pi05_yam_abc/<exp>
#   scripts/sync_checkpoints.sh push checkpoints/pi05_yam_abc/<exp> s3://<bucket>/openpi/pi05_yam_abc/<exp>
#   scripts/sync_checkpoints.sh push checkpoints/pi05_yam_abc/<exp> hf://<user>/<repo>/pi05_yam_abc/<exp>
# Restore the latest complete checkpoint on a new machine, then train with --resume:
#   scripts/sync_checkpoints.sh pull checkpoints/pi05_yam_abc/<exp> <same destination as push>
#
# push options:
#   --interval SEC   Seconds between passes (default 600).
#   --pid PID        Do a final pass and exit once this process (the training run) exits.
#   --once           Do a single pass and exit.
#   --prune          Delete backed-up steps that training already deleted locally (orbax keeps the latest step and
#                    every keep_period-th step). Never deletes the newest backed-up step.
#
# Only finalized checkpoints are backed up: orbax writes to "<step>.orbax-checkpoint-tmp-*" and renames the directory
# when the save is complete. After a step is fully copied, a marker file is written next to it; `pull` only restores
# steps with the marker, so a step cut off mid-copy (e.g. the machine died) is never used.

set -euo pipefail

MARKER="_BACKUP_COMPLETE"

usage() {
    sed -n '2,22p' "$0"
    exit 1
}

[[ $# -ge 3 ]] || usage
MODE=$1
LOCAL=${2%/}
REMOTE=${3%/}
shift 3

INTERVAL=600
PID=""
ONCE=0
PRUNE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval) INTERVAL=$2; shift 2 ;;
        --pid) PID=$2; shift 2 ;;
        --once) ONCE=1; shift ;;
        --prune) PRUNE=1; shift ;;
        *) usage ;;
    esac
done

log() { echo "[$(date '+%F %T')] $*"; }

# Destination backends. Each takes paths relative to $REMOTE.
if [[ "$REMOTE" == hf://* ]]; then
    REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    command -v uv >/dev/null || { echo "uv not found" >&2; exit 1; }
    # hf://<user>/<repo>[/<path>]: a Hugging Face model repo, plus an optional path inside it.
    hf_path=${REMOTE#hf://}
    HF_REPO=$(cut -d/ -f1-2 <<<"$hf_path")
    HF_PREFIX=$(cut -s -d/ -f3- <<<"$hf_path")
    [[ "$HF_REPO" == */* ]] || { echo "Expected hf://<user>/<repo>[/<path>], got $REMOTE" >&2; exit 1; }
    # Runs one operation against the repo with huggingface_hub (log in with `uv run hf auth login`, or set HF_TOKEN).
    hf_op() {
        uv run --project "$REPO_ROOT" python - "$HF_REPO" "$HF_PREFIX" "$LOCAL" "$MARKER" "$@" <<'EOF'
import os
import pathlib
import shutil
import sys

import huggingface_hub as hf
from huggingface_hub.utils import EntryNotFoundError
from huggingface_hub.utils import RepositoryNotFoundError

repo, prefix, local, marker, op, *args = sys.argv[1:]
local = pathlib.Path(local)
api = hf.HfApi()


def remote(rel=""):
    return "/".join(p for p in (prefix, rel) if p)


def tree(rel=""):
    try:
        return list(api.list_repo_tree(repo, remote(rel) or None))
    except (EntryNotFoundError, RepositoryNotFoundError):
        return []


def download(allow_patterns, rel=""):
    # Download into a staging dir, then move the files under remote(rel) to local/rel.
    stage = local / ".hf_download"
    hf.snapshot_download(repo, allow_patterns=allow_patterns, local_dir=stage)
    src, dest = stage / remote(rel), local / rel
    for f in src.rglob("*"):
        rel = f.relative_to(src)
        if f.is_file() and rel.parts[0] != ".cache" and rel.name != marker:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            os.replace(f, dest / rel)
    shutil.rmtree(stage)


if op == "list_dirs":
    for e in tree():
        if isinstance(e, hf.hf_api.RepoFolder):
            print(e.path.rsplit("/", 1)[-1])
elif op == "has_file":
    sys.exit(0 if api.file_exists(repo, remote(args[0])) else 1)
elif op == "top_files_to":
    files = [f for f in local.iterdir() if f.is_file()]
    if files:
        api.create_repo(repo, private=True, exist_ok=True)
        ops = [hf.CommitOperationAdd(remote(f.name), str(f)) for f in files]
        api.create_commit(repo, ops, commit_message="Back up top-level files")
elif op == "top_files_from":
    names = [e.path.rsplit("/", 1)[-1] for e in tree() if isinstance(e, hf.hf_api.RepoFile)]
    names = [n for n in names if n != ".gitattributes"]
    if names:
        download([remote(n) for n in names])
elif op == "step_to":
    api.create_repo(repo, private=True, exist_ok=True)
    # delete_patterns removes leftovers of an earlier interrupted upload of this step.
    api.upload_folder(
        repo_id=repo,
        folder_path=local / args[0],
        path_in_repo=remote(args[0]),
        delete_patterns="*",
        ignore_patterns=marker,
        commit_message=f"Back up step {args[0]}",
    )
elif op == "step_from":
    download(remote(args[0]) + "/*", args[0])
elif op == "write_marker":
    api.upload_file(path_or_fileobj=b"", path_in_repo=remote(f"{args[0]}/{marker}"), repo_id=repo,
                    commit_message=f"Mark step {args[0]} complete")
elif op == "remove":
    api.delete_folder(remote(args[0]), repo, commit_message=f"Prune step {args[0]}")
else:
    sys.exit(f"unknown op {op}")
EOF
    }
    dest_list_dirs() { hf_op list_dirs; }
    dest_has_file() { hf_op has_file "$1"; }
    copy_top_files_to_dest() { hf_op top_files_to; }
    copy_top_files_from_dest() { hf_op top_files_from; }
    copy_step_to_dest() { hf_op step_to "$1"; }
    copy_step_from_dest() { hf_op step_from "$1"; }
    write_marker() { hf_op write_marker "$1"; }
    remove_step() { hf_op remove "$1"; }
elif [[ "$REMOTE" == s3://* ]]; then
    command -v aws >/dev/null || { echo "aws CLI not found" >&2; exit 1; }
    # Subdirectory names directly under $REMOTE.
    dest_list_dirs() {
        { aws s3 ls "$REMOTE/" 2>/dev/null || true; } | awk '$1 == "PRE" { sub("/$", "", $2); print $2 }'
    }
    dest_has_file() { aws s3 ls "$REMOTE/$1" >/dev/null 2>&1; }
    # Copy top-level files only (no subdirectories) between a local dir and $REMOTE.
    copy_top_files_to_dest() { aws s3 sync "$LOCAL" "$REMOTE" --exclude "*/*" --only-show-errors; }
    copy_top_files_from_dest() { aws s3 sync "$REMOTE" "$LOCAL" --exclude "*/*" --only-show-errors; }
    # Mirror a step directory; removes leftovers of an earlier interrupted copy.
    copy_step_to_dest() { aws s3 sync "$LOCAL/$1" "$REMOTE/$1" --delete --only-show-errors; }
    copy_step_from_dest() { aws s3 sync "$REMOTE/$1" "$LOCAL/$1" --exclude "$MARKER" --only-show-errors; }
    write_marker() { date -u '+%FT%TZ' | aws s3 cp - "$REMOTE/$1/$MARKER" --only-show-errors; }
    remove_step() { aws s3 rm "$REMOTE/$1" --recursive --only-show-errors; }
else
    command -v rsync >/dev/null || { echo "rsync not found" >&2; exit 1; }
    dest_list_dirs() { find "$REMOTE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null || true; }
    dest_has_file() { [[ -f "$REMOTE/$1" ]]; }
    copy_top_files_to_dest() { mkdir -p "$REMOTE" && rsync -a --exclude '*/' "$LOCAL/" "$REMOTE/"; }
    copy_top_files_from_dest() { mkdir -p "$LOCAL" && rsync -a --exclude '*/' "$REMOTE/" "$LOCAL/"; }
    copy_step_to_dest() { mkdir -p "$REMOTE/$1" && rsync -a --delete --exclude "$MARKER" "$LOCAL/$1/" "$REMOTE/$1/"; }
    copy_step_from_dest() { mkdir -p "$LOCAL/$1" && rsync -a --exclude "$MARKER" "$REMOTE/$1/" "$LOCAL/$1/"; }
    write_marker() { date -u '+%FT%TZ' > "$REMOTE/$1/$MARKER" && sync "$REMOTE/$1/$MARKER"; }
    remove_step() { rm -rf "${REMOTE:?}/$1"; }
fi

# Step directories (plain integer names) in ascending order. Excludes orbax tmp directories.
local_steps() {
    find "$LOCAL" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' -printf '%f\n' 2>/dev/null | sort -n
}

remote_steps() {
    dest_list_dirs | { grep -E '^[0-9]+$' || true; } | sort -n
}

remote_complete() {
    dest_has_file "$1/$MARKER"
}

push_pass() {
    if [[ ! -d "$LOCAL" ]]; then
        log "waiting for $LOCAL"
        return
    fi

    # Top-level files, e.g. wandb_id.txt (needed to resume the same wandb run).
    copy_top_files_to_dest || log "top-level file copy failed, will retry"

    local uploaded=""
    local step
    for step in $(local_steps); do
        if remote_complete "$step"; then
            uploaded=$step
            continue
        fi
        log "backing up step $step"
        if ! copy_step_to_dest "$step"; then
            log "backup of step $step failed, will retry"
            continue
        fi
        # Orbax may have deleted the step locally (retention) while it was being copied.
        if [[ ! -d "$LOCAL/$step" ]]; then
            log "step $step was removed locally during the copy, skipping"
            continue
        fi
        write_marker "$step"
        uploaded=$step
        log "step $step backed up"
    done

    if [[ $PRUNE -eq 1 && -n "$uploaded" ]]; then
        local keep
        keep=" $(local_steps | tr '\n' ' ') "
        for step in $(remote_steps); do
            if ((step < uploaded)) && [[ "$keep" != *" $step "* ]]; then
                log "pruning step $step from the backup (deleted locally)"
                remove_step "$step" || log "prune of step $step failed"
            fi
        done
    fi
}

push() {
    log "backing up $LOCAL -> $REMOTE every ${INTERVAL}s"
    trap 'log "stopping, final pass"; push_pass; exit 0' INT TERM
    while true; do
        push_pass
        [[ $ONCE -eq 1 ]] && break
        if [[ -n "$PID" ]] && ! kill -0 "$PID" 2>/dev/null; then
            log "process $PID exited, final pass"
            push_pass
            break
        fi
        sleep "$INTERVAL" &
        wait $!
    done
}

pull() {
    local latest="" step
    for step in $(remote_steps); do
        if remote_complete "$step"; then
            latest=$step
        fi
    done
    [[ -n "$latest" ]] || { echo "No complete checkpoint found under $REMOTE" >&2; exit 1; }

    for step in $(local_steps); do
        if ((step > latest)); then
            echo "Local step $step is newer than the latest complete backed-up step $latest." >&2
            echo "Training resumes from the newest local step; remove $LOCAL/$step if it is incomplete." >&2
        fi
    done

    mkdir -p "$LOCAL"
    copy_top_files_from_dest
    log "restoring step $latest"
    copy_step_from_dest "$latest"
    log "restored $LOCAL/$latest; resume training with --resume"
}

case "$MODE" in
    push) push ;;
    pull) pull ;;
    *) usage ;;
esac
