#!/usr/bin/env bash
# Periodically backs up openpi training checkpoints to S3, and restores them on a new machine.
#
# Back up (run next to training, e.g. in a second tmux window):
#   scripts/sync_checkpoints.sh push checkpoints/pi05_yam_abc/<exp> s3://<bucket>/openpi/pi05_yam_abc/<exp>
# Restore the latest complete checkpoint on a new machine, then train with --resume:
#   scripts/sync_checkpoints.sh pull checkpoints/pi05_yam_abc/<exp> s3://<bucket>/openpi/pi05_yam_abc/<exp>
#
# push options:
#   --interval SEC   Seconds between passes (default 600).
#   --pid PID        Do a final pass and exit once this process (the training run) exits.
#   --once           Do a single pass and exit.
#   --prune          Delete S3 steps that training already deleted locally (orbax keeps the latest step and every
#                    keep_period-th step). Never deletes the newest uploaded step.
#
# Only finalized checkpoints are uploaded: orbax writes to "<step>.orbax-checkpoint-tmp-*" and renames the directory
# when the save is complete. After a step is fully uploaded, a marker file is written next to it; `pull` only restores
# steps with the marker, so a step cut off mid-upload (e.g. the machine died) is never used.

set -euo pipefail

MARKER="_S3_UPLOAD_COMPLETE"

usage() {
    sed -n '2,18p' "$0"
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

command -v aws >/dev/null || { echo "aws CLI not found" >&2; exit 1; }

log() { echo "[$(date '+%F %T')] $*"; }

# Step directories (plain integer names) in ascending order. Excludes orbax tmp directories.
local_steps() {
    find "$LOCAL" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' -printf '%f\n' 2>/dev/null | sort -n
}

remote_steps() {
    { aws s3 ls "$REMOTE/" 2>/dev/null || true; } | awk '$1 == "PRE" { sub("/$", "", $2); print $2 }' \
        | { grep -E '^[0-9]+$' || true; } | sort -n
}

remote_complete() {
    aws s3 ls "$REMOTE/$1/$MARKER" >/dev/null 2>&1
}

push_pass() {
    if [[ ! -d "$LOCAL" ]]; then
        log "waiting for $LOCAL"
        return
    fi

    # Top-level files, e.g. wandb_id.txt (needed to resume the same wandb run).
    aws s3 sync "$LOCAL" "$REMOTE" --exclude "*/*" --only-show-errors || log "top-level file sync failed, will retry"

    local uploaded=""
    local step
    for step in $(local_steps); do
        if remote_complete "$step"; then
            uploaded=$step
            continue
        fi
        log "uploading step $step"
        # --delete drops leftovers of an earlier interrupted upload of the same step.
        if ! aws s3 sync "$LOCAL/$step" "$REMOTE/$step" --delete --only-show-errors; then
            log "upload of step $step failed, will retry"
            continue
        fi
        # Orbax may have deleted the step locally (retention) while it was uploading.
        if [[ ! -d "$LOCAL/$step" ]]; then
            log "step $step was removed locally during upload, skipping"
            continue
        fi
        date -u '+%FT%TZ' | aws s3 cp - "$REMOTE/$step/$MARKER" --only-show-errors
        uploaded=$step
        log "step $step uploaded"
    done

    if [[ $PRUNE -eq 1 && -n "$uploaded" ]]; then
        local keep
        keep=" $(local_steps | tr '\n' ' ') "
        for step in $(remote_steps); do
            if ((step < uploaded)) && [[ "$keep" != *" $step "* ]]; then
                log "pruning step $step from S3 (deleted locally)"
                aws s3 rm "$REMOTE/$step" --recursive --only-show-errors || log "prune of step $step failed"
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
            echo "Local step $step is newer than the latest complete S3 step $latest." >&2
            echo "Training resumes from the newest local step; remove $LOCAL/$step if it is incomplete." >&2
        fi
    done

    mkdir -p "$LOCAL"
    aws s3 sync "$REMOTE" "$LOCAL" --exclude "*/*" --only-show-errors
    log "restoring step $latest"
    aws s3 sync "$REMOTE/$latest" "$LOCAL/$latest" --exclude "$MARKER" --only-show-errors
    log "restored $LOCAL/$latest; resume training with --resume"
}

case "$MODE" in
    push) push ;;
    pull) pull ;;
    *) usage ;;
esac
