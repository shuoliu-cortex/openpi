#!/usr/bin/env bash
# Periodically backs up openpi training checkpoints, and restores them on a new machine.
# The backup destination is either a local directory (e.g. a Lambda persistent filesystem) or an S3 prefix.
#
# Back up (run next to training, e.g. in a second tmux window):
#   scripts/sync_checkpoints.sh push checkpoints/pi05_yam_abc/<exp> /lambda/nfs/<fs>/checkpoints/pi05_yam_abc/<exp>
#   scripts/sync_checkpoints.sh push checkpoints/pi05_yam_abc/<exp> s3://<bucket>/openpi/pi05_yam_abc/<exp>
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
    sed -n '2,20p' "$0"
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
if [[ "$REMOTE" == s3://* ]]; then
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
