#!/usr/bin/env bash
# Block L, allocator A/B (LAB_LOG section 33): mini_sched Poisson at rate 4 only, 2x2 interleaved:
#   allocator {on = default expandable segments, off = PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False}
#   x policy order {cs = continuous,static | sc = static,continuous}, 2 reps per cell = 8 runs.
# Rotated order: round 1 on-cs, off-cs, on-sc, off-sc; round 2 off-sc, on-sc, off-cs, on-cs (no cell always first or last,
# allocator setting alternates). Same runner, preflight and sidecar as block J. ~13 to 27 min (derived). No overload tests.
# DRY=1 ./queue_clean3.sh prints every command and runs nothing. PC must not sleep; nothing else on the GPU.
cd ~/inference-lab
C=results/clean
RETRIES=3; RETRY_WAIT_S=300
DRY=${DRY:-0}; N=0

guard() {  # $1 = exit code (no canary for mini_sched)
  [ "$1" = 0 ] || { echo "QUEUE STOPPED: run exited $1 $(date)"; exit 1; }
  return 0
}

# Exit 75 = preflight refused (GPU not idle, nothing was run): wait RETRY_WAIT_S and retry the same run, up to RETRIES
# retries (4 attempts in all). Any other nonzero exit, or 75 after the last retry, stops the queue via guard.
run() {
  local try=0 rc
  if [ "$DRY" = 1 ]; then N=$((N + 1)); printf '%2d  %s\n' "$N" "$*"; return 0; fi
  while :; do
    "$@"; rc=$?
    [ "$rc" = 75 ] && [ "$try" -lt "$RETRIES" ] || break
    try=$((try + 1))
    echo "PREFLIGHT REFUSED (exit 75): retry $try/$RETRIES of the same run in ${RETRY_WAIT_S} s $(date)"
    sleep "$RETRY_WAIT_S"
  done
  guard "$rc"
}

# cell <on|off> <cs|sc> <rep>
cell() {
  local alloc=expandable_segments:True pol=continuous,static
  [ "$1" = off ] && alloc=expandable_segments:False
  [ "$2" = sc ] && pol=static,continuous
  run env PYTORCH_CUDA_ALLOC_CONF=$alloc OUT=$C/mini_sched.csv GPUDIR=$C \
    ./run_mini_sched.sh clean3-alloc-$1-$2-rep$3 --part poisson --rates 4 --policies $pol
}

cell on cs 1; cell off cs 1; cell on sc 1; cell off sc 1
cell off sc 2; cell on sc 2; cell off cs 2; cell on cs 2

[ "$DRY" = 1 ] && { echo "DRY RUN: $N runs, nothing executed"; exit 0; }
echo "QUEUE FINISHED $(date)"
