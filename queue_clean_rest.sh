#!/usr/bin/env bash
# Finish the clean rerun (LAB_LOG section 27): the spec-decoding block D of queue_clean.sh stopped after k5 rep1/rep2 were
# INVALID (k=5 saturates at level 8). Remaining: base rep2/rep3 and k3 rep2/rep3 at LEVELS=1,4,8, plus three new k5 reps at
# LEVELS=1,4 only, named clean-spec-k5-l14-repN so they never mix with the old k5 rep1/rep2 rows (kept as saturation evidence).
# Same helpers as queue_clean.sh: run_config.sh preflight + GPU sidecar, retry on preflight refusal (exit 75), stop on any
# other failure or two INVALID in a row. 7 runs, ~28.5 to 35.5 min (derived, LAB_LOG 27). PC must not sleep; nothing else on the GPU.
cd ~/inference-lab
S=./serve_generic.sh
M3=Qwen/Qwen2.5-3B-Instruct
D=Qwen/Qwen2.5-0.5B-Instruct
CS=results/clean/spec
RETRIES=3; RETRY_WAIT_S=300

guard() {  # $1 = run_config exit code, $2 = OUTDIR. Stop on a failed run or two INVALID in a row
  [ "$1" = 0 ] || { echo "QUEUE STOPPED: run exited $1 $(date)"; exit 1; }
  [ "$(tail -n 2 "$2/runs.csv" | grep -c ',INVALID$')" = 2 ] && { echo "QUEUE STOPPED: two INVALID in a row $(date)"; exit 1; }
  return 0
}

# run OUTDIR VAR=value... ./run_config.sh NAME SERVE
# Exit 75 = preflight refused (GPU not idle, nothing was run): wait RETRY_WAIT_S and retry the same run, up to RETRIES
# retries (4 attempts in all). Any other nonzero exit, or 75 after the last retry, stops the queue via guard.
run() {
  local out=$1; shift
  local try=0 rc
  while :; do
    env OUTDIR="$out" "$@"; rc=$?
    [ "$rc" = 75 ] && [ "$try" -lt "$RETRIES" ] || break
    try=$((try + 1))
    echo "PREFLIGHT REFUSED (exit 75): retry $try/$RETRIES of the same run in ${RETRY_WAIT_S} s $(date)"
    sleep "$RETRY_WAIT_S"
  done
  guard "$rc" "$out"
}

# EXTRA is expanded unquoted by serve_generic.sh: the JSON must not contain spaces.
base() { run $CS MODE=text LEVELS=1,4,8 MODEL=$M3 TAG=clean-spec-base ./run_config.sh clean-spec-base-rep$1 $S; }
k3()   { run $CS MODE=text LEVELS=1,4,8 MODEL=$M3 TAG=clean-spec-k3 \
           EXTRA="--speculative-config {\"method\":\"draft_model\",\"model\":\"$D\",\"num_speculative_tokens\":3}" \
           ./run_config.sh clean-spec-k3-rep$1 $S; }
k5()   { run $CS MODE=text LEVELS=1,4 MODEL=$M3 TAG=clean-spec-k5 \
           EXTRA="--speculative-config {\"method\":\"draft_model\",\"model\":\"$D\",\"num_speculative_tokens\":5}" \
           ./run_config.sh clean-spec-k5-l14-rep$1 $S; }

# Order: each arm takes different positions (base first/last, k5 middle/first, k3 last/middle), k5 rep3 closes.
base 2; k5 1; k3 2
k5 2; k3 3; base 3
k5 3

echo "QUEUE FINISHED $(date)"
