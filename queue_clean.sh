#!/usr/bin/env bash
# Clean rerun of the headline vLLM experiments (LAB_LOG section 24). Every config goes through run_config.sh, so the
# idle-GPU preflight (gpu_preflight.sh, section 23) runs before each server. Results: results/clean/ and results/clean/spec/.
# No overload / rate-16 collapse configs. 30 runs, ~2 h 19 to 2 h 52 (plus 5 min per preflight-refusal retry, section 26):
# close Wallpaper Engine, Remote Desktop, recorders; PC must not sleep.
cd ~/inference-lab
S=./serve_generic.sh
M3=Qwen/Qwen2.5-3B-Instruct
M7=Qwen/Qwen2.5-7B-Instruct
MA=Qwen/Qwen2.5-7B-Instruct-AWQ
D=Qwen/Qwen2.5-0.5B-Instruct
C=results/clean
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

# A. eager vs compiled vs 3B FP8: standard sweep, 3 triplets, arm order rotates. All three arms use serve_generic.sh,
# so they differ only in EXTRA (eager: --enforce-eager; compiled: none; fp8: --quantization fp8).
eager()    { run $C MODEL=$M3 TAG=clean-eager EXTRA="--enforce-eager" ./run_config.sh clean-eager-rep$1 $S; }
compiled() { run $C MODEL=$M3 TAG=clean-compiled EXTRA="" ./run_config.sh clean-compiled-rep$1 $S; }
fp8_3b()   { run $C MODEL=$M3 TAG=clean-fp8-3b EXTRA="--quantization fp8" ./run_config.sh clean-fp8-3b-rep$1 $S; }
eager 1; compiled 1; fp8_3b 1
compiled 2; fp8_3b 2; eager 2
fp8_3b 3; eager 3; compiled 3

# B. 7B FP8 vs 7B AWQ (AWQ stops at rate 12: it collapsed at 16 with default settings; 7B FP8 at 16 was VALID 3/3)
fp8_7b() { run $C RATES=1,4,8,12,16 MODEL=$M7 TAG=clean-fp8-7b EXTRA="--quantization fp8" ./run_config.sh clean-fp8-7b-rep$1 $S; }
awq()    { run $C RATES=1,4,8,12 MODEL=$MA TAG=clean-awq-7b ./run_config.sh clean-awq-7b-rep$1 $S; }
fp8_7b 1; awq 1
awq 2; fp8_7b 2
fp8_7b 3; awq 3

# C. prefix caching on vs off (shared ~1,540-token prefix)
pon()  { run $C MODE=prefix LEVELS=2,4,8 MODEL=$M3 TAG=clean-prefix-on EXTRA="--enable-prefix-caching" ./run_config.sh clean-prefix-on-rep$1 $S; }
poff() { run $C MODE=prefix LEVELS=2,4,8 MODEL=$M3 TAG=clean-prefix-off EXTRA="--no-enable-prefix-caching" ./run_config.sh clean-prefix-off-rep$1 $S; }
pon 1; poff 1
poff 2; pon 2
pon 3; poff 3

# D. speculative decoding, realistic text, rates 1/4/8 only, 3 triplets in rotated order (each arm once in each position).
# EXTRA is expanded unquoted by serve_generic.sh: the JSON must not contain spaces.
CS=$C/spec
base() { run $CS MODE=text LEVELS=1,4,8 MODEL=$M3 TAG=clean-spec-base ./run_config.sh clean-spec-base-rep$1 $S; }
spec() { run $CS MODE=text LEVELS=1,4,8 MODEL=$M3 TAG=clean-spec-k$1 \
           EXTRA="--speculative-config {\"method\":\"draft_model\",\"model\":\"$D\",\"num_speculative_tokens\":$1}" \
           ./run_config.sh clean-spec-k$1-rep$2 $S; }
base 1; spec 3 1; spec 5 1
spec 5 2; base 2; spec 3 2
spec 3 3; spec 5 3; base 3

echo "QUEUE FINISHED $(date)"
