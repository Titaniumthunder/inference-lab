#!/usr/bin/env bash
# Clean rerun of the headline vLLM experiments (LAB_LOG section 24). Every config goes through run_config.sh, so the
# idle-GPU preflight (gpu_preflight.sh, section 23) runs before each server. Results: results/clean/ and results/clean/spec/.
# No overload / rate-16 collapse configs. ~2 h 05 to 2 h 35: close Wallpaper Engine, Remote Desktop, recorders; PC must not sleep.
cd ~/inference-lab
S=./serve_generic.sh
M3=Qwen/Qwen2.5-3B-Instruct
M7=Qwen/Qwen2.5-7B-Instruct
MA=Qwen/Qwen2.5-7B-Instruct-AWQ
D=Qwen/Qwen2.5-0.5B-Instruct
C=results/clean

guard() {  # $1 = run_config exit code, $2 = OUTDIR. Stop on a failed run (incl. preflight refusal) or two INVALID in a row
  [ "$1" = 0 ] || { echo "QUEUE STOPPED: run exited $1 (failed run or preflight refusal) $(date)"; exit 1; }
  [ "$(tail -n 2 "$2/runs.csv" | grep -c ',INVALID$')" = 2 ] && { echo "QUEUE STOPPED: two INVALID in a row $(date)"; exit 1; }
  return 0
}

# A. eager vs compiled vs 3B FP8: standard sweep, 3 triplets, arm order rotates
eager()    { OUTDIR=$C ./run_config.sh clean-eager-rep$1 ./serve_eager.sh; guard $? $C; }
compiled() { OUTDIR=$C ./run_config.sh clean-compiled-rep$1 ./serve_compiled.sh; guard $? $C; }
fp8_3b()   { OUTDIR=$C MODEL=$M3 TAG=clean-fp8-3b EXTRA="--quantization fp8" ./run_config.sh clean-fp8-3b-rep$1 $S; guard $? $C; }
eager 1; compiled 1; fp8_3b 1
compiled 2; fp8_3b 2; eager 2
fp8_3b 3; eager 3; compiled 3

# B. 7B FP8 vs 7B AWQ (AWQ stops at rate 12: it collapsed at 16 with default settings)
fp8_7b() { OUTDIR=$C RATES=1,4,8,12,16 MODEL=$M7 TAG=clean-fp8-7b EXTRA="--quantization fp8" ./run_config.sh clean-fp8-7b-rep$1 $S; guard $? $C; }
awq()    { OUTDIR=$C RATES=1,4,8,12 MODEL=$MA TAG=clean-awq-7b ./run_config.sh clean-awq-7b-rep$1 $S; guard $? $C; }
fp8_7b 1; awq 1
awq 2; fp8_7b 2
fp8_7b 3; awq 3

# C. prefix caching on vs off (shared ~1,540-token prefix)
pon()  { OUTDIR=$C MODE=prefix LEVELS=2,4,8 MODEL=$M3 TAG=clean-prefix-on ./run_config.sh clean-prefix-on-rep$1 $S; guard $? $C; }
poff() { OUTDIR=$C MODE=prefix LEVELS=2,4,8 MODEL=$M3 TAG=clean-prefix-off EXTRA="--no-enable-prefix-caching" ./run_config.sh clean-prefix-off-rep$1 $S; guard $? $C; }
pon 1; poff 1
poff 2; pon 2
pon 3; poff 3

# D. speculative decoding, realistic text, rates 1/4/8 only. EXTRA is expanded unquoted: the JSON must not contain spaces.
CS=$C/spec
base() { OUTDIR=$CS MODE=text LEVELS=1,4,8 MODEL=$M3 TAG=clean-spec-base ./run_config.sh clean-spec-base-rep$1 $S; guard $? $CS; }
spec() { OUTDIR=$CS MODE=text LEVELS=1,4,8 MODEL=$M3 TAG=clean-spec-k$1 \
           EXTRA="--speculative-config {\"method\":\"draft_model\",\"model\":\"$D\",\"num_speculative_tokens\":$1}" \
           ./run_config.sh clean-spec-k$1-rep$2 $S; guard $? $CS; }
base 1; spec 3 1; spec 5 1
spec 5 2; base 2; spec 3 2

echo "QUEUE FINISHED $(date)"
