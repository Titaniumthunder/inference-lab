#!/usr/bin/env bash
# Clean rerun plan 2 (LAB_LOG sections 29, 30). NOT launched without Alex's go. Every run has the idle-GPU preflight and the
# GPU sidecar (run_config.sh, or lab_gpu.sh in the other runners). Results: results/clean/ (profile: results/profiles/clean-healthy/).
# 29 runs, ~2 h 03 to 2 h 35 (plus up to ~19.7 min per preflight-refusal retry). PC must not sleep; nothing else on the GPU.
# Block order: E, F, G, G2 (steps), H, K, J, I. No overload tests. DRY=1 ./queue_clean2.sh prints every command, runs nothing.
cd ~/inference-lab
S=./serve_generic.sh
M3=Qwen/Qwen2.5-3B-Instruct
M7=Qwen/Qwen2.5-7B-Instruct
C=results/clean
RETRIES=3; RETRY_WAIT_S=300
DRY=${DRY:-0}; N=0

guard() {  # $1 = exit code, $2 = OUTDIR whose runs.csv to check, or - for runners without a canary verdict
  [ "$1" = 0 ] || { echo "QUEUE STOPPED: run exited $1 $(date)"; exit 1; }
  [ "$2" = - ] && return 0
  [ "$(tail -n 2 "$2/runs.csv" | grep -c ',INVALID$')" = 2 ] && { echo "QUEUE STOPPED: two INVALID in a row $(date)"; exit 1; }
  return 0
}

# run GUARD_DIR|- command...
# Exit 75 = preflight refused (GPU not idle, nothing was run): wait RETRY_WAIT_S and retry the same run, up to RETRIES
# retries (4 attempts in all). Any other nonzero exit, or 75 after the last retry, stops the queue via guard.
run() {
  local gd=$1; shift
  local try=0 rc
  if [ "$DRY" = 1 ]; then N=$((N + 1)); printf '%2d  %s\n' "$N" "$*"; return 0; fi
  while :; do
    "$@"; rc=$?
    [ "$rc" = 75 ] && [ "$try" -lt "$RETRIES" ] || break
    try=$((try + 1))
    echo "PREFLIGHT REFUSED (exit 75): retry $try/$RETRIES of the same run in ${RETRY_WAIT_S} s $(date)"
    sleep "$RETRY_WAIT_S"
  done
  guard "$rc" "$gd"
}

# E. async scheduling: compiled 3B BF16 standard sweep (RATES 1,4,8,16 as in the clean sweep), default (async ON, section 30)
# vs --no-async-scheduling. 3 interleaved pairs, order rotates.
adef()   { run $C env OUTDIR=$C MODEL=$M3 TAG=clean2-async-default EXTRA="" ./run_config.sh clean2-async-default-rep$1 $S; }
anoasy() { run $C env OUTDIR=$C MODEL=$M3 TAG=clean2-no-async EXTRA="--no-async-scheduling" ./run_config.sh clean2-no-async-rep$1 $S; }
adef 1; anoasy 1
anoasy 2; adef 2
adef 3; anoasy 3

# F. HF generate vs vLLM offline (section 15 workload), fresh process per backend, 3 interleaved pairs.
hfv() { run - env OUT=$C/hf_vs_vllm.csv GPUDIR=$C ./run_hf_vs_vllm.sh $1 clean-$1-rep$2; }
hfv hf 1; hfv vllm 1
hfv vllm 2; hfv hf 2
hfv hf 3; hfv vllm 3

# G. mini_sched continuous vs static (section 17), Poisson rates 1,2,4; both policies per rate in one process,
# policy order rotates across reps (same seeded arrivals).
mini() { run - env OUT=$C/mini_sched.csv GPUDIR=$C ./run_mini_sched.sh clean-mini-rep$1 --part poisson --rates 1,2,4 --policies $2; }
mini 1 continuous,static
mini 2 static,continuous
mini 3 continuous,static

# G2. mini_sched per-step speed (static batches 1/8/32/64, section-15 workload), 3 reps (derived ~2.9 min each, LAB_LOG 31)
steps() { run - env OUT=$C/mini_sched.csv GPUDIR=$C ./run_mini_sched.sh clean-mini-steps-rep$1 --part steps; }
steps 1; steps 2; steps 3

# H. profiling, healthy phase only (section 19 breakdown; the collapse phase is excluded: crash risk)
run - env PHASE=healthy ./profile_run.sh clean-healthy

# K. chunked prefill off vs on (section 13 / queue_chunked.sh flags: off = --no-enable-chunked-prefill, on = default),
# compiled 3B BF16 standard sweep, 3 interleaved pairs, order rotates.
coff() { run $C env OUTDIR=$C MODEL=$M3 TAG=clean2-chunked-off EXTRA="--no-enable-chunked-prefill" ./run_config.sh clean2-chunked-off-rep$1 $S; }
con()  { run $C env OUTDIR=$C MODEL=$M3 TAG=clean2-chunked-on EXTRA="" ./run_config.sh clean2-chunked-on-rep$1 $S; }
coff 1; con 1
con 2; coff 2
coff 3; con 3

# J. mini_sched allocator stall (bullet 4): block G repeated with the default PyTorch allocator (expandable segments off,
# via the env var mini_sched respects; no code change). Compare with block G (not interleaved with it: ~1 h apart).
noxseg() { run - env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False OUT=$C/mini_sched.csv GPUDIR=$C \
             ./run_mini_sched.sh clean-mini-noxseg-rep$1 --part poisson --rates 1,2,4 --policies $2; }
noxseg 1 continuous,static
noxseg 2 static,continuous
noxseg 3 continuous,static

# I. 7B BF16 perplexity (section 11 config: eager + 6 GB CPU offload). The speed canary may come out INVALID again
# (old run: 1.32); perplexity is a fixed computation, report the verdict as is. Last on purpose (heavy host-RAM use).
run $C env OUTDIR=$C PPL=1 RATES=1 DURATION=10 MODEL=$M7 TAG=clean2-bf16-7b-offload-eager \
  EXTRA="--enforce-eager --cpu-offload-gb 6" ./run_config.sh clean2-bf16-7b-offload-eager $S

[ "$DRY" = 1 ] && { echo "DRY RUN: $N runs, nothing executed"; exit 0; }
echo "QUEUE FINISHED $(date)"
