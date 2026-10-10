#!/usr/bin/env bash
# mini_sched benchmark runner with the idle-GPU preflight and GPU sidecar (LAB_LOG section 29).
# Usage: [OUT=results/mini_sched.csv] [GPUDIR=results] ./run_mini_sched.sh LABEL <bench_sched args, e.g. --part poisson>
# Exit 75 = preflight refused (nothing ran). The label is passed as --label; refuses a label already in $OUT.
set -u
LABEL=${1:?need a label}; shift
OUT=${OUT:-results/mini_sched.csv}; GPUDIR=${GPUDIR:-results}
cd ~/inference-lab && source ~/.venv/bin/activate && source ./lab_gpu.sh
pgrep -f "vllm serve" >/dev/null && { echo "a vllm server is running, stop it first"; exit 1; }
[ -e "$OUT" ] && grep -q "^$LABEL," "$OUT" && { echo "label $LABEL already in $OUT, use a new label"; exit 1; }
lab_preflight "$LABEL"
trap lab_sidecar_stop EXIT
lab_sidecar_start "$GPUDIR/gpu_$LABEL.csv"
python -m mini_sched.bench_sched --label "$LABEL" --out "$OUT" "$@"
