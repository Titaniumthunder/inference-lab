#!/usr/bin/env bash
# Item 5: HF generate vs vLLM offline engine, Qwen2.5-3B BF16. Fresh process per backend, two
# interleaved repeats. Same WSL PATH / CUDA_HOME fix as serve_generic.sh (inductor needs it).
# Usage: ./run_hf_vs_vllm.sh                 -> 2 interleaved repeats into results/hf_vs_vllm.csv (original behaviour)
#        ./run_hf_vs_vllm.sh BACKEND LABEL   -> one process (hf|vllm) into $OUT (default results/hf_vs_vllm.csv)
# Every process: idle-GPU preflight first (exit 75 on refusal) and a GPU sidecar to $GPUDIR/gpu_<label>.csv (LAB_LOG 29).
cd ~/inference-lab && source ~/.venv/bin/activate && source ./lab_gpu.sh
OUT=${OUT:-results/hf_vs_vllm.csv}; GPUDIR=${GPUDIR:-results}
trap lab_sidecar_stop EXIT
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd:)
NVCC=$(find ~/.venv -name nvcc -type f 2>/dev/null | head -n1)
if [ -n "$NVCC" ]; then
  export CUDA_HOME=$(dirname "$(dirname "$NVCC")")
  export PATH=$CUDA_HOME/bin:$PATH
fi
gpu_free() {  # no vLLM server or engine left, GPU back near the idle baseline
  pkill -9 -f EngineCor 2>/dev/null; sleep 5
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  echo "GPU memory used: $used MiB"
  [ "$used" -lt 2500 ] || { echo "QUEUE STOPPED: GPU not idle $(date)"; exit 1; }
}
pgrep -f "vllm serve" >/dev/null && { echo "a vllm server is running, stop it first"; exit 1; }
one() {  # one <backend> <label>: preflight, sidecar, one fresh process, GPU back to idle
  [ -e "results/hf_vs_vllm_tokens_$2.json" ] && { echo "results/hf_vs_vllm_tokens_$2.json exists, use a new label"; exit 1; }
  lab_preflight "$2"
  lab_sidecar_start "$GPUDIR/gpu_$2.csv"
  python hf_vs_vllm.py --backend $1 --label $2 --out "$OUT"; local rc=$?
  lab_sidecar_stop
  [ "$rc" = 0 ] || { echo "QUEUE STOPPED: $2 failed $(date)"; exit 1; }
  gpu_free
}
gpu_free
if [ $# -ge 2 ]; then one "$1" "$2"; echo "== done $2 $(date)"; exit 0; fi
for i in 1 2; do
  for b in hf vllm; do one $b $b-rep$i; done
done
echo "QUEUE FINISHED $(date)"
