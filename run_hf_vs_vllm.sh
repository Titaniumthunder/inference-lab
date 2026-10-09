#!/usr/bin/env bash
# Item 5: HF generate vs vLLM offline engine, Qwen2.5-3B BF16. Fresh process per backend, two
# interleaved repeats. Same WSL PATH / CUDA_HOME fix as serve_generic.sh (inductor needs it).
cd ~/inference-lab && source ~/.venv/bin/activate
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
gpu_free
for i in 1 2; do
  for b in hf vllm; do
    python hf_vs_vllm.py --backend $b --label $b-rep$i || { echo "QUEUE STOPPED: $b-rep$i failed $(date)"; exit 1; }
    gpu_free
  done
done
echo "QUEUE FINISHED $(date)"
