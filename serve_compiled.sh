#!/usr/bin/env bash
# Compiled baseline (torch.compile + CUDA graphs): Qwen2.5-3B-Instruct, BF16, WSL2.
source ~/.venv/bin/activate
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd:)
NVCC=$(find ~/.venv -name nvcc -type f 2>/dev/null | head -n1)
if [ -n "$NVCC" ]; then
  export CUDA_HOME=$(dirname "$(dirname "$NVCC")")
  export PATH=$CUDA_HOME/bin:$PATH
fi
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --generation-config vllm --kernel_config.enable_jit_warmup=False \
  2>&1 | tee ~/inference-lab/results/server_compiled-bf16-3b_$(date +%Y%m%d_%H%M%S).log
