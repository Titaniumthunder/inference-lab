#!/usr/bin/env bash
# Eager baseline (no torch.compile, no CUDA graphs): Qwen2.5-3B-Instruct, BF16, WSL2.
source ~/.venv/bin/activate
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd:)
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --enforce-eager --generation-config vllm \
  2>&1 | tee ~/inference-lab/results/server_eager-bf16-3b_$(date +%Y%m%d_%H%M%S).log
