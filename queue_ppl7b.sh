#!/usr/bin/env bash
# Item 1: 7B BF16 perplexity baseline. CUDA-graph capture failed with CPU offload on WSL2
# (unpinned CPU weights cannot be copied during capture), so this runs eager.
# The 3B eager run is a control: its perplexity vs the 3B compiled value shows how much
# eager vs compiled numerics alone move perplexity.
cd ~/inference-lab
S=./serve_generic.sh
PPL=1 RATES=1 DURATION=10 MODEL=Qwen/Qwen2.5-7B-Instruct TAG=bf16-7b-offload-eager EXTRA="--enforce-eager --cpu-offload-gb 6" ./run_config.sh bf16-7b-offload-eager $S
PPL=1 RATES=1 DURATION=10 MODEL=Qwen/Qwen2.5-3B-Instruct TAG=bf16-3b-eager-ppl EXTRA="--enforce-eager" ./run_config.sh bf16-3b-eager-ppl $S
echo "QUEUE FINISHED $(date)"
