#!/usr/bin/env bash
cd ~/inference-lab
S=./serve_generic.sh
PPL=1 MODEL=Qwen/Qwen2.5-3B-Instruct TAG=bf16-3b ./run_config.sh bf16-3b $S
PPL=1 MODEL=Qwen/Qwen2.5-3B-Instruct TAG=fp8-3b EXTRA="--quantization fp8" ./run_config.sh fp8-3b $S
PPL=1 MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ TAG=awq-7b ./run_config.sh awq-7b $S
PPL=1 MODEL=Qwen/Qwen2.5-7B-Instruct TAG=fp8-7b EXTRA="--quantization fp8" ./run_config.sh fp8-7b $S
echo "QUEUE FINISHED $(date)"
