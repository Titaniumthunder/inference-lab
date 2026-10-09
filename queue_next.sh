#!/usr/bin/env bash
cd ~/inference-lab
S=./serve_generic.sh
M3=Qwen/Qwen2.5-3B-Instruct
M7=Qwen/Qwen2.5-7B-Instruct
MA=Qwen/Qwen2.5-7B-Instruct-AWQ
for i in 2 3; do
  MODEL=$M3 TAG=fp8-3b EXTRA="--quantization fp8" ./run_config.sh fp8-3b-rep$i $S
  MODEL=$M7 TAG=fp8-7b EXTRA="--quantization fp8" ./run_config.sh fp8-7b-rep$i $S
done
RATES=1,4,8,12 MODEL=$MA TAG=awq-7b ./run_config.sh awq-7b-b $S
RATES=16 MODEL=$MA TAG=awq-7b ./run_config.sh awq-7b-c $S
PPL=1 RATES=1 DURATION=10 MODEL=$M7 TAG=bf16-7b-offload EXTRA="--cpu-offload-gb 6" ./run_config.sh bf16-7b-offload $S
echo "QUEUE FINISHED $(date)"
