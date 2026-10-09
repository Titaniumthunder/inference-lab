#!/usr/bin/env bash
cd ~/inference-lab
S=./serve_generic.sh
M3=Qwen/Qwen2.5-3B-Instruct
MA=Qwen/Qwen2.5-7B-Instruct-AWQ
MODE=prefix LEVELS=2,4,8 MODEL=$M3 TAG=prefix-on ./run_config.sh prefix-on $S
MODE=prefix LEVELS=2,4,8 MODEL=$M3 TAG=prefix-off EXTRA="--no-enable-prefix-caching" ./run_config.sh prefix-off $S
for n in 256 128 64; do
  MODE=closed LEVELS=16,32,64,128,256 MODEL=$M3 TAG=seqs$n EXTRA="--max-num-seqs $n" ./run_config.sh seqs$n $S
done
RATES=12,16 MODEL=$MA TAG=awq-7b-cap64 EXTRA="--max-num-seqs 64" ./run_config.sh awq-7b-cap64 $S
echo "QUEUE FINISHED $(date)"
