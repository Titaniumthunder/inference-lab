#!/usr/bin/env bash
cd ~/inference-lab
for i in 1 2 3; do
  OVERLOAD=1 ./run_config.sh eager-rep$i ./serve_eager.sh
  OVERLOAD=1 ./run_config.sh compiled-rep$i ./serve_compiled.sh
done
echo "QUEUE FINISHED $(date)"
