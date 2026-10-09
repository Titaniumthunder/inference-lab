#!/usr/bin/env bash
# Item 3: chunked prefill off vs on (default), 3B BF16 compiled, standard sweep, 3 interleaved pairs.
cd ~/inference-lab
S=./serve_generic.sh
M3=Qwen/Qwen2.5-3B-Instruct
guard() {  # stop if a run failed to start or the last two verdicts are INVALID
  [ "$1" = 0 ] || { echo "QUEUE STOPPED: run exited $1 $(date)"; exit 1; }
  [ "$(tail -n 2 results/runs.csv | grep -c ',INVALID$')" = 2 ] && { echo "QUEUE STOPPED: two INVALID in a row $(date)"; exit 1; }
  return 0
}
for i in 1 2 3; do
  MODEL=$M3 TAG=chunked-off EXTRA="--no-enable-chunked-prefill" ./run_config.sh chunked-off-rep$i $S; guard $?
  MODEL=$M3 TAG=chunked-on ./run_config.sh chunked-on-rep$i $S; guard $?
done
echo "QUEUE FINISHED $(date)"
