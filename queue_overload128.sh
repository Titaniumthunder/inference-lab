#!/usr/bin/env bash
# Item 2: Poisson overload (24 and 32 req/s) with --max-num-seqs 128, twice, then a cap-256
# control in the same session. Rates are in the sweep (not OVERLOAD=1), so the after-canary
# measures the server state after overload.
cd ~/inference-lab
S=./serve_generic.sh
M3=Qwen/Qwen2.5-3B-Instruct
guard() {  # stop if a run failed to start or the last two verdicts are INVALID
  [ "$1" = 0 ] || { echo "QUEUE STOPPED: run exited $1 $(date)"; exit 1; }
  [ "$(tail -n 2 results/runs.csv | grep -c ',INVALID$')" = 2 ] && { echo "QUEUE STOPPED: two INVALID in a row $(date)"; exit 1; }
  return 0
}
for i in 1 2; do
  RATES=24,32 MODEL=$M3 TAG=seqs128-ovl EXTRA="--max-num-seqs 128" ./run_config.sh seqs128-ovl-rep$i $S; guard $?
done
RATES=24,32 MODEL=$M3 TAG=seqs256-ovl EXTRA="--max-num-seqs 256" ./run_config.sh seqs256-ovl $S; guard $?
echo "QUEUE FINISHED $(date)"
