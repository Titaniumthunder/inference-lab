#!/usr/bin/env bash
# Item 4: speculative decoding on Qwen2.5-3B BF16 (draft Qwen2.5-0.5B-Instruct, k=3 and k=5) vs no
# speculation, realistic-text mode, 2 interleaved repeats. Results in results/spec/ (text-mode
# rows have extra columns, so they cannot go into results/modes.csv).
# EXTRA is expanded unquoted by serve_generic.sh, so the JSON must not contain spaces.
cd ~/inference-lab
S=./serve_generic.sh
M3=Qwen/Qwen2.5-3B-Instruct
D=Qwen/Qwen2.5-0.5B-Instruct
export OUTDIR=results/spec MODE=text LEVELS=1,4,8,16
guard() {  # stop if a run failed to start or the last two verdicts are INVALID
  [ "$1" = 0 ] || { echo "QUEUE STOPPED: run exited $1 $(date)"; exit 1; }
  [ "$(tail -n 2 $OUTDIR/runs.csv | grep -c ',INVALID$')" = 2 ] && { echo "QUEUE STOPPED: two INVALID in a row $(date)"; exit 1; }
  return 0
}
for i in 1 2; do
  MODEL=$M3 TAG=spec-base ./run_config.sh spec-base-rep$i $S; guard $?
  for k in 3 5; do
    MODEL=$M3 TAG=spec-k$k EXTRA="--speculative-config {\"method\":\"draft_model\",\"model\":\"$D\",\"num_speculative_tokens\":$k}" \
      ./run_config.sh spec-k$k-rep$i $S; guard $?
  done
done
echo "QUEUE FINISHED $(date)"
