#!/usr/bin/env bash
# Item 6: torch-profiler traces of a vLLM server (3B BF16 compiled). Fresh server, auto-stop, like run_config.sh.
# Profiled windows are for breakdowns (GPU kernel time vs step wall time, CUDA API calls), not headline latency:
# the profiler adds overhead.
#   PHASE=healthy  : profile rate-1 decode (10 requests) and a rate-16 burst (10 s)
#   PHASE=collapse : profile a rate-1 canary, overload with 256 closed-loop clients (cap 256, which collapsed in
#                    LAB_LOG section 10), profile the same canary again
set -u
NAME=${1:?need a name}; PHASE=${PHASE:?PHASE=healthy|collapse}
MODEL=Qwen/Qwen2.5-3B-Instruct
DIR=$HOME/inference-lab/results/profiles/$NAME
cd ~/inference-lab && source ~/.venv/bin/activate && mkdir -p "$DIR"
curl -sf localhost:8000/health >/dev/null && { echo "A server is already running. Stop it first."; exit 1; }
[ -e "$DIR/bench.csv" ] && { echo "$DIR already has results, use a new name"; exit 1; }
# Idle-GPU preflight (exit 75 on refusal) and GPU sidecar to $DIR/gpu_$NAME.csv (LAB_LOG section 29)
source ./lab_gpu.sh
lab_preflight "$NAME"
lab_sidecar_start "$DIR/gpu_$NAME.csv"
stop_server() {
  pkill -INT -f "vllm serve" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "vllm serve" >/dev/null || return; sleep 1; done
  pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f EngineCor 2>/dev/null
}
trap 'stop_server; lab_sidecar_stop' EXIT
export TAG=prof-$NAME
export EXTRA="--max-num-seqs 256 --profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"$DIR\",\"max_iterations\":400,\"torch_profiler_with_stack\":false}"
./serve_generic.sh >/dev/null 2>&1 &
for i in $(seq 1 200); do curl -sf localhost:8000/health >/dev/null && break; sleep 3; [ "$i" = 200 ] && { echo "server never came up"; exit 1; }; done
echo "== server up"
prof() {  # prof <label> <bench args...>: profile one bench.py window
  local label=$1; shift
  curl -sf -X POST localhost:8000/start_profile >/dev/null || echo "start_profile failed"
  python bench.py --model $MODEL "$@" --label "$NAME-$label" --out "$DIR/bench.csv"
  curl -sf -X POST localhost:8000/stop_profile >/dev/null || echo "stop_profile failed"
  sleep 20  # trace export
  echo "== profiled $label: $(ls "$DIR" | wc -l) files in $DIR"
}
python bench.py --model $MODEL --rates 1 --num-requests 30 --label "$NAME-warm" --out "$DIR/bench.csv"
if [ "$PHASE" = healthy ]; then
  prof r1 --rates 1 --num-requests 10
  prof r16 --rates 16 --duration 10
else
  prof before --rates 1 --num-requests 10
  echo "== overload: 256 closed-loop clients, 30 s"
  python bench_modes.py --mode closed --model $MODEL --levels 256 --duration 30 --label "$NAME-overload" --out "$DIR/modes.csv"
  prof after --rates 1 --num-requests 10
  python bench.py --model $MODEL --rates 1 --num-requests 30 --label "$NAME-check" --out "$DIR/bench.csv"
fi
echo "== done, stopping server"
