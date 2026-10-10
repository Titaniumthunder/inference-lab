#!/usr/bin/env bash
set -u
NAME=${1:?need a config name}; SERVE=${2:?need a serve script}
MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
RATES=${RATES:-1,4,8,16}
DURATION=${DURATION:-30}
OVERLOAD=${OVERLOAD:-0}
OUTDIR=${OUTDIR:-results}
cd ~/inference-lab && source ~/.venv/bin/activate && mkdir -p "$OUTDIR"

if curl -sf localhost:8000/health >/dev/null; then
  echo "A server is already running. Stop it first."; exit 1
fi

# Idle-GPU preflight (LAB_LOG sections 21-23): thresholds, sampling and retries live in gpu_preflight.sh
# Refusal exits 75 (queue_clean.sh retries on 75); any other preflight failure exits 1.
./gpu_preflight.sh; PF=$?
[ "$PF" = 0 ] || { echo "Not running $NAME: GPU not idle (preflight exit $PF)."; [ "$PF" = 75 ] && exit 75; exit 1; }

stop_server() {
  pkill -INT -f "vllm serve" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "vllm serve" >/dev/null || return; sleep 1; done
  pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f EngineCor 2>/dev/null
}
# GPU sidecar logger (LAB_LOG section 25): 1 Hz nvidia-smi from after the preflight until exit. Best effort: a logging
# failure never fails the run. Never overwrites an existing log (adds a timestamp suffix instead).
GPU_FIELDS=timestamp,utilization.gpu,utilization.decoder,utilization.encoder,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active
GPULOG=$OUTDIR/gpu_$NAME.csv
[ -e "$GPULOG" ] && GPULOG=$OUTDIR/gpu_${NAME}_$(date +%Y%m%d_%H%M%S).csv
GPULOG_PID=
{ nvidia-smi --query-gpu=$GPU_FIELDS --format=csv -l 1 >> "$GPULOG" 2>/dev/null & GPULOG_PID=$!; } 2>/dev/null || true
stop_gpulog() { [ -n "$GPULOG_PID" ] && kill "$GPULOG_PID" 2>/dev/null; wait "$GPULOG_PID" 2>/dev/null; return 0; }

trap 'stop_server; stop_gpulog' EXIT

echo "== starting server ($SERVE)"
"$SERVE" >/dev/null 2>&1 &
for i in $(seq 1 200); do
  curl -sf localhost:8000/health >/dev/null && break
  sleep 3
  [ "$i" = 200 ] && { echo "server never came up, check results/server_*.log"; exit 1; }
done
echo "== server up"

python bench.py --model "$MODEL" --rates 1 --num-requests 30 --label "$NAME-canary-before" --out "$OUTDIR/canary.csv"
if [ -n "${MODE:-}" ]; then python bench_modes.py --mode "$MODE" --model "$MODEL" --levels "$LEVELS" --duration "$DURATION" --label "$NAME" --out "$OUTDIR/modes.csv"; else python bench.py --model "$MODEL" --rates "$RATES" --duration "$DURATION" --label "$NAME" --out "$OUTDIR/main.csv"; fi
python bench.py --model "$MODEL" --rates 1 --num-requests 30 --label "$NAME-canary-after" --out "$OUTDIR/canary.csv"

python - "$NAME" "$OUTDIR" << 'PY'
import csv, sys, os
n, d = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(f"{d}/canary.csv")))
g = lambda l: [float(r["itl_p50_ms"]) for r in rows if r["label"] == l][-1]
b, a = g(n + "-canary-before"), g(n + "-canary-after")
v = "VALID" if a / b <= 1.3 else "INVALID"
print(f"\nCANARY ITL p50 before {b:.1f} ms, after {a:.1f} ms, ratio {a/b:.2f} -> {v}")
new = not os.path.exists(f"{d}/runs.csv")
with open(f"{d}/runs.csv", "a") as f:
    if new: f.write("name,canary_before_ms,canary_after_ms,ratio,verdict\n")
    f.write(f"{n},{b:.2f},{a:.2f},{a/b:.3f},{v}\n")
PY

if [ "$OVERLOAD" = 1 ]; then
  echo "== overload run (rate 32, last on purpose)"
  python bench.py --model "$MODEL" --rates 32 --duration "$DURATION" --label "$NAME-r32" --out "$OUTDIR/main.csv"
fi
if [ "${PPL:-0}" = 1 ]; then python ppl.py --model "$MODEL" --label "$NAME" --out "$OUTDIR/ppl.csv"; fi
echo "== done, stopping server"
