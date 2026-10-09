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

# Idle-GPU preflight (LAB_LOG sections 21-22): 10 one-second nvidia-smi samples; refuse if ANY sample has utilization > 3%
# or memory used > 2048 MiB. In WSL these numbers include Windows processes. Decoder/encoder utilization is printed but
# not checked: Wallpaper Engine's video decode runs there and does not show up in utilization.gpu.
busy=0
for _ in $(seq 1 10); do
  IFS=', ' read -r util mem dec enc <<< "$(nvidia-smi --query-gpu=utilization.gpu,memory.used,utilization.decoder,utilization.encoder --format=csv,noheader,nounits)"
  echo "preflight: util ${util}%  mem ${mem} MiB  dec ${dec}%  enc ${enc}%"
  if ! [[ "$util" =~ ^[0-9]+$ && "$mem" =~ ^[0-9]+$ ]] || [ "$util" -gt 3 ] || [ "$mem" -gt 2048 ]; then busy=1; fi
  sleep 1
done
if [ "$busy" = 1 ]; then
  echo "GPU not idle (a sample had utilization > 3%, memory > 2048 MiB, or an unreadable value). Not running $NAME."; exit 1
fi

stop_server() {
  pkill -INT -f "vllm serve" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "vllm serve" >/dev/null || return; sleep 1; done
  pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f EngineCor 2>/dev/null
}
trap stop_server EXIT

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
