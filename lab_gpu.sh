# Shared GPU guards for runners that do not go through run_config.sh (LAB_LOG section 29). Source it, do not execute it.
#   lab_preflight NAME        idle-GPU preflight (gpu_preflight.sh); exits the caller with 75 on refusal, 1 on any other failure
#   lab_sidecar_start CSV     1 Hz nvidia-smi logger, same fields as run_config.sh (section 25); best effort, never fails the run,
#                             never overwrites (adds a timestamp suffix if CSV exists)
#   lab_sidecar_stop          kill the logger (call it from the caller's EXIT trap)
LAB_GPU_FIELDS=timestamp,utilization.gpu,utilization.decoder,utilization.encoder,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active
LAB_SIDECAR_PID=

lab_preflight() {
  ~/inference-lab/gpu_preflight.sh; local pf=$?
  [ "$pf" = 0 ] && return 0
  echo "Not running $1: GPU not idle (preflight exit $pf)."
  [ "$pf" = 75 ] && exit 75
  exit 1
}

lab_sidecar_start() {
  local f=$1
  [ -e "$f" ] && f=${f%.csv}_$(date +%Y%m%d_%H%M%S).csv
  { nvidia-smi --query-gpu=$LAB_GPU_FIELDS --format=csv -l 1 >> "$f" 2>/dev/null & LAB_SIDECAR_PID=$!; } 2>/dev/null || true
  echo "GPU sidecar -> $f"
}

lab_sidecar_stop() {
  [ -n "$LAB_SIDECAR_PID" ] && kill "$LAB_SIDECAR_PID" 2>/dev/null
  wait "$LAB_SIDECAR_PID" 2>/dev/null
  LAB_SIDECAR_PID=
  return 0
}
