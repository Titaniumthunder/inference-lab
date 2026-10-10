#!/usr/bin/env bash
# Idle-GPU preflight (LAB_LOG sections 21-23), called by run_config.sh before it starts a server; can also run on its own.
# One attempt = 10 one-second nvidia-smi samples. An attempt fails if ANY sample has utilization.gpu > 3%,
# utilization.decoder > 0% (Wallpaper Engine's video decode shows up only there), memory.used > 1500 MiB, or an
# unreadable value. In WSL these numbers include Windows processes. A failed attempt waits 20 s and re-samples;
# exit 75 (EX_TEMPFAIL, "try again later") only if all 3 attempts fail. Exit 0 as soon as one attempt passes.
MAX_UTIL=3; MAX_DEC=0; MAX_MEM_MIB=1500; SAMPLES=10; ATTEMPTS=3; WAIT_S=20
for attempt in $(seq 1 $ATTEMPTS); do
  reasons=()
  for s in $(seq 1 $SAMPLES); do
    IFS=', ' read -r util mem dec enc <<< "$(nvidia-smi --query-gpu=utilization.gpu,memory.used,utilization.decoder,utilization.encoder --format=csv,noheader,nounits)"
    echo "preflight attempt $attempt sample $s: util ${util}%  mem ${mem} MiB  dec ${dec}%  enc ${enc}%"
    if ! [[ "$util" =~ ^[0-9]+$ && "$mem" =~ ^[0-9]+$ && "$dec" =~ ^[0-9]+$ ]]; then
      reasons+=("sample $s unreadable (util='$util' mem='$mem' dec='$dec')")
    else
      [ "$util" -gt $MAX_UTIL ] && reasons+=("sample $s util ${util}% > ${MAX_UTIL}%")
      [ "$dec" -gt $MAX_DEC ] && reasons+=("sample $s decoder ${dec}% > ${MAX_DEC}%")
      [ "$mem" -gt $MAX_MEM_MIB ] && reasons+=("sample $s memory ${mem} MiB > ${MAX_MEM_MIB} MiB")
    fi
    sleep 1
  done
  if [ ${#reasons[@]} -eq 0 ]; then
    echo "preflight PASSED on attempt $attempt"; exit 0
  fi
  echo "preflight attempt $attempt FAILED: $(printf '%s; ' "${reasons[@]}")"
  [ "$attempt" -lt $ATTEMPTS ] && { echo "waiting ${WAIT_S} s before re-sampling"; sleep $WAIT_S; }
done
echo "preflight REFUSED: GPU not idle in all $ATTEMPTS attempts"
exit 75
