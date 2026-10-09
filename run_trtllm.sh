#!/usr/bin/env bash
# Item 7: TensorRT-LLM (1.2.1, PyTorch backend) on the same workload as run_hf_vs_vllm.sh.
# Own venv (~/.venv-trtllm: torch 2.9.1+cu130, pyarrow<20), same WSL PATH / CUDA_HOME fix.
# Usage: ./run_trtllm.sh            -> 2 repeats into results/hf_vs_vllm.csv
#        ./run_trtllm.sh smoke      -> B=1, 1 trial into results/smoke/
cd ~/inference-lab
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/' | paste -sd:)
NVCC=$(find ~/.venv -name nvcc -type f 2>/dev/null | head -n1)
export CUDA_HOME=$(dirname "$(dirname "$NVCC")")
export PATH=$CUDA_HOME/bin:$PATH
PY=~/.venv-trtllm/bin/python
# Single GPU: keep MPI on loopback + shared memory. Unrestricted, OpenMPI hung trying eth0 + the docker bridge (results/trtllm_smoke_mpi_hang.log).
export OMPI_MCA_btl=self,vader OMPI_MCA_btl_tcp_if_include=lo OMPI_MCA_oob_tcp_if_include=lo
gpu_free() {  # no engine left, GPU back near the idle baseline
  sleep 5
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  echo "GPU memory used: $used MiB"
  [ "$used" -lt 2500 ] || { echo "QUEUE STOPPED: GPU not idle $(date)"; exit 1; }
}
pgrep -f "vllm serve" >/dev/null && { echo "a vllm server is running, stop it first"; exit 1; }
gpu_free
if [ "${1:-}" = smoke ]; then
  mkdir -p results/smoke
  $PY hf_vs_vllm.py --backend trtllm --label trtllm-smoke --batch-sizes 1 --trials 1 --out results/smoke/trtllm_smoke.csv
  gpu_free; exit
fi
for i in 1 2; do
  $PY hf_vs_vllm.py --backend trtllm --label trtllm-rep$i || { echo "QUEUE STOPPED: trtllm-rep$i failed $(date)"; exit 1; }
  gpu_free
done
echo "QUEUE FINISHED $(date)"
