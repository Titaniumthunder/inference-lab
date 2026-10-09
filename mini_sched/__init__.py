"""Mini continuous-batching scheduler with a paged KV cache (LAB_LOG section 17)."""
import os

# Must happen before torch initializes CUDA. With the default allocator, growing per-step tensor shapes
# kept creating new device segments (cudaMalloc), and each cost ~0.5 to 1 s under WSL2: 32 of 313 steps
# took >0.5 s. Expandable segments removed all but the warmup stall (LAB_LOG section 17).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
