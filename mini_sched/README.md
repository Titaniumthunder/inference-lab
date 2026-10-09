# mini_sched: continuous batching over a paged KV cache

A small, readable re-implementation of the core of an LLM serving engine, for Qwen2.5-3B-Instruct (BF16) in plain PyTorch.
Measured results and caveats: LAB_LOG.md section 17. Raw rows: `results/mini_sched.csv`.

| File | What it does |
|---|---|
| `kv_cache.py` | Fixed pool of 16-token KV blocks per layer, free-list allocator with double-free checks, slot mapping |
| `model.py` | Qwen2 forward pass on the HF weights: new K/V scattered into their slots, each sequence attends over its own blocks (padded SDPA with GQA), prefill and decode mixed in one step, FP32 logits for the sampled positions |
| `scheduler.py` | vLLM-style step policy: decode running sequences, preempt the newest by recompute when blocks run out, admit waiting requests FCFS under `max_num_seqs`, a per-step token budget and free blocks. `policy="static"` is the request-level baseline |
| `test_mini.py` | Allocator tests; greedy outputs vs HF `generate` with sequences joining mid-batch, and with a 70-block pool that forces preemption |
| `bench_sched.py` | `--part steps` (static batches, section-15 workload) and `--part poisson` (continuous vs static under Poisson arrivals) |

```bash
python -m mini_sched.test_mini
```

```bash
python -m mini_sched.bench_sched --part poisson --rates 1,2,4
```

Run from the repo root with `~/.venv` active. Importing the package sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(without it, new cudaMalloc segments stalled steps for ~1 s each under WSL2).

Not implemented: chunked prefill, prefix caching, CUDA graphs, custom attention kernels, sampling other than greedy.
