# Lab log (condensed)

Working notes from the first day of the project (2026-10-05). Wrong turns and refuted hypotheses are kept on purpose:
they are part of how the measurements got to be trustworthy.

## Environment
| Item | Value |
|---|---|
| GPU | RTX 5080, 16 GB, compute capability 12.0 (sm_120) |
| OS | Windows host, WSL2, Ubuntu 22.04.5 |
| Stack | Python 3.12.15, vLLM 0.31.0, PyTorch 2.13.0+cu132, FlashAttention 2 backend |
| Model | Qwen2.5-3B-Instruct, BF16, `--max-model-len 4096`, `--gpu-memory-utilization 0.85` |

## 1. Getting the compiled mode to start (two failures)
- **First failure:** `PermissionError: 'nvcc'` during torch.compile. Cause: a PyTorch helper that calls `nvcc --version` only catches
  FileNotFoundError, and on WSL2 the lookup through the appended Windows folders raises PermissionError (same traceback as
  pytorch/pytorch issue #185733). Fix: remove the `/mnt/...` folders from PATH for the server process.
- **Second failure:** FlashInfer's JIT build of its sampling kernel failed with "CUDA compiler and CUDA toolkit headers are
  incompatible". Likely cause (inferred from the pip install list): nvcc 13.4 packages with CUDA 13.2 runtime headers.
  Workaround: `--kernel_config.enable_jit_warmup=False` (the benchmark uses temperature 0, so that kernel is not needed).
- Lesson: the console only shows the end of a traceback ("see root cause above"); always `tee` server output to a log file.

## 2. KV-cache formula check
For Qwen2.5-3B (36 layers, 2 KV heads, head_dim 128, BF16): 2 x 36 x 2 x 128 x 2 B = 36,864 B/token.
207,984 tokens x 36,864 B = 7.14 GiB, exactly what vLLM reported in eager mode. Compiled mode holds ~11% fewer tokens
(184,944) because CUDA graphs take memory.

## 3. Harness iterations
| Version | Problem found | Fix |
|---|---|---|
| v0 | Fixed 100 requests per rate: at high rates a run lasts seconds, so ramp-up and drain dominate and tok/s is understated | v1: size each run by duration (30 s, at least 100 requests) |
| v1 | Every run sent identical prompts (same seed) and vLLM's prefix cache is on by default, so runs depended on earlier runs | v2: unique per-request prefix |
Compiled rates 1 to 16 reproduced between a v1 sweep and a v2 sweep (p50 ITL within ~7%, tok/s within ~1%), so those numbers were not materially affected.

## 4. Eager vs compiled (valid runs: fresh server, canaries passed)
| Rate | ITL p50 eager / compiled (ms) | Ratio | ITL p99 ratio | TTFT p50 eager / compiled (ms) |
|---|---|---|---|---|
| 1 | 16.5 / 10.8 | 1.53x | 2.49x | 53 / 41 |
| 4 | 18.6 / 11.3 | 1.65x | 1.77x | 65 / 45 |
| 8 | 19.6 / 11.7 | 1.68x | 1.77x | 66 / 47 |
| 16 | 25.6 / 13.3 | 1.92x | 1.65x | 85 / 57 |
Single-stream decode at 1 req/s: ~61 tok/s eager (~39% of the ~154 tok/s bandwidth ceiling), ~93 tok/s compiled (~60%).
CUDA graphs and compilation remove 5.7 ms of the 10 ms gap (~57%); about 4.3 ms is unexplained and needs profiling.

## 5. Overload behaviour (32 req/s) and a persistent slowdown
Output throughput at 32 req/s in different runs: 1,982 (cached prompts, probably flattered), 1,358, 1,189, and 1,309 tok/s,
all below the 1,971 tok/s the same server handled at 16 req/s. The eager server collapsed to 367 tok/s with p50 TTFT of 58 s.
Observed during overload: running sequences pinned at 256 and up to 541 requests waiting (reason: capacity), KV cache usage at most 40%.
The engine process used ~100% of one CPU core; the API server (1 to 26%) and the load generator (1 to 11%) were not the bottleneck.

After an overload run the server stayed slow even at low load until it was restarted (rate-1 canary, ITL p50):
| State | ITL p50 |
|---|---|
| fresh server (canaries) | 10.9 ms, 11.2 ms |
| after an overload run | 53.7 ms, 48.7 ms, 50.0 ms |
It followed every overload run that was checked afterwards, and a restart cured it each time. Hypotheses tested and **refuted**:
| Hypothesis | Evidence against |
|---|---|
| KV-cache exhaustion / preemption | 0 preemptions across 2,430 requests; KV usage peaked at 40% |
| Thermal or low-clock throttling | 36 C idle, 41 to 47 C under load, SM clock 2.8 GHz of 3.09 GHz max |
| WSL memory pressure | swap used 0 B, 11 GiB available |
| Windows paging VRAM to system RAM | Task Manager shared GPU memory 0.1 / 15.6 GB |
| Too little VRAM headroom | Still degraded at `--gpu-memory-utilization 0.75` (verified from the server log) |
**Cause: unknown.** GPU memory used grew from 14,951 MiB (after startup) to 15,399 MiB (after rate 16) and ~15,943 MiB (after overload),
a lead not yet tested. Practical workaround: fresh server per configuration, rate-1 canary before and after each sweep, overload last.

## 6. Corrections made along the way
- Reported throughput at high rates was understated by the v0 harness (see section 3).
- An early inference about queueing in one rate-32 run was attached to the wrong run: the gauge trace showing a queue of 541 belonged to the second run, not the first.
- A first guess that VRAM was nearly full because Windows apps had grown was wrong: Windows used ~1 GB; vLLM itself used ~14.4 GiB.
- Early eager-vs-compiled numbers were pilot data (shared prompts); they were redone and agreed within ~7% (p50 ITL) for compiled.

## 7. Next
Hugging Face `generate` vs vLLM; quantization on an 8B model (FP8, INT4, NVFP4); prefix caching, chunked prefill, speculative decoding,
`--max-num-seqs` sweep; profiling and roofline analysis (the unexplained 4.3 ms and the overload slowdown); a small continuous-batching
scheduler with a paged KV cache.
