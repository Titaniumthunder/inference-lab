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

## 7. Repeats (2026-10-07, overnight queue)
3 eager + 3 compiled runs, interleaved, each with canaries (all 6 valid, ratios 0.93 to 1.02).
ITL p50 eager vs compiled (mean of 3): 16.9/11.3 ms (1.49x), 18.0/11.8 (1.52x), 19.9/12.2 (1.62x), 27.1/14.1 (1.92x) at rates 1/4/8/16.
TTFT p50 is 28 to 35% lower with compiled. The earlier single-run result held.
Rate 32 did NOT reproduce: 273 to 558 tok/s now vs 1,189 to 1,358 earlier, ITL p50 400 to 800 ms. Not usable as a headline number.

## 8. Tooling added
run_config.sh (fresh server, canary, sweep, canary, verdict, auto-stop), queue scripts, serve_generic.sh (model/flags via env),
ppl.py (WikiText-2 perplexity through the server), bench_modes.py (shared-prefix and closed-loop workloads).

## 9. Quantization (Qwen2.5, compiled, 2026-10-08)
3B BF16 (4 runs) vs 3B FP8 (3 runs), ITL p50 at rates 1/4/8/16: 11.4/11.8/12.3/14.3 -> 7.4/7.4/7.6/8.2 ms (1.5x to 1.75x faster).
TTFT p50 34 to 41% lower. Perplexity 7.981 -> 8.054 (+0.9%). Ranges do not overlap.
7B FP8 (3 runs): ITL 12.8/13.1/13.7/16.7 ms, perplexity 6.949. About 12% slower per token than 3B BF16 at rate 1.
7B AWQ 4-bit: valid sweep ITL 8.2/8.3/9.1 at rates 1/4/8 (34 to 37% below FP8), 16.8 at rate 12, TTFT 14 to 18% worse, perplexity 7.276 (+4.7% vs FP8 7B).
AWQ collapsed at rate 16 twice (166 and 650 tok/s, canary INVALID).
7B BF16 does not fit in 16 GB. CPU-offload baseline attempt failed to start (cause not yet checked).

## 10. Prefix caching and --max-num-seqs
Prefix caching (3B, ~1,540-token shared prefix, best case): TTFT p50 31/35/38 ms on vs 133/138/544 ms off at 2/4/8 req/s.
Off at 8 req/s also pushes ITL p50 from 11 to 142 ms (probably prefill saturation, not measured).
Closed loop, 3B, throughput at 16/32/64/128/256 clients:
cap 256: 1196/2278/3250/4199/2302 (collapsed at 256, canary INVALID, 3.86x)
cap 128: 1157/2213/3121/4059/3979 (valid)
cap 64:  1152/2125/3109/2994/2981 (valid)
AWQ 7B at rate 16 with cap 64: 1634 tok/s, valid (uncapped collapsed twice).
Every degraded run so far had ~255 sequences running. Caps of 128 and 64 avoided it. Root cause still unknown.

## 11. 7B BF16 perplexity baseline (2026-10-08, queue_ppl7b.sh)
Why the first attempt failed (results/server_bf16-7b-offload_20261008_132510.log): `--cpu-offload-gb 6` plus CUDA graphs.
vLLM keeps pinned memory off on WSL2 unless `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, so the offloader copies the CPU-resident weights
to the GPU on every forward pass, and copies from unpinned memory are not allowed during CUDA-graph capture
("Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned").
Enabling pinned memory was rejected: vLLM's own warning says it can make Windows unresponsive (WDDM pinned-memory cap).
Fix used: `--enforce-eager --cpu-offload-gb 6`. No methodology files changed.

| Config | Engine mode | Perplexity (WikiText-2 test, 81,880 tokens) | vs 7B BF16 | Canary of that run (runs.csv) |
|---|---|---|---|---|
| 7B BF16, 6 GB offloaded to CPU | eager | 6.8988 | baseline | **INVALID** (598.9 -> 790.6 ms, 1.32) |
| 7B FP8 | compiled | 6.9494 | +0.73% | VALID (1.00) |
| 7B AWQ 4-bit | compiled | 7.2758 | +5.46% | **INVALID** (8.62 -> 137.60 ms, 15.96) |
| Control: 3B BF16 | eager | 7.9753 | | VALID (0.96) |
| 3B BF16 (section 9) | compiled | 7.9808 | | VALID (0.96) |
Source: results/ppl.csv, results/runs.csv.

- Eager vs compiled numerics alone move 3B perplexity by -0.07%, about 10x smaller than the FP8 gap, so comparing the eager
  7B BF16 run with the compiled 7B FP8 run is fair to within ~0.1%.
- The 7B BF16 run is INVALID by the lab rule (ratio 1.32 > 1.3). Perplexity is a fixed computation over fixed text, so latency
  drift should not change it, but this has not been checked by a repeat. The AWQ perplexity used in section 9 came from an
  INVALID run too (not flagged there before). Latency numbers from the offload run are meaningless anyway (ITL ~600 to 790 ms,
  6 GB crosses PCIe every step).
- Open question: the offloaded server slowed 32% after only a rate-1 sweep (at most 98 running requests per the server log),
  which is below the ~256-sequence pattern from sections 5 and 10. Not investigated.

## 12. Overload with --max-num-seqs 128 (2026-10-08, queue_overload128.sh)
3B BF16 compiled, Poisson 24 and 32 req/s inside the sweep, so the after-canary measures the post-overload state.
| Run | 24 req/s tok/s | 32 req/s tok/s | TTFT p50 at 32 (ms) | ITL p50 at 32 (ms) | Canary (ms) | Verdict |
|---|---|---|---|---|---|---|
| seqs128-ovl-rep1 | 2488.0 | 2717.9 | 5838 | 31.6 | 14.1 -> 14.7 (1.04) | VALID |
| seqs128-ovl-rep2 | 317.8 | 224.0 | 229074 | 254.9 | 13.8 -> 105.0 (7.61) | **INVALID** |
Source: results/main.csv, results/runs.csv, server logs results/server_seqs128-ovl_20261008_224323.log / _224621.log.

- **The cap does not reliably fix the collapse.** Rep 2 degraded as soon as the batch filled: 127 running, 369 tok/s generation
  (rep 1 at the same point: 126 running, 2,590 tok/s). Up to 832 waiting, KV usage at most 21.1%, no preemptions.
- This refutes section 10's "every degraded run had ~255 sequences running": the collapse also happens at 128 and is stochastic
  (one of two runs). The closed-loop cap-128 result in section 10 was one lucky-or-not run.
- Cap-256 control: started, then the PC crashed at about 23:07 during its overload sweep while that server was collapsed
  (256 running, 128 to 205 tok/s; partial log results/server_seqs256-ovl_20261008_230111.log). No result rows, no verdict.
  Not rerun. Whether the crash was related to the collapsed overload state is unknown.
- **Caveat, GPU contention:** during this queue, Windows processes were also using the GPU (Wallpaper Engine `wallpaper32`
  video decode ~4% x2, Chrome Remote Desktop ~4% 3D, dwm, WebView). Fresh-server canaries were 13.7 to 14.1 ms vs 10.8 to
  11.9 ms in all earlier compiled 3B runs (results/canary.csv), about 23% slower. Absolute numbers from this queue are not
  comparable with earlier sections. Contention may also have contributed to the collapse; rep 1 survived under the same conditions.

## 13. Chunked prefill off vs on (2026-10-08, queue_chunked.sh)
3B BF16 compiled, standard sweep, 3 interleaved off/on pairs, all 6 canaries VALID (ratios 0.98 to 1.02). Mean of 3 [min-max].
| Rate | ITL p50 off / on (ms) | ITL p99 off / on | TTFT p50 off / on (ms) | TTFT p99 off / on | tok/s off / on |
|---|---|---|---|---|---|
| 1 | 12.1 [12.0-12.2] / 12.1 [12.0-12.1] | 19.5 / 19.2 | 46.3 / 46.5 | 59.3 / 64.0 | 121.0 / 121.1 |
| 4 | 12.5 [12.4-12.6] / 12.6 [12.5-12.8] | 26.0 / 26.9 | 50.1 / 49.7 | 67.3 / 68.0 | 500.3 / 500.5 |
| 8 | 13.1 [13.0-13.2] / 13.1 [13.0-13.1] | 30.5 / 29.8 | 53.4 / 53.5 | 80.3 / 80.9 | 872.3 / 873.0 |
| 16 | 15.6 [15.4-15.9] / 15.6 [15.6-15.7] | 45.7 / 45.1 | 68.2 / 68.1 | 125.2 / 126.5 | 1947.9 / 1950.6 |
Source: results/main.csv (labels chunked-off-rep1..3, chunked-on-rep1..3), results/runs.csv.

- **No measurable difference** on this workload: all off/on ratios are within 0.93 to 1.02 and every off range overlaps the on range.
  Expected: prompts are ~250 tokens with 128 output tokens, so a prefill never stalls decodes for long. Chunked prefill
  should matter with long prompts (e.g. the 1,540-token prefix workload with caching off, section 10). Not tested.
- Confound: turning chunked prefill off also raises the per-step token budget from 2,048 to 4,096 (vLLM sets it to
  max(max_model_len, default); server logs show compile_ranges_endpoints [4096] vs [2048]).
- vLLM warns "This model does not officially support disabling chunked prefill ... may crash or produce incorrect outputs"
  (generic warning for any generation model). No crashes; outputs were not checked.
- Caveat: Wallpaper Engine was still using the GPU (video decode, checked before launch). Fresh canaries were 12.0 to 12.2 ms,
  vs 10.8 to 11.9 ms before tonight and 13.7 to 14.1 ms during section 12. Pairs were interleaved, so the off/on comparison holds;
  absolute numbers are not comparable with sections 4 and 9.

## Methodology change (2026-10-08, not yet used for any result)
bench_modes.py: added `--mode text` for item 4 (speculative decoding). WikiText-2 *train* paragraphs (150 words, lightly
detokenized, ~249 prompt tokens median) in the chat template as "continue this article"; every request gets a different
paragraph (3,627 in the pool, the run aborts rather than reuse one). Adds TPOT columns and per-level speculative-decoding
acceptance from /metrics counter deltas. Also added a guard that refuses to append rows whose columns differ from the
CSV header. Existing prefix/closed modes are unchanged (their header still matches results/modes.csv), so no reruns needed.
Text-mode runs write to results/spec/ (OUTDIR) because of the extra columns.
New, not yet run: queue_spec.sh (base vs draft Qwen2.5-0.5B k=3/k=5, 2 repeats), hf_vs_vllm.py + run_hf_vs_vllm.sh (item 5).
