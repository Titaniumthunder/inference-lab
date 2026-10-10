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
CUDA graphs and compilation remove 5.7 ms of the 10 ms gap (~57%); about 4.3 ms is unexplained and needs profiling. **[superseded 2026-10-09, see section 20]**

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

## Plan as of 2026-10-05 (superseded by CLAUDE.md "Remaining work")
Hugging Face `generate` vs vLLM; quantization on an 8B model (FP8, INT4, NVFP4); prefix caching, chunked prefill, speculative decoding,
`--max-num-seqs` sweep; profiling and roofline analysis (the unexplained 4.3 ms and the overload slowdown); a small continuous-batching
scheduler with a paged KV cache.

## 7. Repeats (2026-10-07, overnight queue)
3 eager + 3 compiled runs, interleaved, each with canaries (all 6 valid, ratios 0.93 to 1.02).
ITL p50 eager vs compiled (mean of 3): 16.9/11.3 ms (1.49x), 18.0/11.8 (1.52x), 19.9/12.2 (1.62x), 27.1/14.1 (1.92x) at rates 1/4/8/16. **[superseded 2026-10-09, see section 20]**
TTFT p50 is 28 to 35% lower with compiled. The earlier single-run result held. **[superseded 2026-10-09, see section 20]**
Rate 32 did NOT reproduce: 273 to 558 tok/s now vs 1,189 to 1,358 earlier, ITL p50 400 to 800 ms. Not usable as a headline number. **[superseded 2026-10-09, see section 20]**

## 8. Tooling added
run_config.sh (fresh server, canary, sweep, canary, verdict, auto-stop), queue scripts, serve_generic.sh (model/flags via env),
ppl.py (WikiText-2 perplexity through the server), bench_modes.py (shared-prefix and closed-loop workloads).

## 9. Quantization (Qwen2.5, compiled, 2026-10-08)
3B BF16 (4 runs) vs 3B FP8 (3 runs), ITL p50 at rates 1/4/8/16: 11.4/11.8/12.3/14.3 -> 7.4/7.4/7.6/8.2 ms (1.5x to 1.75x faster). **[superseded 2026-10-09, see section 20]**
TTFT p50 34 to 41% lower. Perplexity 7.981 -> 8.054 (+0.9%). Ranges do not overlap.
7B FP8 (3 runs): ITL 12.8/13.1/13.7/16.7 ms, perplexity 6.949. About 12% slower per token than 3B BF16 at rate 1. **[superseded 2026-10-09, see section 20]**
7B AWQ 4-bit: valid sweep ITL 8.2/8.3/9.1 at rates 1/4/8 (34 to 37% below FP8), 16.8 at rate 12, TTFT 14 to 18% worse, perplexity 7.276 (+4.7% vs FP8 7B). **[superseded 2026-10-09, see section 20]**
AWQ collapsed at rate 16 twice (166 and 650 tok/s, canary INVALID).
7B BF16 does not fit in 16 GB. CPU-offload baseline attempt failed to start (cause not yet checked). **[superseded 2026-10-09, see section 20]**

## 10. Prefix caching and --max-num-seqs
Prefix caching (3B, ~1,540-token shared prefix, best case): TTFT p50 31/35/38 ms on vs 133/138/544 ms off at 2/4/8 req/s. **[superseded 2026-10-09, see section 20]**
Off at 8 req/s also pushes ITL p50 from 11 to 142 ms (probably prefill saturation, not measured).
Closed loop, 3B, throughput at 16/32/64/128/256 clients:
cap 256: 1196/2278/3250/4199/2302 (collapsed at 256, canary INVALID, 3.86x)
cap 128: 1157/2213/3121/4059/3979 (valid)
cap 64:  1152/2125/3109/2994/2981 (valid)
AWQ 7B at rate 16 with cap 64: 1634 tok/s, valid (uncapped collapsed twice).
Every degraded run so far had ~255 sequences running. Caps of 128 and 64 avoided it. Root cause still unknown. **[superseded 2026-10-09, see section 20]**

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
  11.9 ms in all earlier compiled 3B runs (results/canary.csv), about 23% slower. Absolute numbers from this queue are not **[superseded 2026-10-09, see section 20]**
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
  vs 10.8 to 11.9 ms before tonight and 13.7 to 14.1 ms during section 12. Pairs were interleaved, so the off/on comparison holds; **[superseded 2026-10-09, see section 20]**
  absolute numbers are not comparable with sections 4 and 9.

## Methodology change (2026-10-08, not yet used for any result)
bench_modes.py: added `--mode text` for item 4 (speculative decoding). WikiText-2 *train* paragraphs (150 words, lightly
detokenized, ~249 prompt tokens median) in the chat template as "continue this article"; every request gets a different
paragraph (3,627 in the pool, the run aborts rather than reuse one). Adds TPOT columns and per-level speculative-decoding
acceptance from /metrics counter deltas. Also added a guard that refuses to append rows whose columns differ from the
CSV header. Existing prefix/closed modes are unchanged (their header still matches results/modes.csv), so no reruns needed.
Text-mode runs write to results/spec/ (OUTDIR) because of the extra columns.
New, not yet run: queue_spec.sh (base vs draft Qwen2.5-0.5B k=3/k=5, 2 repeats), hf_vs_vllm.py + run_hf_vs_vllm.sh (item 5).

## 14. Speculative decoding, realistic text (2026-10-09, queue_spec.sh, results/spec/)
3B BF16 compiled target, draft Qwen2.5-0.5B-Instruct (`method: draft_model`), `--mode text` (WikiText-2 continuation, ~248
prompt tokens, 128 output tokens), Poisson 1/4/8/16 req/s, 2 interleaved repeats. Wallpaper Engine closed (no Windows GPU
users above 0.5%); baseline canaries back to 10.1 to 10.3 ms. Values: both repeats.
| Rate | TPOT p50 no spec (ms) | k=3 | k=5 | TTFT p50 no spec / k=3 / k=5 (ms) | tok/s at 16: no spec / k=3 / k=5 |
|---|---|---|---|---|---|
| 1 | 10.6, 10.3 | 13.3, 13.5 | 20.6, 19.6 | 38-39 / 90-99 / 137-154 | |
| 4 | 11.1, 11.0 | 14.5, 14.4 | 23.2, 20.8 | 41-42 / 103-107 / 162-170 | |
| 8 | 12.3, 12.1 | 16.0, 16.8 | 23.3, 23.2 | 46-47 / 120-121 / 181-186 | |
| 16 | 15.5, 15.2 | 1165, 970 (collapsed) | 196, 3430 (collapsed) | | 2103, 2106 / 191, 234 / 788, 52 |
Draft acceptance rate: k=3 0.497 to 0.516 (mean acceptance length 2.49 to 2.55); k=5 0.372 to 0.391 (2.86 to 2.96).
Canaries (results/spec/runs.csv): base 0.99, 0.97; k=3 **1.68 INVALID**, 1.07; k=5 0.90, 1.23.

- **Speculative decoding with this draft is slower at every load**: TPOT p50 +28.1% to +34.6% (k=3) and +90.9% to +99.2% (k=5),
  means of 2 repeats at rates 1 to 8 (corrected 2026-10-09: an earlier draft of this line said +26 to 37% / +85 to 102%, computed
  from rounded values; pair-by-pair ranges are +25.3 to +38.7% and +89.6 to +108.7%); TTFT 2.4x to 4.1x **[superseded 2026-10-09, see section 20]**
  worse; at 16 req/s both spec configs exceed capacity and the queue explodes (no-spec handles it at 2,103 tok/s).
- Derived, not profiled: time per engine step = TPOT x mean acceptance length = ~33.5 ms for k=3 at rate 1 vs ~10.5 ms without
  spec, i.e. ~7.7 ms per 0.5B draft step, ~70% of a 3B target step for a model 6x smaller. Small-batch decode here is dominated
  by per-layer fixed costs (the 0.5B has 24 layers vs 36), not by weight bytes, so a small draft is not cheap. Same idea as the
  unexplained ~4.3 ms in section 4; profiling (item 6) should check it.
- Random-word prompts would have flattered spec decoding: in the smoke test the random-word canary intervals showed per-position
  acceptance 0.77 to 0.92 vs ~0.5 on real text (server log results/server_smoke-spec-k3_*.log). Smoke run is in results/smoke/.
- spec-k3-rep1 is INVALID (after-canary 43.6 vs 25.9 ms) following the rate-16 collapse; rep 2 recovered. Spec-config canaries
  measure ITL per streamed chunk (several tokens), so they are only comparable within a run.
- Not tried: EAGLE-style heads, n-gram drafting, larger --max-num-batched-tokens (vLLM warned the 2,048 budget "may lead to
  suboptimal performance" with spec decoding).

## 15. Hugging Face generate vs vLLM (2026-10-09, run_hf_vs_vllm.sh, results/hf_vs_vllm.csv)
Qwen2.5-3B BF16, in-process on both sides (HF `generate` with sdpa vs vLLM `LLM.generate`, compiled + CUDA graphs), no HTTP.
Identical 256-token prompts, greedy, exactly 128 new tokens, repetition_penalty 1.0 on HF (Qwen's generation_config sets 1.05,
which vLLM ignores). Fresh process per backend, 2 interleaved repeats x 3 trials = 6 per cell, medians [tok/s min-max].
| Batch | TPOT HF / vLLM (ms) | TTFT HF / vLLM (ms) | tok/s HF | tok/s vLLM | vLLM speedup |
|---|---|---|---|---|---|
| 1 | 28.00 / 9.67 | 32.1 / 22.4 | 35.6 [33.2-39.5] | 102.4 [100.6-103.3] | 2.87x |
| 8 | 29.22 / 10.69 | 141.4 / 134.5 | 266.3 [231.6-291.6] | 685.3 [667.2-711.5] | 2.57x |
| 32 | 32.30 / 11.33 | 551.9 / 486.6 | 881.5 [770.2-894.5] | 2130.2 [2088.4-2166.3] | 2.42x |
| 64 | 33.69 / 12.42 | 1104.3 / 948.4 | 1522.0 [1458.3-1626.6] | 3222.0 [3156.8-3308.6] | 2.12x |
- Correctness: the B=1 greedy outputs are identical, all 128 of 128 tokens, between HF and vLLM, and each backend reproduced
  itself across processes (results/hf_vs_vllm_tokens_*.json).
- The gap is mostly decode (TPOT 2.7x to 2.9x); prefill (TTFT) is only 1.05x to 1.43x, as expected for compute-bound prefill.
  HF decode is ~28 ms/token even at batch 1, i.e. Python/launch overhead without CUDA graphs.
- Static batches only: this does not exercise continuous batching, where vLLM's advantage under mixed arrivals should be larger.
  HF used default settings (dynamic cache, no torch.compile / static cache); a tuned HF would close part of the gap.

## 16. TensorRT-LLM (2026-10-09): stopped, no result
Goal: run tensorrt_llm 1.2.1 (PyTorch backend, CUDA graphs) on the section-15 workload (backend `trtllm` in hf_vs_vllm.py,
runner run_trtllm.sh, separate venv ~/.venv-trtllm so ~/.venv is untouched).
| Step | Outcome | Log |
|---|---|---|
| Install attempt 1 (plain pip) | Installed, import failed: `libcublasLt.so.13` missing (pip picked torch 2.9.1+cu128, wheel needs CUDA 13) | results/trtllm_install_attempt1.log |
| Install attempt 2 (torch 2.9.1+cu130 first) | Import failed on pyarrow 26 needing NumPy 2 (TRT-LLM pins 1.26); fixed with `pyarrow<20`; then needed CUDA_HOME (same fix as serve scripts). Import OK | results/trtllm_install_attempt2.log |
| Run, default MPI | Hung >10 min at 0% GPU: OpenMPI dropped TCP connections across eth0 and the docker bridge 172.17.0.1 | results/trtllm_smoke_mpi_hang.log |
| Run, TLLM_DISABLE_MPI=1 | Refused: needs Ray (not installed, not added) | results/trtllm_smoke_noray.log |
| Run, MPI on loopback + shared memory | `MPI_INIT failed ... ompi_dpm_dyn_init() failed: Unreachable` (worker spawn) | results/trtllm_smoke.log |
Stopped per the two-attempt rule. Nothing measured; no TensorRT-LLM numbers exist. Possible next tries (not done): install Ray and
use TLLM_DISABLE_MPI=1, or exclude only docker0 (`OMPI_MCA_btl_tcp_if_exclude=docker0,lo`) instead of restricting to loopback,
or the official NGC container (needs Docker GPU support in WSL).

## 17. Mini scheduler with a paged KV cache (2026-10-09, mini_sched/)
Plain-PyTorch engine for Qwen2.5-3B BF16: 16-token KV blocks with a free-list allocator, a hand-written Qwen2 forward pass that
scatters K/V into slots and attends over each sequence's blocks (padded SDPA, GQA), and a vLLM-style continuous-batching
scheduler (decode first, preempt newest by recompute, FCFS admission under max_num_seqs 64 / 4,096-token budget / free blocks).
`policy="static"` (admit only when the batch is empty) is the baseline with identical model code. See mini_sched/README.md.

**Correctness** (mini_sched/test_mini.py, 6 prompts of 20 to 600 tokens, 32 greedy tokens, vs HF generate): allocator tests pass;
continuous batching with mid-batch joins: 5/6 exact, 1 diverges at token 1; with a 70-block pool forcing 2 preemptions: 3/6 exact,
3 diverge at tokens 1, 2, 6. Every divergence is an exact tie under HF itself (teacher-forced logit gap 0.000), so the HF choice
there is arbitrary too. Found on the way: with BF16 logits, a 5-prompt prefill turned a 0.125 gap into an exact tie (batch-shape
rounding), so sampled-position logits are computed in FP32.

**Per-step speed** (static batches, 256-token prompts, 128 new tokens, 3 trials each, label mini-xseg in results/mini_sched.csv):
| Batch | TPOT mini (ms) | TPOT HF / vLLM, section 15 (ms) | tok/s mini |
|---|---|---|---|
| 1 | 52.1 to 60.9 | 28.00 / 9.67 | 16.4 to 19.2 |
| 8 | 56.9 to 61.7 | 29.22 / 10.69 | 128.0 to 138.5 |
| 32 | 82.4 to 83.1 | 32.30 / 11.33 | 363.6 to 367.5 |
| 64 | 108.1 to 113.7 | 33.69 / 12.42 | 517.3 to 540.2 |
The mini engine is ~2x slower than HF and ~5 to 9x slower than vLLM per step. Unverified guess: ~20 extra kernel launches per layer
(gathers, scatters, masked attention) with high launch cost under WSL2; no CUDA graphs. Item 6 profiling can check it.

**Continuous vs static batching** (Poisson arrivals for 30 s, same prompts for both policies, single run each, all requests finished):
| Rate (req/s) | tok/s continuous / static | TTFT p50 continuous / static (ms) | TTFT p99 continuous / static (ms) | TPOT p50 c / s (ms) |
|---|---|---|---|---|
| 1 | 95.5 / 96.2 | 141 / 5017 | 854 / 7832 | 58.2 / 59.1 |
| 2 | 215.0 / 178.4 | 145 / 6164 | 268 / 11745 | 80.3 / 82.9 |
| 4 | 310.6 / 300.1 | 188 / 5345 | 289 / 11903 | 105.1 / 84.1 |
- Continuous batching cuts median TTFT 28.4x to 42.5x: a new request starts at the next step instead of waiting for the running batch **[superseded 2026-10-09, see section 20]**
  to finish. Throughput is similar (open loop; both keep up). TPOT is higher at rate 4 because batches are larger (peak 60 vs 44).
- Paged KV memory at the peak: 3,760 / 9,728 / 19,200 tokens (0.13 / 0.33 / 0.66 GiB at 36,864 B/token) vs 45,056 / 122,880 /
  245,760 tokens (1.55 / 4.22 / 8.44 GiB) if each running sequence reserved max_model_len 4,096: 12.0x to 12.8x less. Caveat: peak
  blocks and peak batch need not coincide; it compares against a reservation policy, not a measured contiguous engine.
- **Allocator stall (fixed):** the first Poisson run (label `mini`, rows kept in results/mini_sched.csv, logs
  results/mini_sched_poisson.log) gave continuous 112.7 tok/s and TPOT p50 558 ms at rate 4, worse than static. Per-step timing
  showed 32 of 313 steps taking >0.5 s (~1 s each) and 299 device allocations; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  cut that to 1 slow step (warmup) and the 20 s test run from 53.9 to 31.2 s wall. Now set in mini_sched/__init__.py. All numbers
  above are from the rerun (label mini-xseg). **Lead for item 6:** vLLM's activations also go through the PyTorch caching
  allocator; slow cudaMalloc under WSL2 is a candidate contributor to the overload collapse (untested).

## 18. Docs pass (2026-10-09)
plot.py `--summary` writes results/summary_{quantization,engines,speculative,batching}.png from the CSVs and prints every plotted
value. README rewritten around current results; RESUME_BULLETS.md drafted (measured numbers only, sources cited).
Discrepancies found while recomputing from the CSVs (docs now use the recomputed values):
- Section 9 ITL means: 3B BF16 11.4/11.8/12.3/14.3 vs 11.3/11.8/12.2/14.3 recomputed over bf16-3b + compiled-rep1..3; 7B FP8 **[superseded 2026-10-09, see section 20]**
  12.8/13.1/13.7/16.7 vs 12.7/13.1/13.8/16.7. Within 0.1 ms; probably a different run set or rounding. FP8 speedup is 1.53x to 1.74x. **[superseded 2026-10-09, see section 20]**
- Prefix caching TTFT reduction is 3.9x to 14.3x (138/35 at 4 req/s), not "4x to 14x" as CLAUDE.md says. **[superseded 2026-10-09, see section 20]**
- Section 14 and 17 ratios corrected (see the notes there): spec TPOT +28.1 to +34.6% (k=3), TTFT 2.4x to 4.1x; batching TTFT 28.4x to 42.5x. **[superseded 2026-10-09, see section 20]**
- CLAUDE.md "Findings so far" still says caps 128 and 64 avoided the collapse; section 12 refutes that for 128.

## 19. Profiling (2026-10-09, profile_run.sh, profile_analyze.py, results/profiles/)
torch profiler via vLLM `--profiler-config` (3B BF16 compiled, cap 256), fresh server per run. Profiled windows are for
breakdowns only (profiler start delays TTFT by ~12 s; profiled batch-1 ITL 11.9 to 12.3 ms vs 11.5 ms unprofiled). Per step:
period = start to next start; GPU busy = union of kernel/memcpy intervals in that period. Medians, results/profiles/summary.csv.
Windows GPU users at launch: Chrome Remote Desktop 2.1%, Razer.VideoRecorder 0.5% (small; not closed).

**The "unexplained ~4.3 ms" (section 4), batch-1 decode, healthy server (269 steps):**
| Part of a 11.70 ms step | ms |
|---|---|
| GPU busy (474 GPU ops incl. CUDA-graph kernels) | 8.51 |
| GPU idle between steps (CPU work not overlapped with the GPU) | 3.41 |
Against the ~6.5 ms bandwidth floor (section 4, approximate): ~3.4 ms is GPU idle time while the CPU schedules, samples and
processes outputs (the CPU then spends 7.5 ms/step in cudaStreamSynchronize, i.e. it waits for the GPU and the GPU waits for it,
no overlap), and ~2 ms is kernels running below peak bandwidth (8.5 ms of GPU time vs ~6.5 ms). Batches of 25 to 34 decodes:
period 13.3 to 14.5 ms, GPU busy 9.3 to 10.7, idle 2.8 to 4.2. Mixed prefill+decode steps: idle up to 7.2 ms.

**Collapse reproduction: not reproduced (2 attempts).** Both overloads (256 closed-loop clients, 30 s) stayed healthy: 4,034.4 and
4,001.2 tok/s (the collapsed seqs256 run in section 10 gave 2,302); unprofiled checks after: 12.2 and 12.1 ms. Stopped after 2
attempts (the PC crashed during a collapsed run on 2026-10-08). Collapses so far happened 1 of 2 (cap 128) and 1 of 1 (cap 256)
on 2026-10-08, partly with Wallpaper Engine on the GPU; 0 of 2 today with it closed. GPU contention as a trigger is a hypothesis.

**But a milder post-overload slowdown is visible, and it is on the GPU, in cuBLAS GEMV:** batch-1 step GPU busy rose from 8.72 to
9.85 ms and from 8.90 to 10.07 ms after the overload (same 474 ops; GPU idle unchanged at ~3 ms). Per-kernel: cuBLAS bf16 `gemvx`
(the weight matrix-vector products) went 7.18 -> 8.24 and 7.31 -> 8.39 ms/step (+15%); flash-attention splitkv was flat
(0.226 -> 0.238, 0.230 -> 0.236), which argues against KV-block fragmentation. GEMV is memory-bandwidth-bound, so identical work
got ~15% less effective bandwidth after heavy load. Leading hypothesis (untested): GPU memory clock / power state after sustained
load. Next test: log `nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,temperature.memory` at 1 Hz through a
canary-overload-canary sequence, and try a 60 s idle pause before the after-canary.

## 20. Corrections (2026-10-09)
Every number in sections 1-19 that appears in a summary was recomputed from the CSVs (results/main.csv, modes.csv, canary.csv,
runs.csv, spec/modes.csv, mini_sched.csv). Policy from now on: ratios and percentages are **truncated** to the decimals shown,
never rounded up; exact values are given here. Superseded lines above are tagged in place and not deleted.

| Where | Old claim | Status | Exact value / source |
|---|---|---|---|
| 4 | "about 4.3 ms is unexplained" | Superseded by 19 | Batch-1 step: GPU busy 8.51 ms, GPU idle 3.41 ms of 11.70 ms (profiled) |
| 7 | ITL ratios 1.49x/1.52x/1.62x/1.92x | Last one rounded up | 1.496 / 1.524 / 1.627 / 1.916 -> range **1.49x to 1.91x** |
| 7 | TTFT "28 to 35% lower" | Upper bound rounded up | 34.2 / 28.3 / 29.7 / 33.0% -> **28.3 to 34.2%** |
| 7 | rate-32 "ITL p50 400 to 800 ms" | Rounded | 397.7 to 799.3 ms (tok/s 273 to 558 is correct: 273.3 to 558.2) |
| 9 | 3B BF16 ITL 11.4/11.8/12.3/14.3 | Source unknown | Mean of the 4 valid runs (bf16-3b, compiled-rep1..3): 11.35/11.83/12.24/14.30. bf16-3b alone: 11.41/11.90/12.32/14.79. Neither matches the old line exactly |
| 9 | FP8 "1.5x to 1.75x faster" | Upper bound rounded up | 1.523 / 1.601 / 1.611 / 1.747 -> **1.52x to 1.74x** |
| 9 | FP8 TTFT "34 to 41% lower" | OK (truncated) | 34.7 / 35.2 / 35.6 / 41.6% |
| 9 | 7B FP8 ITL 12.8/13.1/13.7/16.7 | Rounded | 12.74 / 13.10 / 13.76 / 16.72 |
| 9 | AWQ "34 to 37% below FP8" | Both bounds rounded up | 35.5 / 36.3 / 33.5% -> **33.5 to 36.3%** |
| 9 | AWQ "TTFT 14 to 18% worse" | Lower bound rounded up | 15.5 / 13.4 / 18.6% -> **13.4 to 18.6%** |
| 9 | AWQ perplexity 7.276 (+4.7% vs FP8) | Value OK; run was INVALID | runs.csv awq-7b ratio 15.959; flagged in 11 |
| 9 | 7B BF16 "cause not yet checked" | Superseded by 11 | Unpinned CPU weights cannot be copied during CUDA-graph capture |
| 10 | prefix TTFT 31/35/38 vs 133/138/544 | Rounded | 30.8/35.0/38.3 vs 133.0/138.2/543.9 ms; ratio **3.9x to 14.1x** (3.952 to 14.188) |
| 10 | "Caps of 128 and 64 avoided it" | **Superseded by 12** | n=1 closed-loop run per cap. Poisson cap 128: 1 of 2 runs collapsed (seqs128-ovl-rep2, INVALID 7.609) with 127-128 running, not ~255 |
| 12 | "10.8 to 11.9 ms in all earlier compiled 3B runs" | Wrong range | canary-before of all compiled 3B BF16 runs before 2026-10-08 22:24 (hr075, compiled-rep1..3, bf16-3b, prefix-on/off, seqs256/128/64): 10.57 to 11.97 ms |
| 13 | "10.8 to 11.9 ms before tonight" | Same error | 10.57 to 11.97 ms |
| 14 | "TTFT 2.4x to 4.1x worse" (my 2026-10-09 fix) | Lower bound rounded up | per repeat, rates 1-8: k=3 2.350 to 2.637, k=5 3.525 to 4.171 -> **2.3x to 4.1x** |
| 17 | "TTFT 28.4x to 42.5x" | Upper bound understated | 35.515 / 42.608 / 28.433 -> **28.4x to 42.6x** |
| 18 | prefix "3.9x to 14.3x" | Wrong (used rounded inputs) | **3.9x to 14.1x** |
| 18 | FP8 "1.53x to 1.74x" | Lower bound rounded up | **1.52x to 1.74x** |
| CLAUDE.md | "--max-num-seqs ... Caps 128 and 64 avoided it" | Not confirmed | Cause unknown (sections 12, 19); CLAUDE.md updated |

Consistency between old and new sections, beyond numbers:
- Section 5's refuted hypotheses still stand. New since: the collapse also happens at 128 running (12); it did not reproduce in 2
  profiled cap-256 attempts with Wallpaper Engine closed (19); post-overload GEMV slowdown is a new, untested lead (19).
- Section 8 lists bench_modes.py with two modes; it now has a third (text mode; see the "Methodology change" note between sections 13 and 14).
- Section 6 "corrections" and this section do not conflict.
- README.md and RESUME_BULLETS.md still contain several of the old values above (1.92x, 1.53x, 14.3x, 2.4x, 42.5x); not edited
  yet (out of scope for this pass). Section 21 lists them together with the contamination dependencies.

## 21. Contamination audit (2026-10-09, nothing rerun)
**Evidence available.** (a) Direct: four queries of Windows per-process GPU-engine counters (`Get-Counter '\GPU Engine(*)\Utilization
Percentage'`, processes above 0.5%), times from the session record. (b) Indirect: canary-before ITL p50 of fresh servers with the
same config (results/canary.csv, spec/canary.csv). Nothing else in the logs records other GPU users; anything not covered is "unknown".

| # | Time | Processes on the GPU besides WSL (vmwp) |
|---|---|---|
| Q1 | 10-08 ~22:47 (during seqs128-ovl-rep1/rep2) | wallpaper32 (Wallpaper Engine) videodecode 4.2% + 3.4%, 3D 0.9%; remoting_desktop (Chrome Remote Desktop) 3D 3.6%; dwm 3.6%; msedgewebview2 2.2%; claude 1.8% |
| Q2 | 10-08 ~23:15 (after the crash/reboot, before chunked runs) | wallpaper32 videodecode 4.3% + 3.1%, 3D 0.8%; dwm 2.7%; System copy 1.2%. No remoting_desktop |
| Q3 | 10-09 ~09:44 (before smoke/spec runs) | none above 0.5% |
| Q4 | 10-09 ~12:49 (before profiling) | remoting_desktop 2.1%; Razer.VideoRecorder 0.5%; dwm 1.7%; claude 1.4%; System 4.0% |

**How "~23%" was measured.** Not an A/B test. It compares canary-before ITL p50 of fresh compiled 3B BF16 servers in the Q1 window
with same-flag servers earlier the same evening: seqs128-ovl-rep1 14.12 and rep2 13.80 vs seqs128 11.48 ms (+23.0%, +20.2%);
seqs256-ovl 13.65 vs seqs256 11.23 ms (+21.5%). So the measured range is **+20.2% to +23.0%**, and "~23%" quoted the maximum.
During Q1 both Wallpaper Engine **and** Chrome Remote Desktop were active, so attributing it to Wallpaper Engine alone is not
established. With Wallpaper Engine only (Q2 window), chunked-on canaries were 12.11 to 12.24 ms: +7.8% to +8.9% vs the mean of the
six default-config canaries from 10-07/08 (11.235 ms), +1.2% to +2.3% vs their maximum (11.97).
**Also:** the clean 10-09 baseline (Q3; spec-base 10.09 and 10.27 ms, default config) is 8.6% to 10.2% *faster* than that 10-07/08 mean.
So the earlier runs may also have been slowed by something; whether they were is unknown.

**Runs and their status**
| Runs (section) | Time | Evidence | Status |
|---|---|---|---|
| Eager vs compiled, repeats (4, 7) | 10-05, 10-07 | none direct; compiled canaries 10.84 to 11.72 | unknown |
| Quantization 3B/7B, AWQ, 7B offload attempt (9) | 10-08 11:07 to 13:35 | none direct; bf16-3b canary 11.97 | unknown |
| Prefix caching, max-num-seqs closed loop, AWQ cap64 (10) | 10-08 21:37 (first server) to after 21:56 (last server start) | none direct; canaries 10.57 to 11.49 | unknown |
| 7B BF16 ppl + 3B eager ppl control (11) | 10-08 22:28 to 22:42 | none direct; eager canary 21.29 vs 15.89 to 18.88 in section 7 (but different flags) | likely contaminated (indirect); perplexity values not affected (fixed computation) |
| Overload cap 128 x2, cap-256 control (12) | 10-08 22:43 to 23:07 | Q1 | **contaminated** (Wallpaper Engine + Remote Desktop) |
| Chunked prefill 3 pairs (13) | 10-08 23:18 to 23:47 | Q2 at launch only; all 6 canaries 12.00 to 12.24 | **contaminated** (Wallpaper Engine at launch; elevated canaries throughout suggest it stayed) |
| Spec smoke, spec decoding (14) | 10-09 09:44 to 10:48 | Q3 at launch; base canaries 10.09, 10.27 | clean at launch; during: unknown |
| HF vs vLLM (15) | 10-09 ~10:48 to 10:54 | Q3 (1 h earlier) | unknown |
| TensorRT-LLM (16) | 10-09, attempt-1 log written 11:44, last log 12:16 | n/a | no measurements |
| mini_sched tests + benchmarks (17) | 10-09 ~12:20 to 12:43 | none; Q4 six minutes later saw Remote Desktop + Razer recorder | unknown (possibly contaminated) |
| Profiling healthy + 2 collapse attempts (19) | 10-09 12:49 to 13:00 | Q4 at launch | **contaminated (minor)**: Remote Desktop 2.1%, Razer 0.5% |

**Numbers in README.md and RESUME_BULLETS.md and what they depend on**
| Number (where) | Runs | Contamination | Other issue (section 20) |
|---|---|---|---|
| Compiled vs eager 1.49x to 1.92x (README, bullet 1) | section 7 | unknown | upper bound should be 1.91x |
| FP8 ITL table, 1.53x to 1.74x, 7B FP8 / AWQ ITL (README, bullet 2) | section 9 | unknown | lower bound should be 1.52x |
| Perplexities 6.8988 / +0.73% / +5.46% / +0.92% (README, bullet 2) | sections 9, 11 | 7B BF16 run likely contaminated; perplexity is not a timing number | 7B BF16 and AWQ runs INVALID |
| HF vs vLLM 2.12x to 2.87x, TPOT 2.7x to 2.9x (README, bullet 5) | section 15 | unknown | none |
| mini_sched tok/s, TTFT 28.4x to 42.5x, 12x KV memory (README, bullet 3) | section 17 | unknown | upper bound should be 42.6x; KV memory ratio does not depend on timing |
| Allocator stall 112.7 -> 310.6 tok/s (bullet 4) | section 17, two separate runs 10 min apart | unknown for both | cross-run comparison |
| Speculative decoding +28.1% to +34.6% TPOT, TTFT 2.4x to 4.1x, acceptance 0.50 to 0.52 (README, bullet 6) | section 14 | clean at launch, unknown during; both arms interleaved | TTFT lower bound should be 2.3x; acceptance rates do not depend on timing |
| Prefix caching 3.9x to 14.3x (README, bullet 7) | section 10 | unknown | upper bound should be 14.1x |
| Chunked prefill "no measurable difference" (README) | section 13 | **contaminated**; pairs interleaved, so the relative claim stands, absolute values are not comparable | none |
| Overload collapse "1 of 2 runs at cap 128" (README) | section 12 | **contaminated**: the cap-128 collapse happened with Wallpaper Engine + Remote Desktop on the GPU | contention as a trigger is untested |
| Decode-step breakdown 8.51 / 3.41 / 11.70 ms, GEMV +15% (README) | section 19 | **contaminated (minor)**; before/after comparison is within the same server | profiler overhead |
| "Wallpaper Engine made fresh canaries ~23% slower" (README Method) | section 12 | n/a | measured +20.2% to +23.0% with Remote Desktop also active; attribution to Wallpaper Engine alone not established |

## 22. Methodology change: idle-GPU preflight in run_config.sh (2026-10-09)
Before starting a server, run_config.sh now takes 10 one-second nvidia-smi samples and refuses to run if any sample has
utilization.gpu > 3%, memory.used > 2048 MiB, or an unreadable value. Decoder/encoder utilization is printed, not checked.
No measurement code changed, so no configs need rerunning. Known limits, measured while writing it:
- Wallpaper Engine passed both thresholds in the logs we have: idle memory 1,950 MiB at Q2 (section 21), and its video decode runs on
  the decoder engine, which utilization.gpu does not count. Utilization at Q2 was not recorded.
- At idle on 10-09 (no vLLM, Remote Desktop 0.5% and Razer.VideoRecorder 0.8% on the GPU), 10 samples read 0,0,1,0,1,1,1,1,1,6 %:
  one 6% spike would have refused the run.

## 23. Methodology change: stricter preflight with retries (2026-10-09; supersedes the thresholds in section 22)
The preflight moved to gpu_preflight.sh (run_config.sh calls it before starting a server; it also runs on its own).
One attempt = 10 one-second nvidia-smi samples; it fails if any sample has utilization.gpu > 3%, **utilization.decoder > 0%**
(catches Wallpaper Engine's video decode, which utilization.gpu does not count), **memory.used > 1500 MiB** (clean idle on
10-09: ~420 to 1,050 MiB; Wallpaper Engine idle on 10-08: 1,950 MiB), or an unreadable value. A failed attempt prints every
failing sample with the reason, waits 20 s and re-samples; the run is refused only if all 3 attempts fail. No measurement code
changed; no reruns needed for this change.
Standalone test (10-09, no vLLM running): attempt 1 FAILED (util 2 to 11%, 8 of 10 samples > 3%; memory 661 to 699 MiB; decoder 0%),
attempt 2 PASSED (util 0 to 2%, memory 601 to 605 MiB, decoder 0%). Windows users seen earlier the same day: Remote Desktop,
Razer.VideoRecorder (section 21, Q4).

## 24. Clean-rerun plan (2026-10-09, NOT run; script: queue_clean.sh)
Goal: rerun the headline vLLM experiments under the section-23 preflight, with A/B arms interleaved in one session, so
README/RESUME_BULLETS numbers no longer rest on runs whose contamination is "unknown" (section 21). Results go to
results/clean/ (results/clean/spec/ for text mode, which has extra columns), so nothing in results/ is overwritten.
Perplexity is not rerun (a fixed computation; contention changes timing, not log-probs). No overload or rate-16 collapse
configs are included (crash risk, section 12; not needed for the headline numbers).

Per-config wall time below = gap between consecutive server-log start times (results/server_*_<timestamp>.log), which includes
startup, both canaries, the sweep and shutdown; "derived" marks configs with no directly comparable old timing.
| # | Experiment | Config (run_config.sh) | Old sessions | Feeds | Old wall time per config | Planned runs | Est. GPU time |
|---|---|---|---|---|---|---|---|
| A | Eager vs compiled vs 3B FP8 (sections 4, 7, 9) | all via serve_generic.sh MODEL=3B: eager EXTRA="--enforce-eager"; compiled no EXTRA; fp8 EXTRA="--quantization fp8"; RATES=1,4,8,16, no OVERLOAD | 10-05 15:30 to after 15:55 (last server start); 10-07 22:02 to 23:09; 10-08 12:14 to 13:12 | README eager/compiled table and quant table 3B rows; bullets 1, 2 (ITL ratios); CLAUDE.md findings | compiled 3B standard sweep: 4.9 to 5.3 min (chunked-on/off, 10-08 23:18 to 23:47); 3B FP8: 4.8 to 5.0 (10-08 12:57 to 13:12); eager: old runs all had OVERLOAD=1 (8.1 to 11.6 min), no clean timing, derived: assumed = compiled 4.9 to 5.3 | 3 triplets = 9 | 43.8 to 46.8 min |
| B | 7B FP8 vs 7B AWQ (section 9) | serve_generic.sh MODEL=7B EXTRA="--quantization fp8" RATES=1,4,8,12,16; MODEL=7B-AWQ RATES=1,4,8,12 (AWQ collapsed at 16) | 10-08 12:25 to 13:22 | README quant table 7B rows (bullet 2 uses only 7B perplexities) | 7B FP8 at 1,4,8,16: 4.9 min (10-08 13:02, 13:12); AWQ at 1,4,8,12: 4.8 min (10-08 13:17); 7B FP8 with rate 12 added: derived ~5.5 | 3 pairs = 6 | ~30.9 min |
| C | Prefix caching on vs off (section 10) | MODE=prefix LEVELS=2,4,8; on = EXTRA="--enable-prefix-caching"; off = EXTRA="--no-enable-prefix-caching" | 10-08 21:37 to 21:43 | README other findings; bullet 7 | 3.0 min each (10-08 21:37, 21:40) | 3 pairs = 6 | 18.0 min |
| D | Speculative decoding (section 14) | MODE=text LEVELS=1,4,8, OUTDIR=results/clean/spec; base; draft 0.5B k=3; k=5 | 10-09 09:44 to 10:48 | README spec section (TPOT, TTFT, acceptance); bullet 6 | base at 1,4,8,16: 4.3 and 4.6 min; spec runs with rate 16 took 6.0 to 25.8 min because of the collapse; at 1,4,8 only: derived <= 4.6 | 3 triplets = 9 | <= 41.4 min |
| | Preflight | gpu_preflight.sh before each of 30 runs | | | 10 s (pass) to ~70 s (3 attempts) per run | 30 | 5.0 to 35.0 min |
| | **Total** | | | | | **30 runs** | **~2 h 19 to 2 h 52** |

Not in this plan: continuous vs static batching (mini_sched, does not use vLLM); HF vs vLLM (in-process, not run_config.sh;
run_hf_vs_vllm.sh would need the preflight added first); chunked prefill (README claim is relative and interleaved); overload
collapse and profiling (crash risk; separate decision). Claims that stay on old runs: "AWQ collapsed at 16", "spec collapses at
16", 7B FP8 at rate 16 only if the B block fails.

**Proposed order** (arms rotate inside each block so drift does not always hit the same arm; the queue stops on a failed run,
a preflight refusal, or two INVALID verdicts in a row):
1. A: rep1 eager, compiled, fp8; rep2 compiled, fp8, eager; rep3 fp8, eager, compiled
2. B: rep1 fp8-7b, awq; rep2 awq, fp8-7b; rep3 fp8-7b, awq
3. C: rep1 on, off; rep2 off, on; rep3 on, off
4. D: rep1 base, k3, k5; rep2 k5, base, k3; rep3 k3, k5, base
Before launch: Wallpaper Engine, Remote Desktop and recorders closed; PC set not to sleep for ~3 h.

**Amended 10-09 (still NOT run).** (1) Block A: eager and compiled now run through serve_generic.sh like the FP8 arm, so the arms
differ only in EXTRA. The old serve_eager.sh differed from serve_generic.sh in two more ways: it lacks the venv-nvcc/CUDA_HOME setup
and lacks `--kernel_config.enable_jit_warmup=False` (a required flag per CLAUDE.md). serve_compiled.sh differs from serve_generic.sh
only in the hard-coded model and log name. So the old eager arm (sections 4, 7) was not flag-matched; the new one is.
(2) Block C: the "on" arm passes `--enable-prefix-caching` explicitly (it was on by default: server_prefix-on_20261008_213717.log
shows enable_prefix_caching=True). (3) Block D: third triplet added (n=3 per arm); D is now 9 runs, total 30 runs.
Totals: low = 43.8 + 30.9 + 18.0 + 41.4 + 5.0 = 139.1 min; high = 46.8 + 30.9 + 18.0 + 41.4 + 35.0 = 172.1 min.
(4) Block B: 7B FP8 at rate 16 in the old runs was VALID 3/3 and did not collapse (results/main.csv fp8-7b, -rep2, -rep3:
480/480 ok each, 15.19 to 15.27 req/s achieved, ITL p50 15.49 to 18.91 ms; results/runs.csv canary ratios 0.998, 1.009, 1.001).
FP8 keeps RATES=1,4,8,12,16.

## 25. Methodology change: GPU sidecar logger in run_config.sh (2026-10-09)
After the preflight passes and until the EXIT trap, run_config.sh runs `nvidia-smi --query-gpu=... --format=csv -l 1` in the
background, appending to $OUTDIR/gpu_<NAME>.csv (if that file exists, gpu_<NAME>_<timestamp>.csv instead, so nothing is
overwritten). Fields: timestamp, utilization.gpu, utilization.decoder, utilization.encoder, memory.used, clocks.sm, clocks.mem,
power.draw, temperature.gpu, clocks_event_reasons.active (driver 616.64 supports this name; clocks_throttle_reasons.active is
its alias). The EXIT trap now runs stop_server, then kills the logger. Logging is best effort: errors are discarded and never
change run_config.sh's exit code. No measurement code changed. Possible cost: one NVML query per second on the host; its effect
on benchmark numbers was not measured. Purpose: open item 2 (clocks/power/temps through canary -> sweep -> canary).
Standalone test (no vLLM, 17:46:02 to 17:46:17, 15 rows): util 0 to 4%, decoder/encoder 0%, memory 617 to 629 MiB, SM 2662 to
2700 MHz, mem clock 15001 MHz, power 55.38 to 56.58 W, temp 49 to 50 C, event reasons 0x400 every row (`nvidia-smi -q -d
PERFORMANCE` at 17:46:35: only "Reliability" Active). Exit code of the test script was preserved (7) and no nvidia-smi process
was left. Also tested: unwritable OUTDIR and a failing nvidia-smi; the script continued and exited 0 both times.
Observation, not explained: at idle the GPU sat in P0 at ~2.7 GHz SM and ~56 W.

## 26. Preflight refusal exit code 75 + queue retry (2026-10-09)
gpu_preflight.sh now exits 75 (EX_TEMPFAIL) instead of 1 when all 3 attempts fail. run_config.sh passes 75 through and turns
any other preflight failure into 1. queue_clean.sh wraps every run in run(): on exit 75 (nothing was run) it waits 300 s and
retries the same run, up to 3 retries (4 attempts); 75 after the last retry, or any other nonzero exit, stops the queue as before.
Worst case per refused run: 4 x ~70 s preflight + 3 x 300 s wait = ~19.7 min, not included in the section-24 estimate.
No measurement code changed; no reruns needed. Tested without GPU work: stubbed run_config (exit codes 75,75,0 -> continues;
75 x4 -> stops after retry 3/3; 75,2 -> stops; 3 -> stops) and stubbed preflight (0/75/1 -> run_config exit 0/75/1). bash -n OK.

## 27. Clean queue stop and remaining runs (2026-10-10)
queue_clean.sh ran 25 of 30 runs (first server 09:44:17, results/server_clean-eager_20261010_094417.log) and stopped at
11:40:43 with "QUEUE STOPPED: two INVALID in a row" (results/queue_clean.log). All 21 runs in results/clean/runs.csv are VALID.
The two INVALID runs are both k=5 speculative decoding (results/clean/spec/runs.csv):
| run | canary before (ms) | canary after (ms) | ratio | verdict |
|---|---|---|---|---|
| clean-spec-k5-rep1 | 45.11 | 62.03 | 1.375 | INVALID |
| clean-spec-k5-rep2 | 45.56 | 62.30 | 1.367 | INVALID |
At level 8, k5 rep1 saturated: 367.45 tok/s, 2.87 req/s achieved, 93.0 s wall for a 30 s window, TTFT p50 1,409 ms (results/clean/spec/
modes.csv), vs base rep1 1,110.70 tok/s at the same level. **Tag: clean-spec-k5-rep1 and -rep2 are "saturation evidence at level 8,
run verdict INVALID".** They are kept in the CSVs and never used for level-1/4 numbers. The next server's canary-before (rep2 45.56 ms
vs rep1 45.11) shows no carry-over between fresh servers.
Remaining runs: queue_clean_rest.sh (same helpers: preflight, retry on 75, GPU sidecar), OUTDIR results/clean/spec, order
base rep2, k5 rep1, k3 rep2, k5 rep2, k3 rep3, base rep3, k5 rep3. base/k3 at LEVELS=1,4,8; k5 at LEVELS=1,4 only, named
clean-spec-k5-l14-rep1..3 (these are the valid k5 runs for levels 1 and 4). Estimate: 7 x ~3.9 min (gaps between the 10-10
base/k3/k5 server-log start times: 3.8, 3.9 min) + 7 preflights (10 to 70 s) = ~28.5 to 35.5 min (derived).

## 28. Clean rerun results (2026-10-10)
**Runs.** queue_clean.sh 09:44 to 11:40 (25 runs, stopped on two INVALID in a row, section 27), queue_clean_rest.sh finished
12:35:11 (7 runs, all VALID; results/queue_clean_rest.log). 32 runs total, 30 VALID, 2 INVALID (clean-spec-k5-rep1/-rep2, tagged
"saturation evidence at level 8, run verdict INVALID"). Preflight in the rest-queue: 3 runs passed on attempt 2 or 3, and one was
refused (all 3 attempts failed, 12:13:59) and passed on the first retry 300 s later. Every failing sample was utilization.gpu 4 to 9%; decoder 0% and
memory under the limit throughout. Sidecar logs: decoder/encoder 0 in every sample of all 32 runs (results/clean/gpu_summary.md).
Full tables: results/clean/summary_20261010_123633.md (summarize_clean.py; results/clean/summary.md is the first output, which
missed the INVALID rows; kept, not overwritten). Values are median [min-max] of n=3 VALID reps, truncated.

**Headline numbers (clean)**
| Experiment | 1 / 4 / 8 / 16 (A, B: req/s; C: level 2 / 4 / 8; D: level 1 / 4 / 8) | Source |
|---|---|---|
| A ITL p50 eager (ms) | 16.70 / 20.16 / 20.52 / 24.31 | results/clean/main.csv |
| A ITL p50 compiled (ms) | 10.32 / 10.51 / 10.89 / 12.40 | same |
| A ITL p50 3B FP8 (ms) | 6.97 / 7.05 / 7.05 / 7.57 | same |
| A eager / compiled ITL p50 | 1.618x / 1.917x / 1.884x / 1.959x | same |
| A compiled TTFT p50 vs eager | -33.0% / -39.0% / -37.0% / -33.6% | same |
| A compiled / FP8 ITL p50 | 1.479x / 1.490x / 1.543x / 1.638x | same |
| A FP8 TTFT p50 vs compiled | -35.0% / -30.2% / -32.1% / -37.9% | same |
| B ITL p50 7B FP8 (ms), rates 1/4/8/12/16 | 11.79 / 12.00 / 12.42 / 13.41 / 14.21 | same |
| B ITL p50 7B AWQ (ms), rates 1/4/8/12 | 7.62 / 7.84 / 8.64 / 13.23 | same |
| B AWQ TTFT p50 vs FP8, rates 1/4/8/12 | +19.7% / +17.8% / +21.7% / +66.4% | same |
| C TTFT p50 on / off (ms) | 28.04 / 121.05; 32.82 / 126.21; 36.63 / 295.22 | results/clean/modes.csv |
| C off / on TTFT p50 | 4.316x / 3.844x / 8.057x | same |
| D TPOT p50 base / k3 / k5 (ms) | 10.81 / 13.64 / 20.43; 11.41 / 14.16 / 20.27; 12.51 / 16.02 / (k5 not run at 8) | results/clean/spec/modes.csv |
| D TPOT k3 vs base | +26.1% / +24.0% / +28.0% | same |
| D TPOT k5 vs base | +88.9% / +77.6% / - | same |
| D TTFT p50 k3 / base; k5 / base | 2.430x / 2.381x / 2.528x; 3.561x / 3.443x / - | same |
| D draft acceptance (per rep, all levels) | k3 0.506 to 0.517; k5 0.367 to 0.382 | same |

**k=5 saturation.** At level 8, k5 rep1 saturated: 367.45 tok/s (base median 1,108.55), 2.87 req/s achieved, TTFT p50 1,409.13 ms,
p99 22,614.57 ms. k5 rep2 at level 8 did NOT saturate the same way: 970.72 tok/s, TTFT p50 330.55 ms, p99 686.60 ms (base median
47.84 / 76.24), so it was degraded but not collapsed. Both had canary-after 1.37x canary-before (INVALID). So "k=5 saturates at 8 req/s"
holds for 1 of 2 runs; "k=5 degrades the server at level 8" holds for 2 of 2. The new k5 reps (1, 4 only) were VALID, ratios 1.039 to 1.057.

**GPU sidecar (results/clean/gpu_summary.md, summarize_gpu.py).** No run flagged: busy-sample median SM clock 2790 to 2805 MHz
in every run (within 2% of sibling reps), and clocks_event_reasons.active was 0x400 in 100% of samples of every run (the value
seen with only "Reliability" Active, section 25; no power-cap, thermal or slowdown bits). Max temperature 46 to 64 C.
**Eager variance** (canary-before ITL p50 16.58 / 19.20 / 19.86 ms vs compiled 10.11 to 10.25; results/clean/runs.csv): the GPU logs
do not explain it. In the 30 s after server start (the canary-before window, canary_gpu_windows.py), the 3 eager reps ran at SM median
2805 to 2812 MHz with GPU utilization median 40 to 43%, vs compiled 80 to 82%. So in eager mode the GPU is idle over half the time and
the spread does not come from GPU clocks, throttling or other GPU users. The eager step time is set by host-side work (Python kernel
launches), so the likely source is the host CPU (clocks/scheduling on Windows + WSL2), which nothing logs. Not established.

**Old 14.1x prefix-caching number.** It is level 8 of the single old run pair (results/modes.csv, prefix-off and prefix-on, 10-08
21:37 and 21:40, both VALID): TTFT p50 543.94 / 38.34 ms = 14.188x. Clean: on 36.63 ms (old 38.34, similar), off 295.22 ms [280.96
to 301.21]. The difference is entirely the "off" arm at level 8. The old off run at 8 was congested: ITL p50 142.35 ms (clean 21.80 to
25.46), 7.42 req/s achieved (clean 7.61 to 7.64). Without caching, level 8 means ~8 x 1,543 prompt tokens/s of prefill, close to the
server's limit, where queueing makes TTFT very sensitive to small slowdowns. The old session was ~9% slower at the lower levels too
(off TTFT p50 133.04 / 138.25 at 2 / 4 vs clean 121.05 / 126.21; old prefix-off canary-before 10.97 ms vs clean 9.92 to 10.21).
Likely cause: a slightly slower system that session (contamination status "unknown", section 21) pushed a near-saturated
config into queueing, and n=1 could not show it. Not proven. n_req also differed (230 old vs 232 clean at level 8).

**Old vs clean** (old = README / RESUME_BULLETS as of 2026-10-09; old values were means of runs, clean = median of 3, truncated)
| Claim (where) | Old | Clean | Same / changed |
|---|---|---|---|
| Compiled vs eager ITL p50 speedup, rates 1-16 (README table, bullet 1) | 1.49x to 1.92x (16.9/11.3, 18.0/11.8, 19.9/12.2, 27.1/14.1 ms) | 1.61x to 1.95x (16.70/10.32, 20.16/10.51, 20.52/10.89, 24.31/12.40) | changed (higher; eager now flag-matched, section 24) |
| Compiled TTFT p50 lower (README) | 28 to 35% | 33.0% to 39.0% | changed |
| 3B BF16 ITL p50 1/4/8/16 (README quant table) | 11.3 / 11.8 / 12.2 / 14.3 ms | 10.32 / 10.51 / 10.89 / 12.40 | changed (faster) |
| 3B FP8 ITL p50 1/4/8/16 (README) | 7.4 / 7.4 / 7.6 / 8.2 ms | 6.97 / 7.05 / 7.05 / 7.57 | changed (faster) |
| 3B FP8 vs BF16 ITL speedup (README, bullet 2) | 1.53x to 1.74x (corrected 1.52x, section 20) | 1.47x to 1.63x | changed (lower) |
| 3B FP8 TTFT lower (CLAUDE.md findings) | 34.7 to 41.6% | 30.2% to 37.9% | changed |
| 7B FP8 ITL p50 1/4/8/16 (README) | 12.7 / 13.1 / 13.8 / 16.7 ms | 11.79 / 12.00 / 12.42 / 14.21 (12 req/s: 13.41) | changed (faster) |
| 7B AWQ ITL p50 1/4/8, 12 (README) | 8.2 / 8.3 / 9.1, 16.8 at 12 (1 valid run) | 7.62 / 7.84 / 8.64, 13.23 at 12 (n=3) | changed (faster) |
| AWQ faster at low load, TTFT worse (README) | qualitative | ITL p50 1.43x to 1.54x faster at 1-8, equal at 12 (1.013x); TTFT p50 +17.8% to +66.4% | same (now quantified) |
| AWQ collapsed at 16 (README) | 2 runs | not rerun | not retested |
| Prefix caching TTFT p50 reduction (README, bullet 7) | 3.9x to 14.3x (corrected 14.1x); 31/35/38 vs 133/138/544 ms | 3.84x to 8.05x; 28.04/32.82/36.63 vs 121.05/126.21/295.22 | changed (upper bound 14.1x -> 8.05x) |
| Spec k=3 TPOT slower (README, bullet 6) | +28.1% to +34.6% | +24.0% to +28.0% (levels 1/4/8) | changed (smaller) |
| Spec k=5 TPOT slower (README) | +90.9% to +99.2% | +77.6% to +88.9% (levels 1/4 only) | changed (smaller; level 8 not comparable) |
| Spec TTFT worse (README) | 2.4x to 4.1x (corrected 2.3x lower bound) | 2.38x to 3.56x (k3 1/4/8, k5 1/4) | changed (upper bound) |
| Draft acceptance k=3 (README) | 0.50 to 0.52 | 0.506 to 0.517 | same |
| Spec "collapses at 16" (README) | both configs | not rerun at 16; k5 saturated at 8 in 1 of 2 runs | not retested at 16 |
| Compiled 3B canary-before ITL p50 (LAB_LOG 20) | 10.57 to 11.97 ms | 10.11 to 10.25 | changed (faster, tighter) |
Not rerun, unchanged as claims: perplexities, HF vs vLLM, mini_sched, overload collapse, profiling, chunked prefill.
README.md and RESUME_BULLETS.md NOT edited (waiting for Alex).

## 29. Clean rerun plan 2 (2026-10-10, NOT run) + methodology change: preflight/sidecar for non-run_config runners
Experiments that feed README.md / RESUME_BULLETS.md and are not in results/clean/. Status from section 21. GPU time from old
timestamps (server-log name = start, file mtime = end; or gaps between consecutive output files).
| Experiment (section) | Numbers it feeds | Status | Old GPU time | Preflight + sidecar before today | Now |
|---|---|---|---|---|---|
| HF vs vLLM (15) | README engines table (HF/vLLM tok/s, 2.12x to 2.87x), TPOT 2.7x to 2.9x, TTFT 1.05x to 1.43x, identical greedy outputs; bullet 5 | unknown | per process from token-JSON mtimes (10-09): vllm 10:49:07 -> 10:51:13 (2.1 min), hf 10:51:13 -> 10:52:08 (0.9), vllm 10:52:08 -> 10:54:00 (1.9); ~2.8 to 3.0 min per hf+vllm pair | no (own memory check only) | run_hf_vs_vllm.sh: yes |
| mini_sched continuous vs static (17) | README batching table (TTFT, tok/s), TTFT 28.4x to 42.5x; paged KV 12.0x to 12.8x; bullet 3 | unknown (possibly contaminated) | poisson run <= 4.2 min (steps log mtime 12:39:38 -> poisson log mtime 12:43:51, 10-09; upper bound) | no (plain python) | run_mini_sched.sh: yes |
| mini_sched per-step speed (17) | README engines table "mini_sched tok/s" 17.5 / 132.6 / 364.4 / 531.0 | unknown | unknown (start not logged) | no | run_mini_sched.sh: yes; not in queue_clean2.sh (proposal) |
| mini_sched allocator stall (17) | bullet 4: 112.7 -> 310.6 tok/s, 32 of 313 steps > 0.5 s | unknown, two runs 10 min apart | n/a | no | not rerun: needs a run without expandable_segments (code change), decision for Alex |
| Profiling breakdown (19) | README: 11.70 / 8.51 / 3.41 ms, GEMV +15% after overload | contaminated (minor) | healthy: server log 12:49:18 -> mtime 12:52:20 = 3.0 min; collapse attempts ~4 min each | no | profile_run.sh: yes. Only PHASE=healthy planned (collapse = crash risk) |
| Chunked prefill (13) | README "no measurable difference" | contaminated | 6 runs 23:18:08 -> 23:47:53 = 29.7 min (4.85 to 5.3 each) | yes (run_config.sh) | unchanged; not in queue_clean2.sh |
| Perplexity (9, 11) | README ppl column; bullet 2: 6.8988, +0.73%, +5.46%, +0.92% | 7B BF16 run INVALID and likely contaminated; AWQ ppl run INVALID; perplexity is not a timing number | 7B BF16 offload: 22:28:39 -> 22:38:41 = 10.0 min; 3B eager control 3.5 min | yes (run_config.sh PPL=1) | unchanged; 7B BF16 in queue_clean2.sh |
| TensorRT-LLM (16) | none (README says it never ran) | no numbers | n/a | no | run_trtllm.sh: yes |

**Methodology change (runners only; no measurement code changed except one optional argument):**
- New lab_gpu.sh (sourced): `lab_preflight NAME` runs gpu_preflight.sh and exits 75 on refusal, 1 on other failure;
  `lab_sidecar_start CSV` / `lab_sidecar_stop` = the run_config.sh sidecar (same fields, best effort, never overwrites).
- run_hf_vs_vllm.sh: preflight before every process, sidecar per process ($GPUDIR/gpu_<label>.csv). New single-run mode
  `./run_hf_vs_vllm.sh BACKEND LABEL` with OUT=<csv>. Refuses a label whose results/hf_vs_vllm_tokens_<label>.json exists
  (hf_vs_vllm.py writes that path unconditionally, so a reused label would have overwritten a results file).
- New run_mini_sched.sh LABEL <bench_sched args>: preflight, sidecar, refuses a label already in $OUT.
- mini_sched/bench_sched.py: new `--policies` (default "continuous,static" = the old fixed order), so reps can rotate
  which policy runs first at each rate. Arrival times are seeded per rate, unchanged.
- profile_run.sh: preflight + sidecar ($DIR/gpu_<name>.csv), refuses an existing results/profiles/<name>/bench.csv.
- run_trtllm.sh: preflight + sidecar per process.
Tests (no GPU work): bash -n on all; lab_preflight with stub preflight exit 75/1/0 -> caller exits 75/1/continues; real
preflight + 4 s sidecar wrote 5 lines and left no nvidia-smi process; run_mini_sched.sh refused existing label mini-xseg.
One real preflight during the tests refused (all 3 attempts failed, reasons not captured); the next one passed on attempt 1.
No existing results depend on these runners' new behaviour; no reruns needed for the change itself.

## 30. Async scheduling (2026-10-10, no GPU run)
`vllm serve --help=all`: `--async-scheduling, --no-async-scheduling`, default None. vllm/config/vllm.py (vLLM 0.31.0, lines
1576 to 1635): None resolves to True unless pooling model, unsupported speculative method, disable_padded_drafter_batch,
an executor without support, ROCm DeepEP DBO, or pipeline parallelism; each of those logs a warning. Both uniproc and
multiproc executors return supports_async_scheduling() = True; draft_model speculative decoding is supported.
Server logs: no line in any of the 83 results/server_*.log mentions async scheduling (the engine-config line does not print
it, and none of the "will be disabled" warnings appear). **So async scheduling was ON in every vLLM server run, by default,
inferred from the code path; no log line states it directly.** Consequence: "compiled + --async-scheduling" equals the
baseline. The informative A/B is default vs `--no-async-scheduling`. Also: section 19's "GPU idle ~3.4 ms per step, no
CPU/GPU overlap" was measured with async scheduling on.

**queue_clean2.sh (2026-10-10, NOT launched).** 17 runs, order: E async default vs --no-async-scheduling (3 interleaved
pairs, RATES 1,4,8,16), F HF vs vLLM (3 interleaved pairs), G mini_sched poisson (3 reps, policy order rotated), H profile
healthy (1), I 7B BF16 perplexity (1, last). Estimate: E 6 x 4.9 to 5.0 min (clean compiled runs 10-10: server-log gaps 5.0,
5.0, 4.9) = 29.4 to 30.0; F 3 x 2.8 to 3.0 = 8.4 to 9.0; G 3 x <= 4.2 = <= 12.6; H 3.0; I 10.0; preflight 17 x 10 to 70 s =
2.8 to 19.8. Total ~66.2 to 84.4 min (~1 h 06 to 1 h 25), plus up to ~19.7 min per refused run. Block E deviates from the
request ("+ --async-scheduling"): async is already on by default (section 30), so the meaningful arm is --no-async-scheduling.

## 31. queue_clean2.sh final plan (2026-10-10)
Decisions (Alex): E = default vs --no-async-scheduling; add mini_sched --part steps; allocator stall as block J if non-invasive;
chunked prefill as block K. Order E, F, G, G2 (steps), H, K, J, I. 29 runs. `DRY=1 ./queue_clean2.sh` prints the command list.
- **J needs no code change:** mini_sched/__init__.py uses os.environ.setdefault, so PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
  in the environment restores the default allocator; default behaviour is unchanged. Nothing else in the repo or the shell
  environment sets PYTORCH_CUDA_ALLOC_CONF (checked). Caveat: J is compared with G about 1 h earlier, not interleaved.
- run_mini_sched.sh: the label check now also catches LABEL-steps (what --part steps writes); runner-only change.
- K: same flags as queue_chunked.sh (off = --no-enable-chunked-prefill, on = default), pair order now rotates.
Estimates (derived where marked):
| Block | Runs | Basis | Est. min |
|---|---|---|---|
| E async | 6 | clean compiled runs 4.9 to 5.0 each | 29.4 to 30.0 |
| F HF vs vLLM | 6 | 2.8 to 3.0 per pair (section 29) | 8.4 to 9.0 |
| G mini poisson | 3 | <= 4.2 each (section 29) | <= 12.6 |
| G2 mini steps | 3 | derived: 124.9 s measured walls (mini-xseg-steps) + 4 warm-up trials ~42 s (assumed = t0) + load ~5 s = ~2.9 min, close to the 3 min limit | ~8.6 |
| H profile | 1 | 3.0 (section 29) | 3.0 |
| K chunked | 6 | 4.85 to 5.3 each (section 29) | 29.1 to 31.8 |
| J mini no-xseg | 3 | derived: old stall run walls 329.8 s (label mini) + ~5 s load = ~5.6 min | ~16.6 |
| I 7B ppl | 1 | 10.0 (section 29) | 10.0 |
| Preflight | 29 | 10 to 70 s each | 4.8 to 33.8 |
| **Total** | **29** | | **~122.5 to 155.4 (~2 h 03 to 2 h 35)** |

## 32. Clean rerun results 2 (2026-10-10, queue_clean2.sh)
**Queue events.** First server 13:39:40 (results/server_clean2-async-default_20261010_133940.log), QUEUE FINISHED 15:42:37
(results/queue_clean2.log): 29 of 29 runs, ~2 h 03 (estimate was 2 h 03 to 2 h 35). All 13 canary-checked runs VALID (ratios 0.97 to
1.04, results/clean/runs.csv), including the 7B BF16 offload perplexity run (542.07 -> 562.55 ms, 1.038; it was INVALID at 1.32 on
10-08). Preflight: no refusals; 8 runs needed a 2nd attempt (7) or 3rd attempt (1); every failing sample was utilization.gpu 4%.
GPU sidecar (results/clean/gpu_summary_20261010_154555.md): no run flagged for clocks or event reasons (busy median SM 2790 to
2812 MHz, 0x400 in 100% of samples, decoder/encoder 0 everywhere). One implausible sample: 437 C in gpu_clean-mini-rep2.csv at
14:26:21.976 (0.25 s after the previous sample, between readings of 40 and 41 C), treated as a logging glitch, excluded from max temp
and flagged; the CSV is unchanged. Allocator-stall reps (J rep1, rep3) show more busy samples at lower power (busy median 79.8 and
94.8 W vs 136.3 W in rep2): GPU "busy" while doing less work.
Full tables: results/clean/summary_20261010_154516.md (summarize_clean.py). Profiling: results/profiles/summary.csv rows for
results/profiles/clean-healthy (profile_analyze.py).

**Block E, async scheduling** (default = on, section 30, vs --no-async-scheduling; 3B BF16 compiled; medians of 3, all VALID):
| Rate | ITL p50 on / off (ms) | TPOT p50 on / off | TTFT p50 on / off (ms) | TTFT p99 on / off | tok/s on / off |
|---|---|---|---|---|---|
| 1 | 10.07 / 10.56 | 10.34 / 10.81 | 37.17 / 28.54 | 49.63 / 41.01 | 121.39 / 121.42 |
| 4 | 10.15 / 10.77 | 10.78 / 11.44 | 40.46 / 31.54 | 55.91 / 43.71 | 506.81 / 505.61 |
| 8 | 10.60 / 11.19 | 11.49 / 12.23 | 42.41 / 33.00 | 61.19 / 51.09 | 881.36 / 879.61 |
| 16 | 11.87 / 12.63 | 13.98 / 15.32 | 51.62 / 39.89 | 101.38 / 82.81 | 1975.67 / 1969.38 |
**What E shows:** async scheduling is a trade-off on this setup, not a free win. With it on (the default), per-token latency is
lower at every rate (ITL p50 -4.6% to -6.0%, TPOT p50 -4.3% to -8.7%) but time to first token is higher (TTFT p50 +28.2% to +30.2%,
i.e. ~8.6 to 11.7 ms; TTFT p99 +19.7% to +27.9%). Per-rate ranges do not overlap for ITL p50 or TTFT p50. Throughput is unchanged
(open loop; both keep up). All earlier vLLM results were measured with it on. Mechanism of the TTFT increase not measured.
Profiling (clean, async on) still shows GPU idle 2.61 ms per batch-1 step, so async does not remove the CPU gap here.

**Block K, chunked prefill** (off = --no-enable-chunked-prefill, on = default): at rates 1 to 8 all off/on median ratios are within
0.96 to 1.04 and ranges overlap. At rate 16, chunked-on is slightly faster with non-overlapping ranges: ITL p50 11.74 [11.33-11.85]
vs 12.24 [12.03-12.24] ms (-4.0%), TPOT p50 13.95 [13.31-13.97] vs 14.29 [14.23-14.35] (-2.3%). Same confound as section 13 (off
also doubles the per-step token budget).

**Block J, allocator stall:** expandable segments off reproduced the rate-4 stall in 2 of 3 reps: continuous 140.63 and 137.24 tok/s
(TPOT p50 394.88, 432.93 ms) vs 320.90 in the 3rd rep, and 326.24 to 330.88 with expandable segments on (block G). The 2 stalled reps
both ran continuous before static at rate 4; in those processes static at rate 4 then also degraded (TTFT p50 19,935 and 19,764 ms
vs 5,690 to 6,374 elsewhere). The rep that ran static first stalled in neither. n=3, not interleaved with G (~1 h apart), and per-step
stall counts were not re-measured.

**Old vs clean, numbers not covered in section 28** (old = README / RESUME_BULLETS as of 2026-10-09; clean = medians, truncated)
| Claim (where) | Old | Clean | Same / changed |
|---|---|---|---|
| HF tok/s B=1/8/32/64 (README engines table) | 35.6 / 266.3 / 881.5 / 1522.0 (median of 6) | 33.71 / 242.74 / 844.92 / 1530.82 (median of 9) | changed (slightly) |
| vLLM tok/s B=1/8/32/64 (README) | 102.4 / 685.3 / 2130.2 / 3222.0 | 100.93 / 692.54 / 2148.65 / 3275.73 | changed (slightly) |
| vLLM speedup vs HF (README, bullet 5) | 2.87x / 2.57x / 2.42x / 2.12x (2.1x to 2.9x) | 2.99x / 2.85x / 2.54x / 2.13x (2.13x to 2.99x) | changed (higher at B=1-32) |
| Decode TPOT ratio HF/vLLM (README, bullet 5) | 2.7x to 2.9x | 2.74x to 3.03x | changed (upper bound) |
| Prefill TTFT ratio HF/vLLM (README) | 1.05x to 1.43x | 1.04x to 1.16x | changed (upper bound much lower) |
| Identical greedy outputs B=1 (README, bullet 5) | 128/128 | 128/128 in all 6 clean processes, and equal to the old output | same |
| mini_sched tok/s B=1/8/32/64 (README engines table) | 17.5 / 132.6 / 364.4 / 531.0 | 18.00 / 134.45 / 383.76 / 533.31 (median of 9) | changed (slightly) |
| TTFT p50 continuous / static, rates 1/2/4 (README batching table) | 141/5017, 145/6164, 188/5345 ms | 120.74/4682.48, 137.89/5109.43, 157.31/5850.52 | changed |
| tok/s continuous / static (README batching table) | 95.5/96.2, 215.0/178.4, 310.6/300.1 | 96.98/85.11, 221.79/201.81, 328.14/277.29 | changed (static lower) |
| Continuous vs static TTFT reduction (README, bullet 3) | 28.4x to 42.5x (corrected 42.6x, section 20) | 37.05x to 38.78x | changed (narrower) |
| Paged vs reserved KV (README, bullet 3) | 12.0x to 12.8x | 11.79x to 12.83x per run (medians 12.23x to 12.82x) | changed; **and the old "12.0x" was rounded up: old rate-1 value 45,056 / 3,760 = 11.982x** (results/mini_sched.csv, mini-xseg) |
| Allocator stall, 112.7 -> 310.6 tok/s at 4 req/s (bullet 4) | 1 run each, 10 min apart | off 140.63 / 137.24 / 320.90 (stall in 2 of 3) vs on 326.24 to 330.88 | same direction; stall is intermittent |
| 32 of 313 steps > 0.5 s (bullet 4) | measured once | not re-measured | not retested |
| Decode step at batch 1: period / GPU busy / GPU idle (README) | 11.70 / 8.51 / 3.41 ms | 10.83 / 8.12 / 2.61 ms (277 steps) | changed |
| GEMV +15% after overload (README) | collapse-phase profile | not rerun (no overload) | not retested |
| Chunked prefill "no measurable difference" (README) | 3 pairs, contaminated | none at rates 1-8; at 16 chunked-on ITL p50 -4.0%, TPOT -2.3% (ranges do not overlap) | changed (small effect at 16) |
| 7B BF16 perplexity (README, bullet 2) | 6.8988 (canary INVALID) | 6.8988 (canary VALID, 1.038) | same; now from a VALID run |
| 7B FP8 +0.73%, AWQ +5.46% vs 7B BF16 (README, bullet 2) | base 6.8988 | base unchanged; FP8/AWQ ppl not rerun | same (AWQ ppl run still INVALID) |
| Spec base 2,103 tok/s at 16, both spec configs collapse at 16 (README) | section 14 | not rerun (no overload) | not retested |
| Overload collapse claims (README) | section 12 | not rerun (no overload) | not retested |
README.md and RESUME_BULLETS.md NOT edited (waiting for Alex).
