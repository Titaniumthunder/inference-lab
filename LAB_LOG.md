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

## Plan as of 2026-10-05 (superseded by CLAUDE.md "Remaining work")
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
  from rounded values; pair-by-pair ranges are +25.3 to +38.7% and +89.6 to +108.7%); TTFT 2.4x to 4.1x
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
- Continuous batching cuts median TTFT 28.4x to 42.5x: a new request starts at the next step instead of waiting for the running batch
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
- Section 9 ITL means: 3B BF16 11.4/11.8/12.3/14.3 vs 11.3/11.8/12.2/14.3 recomputed over bf16-3b + compiled-rep1..3; 7B FP8
  12.8/13.1/13.7/16.7 vs 12.7/13.1/13.8/16.7. Within 0.1 ms; probably a different run set or rounding. FP8 speedup is 1.53x to 1.74x.
- Prefix caching TTFT reduction is 3.9x to 14.3x (138/35 at 4 req/s), not "4x to 14x" as CLAUDE.md says.
- Section 14 and 17 ratios corrected (see the notes there): spec TPOT +28.1 to +34.6% (k=3), TTFT 2.4x to 4.1x; batching TTFT 28.4x to 42.5x.
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
