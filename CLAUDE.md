# Inference Lab

Rigorous LLM-serving benchmarks on one RTX 5080 (16 GB), Windows + WSL2 Ubuntu 22.04, vLLM 0.31.0, venv at ~/.venv.
Results become resume bullets, so ONLY real measured numbers. Owner: Alex (LMU CompE senior, applying to ML-systems roles).

## Hard rules
- ONE GPU job at a time. Before any run: `pgrep -f "vllm serve"` must be empty, and the GPU must be idle: sample nvidia-smi for 10 s and
  refuse to run if utilization > 3% or memory used > 2048 MiB in any sample (run_config.sh enforces this). nvidia-smi in WSL sees
  Windows processes' memory and utilization too. Caveat: with Wallpaper Engine running, idle memory was 1,950 MiB, which passes a
  2048 MiB limit (LAB_LOG section 21); its video decode runs on the decoder engine, which utilization.gpu does not count.
- Fresh server per config. Use ./run_config.sh for every benchmark (canary before and after, verdict into runs.csv, auto-stops the server). Never hand-roll a benchmark.
- A config is valid only if canary ITL ratio (after/before) <= 1.3. Report INVALID as INVALID. Never drop it silently or average it in.
- Never invent, estimate or round up numbers. Every number in docs must come from results/*.csv or logs. Cite the file.
- Do not change methodology files (bench.py, bench_modes.py, ppl.py, run_config.sh, serve_*.sh) without saying so and logging it in LAB_LOG.md. Rerun configs that the change affects.
- Anything over ~2 minutes: `nohup ... > results/<name>.log 2>&1 &`, then poll with tail. Tell Alex when it finishes.
- Never delete or overwrite files in results/. Commit locally after each finished experiment. NEVER git push (Alex pushes with GitHub Desktop).
- Alex's PC must not sleep during queues. Remind him before long ones. Nothing else may use the GPU meanwhile.

## Environment quirks
- WSL PATH includes /mnt/c folders, which breaks torch inductor (PermissionError: nvcc). The serve scripts strip /mnt/ from PATH and use the venv nvcc. Keep that.
- Required server flags: `--kernel_config.enable_jit_warmup=False --generation-config vllm --gpu-memory-utilization 0.85 --max-model-len 4096`.
- serve_generic.sh reads env vars MODEL, EXTRA, TAG. Logs go to results/server_<TAG>_<timestamp>.log.
- run_config.sh usage:
  `[MODE=prefix|closed LEVELS=..] [RATES=1,4,8,16] [DURATION=30] [OVERLOAD=1] [PPL=1] [OUTDIR=results] MODEL=.. TAG=.. EXTRA=".." ./run_config.sh NAME ./serve_generic.sh`
  Outputs: results/main.csv, canary.csv, runs.csv, ppl.csv, modes.csv.
- sed pitfall: `&` in a replacement means the matched text, a literal needs `\&`. This already broke run_config.sh once. Prefer the Edit tool, then `bash -n`.
- Prompts in bench.py are random words with a unique per-request prefix (defeats prefix caching on purpose).

## Files
bench.py (open-loop Poisson), bench_modes.py (shared-prefix, closed-loop, realistic text), ppl.py (WikiText-2 perplexity), plot.py,
run_config.sh, serve_eager.sh, serve_compiled.sh, serve_generic.sh, queue_*.sh, hf_vs_vllm.py, run_hf_vs_vllm.sh, run_trtllm.sh,
profile_run.sh, profile_analyze.py, mini_sched/, RESUME_BULLETS.md, results/, LAB_LOG.md (running log), README.md.

## Findings so far (clean reruns: LAB_LOG 28, 32, 33; perplexity 34-36; older sections 1-21 are superseded where they differ)
Ratios below are ratios of medians, truncated, never rounded up; n=3 runs per arm unless stated. Same numbers as README.md /
RESUME_BULLETS.md (Sources table there).
- Compiled vs eager (3B BF16, 1-16 req/s): ITL p50 1.61x to 1.95x lower, TTFT p50 33.0% to 39.0% lower. Eager canary ITL p50
  spread 16.58 to 19.86 ms across reps (compiled 10.11 to 10.25); cause not established.
- 3B FP8 vs BF16: ITL p50 1.47x to 1.63x lower, TTFT p50 30.2% to 37.9% lower, perplexity +0.91% (8.0540 / 7.9808, truncated;
  LAB_LOG 35 corrects an earlier rounded-up value).
- 7B: FP8 perplexity +0.73% vs 7B BF16 (6.8988, eager + CPU offload, canary VALID in the clean rerun). AWQ perplexity NOT reported
  (its only run was INVALID; reruns refused by the preflight on 10-11 and 10-12). AWQ ITL p50 1.43x to 1.54x lower than FP8 at 1-8
  req/s, equal at 12; TTFT p50 +17.8% to +66.4% (1-12 req/s). AWQ not measured at 16 in the clean rerun.
- Prefix caching (best case, shared ~1,540-token prefix): TTFT p50 3.84x to 8.05x lower at 2-8 req/s.
- Async scheduling (on by default in vLLM 0.31.0): ITL p50 4.6% to 6.0% lower, TTFT p50 28.2% to 30.2% higher vs
  --no-async-scheduling. Cause not measured.
- Chunked prefill: no difference at 1-8 req/s; at 16 the ranges do not overlap (on ITL p50 11.74 vs off 12.24 ms).
- HF generate vs vLLM: vLLM 2.13x to 2.99x throughput (batch 1-64), identical greedy outputs (6 of 6 processes).
- mini_sched: continuous vs own static baseline TTFT p50 37.05x to 38.78x; paged vs own preallocation KV 11.79x to 12.83x.
  Allocator stall with expandable segments off: 4 of 7 runs (all with continuous first), 0 of 7 with them on; cause not established.
- Speculative decoding (0.5B draft, realistic text): TPOT +24.0% to +28.0% (k=3, 1/4/8 req/s), +77.6% to +88.9% (k=5, 1/4 req/s).
- Profiling (1 clean run, batch-1 decode, async on): step 10.83 ms, GPU busy 8.12, GPU idle 2.61 ms. Measurement only.
- Overload collapse: ROOT CAUSE UNKNOWN, not rerun under the clean protocol. "--max-num-seqs 128 fixes it" is NOT confirmed
  (LAB_LOG 12, 19). Do not use overload or rate-32 numbers as results.
- The PC crashed (hard, needed reboot) during a collapsed overload run on 2026-10-08 ~23:07. Treat deliberate collapse runs as risky.
- Video decode on the GPU (Wallpaper Engine) contaminated some 2026-10-08 runs (LAB_LOG 21); the preflight now refuses it
  (decoder > 0%). It refused on 10-11 and twice on 10-12 for that reason.
- TensorRT-LLM never ran (section 16).

## Remaining work
Items 1-9 of the original list are done (TensorRT-LLM stopped without numbers). Open:
1. Done 2026-10-11/12: README.md / RESUME_BULLETS.md rewritten from the clean reruns (LAB_LOG 28-36).
   Pending a GPU with no other users (preflight refused 10-11, 10-12): 7B AWQ perplexity rerun; python -m mini_sched.test_mini rerun.
2. 1 Hz GPU logging exists (run_config.sh / lab_gpu.sh sidecar); not yet run through canary -> overload -> canary (GEMV-slowdown lead).
3. Does GPU contention trigger the collapse? Needs deliberate overload runs (crash risk); Alex decides.
4. Optional: long-prompt chunked-prefill test; TensorRT-LLM with Ray.

## Stop and ask Alex when
- A config is INVALID twice in a row, or a run OOMs unexpectedly.
- A methodology change is needed.
- Anything would delete or overwrite results.

## Style
Alex prefers short answers for simple things, real explanation when needed, tables for comparisons, and being told plainly when he or the code is wrong.
After each experiment: append a short section to LAB_LOG.md (numbers, validity, caveats, open questions), then summarize in a small table.
