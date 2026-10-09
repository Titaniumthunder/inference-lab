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

## Findings so far (details in LAB_LOG.md sections 1-19; corrections in section 20, contamination audit in section 21)
Ratios below are truncated, never rounded up.
- Compiled vs eager (3 repeats each): ITL p50 1.49x to 1.91x faster, TTFT 28.3 to 34.2% lower.
- 3B FP8 vs BF16: ITL 1.52x to 1.74x faster, TTFT 34.7 to 41.6% lower, perplexity +0.92%.
- 7B perplexity vs 7B BF16 (6.8988, eager + CPU offload, canary INVALID): FP8 +0.73%, AWQ +5.46% (AWQ perplexity run also INVALID).
  AWQ is faster at low load, TTFT worse, and collapsed at rate 16 with default settings.
- Prefix caching (best-case shared prefix): TTFT 3.9x to 14.1x lower.
- Overload collapse: ROOT CAUSE UNKNOWN. "--max-num-seqs 128 fixes it" is NOT confirmed: the closed-loop evidence was n=1 per cap,
  and one of two later Poisson runs with cap 128 still collapsed (LAB_LOG section 12). It did not reproduce in 2 profiled attempts
  at cap 256 on 2026-10-09 (section 19). After a heavy overload, batch-1 cuBLAS GEMV kernels ran ~15% slower (section 19); untested lead.
- The PC crashed (hard, needed reboot) during a collapsed overload run on 2026-10-08 ~23:07. Treat deliberate collapse runs as risky.
- Wallpaper Engine (video decode on the GPU) slowed some runs on 2026-10-08: fresh compiled-3B canaries ~23% slower (section 21 says
  exactly how that was measured and which runs it affects).
- Profiling: in a batch-1 decode step (11.70 ms) the GPU sits idle ~3.4 ms while the CPU prepares the next step (no overlap);
  this plus kernels below peak bandwidth explains the old "unexplained ~4.3 ms" (section 19).
- Speculative decoding with a 0.5B draft is slower than none on realistic text (TPOT +28.1% to +34.6% at k=3).
- Rate-32 overload numbers are not reproducible. Do not use them as results. TensorRT-LLM never ran (section 16).

## Remaining work
Items 1-9 of the original list are done (TensorRT-LLM stopped without numbers). Open:
1. Fix README.md / RESUME_BULLETS.md numbers flagged in LAB_LOG sections 20-21 (waiting for Alex).
2. Log GPU clocks/power/temps at 1 Hz through canary -> overload -> canary (tests the GEMV-slowdown lead).
3. Does GPU contention trigger the collapse? Needs deliberate overload runs (crash risk); Alex decides.
4. Optional: long-prompt chunked-prefill test; TensorRT-LLM with Ray.

## Stop and ask Alex when
- A config is INVALID twice in a row, or a run OOMs unexpectedly.
- A methodology change is needed.
- Anything would delete or overwrite results.

## Style
Alex prefers short answers for simple things, real explanation when needed, tables for comparisons, and being told plainly when he or the code is wrong.
After each experiment: append a short section to LAB_LOG.md (numbers, validity, caveats, open questions), then summarize in a small table.
