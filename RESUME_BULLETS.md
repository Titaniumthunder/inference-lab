# Resume bullet drafts (Alex edits these; every number is measured, source in brackets)

Pick 3 to 4. Numbers are from `results/*.csv` via the LAB_LOG sections cited; ratios were computed from those values without rounding up.

1. Built a reproducible LLM-serving benchmark for vLLM on an RTX 5080 (open-loop Poisson load, fresh server per config, before/after
   canary runs that reject degraded servers); showed torch.compile + CUDA graphs cut median inter-token latency 1.49x to 1.92x
   versus eager mode across 1 to 16 req/s (3 interleaved repeats). [LAB_LOG 7; results/main.csv, runs.csv]

2. Quantified quantization trade-offs on Qwen2.5 3B/7B: FP8 lowered median inter-token latency 1.53x to 1.74x at +0.92% WikiText-2
   perplexity (3B); 7B FP8 cost +0.73% and AWQ 4-bit +5.46% perplexity against a BF16 baseline obtained via CPU offload.
   [LAB_LOG 9, 11; results/main.csv, ppl.csv]

3. Implemented a continuous-batching inference engine with a paged KV cache in PyTorch (block allocator, preemption by recompute,
   token-budget scheduling); greedy outputs match Hugging Face generate (differences only at exact logit ties, also under forced
   preemption); continuous batching cut median time-to-first-token 28.4x to 42.5x versus static batching, and paged allocation used
   12x less peak KV memory than max-length reservation. [LAB_LOG 17; results/mini_sched.csv]

4. Diagnosed a 1 s-per-step stall in that engine to CUDA allocator segment growth under WSL2 (32 of 313 steps over 0.5 s);
   expandable segments removed it and raised throughput at 4 req/s from 112.7 to 310.6 tok/s. [LAB_LOG 17]

5. Measured vLLM at 2.1x to 2.9x the throughput of Hugging Face generate (batch 1 to 64) with identical greedy outputs, and traced
   the gap to decode (2.7x to 2.9x per-token time) rather than prefill. [LAB_LOG 15; results/hf_vs_vllm.csv]

6. Showed speculative decoding with a 0.5B draft model slows a 3B target by 28.1% to 34.6% per token on realistic text, and that the
   common random-word benchmark prompt inflates draft acceptance (0.77 to 0.92 per position vs ~0.5); added a realistic-text
   workload to the harness. [LAB_LOG 14; results/spec/modes.csv]

7. Prefix caching: 3.9x to 14.3x lower median time-to-first-token on a shared 1,540-token prefix (best case). [LAB_LOG 10; results/modes.csv]

Not usable as results: rate-32 overload numbers (not reproducible), TensorRT-LLM (never ran), and anything from an INVALID run
except the two perplexities flagged in LAB_LOG 11.
