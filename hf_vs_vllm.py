#!/usr/bin/env python3
"""Hugging Face `generate` vs the vLLM offline engine (LLM.generate) vs TensorRT-LLM, one backend per process.

Both backends get identical token-ID prompts (exactly --input-tokens tokens of random words with a
unique prefix, so vLLM's prefix cache never hits), greedy decoding, exactly --output-len new tokens.
No HTTP server is involved on either side. For every batch size B and trial:
  ttft_ms   wall time of a 1-token generation of a fresh batch of B prompts (prefill + first token)
  total_s   wall time of a full --output-len generation of another fresh batch of B prompts
  tpot_ms   (total - ttft) / (output_len - 1), per decode step
  out_tok_s B * output_len / total
One row per trial is appended to --out. The output token IDs of the first B=1 sequence are saved to
results/hf_vs_vllm_tokens_<label>.json so the two backends' greedy outputs can be compared.
Run it through run_hf_vs_vllm.sh (same WSL PATH fix as the serve scripts).
"""
import argparse, csv, json, os, random, time

WORDS = ("the of and to in is that for it as with was on be by this at from or "
         "an are not but they which one had all their there when can more if "
         "system memory signal clock cache thread queue buffer kernel tensor "
         "layer token batch latency bandwidth compute register pipeline").split()


def prompts(tok, b, n_tok, tag):
    """b prompts of exactly n_tok tokens; deterministic in tag, so both backends see the same IDs."""
    rng = random.Random(tag)
    out = []
    for i in range(b):
        text = f"[{tag}-{i}] " + " ".join(rng.choice(WORDS) for _ in range(n_tok + 50))
        ids = tok(text)["input_ids"][:n_tok]
        assert len(ids) == n_tok
        out.append(ids)
    return out


def hf_backend(args):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()

    def gen(batch, n):
        x = torch.tensor(batch, device="cuda")
        torch.cuda.synchronize()
        t = time.perf_counter()
        # repetition_penalty=1.0: Qwen's generation_config.json sets 1.05, which vLLM ignores
        # (--generation-config vllm), so HF must not apply it either.
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=n,
                             min_new_tokens=n, do_sample=False, temperature=None, top_p=None,
                             top_k=None, repetition_penalty=1.0, pad_token_id=0)
        torch.cuda.synchronize()
        return time.perf_counter() - t, out[:, x.shape[1]:].tolist()
    return gen


def vllm_backend(args):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=0.85,
              generation_config="vllm", kernel_config={"enable_jit_warmup": False})

    def gen(batch, n):
        sp = SamplingParams(max_tokens=n, min_tokens=n, ignore_eos=True, temperature=0)
        t = time.perf_counter()
        outs = llm.generate([TokensPrompt(prompt_token_ids=b) for b in batch], sp, use_tqdm=False)
        return time.perf_counter() - t, [list(o.outputs[0].token_ids) for o in outs]
    return gen


def trtllm_backend(args):
    # TensorRT-LLM 1.x LLM API, default PyTorch backend, CUDA graphs for the measured batch sizes.
    # Runs from its own venv (~/.venv-trtllm), see run_trtllm.sh.
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
    sizes = [int(b) for b in args.batch_sizes.split(",")]
    llm = LLM(model=args.model, max_seq_len=4096, max_batch_size=max(sizes),
              cuda_graph_config=CudaGraphConfig(batch_sizes=sizes),
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.85))

    def gen(batch, n):
        sp = SamplingParams(max_tokens=n, min_tokens=n, ignore_eos=True, temperature=0)
        t = time.perf_counter()
        outs = llm.generate(batch, sp, use_tqdm=False)
        return time.perf_counter() - t, [list(o.outputs[0].token_ids) for o in outs]
    return gen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["hf", "vllm", "trtllm"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--batch-sizes", default="1,8,32,64")
    ap.add_argument("--input-tokens", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="results/hf_vs_vllm.csv")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    gen = {"hf": hf_backend, "vllm": vllm_backend, "trtllm": trtllm_backend}[args.backend](args)
    n, rows = args.output_len, []
    for b in map(int, args.batch_sizes.split(",")):
        gen(prompts(tok, b, args.input_tokens, f"warm{b}"), n)  # warmup, not measured
        for t in range(args.trials):
            ttft, _ = gen(prompts(tok, b, args.input_tokens, f"ttft{b}-{t}"), 1)
            total, out = gen(prompts(tok, b, args.input_tokens, f"full{b}-{t}"), n)
            assert all(len(o) == n for o in out), "wrong number of output tokens"
            if b == 1 and t == 0:
                with open(f"results/hf_vs_vllm_tokens_{args.label}.json", "w") as f:
                    json.dump(out[0], f)
            row = {"label": args.label, "backend": args.backend, "batch": b,
                   "input_tokens": args.input_tokens, "output_len": n, "trial": t,
                   "ttft_ms": ttft * 1e3, "total_s": total, "tpot_ms": (total - ttft) / (n - 1) * 1e3,
                   "out_tok_s": b * n / total}
            rows.append(row)
            print(f"[{args.label}] B={b:>3} trial {t}  TTFT {row['ttft_ms']:7.1f} ms  "
                  f"TPOT {row['tpot_ms']:6.2f} ms  {row['out_tok_s']:8.1f} tok/s", flush=True)

    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
