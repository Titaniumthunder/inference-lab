"""Benchmarks for the mini scheduler. Run from the repo root: python -m mini_sched.bench_sched --part steps|poisson
steps   : static batches of B identical-shape requests (section 15 workload: 256-token prompts, 128 new
          tokens), TTFT and TPOT per batch size, to place the engine next to HF and vLLM
poisson : Poisson arrivals at each rate for --duration s, continuous vs static policy, same model code.
          Arrivals stop after --duration; the run drains for at most --max-drain s, and requests not
          finished by then are reported as unfinished (not dropped silently).
Rows are appended to results/mini_sched.csv.
"""
import argparse, csv, os, random, sys, time

from transformers import AutoTokenizer

from .model import PagedQwen2
from .scheduler import Request, Scheduler

sys.path.insert(0, ".")
from hf_vs_vllm import prompts  # noqa: E402

KV_BYTES_PER_TOKEN = 2 * 36 * 2 * 128 * 2  # K+V, 36 layers, 2 KV heads, head_dim 128, BF16 (LAB_LOG section 2)


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def drive(model, reqs, args, policy, max_wall):
    """Real-time loop: requests become visible at their arrival offset. The static policy gets a budget
    that fits a whole batch of prompts in one step (it never admits mid-batch)."""
    budget = args.budget if policy == "continuous" else args.max_num_seqs * args.input_tokens
    s = Scheduler(model, max_num_seqs=args.max_num_seqs, max_num_batched_tokens=budget, policy=policy)
    pending = sorted(reqs, key=lambda r: r.arrival)
    t0 = time.perf_counter()
    while pending or s.has_work():
        now = time.perf_counter() - t0
        if now > max_wall:
            break
        while pending and pending[0].arrival <= now:
            s.add(pending.pop(0))
        if not s.step(now, lambda: time.perf_counter() - t0):
            if pending:
                time.sleep(max(0.0, min(0.005, pending[0].arrival - (time.perf_counter() - t0))))
            elif not s.has_work():
                break
    return s, time.perf_counter() - t0


def row(label, policy, level, reqs, sched, wall, args):
    fin = [r for r in reqs if r.finish_t is not None]
    ttft = [(r.first_token_t - r.arrival) * 1e3 for r in fin]
    tpot = [(r.finish_t - r.first_token_t) / (r.max_tokens - 1) * 1e3 for r in fin]
    st = sched.stats
    return {"label": label, "policy": policy, "level": level, "n_req": len(reqs), "finished": len(fin),
            "wall_s": round(wall, 2), "out_tok_s": sum(len(r.output) for r in fin) / wall,
            "ttft_p50_ms": pct(ttft, 50), "ttft_p99_ms": pct(ttft, 99),
            "tpot_p50_ms": pct(tpot, 50), "tpot_p99_ms": pct(tpot, 99),
            "steps": st["steps"], "preemptions": st["preemptions"], "peak_batch": st["peak_batch"],
            "peak_kv_tokens": st["peak_blocks"] * model_block_size,
            "contiguous_kv_tokens_same_peak": st["peak_batch"] * args.max_model_len}


def main():
    global model_block_size
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["steps", "poisson"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--num-blocks", type=int, default=4096)       # 65,536 tokens, 2.25 GiB of KV
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--budget", type=int, default=4096)           # max tokens per step
    ap.add_argument("--max-model-len", type=int, default=4096)    # for the contiguous-reservation comparison
    ap.add_argument("--input-tokens", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--batch-sizes", default="1,8,32,64")
    ap.add_argument("--rates", default="1,2,4")
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--max-drain", type=float, default=120)
    ap.add_argument("--label", default="mini")
    ap.add_argument("--out", default="results/mini_sched.csv")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = PagedQwen2(args.model, num_blocks=args.num_blocks)
    model_block_size = model.cache.block_size
    n = args.output_len
    rows = []
    if args.part == "steps":
        for b in map(int, args.batch_sizes.split(",")):
            args.max_num_seqs = max(args.max_num_seqs, b)
            for tag in ("warm", "t0", "t1", "t2"):
                ps = prompts(tok, b, args.input_tokens, f"ministep{b}-{tag}")
                reqs = [Request(i, p, n) for i, p in enumerate(ps)]
                s, wall = drive(model, reqs, args, "static", 1e9)
                if tag != "warm":
                    rows.append(row(f"{args.label}-steps", "static", b, reqs, s, wall, args))
                    print(f"B={b:>3} {tag}  TTFT p50 {rows[-1]['ttft_p50_ms']:7.1f} ms  TPOT p50 "
                          f"{rows[-1]['tpot_p50_ms']:6.2f} ms  {rows[-1]['out_tok_s']:7.1f} tok/s", flush=True)
    else:
        for rate in map(float, args.rates.split(",")):
            for policy in ("continuous", "static"):
                rng = random.Random(int(rate * 100))
                t, arr = 0.0, []
                while t < args.duration:
                    arr.append(t)
                    t += rng.expovariate(rate)
                ps = prompts(tok, len(arr), args.input_tokens, f"minipois{rate}")
                reqs = [Request(i, p, n, arrival=a) for i, (p, a) in enumerate(zip(ps, arr))]
                s, wall = drive(model, reqs, args, policy, args.duration + args.max_drain)
                rows.append(row(args.label, policy, rate, reqs, s, wall, args))
                r = rows[-1]
                print(f"rate {rate:>4} {policy:10s} finished {r['finished']}/{r['n_req']}  {r['out_tok_s']:7.1f} tok/s  "
                      f"TTFT p50/p99 {r['ttft_p50_ms']:8.0f}/{r['ttft_p99_ms']:8.0f} ms  TPOT p50 {r['tpot_p50_ms']:6.1f} ms  "
                      f"peak batch {r['peak_batch']}  preempt {r['preemptions']}", flush=True)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
