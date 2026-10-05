#!/usr/bin/env python3
"""
Open-loop LLM serving benchmark for OpenAI-compatible /v1/completions endpoints
(vLLM, SGLang, llama.cpp server, ...).

Requests arrive as a Poisson process at a fixed rate (open loop), so the server
cannot slow the arrival of load. That is what exposes queueing and the
latency-throughput knee. Use --rate inf for a closed-loop saturation test
(combine with --max-concurrency).

Metrics per run:
  TTFT  time to first token (ms)          -> prefill + queueing
  ITL   inter-token latency (ms)          -> decode speed as the user sees it
  TPOT  (e2e - ttft) / (out_tokens - 1)   -> per-request average decode time
  E2E   full request latency (ms)
  throughput: output tokens/s and requests/s over the whole run
  optional: USD per million output tokens (--usd-per-hour)

Caveat: one streamed chunk is treated as one token. Servers that bundle several
tokens per chunk (e.g. some speculative-decoding configs) make ITL look larger
and rarer; use TPOT for cross-config comparison in that case.
"""
import argparse
import asyncio
import csv
import json
import os
import random
import time
import uuid

import aiohttp

WORDS = ("the of and to in is that for it as with was on be by this at from or "
         "an are not but they which one had all their there when can more if "
         "system memory signal clock cache thread queue buffer kernel tensor "
         "layer token batch latency bandwidth compute register pipeline").split()


def make_prompt(n_words, rng):
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


async def one_request(session, url, model, prompt, out_len):
    payload = {
        "model": model, "prompt": prompt,
        "max_tokens": out_len, "min_tokens": out_len, "ignore_eos": True,
        "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft, last, itls = None, t0, []
    n_out, n_in, ok, err = 0, None, True, None
    try:
        async with session.post(url, json=payload) as r:
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:200]}")
            async for raw in r.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    n_out = obj["usage"]["completion_tokens"]
                    n_in = obj["usage"]["prompt_tokens"]
                ch = obj.get("choices") or []
                if ch and ch[0].get("text"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    else:
                        itls.append(now - last)
                    last = now
    except Exception as e:  # noqa: BLE001 - record and continue
        ok, err = False, str(e)
    return {"ok": ok and ttft is not None, "err": err, "ttft": ttft, "itls": itls,
            "e2e": time.perf_counter() - t0, "n_out": n_out, "n_in": n_in}


def n_requests(args, rate):
    """Requests per rate: fixed count if given, else enough to cover --duration seconds."""
    if args.num_requests:
        return args.num_requests
    if rate == float("inf"):
        return args.min_requests
    return max(args.min_requests, int(round(rate * args.duration)))


async def run_once(args, rate):
    rng = random.Random(args.seed)
    url = args.base_url.rstrip("/") + "/v1/completions"
    # Unique per-run, per-request prefix: identical prompts across runs would otherwise hit vLLM's
    # prefix cache (on by default) and make runs depend on what the previous run left in the cache.
    nonce = uuid.uuid4().hex[:8]
    prompts = [f"[{nonce}-{i}] " + make_prompt(args.input_words, rng)
               for i in range(n_requests(args, rate))]
    sem = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                     timeout=timeout) as s:
        for _ in range(args.warmup):  # warm caches / CUDA graphs, not measured
            await one_request(s, url, args.model, prompts[0], min(args.output_len, 16))

        async def guarded(p):
            if sem is None:
                return await one_request(s, url, args.model, p, args.output_len)
            async with sem:
                return await one_request(s, url, args.model, p, args.output_len)

        tasks, t_start = [], time.perf_counter()
        for p in prompts:
            tasks.append(asyncio.create_task(guarded(p)))
            if rate != float("inf"):
                await asyncio.sleep(rng.expovariate(rate))
        results = await asyncio.gather(*tasks)
        dur = time.perf_counter() - t_start
    return results, dur


def summarize(results, dur, rate, args):
    good = [r for r in results if r["ok"]]
    out_tok = sum(r["n_out"] for r in good)
    ttft = [r["ttft"] * 1e3 for r in good]
    itl = [x * 1e3 for r in good for x in r["itls"]]
    tpot = [(r["e2e"] - r["ttft"]) / max(r["n_out"] - 1, 1) * 1e3 for r in good]
    e2e = [r["e2e"] * 1e3 for r in good]
    row = {
        "label": args.label, "rate": rate, "max_conc": args.max_concurrency or 0,
        "n_req": len(results), "wall_s": dur,
        "ok": len(good), "failed": len(results) - len(good),
        "out_tok_s": out_tok / dur if dur else 0, "req_s": len(good) / dur if dur else 0,
    }
    for name, xs in (("ttft", ttft), ("itl", itl), ("tpot", tpot), ("e2e", e2e)):
        for p in (50, 95, 99):
            row[f"{name}_p{p}_ms"] = pct(xs, p)
    if args.usd_per_hour and out_tok:
        row["usd_per_M_out_tok"] = args.usd_per_hour * dur / 3600 / out_tok * 1e6
    errs = {r["err"] for r in results if r["err"]}
    return row, errs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True, help="model name as the server reports it")
    ap.add_argument("--rates", default="1,2,4,8",
                    help="comma list of req/s, or 'inf' for closed loop")
    ap.add_argument("--num-requests", type=int, default=0,
                    help="fixed requests per rate; 0 = size each run by --duration")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="seconds of arrivals per rate (so high rates reach steady state)")
    ap.add_argument("--min-requests", type=int, default=100,
                    help="floor on requests per rate, for meaningful p99")
    ap.add_argument("--input-words", type=int, default=200, help="~prompt length (words)")
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--max-concurrency", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="run", help="config name, e.g. fp8-prefixcache-on")
    ap.add_argument("--usd-per-hour", type=float, default=0.0,
                    help="GPU $/hr (cloud price, or power cost + amortization)")
    ap.add_argument("--out", default="results.csv", help="CSV, appended to")
    args = ap.parse_args()

    rows = []
    for tok in args.rates.split(","):
        rate = float("inf") if tok.strip() == "inf" else float(tok)
        results, dur = asyncio.run(run_once(args, rate))
        row, errs = summarize(results, dur, rate, args)
        rows.append(row)
        print(f"[{args.label}] rate={tok:>4}  ok={row['ok']:>3}/{len(results)}  "
              f"{row['out_tok_s']:7.1f} tok/s  "
              f"TTFT p50/p99 {row['ttft_p50_ms']:7.0f}/{row['ttft_p99_ms']:7.0f} ms  "
              f"ITL p50/p99 {row['itl_p50_ms']:6.1f}/{row['itl_p99_ms']:6.1f} ms")
        for e in list(errs)[:3]:
            print(f"    error: {e}")

    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
