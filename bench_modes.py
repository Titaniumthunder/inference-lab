#!/usr/bin/env python3
"""Extra workloads for Inference Lab.
--mode prefix : Poisson arrivals (levels = req/s). Every prompt = one SHARED long prefix
                + a short unique suffix. Tests prefix caching.
--mode closed : closed loop (levels = concurrent clients). Each client sends requests
                back to back, all prompts unique. Tests --max-num-seqs.
"""
import argparse, asyncio, csv, json, os, random, time
import aiohttp

WORDS = ("the of and to in is that for it as with was on be by this at from or "
         "an are not but they which one had all their there when can more if "
         "system memory signal clock cache thread queue buffer kernel tensor "
         "layer token batch latency bandwidth compute register pipeline").split()


def words(n, rng):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def new_session():
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                 timeout=aiohttp.ClientTimeout(total=None))


async def one_request(session, url, model, prompt, out_len):
    payload = {"model": model, "prompt": prompt, "max_tokens": out_len,
               "min_tokens": out_len, "ignore_eos": True, "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True}}
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
    except Exception as e:
        ok, err = False, str(e)
    return {"ok": ok and ttft is not None, "err": err, "ttft": ttft, "itls": itls,
            "e2e": time.perf_counter() - t0, "n_out": n_out, "n_in": n_in}


async def run_prefix(args, rate):
    rng = random.Random(args.seed)
    url = args.base_url.rstrip("/") + "/v1/completions"
    shared = words(args.prefix_words, rng)
    async with new_session() as s:
        for _ in range(args.warmup):
            await one_request(s, url, args.model, shared + " " + words(args.suffix_words, rng), 16)
        tasks, i, t0 = [], 0, time.perf_counter()
        while not (time.perf_counter() - t0 >= args.duration and i >= args.min_requests):
            prompt = shared + " [" + format(rng.getrandbits(32), "08x") + f"-{i}] " + words(args.suffix_words, rng)
            tasks.append(asyncio.create_task(one_request(s, url, args.model, prompt, args.output_len)))
            i += 1
            await asyncio.sleep(rng.expovariate(rate))
        results = await asyncio.gather(*tasks)
        return results, time.perf_counter() - t0


async def run_closed(args, conc):
    url = args.base_url.rstrip("/") + "/v1/completions"
    results = []
    async with new_session() as s:
        wr = random.Random(args.seed)
        for _ in range(args.warmup):
            await one_request(s, url, args.model, "[w" + format(wr.getrandbits(32), "08x") + "] " + words(args.input_words, wr), 16)
        t0 = time.perf_counter()
        stop_at = t0 + args.duration

        async def worker(w):
            r = random.Random(args.seed * 1000 + w)
            while time.perf_counter() < stop_at:
                prompt = "[" + format(r.getrandbits(32), "08x") + f"-{w}] " + words(args.input_words, r)
                results.append(await one_request(s, url, args.model, prompt, args.output_len))

        await asyncio.gather(*(worker(w) for w in range(conc)))
        return results, time.perf_counter() - t0


def summarize(results, dur, level, args):
    good = [r for r in results if r["ok"]]
    out_tok = sum(r["n_out"] for r in good)
    ttft = [r["ttft"] * 1e3 for r in good]
    itl = [x * 1e3 for r in good for x in r["itls"]]
    e2e = [r["e2e"] * 1e3 for r in good]
    nin = [r["n_in"] for r in good if r["n_in"]]
    row = {"label": args.label, "mode": args.mode, "level": level,
           "n_req": len(results), "ok": len(good), "failed": len(results) - len(good),
           "wall_s": round(dur, 1), "out_tok_s": out_tok / dur if dur else 0,
           "req_s": len(good) / dur if dur else 0, "prompt_tok_p50": pct(nin, 50)}
    for name, xs in (("ttft", ttft), ("itl", itl), ("e2e", e2e)):
        for p in (50, 95, 99):
            row[f"{name}_p{p}_ms"] = pct(xs, p)
    return row, {r["err"] for r in results if r["err"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["prefix", "closed"], required=True)
    ap.add_argument("--levels", required=True)
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--min-requests", type=int, default=60)
    ap.add_argument("--prefix-words", type=int, default=1500)
    ap.add_argument("--suffix-words", type=int, default=30)
    ap.add_argument("--input-words", type=int, default=200)
    ap.add_argument("--output-len", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="results/modes.csv")
    args = ap.parse_args()
    if not args.output_len:
        args.output_len = 32 if args.mode == "prefix" else 128

    rows = []
    for tok in args.levels.split(","):
        level = float(tok)
        if args.mode == "prefix":
            results, dur = asyncio.run(run_prefix(args, level))
        else:
            level = int(level)
            results, dur = asyncio.run(run_closed(args, level))
        row, errs = summarize(results, dur, level, args)
        rows.append(row)
        print(f"[{args.label}] {args.mode} level={tok:>4}  ok={row['ok']}/{row['n_req']}  "
              f"{row['out_tok_s']:7.1f} tok/s  prompt~{row['prompt_tok_p50']:.0f} tok  "
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
