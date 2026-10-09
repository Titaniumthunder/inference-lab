#!/usr/bin/env python3
"""Extra workloads for Inference Lab.
--mode prefix : Poisson arrivals (levels = req/s). Every prompt = one SHARED long prefix
                + a short unique suffix. Tests prefix caching.
--mode closed : closed loop (levels = concurrent clients). Each client sends requests
                back to back, all prompts unique. Tests --max-num-seqs.
--mode text   : Poisson arrivals (levels = req/s). Realistic text: WikiText-2 *train*
                paragraphs (ppl.py uses test), chat template, "continue this article".
                Every request gets a different paragraph (no prefix-cache hits beyond the
                shared chat header). Adds TPOT columns and, if the server runs speculative
                decoding, the draft acceptance rate from /metrics for each level.
"""
import argparse, asyncio, csv, json, os, random, re, sys, time
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


INSTRUCTION = ("Continue the following Wikipedia article. Keep writing in the same style; "
               "do not summarize it or comment on it.\n\n")


def text_prompts(args):
    """Chat-formatted continuation prompts, shuffled with --seed (same prompts for every config)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    paras = []
    for line in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]:
        line = line.strip()
        if not line or line.startswith("="):
            continue
        # light detokenization of WikiText's " @-@ ", " , ", "( x )", " 's"
        line = line.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
        line = re.sub(r" ([,.;:!?)\]])", r"\1", line)
        line = re.sub(r"([(\[]) ", r"\1", line)
        line = re.sub(r" ('s|'t|'re|'ve|'ll|'d|'m)\b", r"\1", line)
        w = line.split()
        if len(w) >= args.text_words:
            paras.append(" ".join(w[:args.text_words]))
    random.Random(args.seed).shuffle(paras)
    return [tok.apply_chat_template([{"role": "user", "content": INSTRUCTION + p}],
                                    tokenize=False, add_generation_prompt=True) for p in paras]


SPEC = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens")


async def spec_counters(session, base_url):
    """vLLM speculative-decoding counters from /metrics, summed over label sets (0 if spec is off)."""
    async with session.get(base_url.rstrip("/") + "/metrics") as r:
        text = await r.text()
    tot = dict.fromkeys(SPEC, 0.0)
    for line in text.splitlines():
        name = line.split("{")[0].split(" ")[0]
        if name.endswith("_total") and name[:-6] in tot:
            tot[name[:-6]] += float(line.rsplit(" ", 1)[1])
    return tot


async def run_text(args, rate, pool, cursor):
    url = args.base_url.rstrip("/") + "/v1/completions"
    rng = random.Random(args.seed + int(rate * 1000))
    async with new_session() as s:
        for p in pool[-args.warmup:] if args.warmup else []:  # warmup prompts are never measured
            await one_request(s, url, args.model, p, 16)
        before = await spec_counters(s, args.base_url)
        tasks, t0 = [], time.perf_counter()
        while not (time.perf_counter() - t0 >= args.duration and len(tasks) >= args.min_requests):
            if cursor[0] >= len(pool) - args.warmup:
                sys.exit("text prompt pool exhausted (would reuse prompts and hit the prefix cache)")
            tasks.append(asyncio.create_task(one_request(s, url, args.model, pool[cursor[0]], args.output_len)))
            cursor[0] += 1
            await asyncio.sleep(rng.expovariate(rate))
        results = await asyncio.gather(*tasks)
        dur = time.perf_counter() - t0
        after = await spec_counters(s, args.base_url)
        return results, dur, {k: after[k] - before[k] for k in SPEC}


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


def summarize(results, dur, level, args, spec=None):
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
    if args.mode == "text":
        # TPOT, because with speculative decoding one streamed chunk can carry several tokens (ITL is per chunk)
        tpot = [(r["e2e"] - r["ttft"]) / max(r["n_out"] - 1, 1) * 1e3 for r in good]
        for p in (50, 95, 99):
            row[f"tpot_p{p}_ms"] = pct(tpot, p)
        spec = spec or dict.fromkeys(SPEC, 0.0)
        drafts, dtok, acc = (spec[k] for k in SPEC)
        row["spec_draft_tokens"] = int(dtok)
        row["spec_accepted_tokens"] = int(acc)
        row["spec_accept_rate"] = acc / dtok if dtok else ""
        row["spec_mean_accept_len"] = 1 + acc / drafts if drafts else ""
    return row, {r["err"] for r in results if r["err"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["prefix", "closed", "text"], required=True)
    ap.add_argument("--levels", required=True)
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--min-requests", type=int, default=60)
    ap.add_argument("--prefix-words", type=int, default=1500)
    ap.add_argument("--suffix-words", type=int, default=30)
    ap.add_argument("--input-words", type=int, default=200)
    ap.add_argument("--text-words", type=int, default=150, help="text mode: words of article per prompt")
    ap.add_argument("--output-len", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="results/modes.csv")
    args = ap.parse_args()
    if not args.output_len:
        args.output_len = 32 if args.mode == "prefix" else 128

    # refuse to append rows whose columns differ from the file's header (text mode adds columns)
    if os.path.exists(args.out):
        with open(args.out) as f:
            header = f.readline().strip().split(",")
        if header != list(summarize([], 0, 0, args)[0].keys()):
            sys.exit(f"{args.out} has different columns for mode {args.mode}; write to another file")

    pool, cursor = (text_prompts(args), [0]) if args.mode == "text" else (None, None)
    rows = []
    for tok in args.levels.split(","):
        level = float(tok)
        spec = None
        if args.mode == "prefix":
            results, dur = asyncio.run(run_prefix(args, level))
        elif args.mode == "text":
            results, dur, spec = asyncio.run(run_text(args, level, pool, cursor))
        else:
            level = int(level)
            results, dur = asyncio.run(run_closed(args, level))
        row, errs = summarize(results, dur, level, args, spec)
        rows.append(row)
        print(f"[{args.label}] {args.mode} level={tok:>4}  ok={row['ok']}/{row['n_req']}  "
              f"{row['out_tok_s']:7.1f} tok/s  prompt~{row['prompt_tok_p50']:.0f} tok  "
              f"TTFT p50/p99 {row['ttft_p50_ms']:7.0f}/{row['ttft_p99_ms']:7.0f} ms  "
              f"ITL p50/p99 {row['itl_p50_ms']:6.1f}/{row['itl_p99_ms']:6.1f} ms"
              + (f"  TPOT p50 {row['tpot_p50_ms']:6.1f} ms  accept {row['spec_accept_rate']}"
                 if args.mode == "text" else ""))
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
