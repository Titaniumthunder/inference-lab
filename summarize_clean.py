#!/usr/bin/env python3
"""Summarize the clean reruns (LAB_LOG sections 24, 27, 28, 29 to 32). CPU only, reads CSVs, never touches the GPU.

Reads results/clean/{main,modes,runs,hf_vs_vllm,mini_sched,ppl}.csv and results/clean/spec/{modes,runs}.csv. For run_config.sh
experiments only runs with verdict VALID in the matching runs.csv are aggregated; INVALID runs are listed separately.
Experiments without a canary (HF vs vLLM, mini_sched) aggregate every row; n = reps / samples (each trial is a sample). Per experiment and rate/level it prints n reps,
median and min-max of each metric, and ratios between arms (ratio of medians). All math uses the unrounded CSV values.
Every printed number is TRUNCATED toward zero (never rounded up). Writes results/clean/summary.md (or a timestamped name
if that file exists, so nothing is overwritten) and prints the same text.
"""
import csv, math, os, re, statistics, sys, time
from collections import defaultdict

ROOT = "results/clean"
METRICS = [("itl_p50_ms", "ITL p50 ms"), ("itl_p99_ms", "ITL p99 ms"), ("ttft_p50_ms", "TTFT p50 ms"),
           ("ttft_p99_ms", "TTFT p99 ms"), ("tpot_p50_ms", "TPOT p50 ms"), ("out_tok_s", "tok/s")]

HF_METRICS = [("ttft_ms", "TTFT ms"), ("tpot_ms", "TPOT ms"), ("out_tok_s", "tok/s")]
MINI_METRICS = [("ttft_p50_ms", "TTFT p50 ms"), ("ttft_p99_ms", "TTFT p99 ms"), ("tpot_p50_ms", "TPOT p50 ms"),
                ("out_tok_s", "tok/s"), ("peak_batch", "peak batch"),
                (lambda r: float(r["contiguous_kv_tokens_same_peak"]) / float(r["peak_kv_tokens"]), "reserved/paged KV")]
STEP_METRICS = [("ttft_p50_ms", "TTFT p50 ms"), ("tpot_p50_ms", "TPOT p50 ms"), ("out_tok_s", "tok/s")]

# (title, csv dir, csv file, rate/level column, [(arm name, label regex[, policy])], [(numerator arm, denominator arm)],
#  optional dict: metrics (default METRICS), verdict (default True), note)
EXPERIMENTS = [
    ("A. Eager vs compiled vs 3B FP8 (3B, open-loop Poisson)", ROOT, "main.csv", "rate",
     [("eager", r"clean-eager-rep\d+"), ("compiled", r"clean-compiled-rep\d+"), ("fp8-3b", r"clean-fp8-3b-rep\d+")],
     [("eager", "compiled"), ("compiled", "fp8-3b"), ("eager", "fp8-3b")]),
    ("B. 7B FP8 vs 7B AWQ (open-loop Poisson)", ROOT, "main.csv", "rate",
     [("fp8-7b", r"clean-fp8-7b-rep\d+"), ("awq-7b", r"clean-awq-7b-rep\d+")],
     [("fp8-7b", "awq-7b")]),
    ("C. Prefix caching on vs off (3B, shared prefix, closed loop)", ROOT, "modes.csv", "level",
     [("prefix-off", r"clean-prefix-off-rep\d+"), ("prefix-on", r"clean-prefix-on-rep\d+")],
     [("prefix-off", "prefix-on")]),
    ("D. Speculative decoding (3B target, 0.5B draft, realistic text)", ROOT + "/spec", "modes.csv", "level",
     [("base", r"clean-spec-base-rep\d+"), ("k3", r"clean-spec-k3-rep\d+"), ("k5", r"clean-spec-k5-l14-rep\d+")],
     [("k3", "base"), ("k5", "base")]),
    ("E. Async scheduling: default (on) vs --no-async-scheduling (3B BF16 compiled, open-loop Poisson)", ROOT, "main.csv",
     "rate", [("async-default", r"clean2-async-default-rep\d+"), ("no-async", r"clean2-no-async-rep\d+")],
     [("no-async", "async-default")]),
    ("K. Chunked prefill off vs on (3B BF16 compiled, open-loop Poisson)", ROOT, "main.csv", "rate",
     [("chunked-off", r"clean2-chunked-off-rep\d+"), ("chunked-on", r"clean2-chunked-on-rep\d+")],
     [("chunked-off", "chunked-on")]),
    ("F. HF generate vs vLLM offline (3B BF16, static batches, 256-token prompts, 128 greedy tokens)", ROOT, "hf_vs_vllm.csv",
     "batch", [("hf", r"clean-hf-rep\d+"), ("vllm", r"clean-vllm-rep\d+")], [("hf", "vllm"), ("vllm", "hf")],
     dict(metrics=HF_METRICS, verdict=False, note="No canary (in-process). Each trial is a sample: 3 reps x 3 trials.")),
    ("G. mini_sched continuous vs static (Poisson, expandable segments on = default)", ROOT, "mini_sched.csv", "level",
     [("continuous", r"clean-mini-rep\d+", "continuous"), ("static", r"clean-mini-rep\d+", "static")],
     [("static", "continuous")],
     dict(metrics=MINI_METRICS, verdict=False, note="No canary. One row per rep, policy and rate. reserved/paged KV = "
          "contiguous_kv_tokens_same_peak / peak_kv_tokens.")),
    ("J. mini_sched allocator: expandable segments off (default PyTorch allocator) vs on (block G)", ROOT, "mini_sched.csv",
     "level", [("cont-xseg-on", r"clean-mini-rep\d+", "continuous"), ("cont-xseg-off", r"clean-mini-noxseg-rep\d+", "continuous"),
               ("static-xseg-on", r"clean-mini-rep\d+", "static"), ("static-xseg-off", r"clean-mini-noxseg-rep\d+", "static")],
     [("cont-xseg-off", "cont-xseg-on"), ("static-xseg-off", "static-xseg-on")],
     dict(metrics=MINI_METRICS, verdict=False, note="No canary. J ran ~1 h after G, not interleaved with it.")),
    ("L. Allocator A/B (block L): expandable segments on/off x policy order, mini_sched Poisson rate 4 only", ROOT,
     "mini_sched.csv", "level",
     [("cont on cs", r"clean3-alloc-on-cs-rep\d+", "continuous"), ("cont on sc", r"clean3-alloc-on-sc-rep\d+", "continuous"),
      ("cont off cs", r"clean3-alloc-off-cs-rep\d+", "continuous"), ("cont off sc", r"clean3-alloc-off-sc-rep\d+", "continuous"),
      ("static on cs", r"clean3-alloc-on-cs-rep\d+", "static"), ("static on sc", r"clean3-alloc-on-sc-rep\d+", "static"),
      ("static off cs", r"clean3-alloc-off-cs-rep\d+", "static"), ("static off sc", r"clean3-alloc-off-sc-rep\d+", "static")],
     [("cont off cs", "cont on cs"), ("cont off sc", "cont on sc")],
     dict(metrics=MINI_METRICS, verdict=False, note="No canary. 2 reps per cell (fresh process each); cs = continuous first, "
          "sc = static first. Metrics other than tok/s cover finished requests only; see failed/unfinished counts.")),
    ("G2. mini_sched per-step speed (static batches, section-15 workload)", ROOT, "mini_sched.csv", "level",
     [("mini-steps", r"clean-mini-steps-rep\d+-steps")], [],
     dict(metrics=STEP_METRICS, verdict=False, note="No canary. Each trial is a sample: 3 reps x 3 trials.")),
]


def trunc(x, d):
    """Truncate toward zero to d decimals (never rounds up in magnitude)."""
    if x is None or math.isnan(x):
        return "n/a"
    f = 10 ** d
    v = math.trunc(x * f) / f
    return f"{v:.{d}f}"


def verdicts(d):
    out = {}
    with open(f"{d}/runs.csv") as f:
        for r in csv.DictReader(f):
            out[r["name"]] = r["verdict"]
    return out


def load(d, fn):
    with open(f"{d}/{fn}") as f:
        return list(csv.DictReader(f))


def main():
    lines = []
    p = lines.append
    p("# Clean rerun summary (2026-10-10, clean reruns 1 and 2)")
    p("")
    p("Generated by summarize_clean.py from results/clean/{main,modes,runs,hf_vs_vllm,mini_sched,ppl}.csv and results/clean/spec/{modes,runs}.csv.")
    p("Only VALID runs are aggregated (where a canary exists). Values: median [min-max] over samples. Ratio = median(A) / median(B) per metric;")
    p("'B vs A' = (median(B) / median(A) - 1) x 100%. All numbers truncated toward zero, never rounded up.")
    invalid = []
    seen = set()
    for exp in EXPERIMENTS:
        title, d, fn, col, arms, pairs = exp[:6]
        opt = exp[6] if len(exp) > 6 else {}
        metrics = opt.get("metrics", METRICS)
        use_ver = opt.get("verdict", True)
        ver = verdicts(d) if use_ver else {}
        rows = load(d, fn)
        # data[arm][level][metric] -> list of values (one per VALID rep)
        data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        reps = defaultdict(lambda: defaultdict(list))
        fails = defaultdict(int)
        for r in (rows if use_ver else []):  # every non-VALID row in this file, listed once per file
            v = ver.get(r["label"], "MISSING")
            if v != "VALID" and (d, fn, r["label"], r[col]) not in seen:
                seen.add((d, fn, r["label"], r[col]))
                invalid.append((title, r["label"], r[col], v, r))
        for arm, rx, *pol in arms:
            for r in rows:
                if not re.fullmatch(rx, r["label"]) or (use_ver and ver.get(r["label"]) != "VALID"):
                    continue
                if pol and r.get("policy") != pol[0]:
                    continue
                lv = float(r[col])
                reps[arm][lv].append(r["label"])
                if "failed" in r:
                    fails[arm] += int(r["failed"])
                elif "finished" in r:
                    fails[arm] += int(r["n_req"]) - int(r["finished"])
                for m, name in metrics:
                    if callable(m):
                        data[arm][lv][name].append(m(r))
                    elif m in r and r[m] != "":
                        data[arm][lv][name].append(float(r[m]))
        p("")
        p(f"## {title}")
        p(f"Source: {d}/{fn}" + (f", verdicts {d}/runs.csv" if use_ver else "") + f". Column '{col}'. "
          + ("Failed/unfinished requests in aggregated runs: " + ", ".join(f"{a[0]} {fails[a[0]]}" for a in arms) + "."
             if fn != "hf_vs_vllm.csv" else ""))
        if opt.get("note"):
            p("Note: " + opt["note"])
        if "spec" in d:
            p("Note: in spec mode one streamed chunk can carry several accepted tokens, so ITL here is time between chunks, "
              "not per token; TPOT is the per-token metric.")
        if fn == "modes.csv" and "spec" not in d and use_ver:
            p("Note: prefix mode has no TPOT column (n/a).")
        p("")
        p("| arm | " + col + " | n | " + " | ".join(name for _, name in metrics) + " |")
        p("|---|---|---|" + "---|" * len(metrics))
        levels = sorted({lv for a in arms for lv in data[a[0]]})
        for lv in levels:
            for arm, *_ in arms:
                if lv not in data[arm]:
                    continue
                cells = []
                for _, m in metrics:
                    vals = data[arm][lv][m]
                    if not vals:
                        cells.append("n/a")
                    else:
                        cells.append(f"{trunc(statistics.median(vals), 2)} [{trunc(min(vals), 2)}-{trunc(max(vals), 2)}]")
                nrep, nsamp = len(set(reps[arm][lv])), len(reps[arm][lv])
                ncell = str(nrep) if nrep == nsamp else f"{nrep} / {nsamp}"
                p(f"| {arm} | {trunc(lv, 0)} | {ncell} | " + " | ".join(cells) + " |")
        for a, b in pairs:
            p("")
            p(f"Ratios {a} / {b} (and {b} vs {a}):")
            p("")
            p(f"| {col} | " + " | ".join(name for _, name in metrics) + " |")
            p("|---|" + "---|" * len(metrics))
            for lv in levels:
                if lv not in data[a] or lv not in data[b]:
                    continue
                cells = []
                for _, m in metrics:
                    va, vb = data[a][lv][m], data[b][lv][m]
                    if not va or not vb:
                        cells.append("n/a")
                        continue
                    ma, mb = statistics.median(va), statistics.median(vb)
                    cells.append(f"{trunc(ma / mb, 3)}x ({b} {'+' if mb >= ma else ''}{trunc((mb / ma - 1) * 100, 1)}%)")
                p(f"| {trunc(lv, 0)} | " + " | ".join(cells) + " |")
    p("")
    p("## I. Perplexity (clean rerun 2)")
    p("")
    p("Source: results/clean/ppl.csv, verdicts results/clean/runs.csv.")
    p("")
    p("| label | model | perplexity | tokens | canary verdict |")
    p("|---|---|---|---|---|")
    ver = verdicts(ROOT)
    for r in load(ROOT, "ppl.csv"):
        p(f"| {r['label']} | {r['model']} | {r['perplexity']} | {r['tokens']} | {ver.get(r['label'], 'MISSING')} |")
    p("")
    p("## INVALID runs (not aggregated)")
    p("")
    p("| experiment | label | level/rate | verdict | ok/n_req | out tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms |")
    p("|---|---|---|---|---|---|---|---|---|")
    for title, label, lv, v, r in invalid:
        p(f"| {title.split('.')[0]} | {label} | {lv} | {v} | {r['ok']}/{r['n_req']} | {trunc(float(r['out_tok_s']), 2)} | "
          f"{trunc(float(r['ttft_p50_ms']), 2)} | {trunc(float(r['ttft_p99_ms']), 2)} | {trunc(float(r['itl_p50_ms']), 2)} |")
    text = "\n".join(lines) + "\n"
    if "--no-write" in sys.argv:
        sys.stdout.write(text)
        return
    out = f"{ROOT}/summary.md"
    if os.path.exists(out):
        out = f"{ROOT}/summary_{time.strftime('%Y%m%d_%H%M%S')}.md"
    with open(out, "w") as f:
        f.write(text)
    sys.stdout.write(text)
    print(f"\n(wrote {out})")


if __name__ == "__main__":
    main()
