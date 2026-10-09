#!/usr/bin/env python3
"""Plot results/main.csv: latency and throughput vs arrival rate, one color per config label.

    python plot.py                         # all labels in results/main.csv
    python plot.py --labels eager-bf16-3b compiled-bf16-3b
"""
import argparse
import collections
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/main.csv")
ap.add_argument("--out", default="results/latency_throughput.png")
ap.add_argument("--output-len", type=int, default=128,
                help="output tokens per request (for the offered-load line)")
ap.add_argument("--labels", nargs="*", help="only plot these config labels")
ap.add_argument("--summary", action="store_true",
                help="write the README summary figures (results/summary_*.png) from all CSVs and exit")
args = ap.parse_args()

# Categorical slots 1-4 of the reference palette, fixed order; every series also gets its own marker.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, MUTED = "#0b0b0b", "#52514e"


def mean_by(rows, key_col, val_col):
    acc = collections.defaultdict(list)
    for r in rows:
        acc[float(r[key_col])].append(float(r[val_col]))
    ks = sorted(acc)
    return ks, [sum(acc[k]) / len(acc[k]) for k in ks]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)


def lines(ax, series):
    """series: list of (name, xs, ys); direct-labels the last point and keeps a legend."""
    for i, (name, xs, ys) in enumerate(series):
        ax.plot(xs, ys, "-", marker=MARKERS[i], color=SERIES[i], lw=2, ms=8,
                markeredgecolor="white", markeredgewidth=2, label=name)
        ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(series))


def summary():
    def read(p):
        with open(p) as f:
            return list(csv.DictReader(f))
    main_rows = read("results/main.csv")
    pick = lambda labels: [r for r in main_rows if r["label"] in labels]  # noqa: E731

    # 1. quantization: ITL p50 vs rate, mean of the VALID runs (LAB_LOG section 9; awq-7b-b is the valid AWQ sweep)
    groups = [("3B BF16", ["bf16-3b", "compiled-rep1", "compiled-rep2", "compiled-rep3"]),
              ("3B FP8", ["fp8-3b", "fp8-3b-rep2", "fp8-3b-rep3"]),
              ("7B FP8", ["fp8-7b", "fp8-7b-rep2", "fp8-7b-rep3"]),
              ("7B AWQ 4-bit", ["awq-7b-b"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    ser = []
    for name, labels in groups:
        xs, ys = mean_by(pick(labels), "rate", "itl_p50_ms")
        ser.append((name, xs, ys))
        print(f"quant {name}: " + ", ".join(f"r{x:g} {y:.1f}" for x, y in zip(xs, ys)))
    lines(ax, ser)
    style(ax, "Inter-token latency p50 by quantization (lower is better)", "arrival rate (req/s)", "ms")
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig("results/summary_quantization.png", dpi=140)

    # 2. HF vs mini scheduler vs vLLM: output tok/s by batch size, medians (sections 15 and 17)
    hv = read("results/hf_vs_vllm.csv")
    mini = [r for r in read("results/mini_sched.csv") if r["label"] == "mini-xseg-steps"]
    def med(rows, key, val):
        acc = collections.defaultdict(list)
        for r in rows:
            acc[int(float(r[key]))].append(float(r[val]))
        ks = sorted(acc)
        return ks, [sorted(acc[k])[len(acc[k]) // 2] if len(acc[k]) % 2 else
                    sum(sorted(acc[k])[len(acc[k]) // 2 - 1:len(acc[k]) // 2 + 1]) / 2 for k in ks]
    ser = [("HF generate", *med([r for r in hv if r["backend"] == "hf"], "batch", "out_tok_s")),
           ("mini_sched (this repo)", *med(mini, "level", "out_tok_s")),
           ("vLLM", *med([r for r in hv if r["backend"] == "vllm"], "batch", "out_tok_s"))]
    for name, xs, ys in ser:
        print(f"engines {name}: " + ", ".join(f"B{x} {y:.1f}" for x, y in zip(xs, ys)))
    fig, ax = plt.subplots(figsize=(8, 5))
    lines(ax, ser)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 32, 64])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    style(ax, "Output throughput, Qwen2.5-3B BF16, static batches", "batch size", "output tokens/s")
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig("results/summary_engines.png", dpi=140)

    # 3. speculative decoding: TPOT p50 at rates 1-8, mean of 2 repeats (section 14; rate 16 collapsed, left out)
    sp = [r for r in read("results/spec/modes.csv") if float(r["level"]) <= 8]
    ser = []
    for name, pre in (("no speculation", "spec-base-rep"), ("draft 0.5B, k=3", "spec-k3-rep"), ("draft 0.5B, k=5", "spec-k5-rep")):
        xs, ys = mean_by([r for r in sp if r["label"].startswith(pre)], "level", "tpot_p50_ms")
        ser.append((name, xs, ys))
        print(f"spec {name}: " + ", ".join(f"r{x:g} {y:.1f}" for x, y in zip(xs, ys)))
    fig, ax = plt.subplots(figsize=(8, 5))
    lines(ax, ser)
    style(ax, "Time per output token p50, realistic text (lower is better)", "arrival rate (req/s)", "ms")
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig("results/summary_speculative.png", dpi=140)

    # 4. continuous vs static batching in mini_sched: TTFT p50 (section 17, rerun label mini-xseg)
    ms = [r for r in read("results/mini_sched.csv") if r["label"] == "mini-xseg"]
    ser = []
    for name, pol in (("static batching", "static"), ("continuous batching", "continuous")):
        xs, ys = mean_by([r for r in ms if r["policy"] == pol], "level", "ttft_p50_ms")
        ser.append((name, xs, ys))
        print(f"batching {name}: " + ", ".join(f"r{x:g} {y:.0f}" for x, y in zip(xs, ys)))
    fig, ax = plt.subplots(figsize=(8, 5))
    lines(ax, ser)
    ax.set_yscale("log")
    ax.set_yticks([100, 200, 500, 1000, 2000, 5000, 10000])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    style(ax, "Time to first token p50, mini_sched (lower is better)", "arrival rate (req/s)", "ms (log scale)")
    fig.tight_layout()
    fig.savefig("results/summary_batching.png", dpi=140)
    print("wrote results/summary_{quantization,engines,speculative,batching}.png")


if args.summary:
    summary()
    raise SystemExit

by = collections.OrderedDict()
with open(args.csv) as f:
    for r in csv.DictReader(f):
        rate = float(r["rate"])
        if not math.isfinite(rate):
            continue
        if args.labels and r["label"] not in args.labels:
            continue
        r["rate"] = rate
        by.setdefault(r["label"], []).append(r)
for rows in by.values():
    rows.sort(key=lambda r: r["rate"])

COLORS = ["#c2410c", "#0369a1", "#15803d", "#7e22ce", "#a16207", "#be123c"]
all_rates = sorted({r["rate"] for rows in by.values() for r in rows})
fig, axes = plt.subplots(1, 3, figsize=(17, 5))


def draw(ax, p50, p99, title, ylabel):
    for i, (label, rows) in enumerate(by.items()):
        x = [r["rate"] for r in rows]
        c = COLORS[i % len(COLORS)]
        ax.plot(x, [float(r[p50]) for r in rows], "-o", color=c, lw=2, ms=5)
        ax.plot(x, [float(r[p99]) for r in rows], "--s", color=c, lw=1.4, ms=4, alpha=0.8)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)


draw(axes[0], "itl_p50_ms", "itl_p99_ms", "Inter-token latency (lower is better)", "ms (log scale)")
draw(axes[1], "ttft_p50_ms", "ttft_p99_ms", "Time to first token (lower is better)", "ms (log scale)")

ax = axes[2]
for i, (label, rows) in enumerate(by.items()):
    ax.plot([r["rate"] for r in rows], [float(r["out_tok_s"]) for r in rows], "-o",
            color=COLORS[i % len(COLORS)], lw=2, ms=5)
ax.plot(all_rates, [x * args.output_len for x in all_rates], ":", color="black", lw=1.5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Output throughput vs offered load", fontsize=12)
ax.set_ylabel("output tokens/s (log scale)")

for a in axes:
    a.set_xscale("log")
    a.set_xticks(all_rates)
    a.xaxis.set_major_formatter(ScalarFormatter())
    a.xaxis.set_minor_formatter(NullFormatter())
    a.set_xlabel("arrival rate (requests/s)")
    a.grid(True, which="major", alpha=0.3)

cfg = [Line2D([0], [0], color=COLORS[i % len(COLORS)], lw=3, label=l) for i, l in enumerate(by)]
style = [Line2D([0], [0], color="gray", lw=2, ls="-", marker="o", label="p50"),
         Line2D([0], [0], color="gray", lw=1.4, ls="--", marker="s", label="p99")]
axes[0].legend(handles=cfg + style, fontsize=9, loc="upper left")
axes[2].legend(handles=[Line2D([0], [0], color="black", ls=":", label="offered load (ideal)")],
               fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(args.out, dpi=140)
print("wrote", args.out)
