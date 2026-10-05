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
args = ap.parse_args()

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
