#!/usr/bin/env python3
"""Figures from the clean-rerun CSVs only (LAB_LOG sections 28, 32, 33). CPU only. Writes results/clean_*.png and never
overwrites: an existing file gets a timestamped sibling instead. Old figures (results/summary_*.png etc.) are untouched.

Aggregation matches summarize_clean.py: run_config.sh experiments use only VALID runs (results/clean/runs.csv); every point
is the median over reps, bands/whiskers are min-max over reps. Palette: reference categorical slots 1-3 (blue, orange,
aqua), which the dataviz reference validates all-pairs in light mode; at most 3 series per chart.
"""
import csv, os, re, statistics, time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = "results/clean"
HW = "RTX 5080 16 GB, Windows + WSL2, vLLM 0.31.0"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
})


def valid(d):
    return {r["name"] for r in csv.DictReader(open(f"{d}/runs.csv")) if r["verdict"] == "VALID"}


def agg(path, rx, col, metric, policy=None, ok=None):
    """{level: (median, min, max, n)} over rows whose label fully matches rx (and policy, and label in ok if given)."""
    vals = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if not re.fullmatch(rx, r["label"]) or (policy and r.get("policy") != policy) or (ok is not None and r["label"] not in ok):
            continue
        vals[float(r[col])].append(float(r[metric]))
    return {lv: (statistics.median(v), min(v), max(v), len(v)) for lv, v in sorted(vals.items())}


def save(fig, name, caption):
    fig.text(0.01, 0.01, caption, fontsize=8, color=INK2, ha="left", va="bottom", wrap=True)
    d = os.environ.get("PLOT_DIR", "results")  # PLOT_DIR=<scratch dir> for previews
    out = f"{d}/{name}.png"
    if os.path.exists(out):
        out = f"{d}/{name}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def line_band(ax, data, color, label, end_label=True):
    xs = list(data)
    med = [data[x][0] for x in xs]
    ax.fill_between(xs, [data[x][1] for x in xs], [data[x][2] for x in xs], color=color, alpha=0.18, linewidth=0)
    ax.plot(xs, med, color=color, marker="o", label=label)
    if end_label:
        ax.annotate(label, (xs[-1], med[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)


def grouped_bars(ax, groups, series, ylabel):
    """series: [(label, color, {group: (med, lo, hi, n)})]. Median bars, min-max whiskers, 2px surface gap between bars."""
    w = 0.8 / len(series)
    for i, (lab, color, data) in enumerate(series):
        xs = [g + (i - (len(series) - 1) / 2) * w for g in range(len(groups))]
        meds = [data[g][0] for g in groups]
        err = [[data[g][0] - data[g][1] for g in groups], [data[g][2] - data[g][0] for g in groups]]
        ax.bar(xs, meds, width=w, color=color, label=lab, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(xs, meds, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g:g}" for g in groups])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(ylabel)


def main():
    ok = valid(C)
    main_csv, modes_csv = f"{C}/main.csv", f"{C}/modes.csv"

    # (a) eager / compiled / FP8 3B: ITL p50 vs rate
    fig, ax = plt.subplots(figsize=(7, 4.4))
    fig.subplots_adjust(bottom=0.22, right=0.85)
    for rx, color, lab in [(r"clean-eager-rep\d+", S2, "eager BF16"), (r"clean-compiled-rep\d+", S1, "compiled BF16"),
                           (r"clean-fp8-3b-rep\d+", S3, "compiled FP8")]:
        line_band(ax, agg(main_csv, rx, "rate", "itl_p50_ms", ok=ok), color, lab)
    ax.set_xlabel("request rate (req/s, open-loop Poisson)")
    ax.set_ylabel("ITL p50 (ms)")
    ax.set_ylim(0, 33)  # headroom so the legend sits above the eager band
    ax.set_title("Qwen2.5-3B: inter-token latency, eager vs compiled vs FP8")
    ax.legend(loc="upper left")
    save(fig, "clean_a_itl_eager_compiled_fp8",
         f"n=3 runs per point (median line, min-max band), all VALID. {HW}. Source: results/clean/main.csv (LAB_LOG 28).")

    # (b) 7B FP8 vs AWQ: ITL p50 and p99 vs rate (two panels, one scale each)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    fig.subplots_adjust(bottom=0.24, wspace=0.3)
    for ax, metric, title in [(axes[0], "itl_p50_ms", "ITL p50"), (axes[1], "itl_p99_ms", "ITL p99")]:
        line_band(ax, agg(main_csv, r"clean-fp8-7b-rep\d+", "rate", metric, ok=ok), S1, "7B FP8", end_label=False)
        line_band(ax, agg(main_csv, r"clean-awq-7b-rep\d+", "rate", metric, ok=ok), S2, "7B AWQ 4-bit", end_label=False)
        ax.set_title(f"Qwen2.5-7B: {title}")
        ax.set_xlabel("request rate (req/s)")
        ax.set_ylabel(f"{title} (ms)")
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="lower right")
    save(fig, "clean_b_7b_fp8_vs_awq",
         f"n=3 runs per point (median, min-max band), all VALID. AWQ measured up to 12 req/s only. {HW}. "
         "Source: results/clean/main.csv (LAB_LOG 28).")

    # (c) prefix caching on/off: TTFT p50 by level
    fig, ax = plt.subplots(figsize=(7, 4.4))
    fig.subplots_adjust(bottom=0.24)
    levels = [2.0, 4.0, 8.0]
    grouped_bars(ax, levels, [
        ("prefix caching off", S2, agg(modes_csv, r"clean-prefix-off-rep\d+", "level", "ttft_p50_ms", ok=ok)),
        ("prefix caching on", S1, agg(modes_csv, r"clean-prefix-on-rep\d+", "level", "ttft_p50_ms", ok=ok))], "TTFT p50 (ms)")
    ax.set_xlabel("request rate (req/s)")
    ax.set_title("Qwen2.5-3B: TTFT with a shared ~1,540-token prefix (best case)")
    ax.legend(loc="upper left")
    save(fig, "clean_c_prefix_caching",
         f"n=3 runs per bar (median, min-max whiskers), all VALID. Every prompt shares one ~1,540-token prefix. {HW}. "
         "Source: results/clean/modes.csv (LAB_LOG 28).")

    # (d) async scheduling on vs off: ITL p50 and TTFT p50 (two panels)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    fig.subplots_adjust(bottom=0.24, wspace=0.3)
    rates = [1.0, 4.0, 8.0, 16.0]
    for ax, metric, title in [(axes[0], "itl_p50_ms", "ITL p50"), (axes[1], "ttft_p50_ms", "TTFT p50")]:
        grouped_bars(ax, rates, [
            ("async on (default)", S1, agg(main_csv, r"clean2-async-default-rep\d+", "rate", metric, ok=ok)),
            ("async off", S2, agg(main_csv, r"clean2-no-async-rep\d+", "rate", metric, ok=ok))], f"{title} (ms)")
        ax.set_xlabel("request rate (req/s)")
        ax.set_title(f"Async scheduling: {title}")
    axes[0].legend(loc="upper left")
    save(fig, "clean_d_async_scheduling",
         f"Qwen2.5-3B BF16 compiled. n=3 runs per bar (median, min-max whiskers), all VALID. {HW}. "
         "Source: results/clean/main.csv (LAB_LOG 32).")

    # (e) mini_sched continuous vs static: TTFT p50 by rate (log scale: values span ~120 ms to ~5.9 s)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    fig.subplots_adjust(bottom=0.24)
    mini = f"{C}/mini_sched.csv"
    grouped_bars(ax, [1.0, 2.0, 4.0], [
        ("static batching", S2, agg(mini, r"clean-mini-rep\d+", "level", "ttft_p50_ms", policy="static")),
        ("continuous batching", S1, agg(mini, r"clean-mini-rep\d+", "level", "ttft_p50_ms", policy="continuous"))],
        "TTFT p50 (ms, log scale)")
    ax.set_yscale("log")
    ax.set_ylim(50, 40000)  # headroom so the legend sits above the bars
    ax.set_xlabel("request rate (req/s, Poisson)")
    ax.set_title("mini_sched (own engine): continuous vs own static-batching baseline")
    ax.legend(loc="upper left")
    save(fig, "clean_e_continuous_vs_static",
         f"Own PyTorch engine, Qwen2.5-3B BF16, same model code for both policies. n=3 runs per bar (median, min-max whiskers). "
         f"{HW.split(', vLLM')[0]}. Source: results/clean/mini_sched.csv (LAB_LOG 32).")

    # (f) HF vs vLLM tok/s vs batch size
    fig, ax = plt.subplots(figsize=(7, 4.4))
    fig.subplots_adjust(bottom=0.24, right=0.85)
    hf = f"{C}/hf_vs_vllm.csv"
    line_band(ax, agg(hf, r"clean-vllm-rep\d+", "batch", "out_tok_s"), S1, "vLLM")
    line_band(ax, agg(hf, r"clean-hf-rep\d+", "batch", "out_tok_s"), S2, "HF generate")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 32, 64])
    ax.set_xticklabels(["1", "8", "32", "64"])
    ax.set_xlabel("static batch size")
    ax.set_ylabel("output tokens/s")
    ax.set_ylim(bottom=0)
    ax.set_title("Qwen2.5-3B BF16: HF generate vs vLLM offline")
    ax.legend(loc="upper left")
    save(fig, "clean_f_hf_vs_vllm",
         f"256-token prompts, 128 greedy tokens. n=3 runs x 3 trials = 9 samples per point (median, min-max band). {HW}. "
         "Source: results/clean/hf_vs_vllm.csv (LAB_LOG 32).")


if __name__ == "__main__":
    main()
