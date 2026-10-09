#!/usr/bin/env python3
"""Summarize vLLM torch-profiler traces (profile_run.sh): per engine-step type, how much of the step period the GPU
is actually busy, and which CUDA runtime calls the CPU spends time in.

Steps are the worker's `execute_context_<n>(...)_generation_<m>(...)` annotations (n prefill and m decode requests).
For each step: period = start of this step to start of the next (what a decoding request experiences per token),
gpu_busy = union of kernel/memcpy/memset intervals that start inside the period. Trace timestamps are microseconds.
    python profile_analyze.py results/profiles/healthy [more dirs...]   -> appends to results/profiles/summary.csv
"""
import collections, csv, glob, gzip, json, os, re, statistics, sys

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def union(intervals):
    tot, end = 0.0, -1.0
    for s, e in sorted(intervals):
        if e <= end:
            continue
        tot += e - max(s, end)
        end = e
    return tot


def analyze(path):
    ev = [e for e in json.load(gzip.open(path))["traceEvents"] if e.get("ph") == "X"]
    steps = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context")),
                   key=lambda e: e["ts"])
    gpu = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("cat") in GPU_CATS)
    rt = sorted((e["ts"], e["dur"], e["name"]) for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver"))
    out = collections.defaultdict(list)
    gi = ri = 0
    for a, b in zip(steps, steps[1:]):
        t0, t1 = a["ts"], b["ts"]
        m = re.match(r"execute_context_(\d+)\(\d+\)_generation_(\d+)", a["name"])
        kind = ("decode" if m[1] == "0" else "mixed") + f" b={m[2] if m[1] == '0' else m[1] + '+' + m[2]}"
        while gi < len(gpu) and gpu[gi][0] < t0:
            gi += 1
        j, iv = gi, []
        while j < len(gpu) and gpu[j][0] < t1:
            iv.append((gpu[j][0], min(gpu[j][1], t1)))
            j += 1
        while ri < len(rt) and rt[ri][0] < t0:
            ri += 1
        k, api = ri, collections.Counter()
        while k < len(rt) and rt[k][0] < t1:
            api[rt[k][2]] += rt[k][1]
            k += 1
        out[kind].append({"period": t1 - t0, "cpu_execute": a["dur"], "gpu_busy": union(iv),
                          "n_gpu_ops": len(iv), "api": api})
    return out


def main():
    rows = []
    for d in sys.argv[1:]:
        for path in sorted(glob.glob(os.path.join(d, "rank0.*.pt.trace.json.gz"))):
            for kind, st in sorted(analyze(path).items(), key=lambda kv: -len(kv[1])):
                if len(st) < 5:
                    continue
                med = lambda k: statistics.median(s[k] for s in st)  # noqa: E731
                api = collections.Counter()
                for s in st:
                    api.update(s["api"])
                top = "; ".join(f"{n} {v / len(st) / 1e3:.2f}" for n, v in api.most_common(4))
                row = {"dir": d, "trace": os.path.basename(path)[:30], "step_kind": kind, "n_steps": len(st),
                       "period_ms_p50": med("period") / 1e3, "cpu_execute_ms_p50": med("cpu_execute") / 1e3,
                       "gpu_busy_ms_p50": med("gpu_busy") / 1e3,
                       "gpu_idle_ms_p50": statistics.median(s["period"] - s["gpu_busy"] for s in st) / 1e3,
                       "gpu_ops_p50": med("n_gpu_ops"), "top_cuda_api_ms_per_step": top}
                rows.append(row)
                print(f"{d.split('/')[-1]:10s} {row['trace'][:22]} {kind:12s} n={len(st):4d}  period {row['period_ms_p50']:6.2f} ms  "
                      f"GPU busy {row['gpu_busy_ms_p50']:6.2f}  idle {row['gpu_idle_ms_p50']:6.2f}  "
                      f"execute(CPU) {row['cpu_execute_ms_p50']:6.2f}  ops {row['gpu_ops_p50']:.0f}\n    API ms/step: {top}")
    out = "results/profiles/summary.csv"
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print("appended", len(rows), "rows to", out)


if __name__ == "__main__":
    main()
