#!/usr/bin/env python3
"""GPU sidecar samples during the canary-before window (LAB_LOG section 28). CPU only.
Window = [last timestamped server-log line before "Application startup complete", +30 s] (canary = 30 requests at rate 1, bench.py). The server log
has second resolution and run_config.sh polls /health every 3 s, so the window start is approximate (a few s).
Usage: python canary_gpu_windows.py <run name> <server log> [<run name> <server log> ...]"""
import csv, re, statistics, sys
from datetime import datetime, timedelta

def gpu_rows(name):
    import glob
    f = glob.glob(f"results/clean/**/gpu_{name}.csv", recursive=True)[0]
    out = []
    for row in list(csv.reader(open(f)))[1:]:
        ts = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
        out.append((ts, float(row[1].split()[0]), float(row[5].split()[0]), float(row[7].split()[0])))
    return f, out

args = sys.argv[1:]
print("| run | window start | samples | util % median [min-max] | SM MHz median | power W median |")
print("|---|---|---|---|---|---|")
for name, log in zip(args[::2], args[1::2]):
    t = None  # that line has no timestamp: use the last timestamped line before it (the route listing)
    for line in open(log, errors="replace"):
        m = re.search(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d)", line)
        if m:
            t = datetime.strptime("2026-" + m.group(1), "%Y-%m-%d %H:%M:%S")
        if "Application startup complete" in line:
            break
    f, rows = gpu_rows(name)
    w = [r for r in rows if t <= r[0] <= t + timedelta(seconds=30)]
    u = [r[1] for r in w]
    print(f"| {name} | {t:%H:%M:%S} | {len(w)} | {int(statistics.median(u))} [{int(min(u))}-{int(max(u))}] | "
          f"{int(statistics.median([r[2] for r in w]))} | {int(statistics.median([r[3] for r in w])*10)/10} |")
