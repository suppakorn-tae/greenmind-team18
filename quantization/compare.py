#!/usr/bin/env python3
"""Side-by-side table of eval_ab.sh outputs.  Usage: python compare.py results/bf16 results/w8a8 ..."""
import glob
import json
import os
import re
import sys

METRICS = ("acc,none", "exact_match,flexible-extract", "exact_match,strict-match", "exact_match,none")
BENCH = {"output tok/s": r"Output token throughput \(tok/s\):\s+([\d.]+)",
         "median TPOT ms": r"Median TPOT \(ms\):\s+([\d.]+)",
         "median TTFT ms": r"Median TTFT \(ms\):\s+([\d.]+)"}

cols, table = [], {}
for d in sys.argv[1:]:
    label = os.path.basename(d.rstrip("/"))
    cols.append(label)
    for f in glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True):
        res = json.load(open(f))["results"]
        for task, m in res.items():
            for k in METRICS:
                if k in m:
                    table.setdefault(f"{task} [{k.split(',')[0]}]", {})[label] = m[k]
                    break
    bench = os.path.join(d, "bench.log")
    if os.path.exists(bench):
        txt = open(bench).read()
        for name, rx in BENCH.items():
            hit = re.search(rx, txt)
            if hit:
                table.setdefault(name, {})[label] = float(hit.group(1))

w = max([len(k) for k in table] + [10])
print(f"{'metric':<{w}} " + " ".join(f"{c:>10}" for c in cols) + "   Δ(last-first)")
for k in sorted(table):
    row = [table[k].get(c) for c in cols]
    cells = " ".join(f"{v:>10.4f}" if isinstance(v, float) and v < 10 else f"{v:>10.1f}" if v is not None
                     else f"{'-':>10}" for v in row)
    delta = row[-1] - row[0] if None not in (row[0], row[-1]) and len(row) > 1 else None
    print(f"{k:<{w}} {cells}   {'' if delta is None else f'{delta:+.4f}'}")
