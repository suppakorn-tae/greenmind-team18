#!/usr/bin/env python3
"""Regenerate EXPERIMENTS.md from experiments/log.jsonl (latest event per id wins per field)."""
import json, pathlib
root = pathlib.Path(__file__).resolve().parent
rows = {}
order = []
for line in (root / "log.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    e = json.loads(line)
    i = e["id"]
    if i not in rows:
        rows[i] = {}
        order.append(i)
    ev = e.get("event")
    for k, v in e.items():
        if v is not None and k != "event":
            rows[i][k] = v
    if ev == "official":
        rows[i]["official_S"] = e.get("official_S", e.get("S"))
cols = [("id", "id"), ("parent", "parent"), ("change", "change"), ("status", "status"), ("val_S", "val S"),
        ("final_S", "final S"), ("official_S", "official S"), ("j_per_req", "J/req"), ("slo_met", "SLO"),
        ("accuracy", "acc"), ("lesson", "lesson")]
out = ["# Experiments\n", "| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
for i in order:
    out.append("| " + " | ".join(str(rows[i].get(k, "")).replace("|", "/") for k, _ in cols) + " |")
kf = root.parent / "KEY_FINDINGS.md"
out += ["", "## Key findings", "", kf.read_text() if kf.exists() else "_(none yet)_"]
(root.parent / "EXPERIMENTS.md").write_text("\n".join(out) + "\n")
