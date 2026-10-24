# E004 result — rejected (ablation informative)
Run 20261024-141006-e004 val fresh: on-time 80/134, fails ttft 35/tpot 21/cutoff 3, p95 TTFT 2.1 s, p95 TPOT 63 ms, 315 J/req, acc 77% (agent 0.10 = same 2/20), eta 0.81x.
vs E001 cold (MTP): fewer TTFT fails but worse TPOT and J/req -> MTP helps decode efficiency; MTP is NOT the agent-accuracy culprit.
Agent 2/20 persists without MTP => AWQ weights (or a shared tuned flag) cause it. AWQ alone is also slower than BF16 baseline on TPOT at this concurrency (Marlin W4A16 at large batch).
Lesson: AWQ line loses agent accuracy deterministically; move mainline to BF16 (+MTP).
