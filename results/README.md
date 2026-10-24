# Results evidence

## `official/`: organizer grader
- `official_list.txt` is the registry of all 9 official results we recorded with `grader.py official add` (8 Final, plus the E008 run and the baseline on Validation), each linked to its local twin run. It was exported with `local_grader/grader.py official list`.
- `E046_final_20261024T164844-c109c6.json` is the full export of the **final submission** (S 99.38).
- `E039_final_20261024T141700-d44fef.json` is the full export of E039 (S 85.97).
- The other official results (E000/E008 val; E017, E023, E026, E030, E033 final) were reported to us as table rows only (S, A, R, sE, sT, sL, η, J/req, p95 TTFT/TPOT, SLO-met, accuracy). **TODO (team):** add their attempt IDs and exports or screenshots from the grader UI, and the official Final baseline numbers if they're available.
- Timestamps in the official registry are as shown by the organizer grader (UTC).

## `local/`: our local grader (`local_grader/`, frozen traces `validation.jsonl` / `final.jsonl`)
- `history_val.txt` and `history_final.txt` are `grader.py history` output: every local run with S, A, R, sE, sT, sL, η, J/req, p95 TTFT/TPOT, SLO-met and accuracy. Rows marked `official` are the official results, for calibration.
- `runs/<run_id>/summary.json` holds the full metrics of one run: per-phase and per-class SLO/latency, energy window, idle W, per-phase joules, class accuracy and score. `runs/<run_id>/report.html` is the self-contained report.
- Raw request records (`records.jsonl`, `items.jsonl`) and the power trace (`power.csv`) are kept for the baselines and the champion runs (E017, E030, E045, E046).
- Local S is relative to **our local baseline** (`vllm-val-baseline`, `baseline_bf16_default`), not the official reference. Local Val S swings with the 20-item agent class (A drops to 0.05–0.74 on identical configs), so compare SLO-met, η and J/req instead.
- Run IDs map to experiments by label. `E033#2` is the second run of E033, and `E041#mix` / `E042#maxutil` are tuned attempts. Runs `20261024-1741xx-e025-2` were an invalid concurrent val+final launch: interrupted, ignore. The runs labelled `E028` (`20261024-181545`, `-181725`) actually measured the E027 server (port conflict); see `experiments/E027/result.md`.
