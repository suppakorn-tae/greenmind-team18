# Local GreenMind grader

Replays a frozen request trace against one OpenAI-compatible endpoint, measures SLOs, GPU energy and accuracy,
computes the Round-1 score S and keeps a history of runs. CLI + one self-contained HTML report per run.
`../greenmind-kit/greenbench.py` is untouched; this package is the refactor (scoring math is a port, verified against
the official tables in `tests/test_score.py`).

```bash
python -m venv .venv && .venv/bin/pip install httpx nvidia-ml-py          # runtime deps
.venv/bin/pip install pytest starlette uvicorn                             # tests only
.venv/bin/python grader.py health                                          # stream+usage, tool call, NVML
.venv/bin/python grader.py baseline --track val --label baseline_400W      # run, then mark as the track's reference
.venv/bin/python grader.py run --track val --label awq-mtp_250W            # scored against the baseline
.venv/bin/python grader.py list | show RUN | diff A B | report RUN --open
.venv/bin/python grader.py rescore --all --set TTFT_FROM=content_only      # recompute from stored records
.venv/bin/python -m pytest tests -q                                        # ~2.5 min (e2e replays a 45 s trace twice)
```
Run ids / labels / unique prefixes are all accepted wherever a RUN_ID is expected. `GRADER_HOME` relocates
`runs/` and `runs.sqlite`.

## Layout
`grader.py` CLI + run orchestration · `gm/{config,trace,client,slo,energy,accuracy,score,store,report}.py` ·
`traces/{validation,final}.jsonl` (+`.meta.json`) · `runs/<id>/{records,items}.jsonl, power.csv, summary.json, report.html` ·
`INTERFACES.md` data contract · `fixtures/` synthetic run folders for report/store tests.

## Design notes
* **Raw timings are stored, verdicts are not.** `records.jsonl` keeps `ttft_any/ttft_content`, token spans, token
  counts, finish reason; `gm.slo` evaluates them. That is what lets `rescore` change `TTFT_FROM`, `TPOT_MODE`,
  `EPISODE_TURNS_SCORED`, `FAILED_IN_ACCURACY`, `ENERGY_WINDOW` after the fact. `ARRIVALS=poisson` needs
  `trace freeze --force --arrivals poisson` (the trace carries the arrival times). `rescore --set` writes the new
  settings back to the run row.
* **Energy window.** `first_arrival_to_last_done` = trace origin t0 → last completion (as greenbench);
  `first_send_to_last_done` = first actual request send → last completion. Both are stored, per-phase joules too.
* **TPOT** always uses the full token span ÷ (completion_tokens−1); `TTFT_FROM` changes TTFT only.
  `p95_over_tokens` = p95 of per-chunk gaps normalised by tokens/chunk (the official definition is unknown).
* **Lateness check.** Every first-turn request records `launch_lag`; the run warns if any is >100 ms late and stores
  max launch lag and max event-loop lag in `summary.launch`.
* **Foreign-process check** treats any process whose cmdline contains `vllm` as "the server"; everything else with a
  compute context on the GPU blocks the run unless `--force`. It cannot see other containers' processes.
* A run killed with SIGKILL stays `running` in the table; Ctrl-C leaves it `interrupted` with partial data.

## Differences from greenbench worth knowing
1. **Trace reproducibility bug fixed.** greenbench built RAG vault ids in a `set`, so prompts changed with
   `PYTHONHASHSEED` (its README's "same seed = identical prompts" was false across processes). `trace.py` uses an
   ordered list; the frozen files are therefore *not* byte-identical to any greenbench trace. Regression test included.
2. `no_stream`/`no_usage` requests are always "wrong" in accuracy under `FAILED_IN_ACCURACY=wrong` (greenbench only
   counted `error`/`length`).
3. The spec said `mock_server.py` already existed in the kit; it did not, so it was written from scratch.
