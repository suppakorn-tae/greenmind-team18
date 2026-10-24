#!/usr/bin/env python3
"""Local GreenMind grader — CLI. See README / spec. `python grader.py -h`."""
import argparse, asyncio, json, sys, time
from pathlib import Path

from gm import client, config, energy, score as score_mod, slo, trace
from gm.store import Store, format_list, format_show

LATE_WARN_S = 0.100


def log(*a):
    print(*a, flush=True)


def _write_progress(run_dir, done, total, started_at):
    """runs/<id>/progress.json (atomic replace) — read by the dashboard while a run executes."""
    import os
    tmp = Path(run_dir) / "progress.json.tmp"
    tmp.write_text(json.dumps({"items_done": done, "items_total": total, "started_at": started_at,
                               "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    os.replace(tmp, Path(run_dir) / "progress.json")


async def _progress_loop(run_dir, prog, total, started_at, every=2.0):
    while True:
        try:
            _write_progress(run_dir, prog["done"], total, started_at)
        except OSError:
            pass
        await asyncio.sleep(every)


# ═════════════════════════════════ scoring against the current baseline ═════════════════════════════════
def compute_score(run, summary, baseline):
    """score_json for a run. Baseline run itself → S = 0. No baseline / no energy → S None ('n/a')."""
    if run["is_baseline"]:
        return {"S": 0.0, "is_baseline": True, "eta_ratio": 1.0, "R": 1.0, "A": 1.0, "s_E": 0.0, "class_factor": 1.0}
    if not baseline or not baseline.get("summary"):
        return {"S": None, "note": "no baseline for this track"}
    sc = score_mod.score(summary, baseline["summary"])
    if not sc:
        return {"S": None, "note": "energy (eta) unavailable for this run or the baseline"}
    sc["baseline_id"] = baseline["id"]
    return sc


# ═════════════════════════════════ run ═════════════════════════════════
async def _execute(args, items, bounds, st, run_id, url, model, settings):
    """Open-loop replay. → (records, item_results, energy_raw, launch_stats, interrupted)."""
    api = client.base_url(url) + "/v1/chat/completions"
    async with client.make_client() as c:
        for _ in range(args.warmup):
            await client.stream_request(c, api, model, [{"role": "user", "content": "Say hi."}], None, 32)
        meter = energy.EnergyMeter(args.gpu)
        idle = None
        if meter.ok and args.settle:
            log(f"measuring idle power for {args.settle:.0f}s ...")
            idle = meter.idle_power(args.settle)
            log(f"idle power ≈ {idle:.0f} W")
        elif not meter.ok:
            log("!! WARNING: no NVML — energy is null; eta/s_E/S will be n/a")
        records, results, lagstats = [], [], {}
        t0 = time.monotonic()
        meter.start(t0)
        monitor = asyncio.create_task(client.loop_lag_monitor(lagstats))
        prog = {"done": 0}
        progress = asyncio.create_task(_progress_loop(st.run_dir(run_id), prog, len(items), time.strftime("%Y-%m-%dT%H:%M:%S")))
        tasks = [asyncio.create_task(client.run_item(c, api, model, it, t0, bounds, records)) for it in items]
        interrupted, done_n = False, 0
        try:
            for fut in asyncio.as_completed(tasks):
                results.append(await fut)
                done_n += 1
                prog["done"] = done_n
                if done_n % max(1, len(items) // 10) == 0:
                    log(f"  {done_n}/{len(items)} items done @ {time.monotonic() - t0:.0f}s")
        except (asyncio.CancelledError, KeyboardInterrupt):
            interrupted = True
            log("!! interrupted — stopping; partial data kept, run excluded from scoring")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        end = meter.stop()
        monitor.cancel()
        progress.cancel()
        try:
            _write_progress(st.run_dir(run_id), prog["done"], len(items), time.strftime("%Y-%m-%dT%H:%M:%S"))
        except OSError:
            pass
    lags = [r["launch_lag"] for r in records if r.get("launch_lag") is not None]
    launch = {"max_launch_lag_ms": max(lags) * 1000 if lags else 0.0, "n_late_gt_100ms": sum(l > LATE_WARN_S for l in lags),
              "max_loop_lag_ms": lagstats.get("max_loop_lag_s", 0.0) * 1000}
    if launch["n_late_gt_100ms"]:
        log(f"!! WARNING: {launch['n_late_gt_100ms']} request(s) launched >100 ms late (max {launch['max_launch_lag_ms']:.0f} ms, "
            f"max loop lag {launch['max_loop_lag_ms']:.0f} ms) — the client may be the bottleneck")
    first_send = min((r["sent"] for r in records), default=0.0)
    raw = meter.raw(bounds, end, first_send, idle) if meter.ok else {"ok": False}
    if meter.ok:
        meter.write_csv(st.run_dir(run_id) / "power.csv")
    return records, sorted(results, key=lambda r: r["id"]), raw, launch, interrupted


async def run_track(args, st=None):
    """Run one track end to end. → run_id (even when interrupted/failed). Raises SystemExit on refusals."""
    st = st or Store()
    items, bounds, meta = trace.load(args.track)
    settings = config.resolve_settings({"ARRIVALS": meta.get("arrivals", "uniform")})
    async with client.make_client() as c:
        try:
            model, checks = await client.health(c, args.url, args.model)
        except Exception as e:
            sys.exit(f"health check failed: cannot talk to {args.url}: {e!r}")
    for name, ok, detail in checks:
        log(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail[:200]}")
    if not all(ok for _, ok, _ in checks):
        sys.exit("refusing to run: health check failed")
    others = energy.other_compute_pids(args.gpu)
    if others:
        msg = f"other compute processes on the GPU (pids {others}) would pollute the energy measurement"
        if not args.force:
            sys.exit(f"refusing to run: {msg}. Use --force to override.")
        log(f"!! WARNING: {msg} (--force)")
    server = energy.server_context()
    server.update(model=model, url=args.url)
    run_id = st.create_run(args.track, args.label or "", args.notes or "", server, settings)
    log(f"run {run_id}: track={args.track} model={model} items={len(items)} bounds="
        f"{ {k: round(b - a, 1) for k, (a, b) in bounds.items()} } s")
    try:
        records, results, raw, launch, interrupted = await _execute(args, items, bounds, st, run_id, args.url, model, settings)
    except BaseException as e:
        st.finish_run(run_id, "failed", None)
        log(f"!! run failed: {e!r}")
        raise
    st.write_jsonl(run_id, "records", records)
    st.write_jsonl(run_id, "items", results)
    summary = score_mod.summarize(records, results, raw, settings, bounds, launch)
    st.write_json(run_id, "summary", summary)
    if interrupted:
        st.finish_run(run_id, "interrupted", summary)
        log(f"interrupted run {run_id} kept (not scored)")
        return run_id
    st.finish_run(run_id, "done", summary)
    run = st.get_run(run_id)
    st.set_score(run_id, compute_score(run, summary, st.get_baseline(args.track)))
    try:
        from gm import report
        report.build_report(st, run_id)
    except Exception as e:  # report problems must never lose a finished run
        log(f"!! report not built: {e!r}")
    log(format_show(st.get_run(run_id)))
    return run_id


# ═════════════════════════════════ rescore ═════════════════════════════════
def rescore(st, ref, overrides, log=log):
    """Recompute summary + score for one run (or all done runs) from stored records, against the current baseline.
    `overrides` = calibration settings to apply to every run recomputed (they are written back to the run row)."""
    ids = [r["id"] for r in st.list_runs(limit=10 ** 6) if r["status"] == "done"] if ref is None else [st.resolve(ref)]
    new = {}
    for rid in ids:
        run = st.get_run(rid)
        if run["status"] != "done":
            log(f"skip {rid}: status {run['status']}")
            continue
        settings = config.resolve_settings({**(run["settings"] or {}), **overrides})
        old = run["summary"]
        bounds = {k: tuple(v) for k, v in old["bounds"].items()}
        new[rid] = (settings, score_mod.summarize(st.read_jsonl(rid, "records"), st.read_jsonl(rid, "items"),
                                                  old.get("energy_raw"), settings, bounds, old.get("launch")))
    for rid, (settings, summary) in new.items():
        run = st.get_run(rid)
        base = st.get_baseline(run["track"])
        if base and base["id"] in new:
            base = {**base, "summary": new[base["id"]][1]}
        st.update_run(rid, settings=settings, summary=summary)
        sc = compute_score(run, summary, base)
        st.set_score(rid, sc)
        st.write_json(rid, "summary", summary)
        log(f"{rid}: S = {'n/a' if sc.get('S') is None else format(sc['S'], '.1f')}")
    return list(new)


# ═════════════════════════════════ CLI ═════════════════════════════════
def _common(p):
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model")


def _run_args(p):
    p.add_argument("--track", choices=list(config.TRACKS), required=True)
    p.add_argument("--label", default="")
    p.add_argument("--notes", default="")
    _common(p)
    p.add_argument("--force", action="store_true", help="run even if other GPU processes are detected")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--settle", type=float, default=5.0, help="seconds of idle-power measurement before the window")


def build_parser():
    ap = argparse.ArgumentParser(prog="grader.py", description="Local GreenMind grader")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("health"); _common(p); p.add_argument("--gpu", type=int, default=0)
    p = sub.add_parser("run"); _run_args(p)
    p.add_argument("--repeat", type=int, default=1, help="N sequential runs, labelled <label>#1 … <label>#N")
    _run_args(sub.add_parser("baseline"))
    sub.add_parser("set-baseline").add_argument("run_id")
    p = sub.add_parser("list"); p.add_argument("--track", choices=list(config.TRACKS)); p.add_argument("--limit", type=int, default=20)
    sub.add_parser("show").add_argument("run_id")
    p = sub.add_parser("report"); p.add_argument("run_id"); p.add_argument("--open", action="store_true")
    p = sub.add_parser("diff"); p.add_argument("run_a"); p.add_argument("run_b")
    p = sub.add_parser("rescore"); p.add_argument("run_id", nargs="?"); p.add_argument("--all", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="calibration override, e.g. TTFT_FROM=content_only")
    p = sub.add_parser("history"); p.add_argument("--track", choices=list(config.TRACKS))
    p = sub.add_parser("dashboard"); p.add_argument("--host", default="0.0.0.0"); p.add_argument("--port", type=int, default=8050)
    p.add_argument("--export", metavar="FILE", help="write a static snapshot instead of serving")
    p = sub.add_parser("official"); os_ = p.add_subparsers(dest="ocmd", required=True)
    a_ = os_.add_parser("add"); a_.add_argument("--track", choices=list(config.TRACKS), required=True)
    a_.add_argument("--started", required=True); a_.add_argument("--model", required=True)
    for k, t in (("S", float), ("A", float), ("R", float), ("sE", float), ("sT", float), ("sL", float), ("eta-vs-base", float),
                 ("j-per-req", float), ("p95-ttft", float), ("p95-tpot-ms", float), ("slo-met", int), ("slo-total", int), ("accuracy", float)):
        a_.add_argument(f"--{k}", type=t, required=True)
    a_.add_argument("--label", default=""); a_.add_argument("--notes", default=""); a_.add_argument("--local")
    l_ = os_.add_parser("list"); l_.add_argument("--track", choices=list(config.TRACKS))
    k_ = os_.add_parser("link"); k_.add_argument("official_id", type=int); k_.add_argument("run_id")
    p = sub.add_parser("trace"); ts = p.add_subparsers(dest="tcmd", required=True)
    f = ts.add_parser("freeze"); f.add_argument("--seed", type=int, default=0); f.add_argument("--force", action="store_true")
    f.add_argument("--arrivals", choices=config.SETTING_CHOICES["ARRIVALS"])
    ts.add_parser("stats").add_argument("--track", choices=list(config.TRACKS), required=True)
    return ap


def cmd_health(a):
    async def go():
        async with client.make_client() as c:
            return await client.health(c, a.url, a.model)
    try:
        model, checks = asyncio.run(go())
    except Exception as e:
        sys.exit(f"[FAIL] cannot reach {a.url}: {e!r}")
    log(f"model: {model}")
    ok = True
    for name, good, detail in checks:
        log(f"[{'PASS' if good else 'FAIL'}] {name}  {detail[:300]}")
        ok &= bool(good)
    meter = energy.EnergyMeter(a.gpu)
    log(f"[{'PASS' if meter.ok else 'WARN'}] NVML energy counter" + (f"  idle ≈ {meter.idle_power(2):.0f} W" if meter.ok else "  (energy will be null)"))
    others = energy.other_compute_pids(a.gpu)
    if others:
        log(f"[WARN] other compute processes on the GPU: {others}")
    return 0 if ok else 1


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "trace":
        if a.tcmd == "freeze":
            try:
                for t, p in trace.freeze(a.seed, a.force, arrivals=a.arrivals).items():
                    log(f"wrote {p}")
            except FileExistsError as e:
                sys.exit(str(e))
        else:
            items, bounds, meta = trace.load(a.track)
            s = trace.stats(items)
            log(f"track {a.track}: n={s['n']} meta={meta}")
            log(f"by class : {s['by_class']}\nby phase : {s['by_phase']}\nbounds   : { {k: [round(x, 1) for x in v] for k, v in bounds.items()} }")
            for c, v in s["prompt_tokens_approx"].items():
                log(f"  {c:7s} prompt tokens ≈ mean {v['mean']:.0f}  max {v['max']}")
            log(f"max prompt ≈ {s['max_prompt_tokens_approx']} tokens (limit 32000)")
        return 0
    if a.cmd == "health":
        return cmd_health(a)
    st = Store()
    if a.cmd == "run" and a.repeat > 1:
        base_label, rc = a.label or "run", 0
        for k in range(1, a.repeat + 1):
            a.label = f"{base_label}#{k}"
            log(f"=== repeat {k}/{a.repeat}: {a.label}")
            try:
                rid = asyncio.run(run_track(a, st))
            except KeyboardInterrupt:
                return 130
            if st.get_run(rid)["status"] != "done":
                return 1
        return 0
    if a.cmd in ("run", "baseline"):
        try:
            rid = asyncio.run(run_track(a, st))
        except KeyboardInterrupt:
            return 130
        run = st.get_run(rid)
        if a.cmd == "baseline":
            if run["status"] != "done":
                sys.exit(f"run {rid} is {run['status']}; not marking it as baseline")
            st.set_baseline(rid)
            rescore(st, None, {})   # baseline S = 0; other runs of this track now score against it
            log(f"{rid} is now the {a.track} baseline")
        return 0 if run["status"] == "done" else 1
    if a.cmd == "set-baseline":
        rid = st.resolve(a.run_id)
        st.set_baseline(rid)
        rescore(st, None, {})
        log(f"{rid} is now the {st.get_run(rid)['track']} baseline")
    elif a.cmd == "official":
        from gm import official
        if a.ocmd == "add":
            oid = official.add(st, a.track, a.started, a.model, a.S, a.A, a.R, a.sE, a.sT, a.sL, a.eta_vs_base, a.j_per_req,
                               a.p95_ttft, a.p95_tpot_ms, a.slo_met, a.slo_total, a.accuracy, label=a.label, notes=a.notes,
                               local_run_id=a.local)
            log(f"official result #{oid} stored" + (f", linked to {a.local}" if a.local else ""))
        elif a.ocmd == "list":
            log(official.format_official_list(official.list_official(st, a.track)))
        else:
            log(f"official #{a.official_id} ↔ {official.link(st, a.official_id, a.run_id)}")
    elif a.cmd == "history":
        from gm import history
        tracks = [a.track] if a.track else list(config.TRACKS)
        for t in tracks:
            rows = history.load_rows(st, t)
            log(f"── track {t} ──")
            log(history.format_history(rows, history.group_repeats(rows), history.noise_spread(rows)["spread"]))
    elif a.cmd == "dashboard":
        from gm import dashboard
        if a.export:
            log(f"wrote {dashboard.export_static(st, a.export)}")
        else:
            log(f"dashboard on http://{a.host}:{a.port}  (tunnel: ssh -L {a.port}:localhost:{a.port} <vm>)")
            dashboard.serve(st, a.host, a.port)
    elif a.cmd == "list":
        log(format_list(st.list_runs(a.track, a.limit)))
    elif a.cmd == "show":
        log(format_show(st.get_run(st.resolve(a.run_id))))
    elif a.cmd == "report":
        from gm import report
        p = report.build_report(st, st.resolve(a.run_id))
        log(f"wrote {p}")
        if a.open:
            import webbrowser
            webbrowser.open(Path(p).resolve().as_uri())
    elif a.cmd == "diff":
        from gm import report
        log(report.diff_text(st, st.resolve(a.run_a), st.resolve(a.run_b)))
    elif a.cmd == "rescore":
        ov = dict(kv.split("=", 1) for kv in a.set)
        rescore(st, None if (a.all or not a.run_id) else a.run_id, ov)
    return 0


if __name__ == "__main__":
    sys.exit(main())
