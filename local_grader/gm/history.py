"""Unified history rows over local runs + official-table rows.
Contract: INTERFACES2.md section 3. Owner: Agent H.

ROW is a flat dict with identical keys for local and official rows
(missing -> None). Pure functions over dicts; only load_rows touches
the store (and imports gm.official lazily).
"""
import re

_REPEAT = re.compile(r"^(.*)#(\d+)$")

KEY_FLAGS = (
    "--max-model-len", "--dtype", "--quantization", "--max-num-seqs",
    "--speculative-config", "--gpu-memory-utilization",
    "--enable-prefix-caching", "--kv-cache-dtype",
)
_BOOL_FLAGS = {"--enable-prefix-caching"}

METRIC_KEYS = ("S", "eta_vs_base", "eta", "j_per_req", "slo_met_frac",
               "accuracy_pct", "p95_ttft_s", "p95_tpot_ms")


# ── small helpers ─────────────────────────────────────────────────────────────
def _as_dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _norm_started(s):
    """ISO string (with or without microseconds) -> 'YYYY-MM-DDTHH:MM:SS'."""
    if not isinstance(s, str) or len(s) < 19:
        return s
    head = s[:19]
    if head[4] != "-" or head[10] not in ("T", " "):
        return s
    return f"{head[:10]}T{head[11:13]}:{head[14:16]}:{head[17:19]}"


def _flags_from_cmdline(cmdline):
    """vllm_cmdline minus everything up to and including the first token
    ending in 'vllm' (the interpreter/path prefix). None when absent."""
    if not cmdline or not isinstance(cmdline, str):
        return None
    tokens = cmdline.split()
    for i, tok in enumerate(tokens):
        if tok.endswith("vllm"):
            rest = tokens[i + 1:]
            return " ".join(rest) if rest else None
    return None


def _key_flags(cmdline):
    """Key flags (with values) present in the vllm cmdline, in order."""
    flags = _flags_from_cmdline(cmdline)
    if not flags:
        return None
    toks = flags.split()
    parts = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in KEY_FLAGS:
            if t in _BOOL_FLAGS or i + 1 >= len(toks) or toks[i + 1].startswith("--"):
                parts.append(t)
            else:
                parts.append(f"{t} {toks[i + 1]}")
                i += 1
        i += 1
    return " ".join(parts) if parts else None


def _server_summary(model, power_limit, cmdline):
    parts = []
    if model is not None and str(model) != "":
        parts.append(str(model))
    if power_limit is not None and str(power_limit) != "":
        parts.append(f"{power_limit} W")
    kf = _key_flags(cmdline)
    if kf:
        parts.append(kf)
    return " | ".join(parts) if parts else None


def _slo_frac(row):
    met, tot = row.get("slo_met"), row.get("slo_total")
    met, tot = _num(met), _num(tot)
    if met is None or not tot:
        return None
    return met / tot


# ── row builders ──────────────────────────────────────────────────────────────
def _base_row(key, source, rid, track, label, started, status, notes,
              model, power_limit, cmdline, group, repeat_k):
    return {
        "key": key, "source": source, "id": rid, "track": track,
        "label": label, "started": started, "status": status,
        "model": model, "power_limit": power_limit,
        "flags": _flags_from_cmdline(cmdline),
        "server_summary": _server_summary(model, power_limit, cmdline),
        "is_baseline": False, "pair_key": None,
        "S": None, "A": None, "R": None, "s_E": None,
        "s_T": None, "s_L": None, "eta_vs_base": None,
        "j_per_req": None, "p95_ttft_s": None, "p95_tpot_ms": None,
        "slo_met": None, "slo_total": None, "accuracy_pct": None,
        "slo_turn_met": None, "slo_turn_total": None, "eta": None,
        "group": group, "repeat_k": repeat_k, "notes": notes,
    }


def _set_metrics(row, m, done):
    """m = dict of metric keys -> value; non-done rows get all-None."""
    for k in ("S", "A", "R", "s_E", "s_T", "s_L", "eta_vs_base", "j_per_req",
              "p95_ttft_s", "p95_tpot_ms", "slo_met", "slo_total",
              "accuracy_pct", "slo_turn_met", "slo_turn_total", "eta"):
        row[k] = m.get(k) if done else None


def local_row(run: dict) -> dict:
    """run = Store.get_run() dict (id, track, label, status, started_at,
    is_baseline, server, settings, summary, score)."""
    run = _as_dict(run)
    server = _as_dict(run.get("server"))
    smi = _as_dict(server.get("nvidia_smi"))
    power = _as_dict(smi.get("power"))
    score = _as_dict(run.get("score"))
    summary = _as_dict(run.get("summary"))
    group, repeat_k = parse_repeat(run.get("label"))
    row = _base_row(
        f"local:{run.get('id')}", "local", run.get("id"), run.get("track"),
        run.get("label"), _norm_started(run.get("started_at")), run.get("status"),
        run.get("notes"), server.get("model"), power.get("limit"),
        server.get("vllm_cmdline"), group, repeat_k)
    row["is_baseline"] = bool(run.get("is_baseline"))
    turn = _as_dict(summary.get("slo_turn"))
    m = {
        "S": score.get("S"), "A": score.get("A"), "R": score.get("R"),
        "s_E": score.get("s_E"), "eta_vs_base": score.get("eta_ratio"),
        "s_T": summary.get("s_T"), "s_L": summary.get("s_L"),
        "j_per_req": summary.get("j_per_req"),
        "p95_ttft_s": summary.get("p95_ttft_s"),
        "p95_tpot_ms": summary.get("p95_tpot_ms"),
        "slo_met": summary.get("slo_met"), "slo_total": summary.get("slo_total"),
        "accuracy_pct": summary.get("accuracy_pct"),
        "slo_turn_met": turn.get("on_time"), "slo_turn_total": turn.get("total"),
        "eta": summary.get("eta"),
    }
    _set_metrics(row, m, run.get("status") == "done")
    return row


def official_row(orow: dict, local_baseline_id=None,
                 baseline_official_id=None, pair_key=None) -> dict:
    """orow = gm.official row dict (COLUMNS keys).

    is_baseline: True when this row's local_run_id links to the local
    baseline run of its track (local_baseline_id), else True when this row
    is the track's fallback baseline (baseline_official_id = earliest-started
    official row whose label starts with 'baseline'); False otherwise.
    pair_key: key of the linked local row, or None.
    """
    orow = _as_dict(orow)
    oid = orow.get("id")
    linked = orow.get("local_run_id")
    is_base = (linked is not None and linked == local_baseline_id
               and local_baseline_id is not None)
    if not is_base and baseline_official_id is not None and oid == baseline_official_id:
        is_base = True
    group, repeat_k = parse_repeat(orow.get("label"))
    row = _base_row(
        f"official:{oid}", "official", oid, orow.get("track"), orow.get("label"),
        _norm_started(orow.get("started_at")), "done", orow.get("notes"),
        orow.get("model"), None, None, group, repeat_k)
    row["is_baseline"] = is_base
    row["pair_key"] = pair_key
    _set_metrics(row, {
        "S": orow.get("S"), "A": orow.get("A"), "R": orow.get("R"),
        "s_E": orow.get("s_E"), "s_T": orow.get("s_T"), "s_L": orow.get("s_L"),
        "eta_vs_base": orow.get("eta_vs_base"), "j_per_req": orow.get("j_per_req"),
        "p95_ttft_s": orow.get("p95_ttft_s"),
        "p95_tpot_ms": orow.get("p95_tpot_ms"),
        "slo_met": orow.get("slo_met"), "slo_total": orow.get("slo_total"),
        "accuracy_pct": orow.get("accuracy_pct"),
    }, True)
    return row


def build_rows(runs: list, officials: list, track=None) -> list:
    """Newest-first (by started, ISO strings compare lexicographically),
    local + official interleaved; sets is_baseline + pair_key per track."""
    lrows = [local_row(r) for r in (runs or [])]
    officials = [_as_dict(o) for o in (officials or [])]
    if track is not None:
        lrows = [r for r in lrows if r["track"] == track]
        officials = [o for o in officials if o.get("track") == track]
    # per-track local baseline id
    base_local = {}
    for r in lrows:
        if r["is_baseline"] and r["track"] not in base_local:
            base_local[r["track"]] = r["id"]
    # per-track fallback: earliest-started official whose label starts with
    # 'baseline', used only when no official links to the local baseline
    off_by_track = {}
    for o in officials:
        off_by_track.setdefault(o.get("track"), []).append(o)
    base_official = {}
    for t, lst in off_by_track.items():
        if t in base_local and any(o.get("local_run_id") == base_local[t]
                                   for o in lst):
            continue
        cands = [o for o in lst
                 if str(o.get("label") or "").lower().startswith("baseline")]
        if cands:
            cands.sort(key=lambda o: (_norm_started(o.get("started_at")) or "",
                                      str(o.get("id"))))
            base_official[t] = cands[0].get("id")
    # newest official per linked local (for the local side's pair_key)
    latest_off = {}
    for o in officials:
        lid = o.get("local_run_id")
        if lid is None:
            continue
        prev = latest_off.get(lid)
        if prev is None or (_norm_started(o.get("started_at")) or "") > \
                (_norm_started(prev.get("started_at")) or ""):
            latest_off[lid] = o
    orows = []
    for o in officials:
        lid = o.get("local_run_id")
        orows.append(official_row(
            o, base_local.get(o.get("track")), base_official.get(o.get("track")),
            f"local:{lid}" if lid is not None else None))
    by_key = {r["key"]: r for r in lrows}
    for lid, o in latest_off.items():
        l = by_key.get(f"local:{lid}")
        if l is not None:
            l["pair_key"] = f"official:{o.get('id')}"
    rows = lrows + orows
    rows.sort(key=lambda r: (r["started"] or "", str(r["id"])), reverse=True)
    return rows


def load_rows(store, track=None) -> list:
    """store.list_runs + official.list_official -> build_rows -> mark_noise
    (with the per-track noise_spread) -> rows."""
    from . import official  # lazy: everything above is pure
    runs = store.list_runs(track, limit=10 ** 6)
    officials = official.list_official(store, track)
    rows = build_rows(runs, officials, track)
    seen = []
    for r in rows:
        if r.get("track") and r["track"] not in seen:
            seen.append(r["track"])
    for t in seen:
        sub = [r for r in rows if r["track"] == t]
        mark_noise(sub, noise_spread(sub)["spread"])
    return rows


# ── repeats / noise ───────────────────────────────────────────────────────────
def parse_repeat(label):
    """label 'X#3' -> ('X', 3); anything else -> (None, None)."""
    if not isinstance(label, str):
        return (None, None)
    m = _REPEAT.match(label)
    if not m:
        return (None, None)
    return (m.group(1), int(m.group(2)))


def _metric_of(row, name):
    if name == "slo_met_frac":
        return _slo_frac(row)
    return _num(row.get(name))


def _stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return (None, None, None)
    return (sum(vals) / len(vals), min(vals), max(vals))


def group_repeats(rows) -> list:
    """Groups of >=2 local done rows sharing track+repeat group."""
    buckets = {}
    for r in rows or []:
        if not isinstance(r, dict) or r.get("source") != "local":
            continue
        if r.get("status") != "done" or not r.get("group"):
            continue
        buckets.setdefault((r.get("track"), r["group"]), []).append(r)
    out = []
    for (track, group) in sorted(buckets, key=lambda k: (str(k[0]), str(k[1]))):
        members = buckets[(track, group)]
        if len(members) < 2:
            continue
        mean, mini, maxi = {}, {}, {}
        for name in METRIC_KEYS:
            vals = [_metric_of(r, name) for r in members]
            m, a, b = _stats(vals)
            mean[name], mini[name], maxi[name] = m, a, b
        etas = [v for v in (_num(r.get("eta")) for r in members) if v is not None]
        spread = None
        if etas and mean.get("eta"):
            spread = (max(etas) - min(etas)) / mean["eta"]
        out.append({"track": track, "group": group, "n": len(members),
                    "members": [r["key"] for r in members],
                    "mean": mean, "min": mini, "max": maxi,
                    "spread_eta": spread})
    return out


def noise_spread(rows, default=0.10) -> dict:
    """{"spread": max group spread_eta else default, "observed": bool, "groups": n}."""
    groups = group_repeats(rows)
    spreads = [g["spread_eta"] for g in groups if g.get("spread_eta") is not None]
    if spreads:
        return {"spread": max(spreads), "observed": True, "groups": len(groups)}
    return {"spread": default, "observed": False, "groups": len(groups)}


def within_noise(a, b, spread) -> bool:
    """|a-b| / max(|b|, tiny) <= spread; any None -> False."""
    a, b, spread = _num(a), _num(b), _num(spread)
    if a is None or b is None or spread is None:
        return False
    return abs(a - b) / max(abs(b), 1e-9) <= spread


def mark_noise(rows, spread) -> None:
    """row['within_noise'] = within_noise(eta_vs_base, baseline ref, spread);
    baseline rows -> None."""
    rows = list(rows or [])
    by_track = {}
    for r in rows:
        by_track.setdefault(r.get("track"), []).append(r)
    for tr in by_track.values():
        local_base = next((r for r in tr if r.get("is_baseline")
                           and r.get("source") == "local"), None)
        any_base = local_base or next((r for r in tr if r.get("is_baseline")), None)
        ref = _num(any_base.get("eta_vs_base")) if any_base else None
        if ref is None:
            ref = 1.0
        for r in tr:
            if r.get("is_baseline"):
                r["within_noise"] = None
            else:
                r["within_noise"] = within_noise(r.get("eta_vs_base"), ref, spread)


# ── calibration / best / formatting ───────────────────────────────────────────
_CALIB_METRICS = (
    ("j_per_req", "J/REQ"), ("slo_met_frac", "SLO"), ("p95_ttft_s", "P95TTFT"),
    ("p95_tpot_ms", "P95TPOT"), ("accuracy_pct", "ACC%"), ("eta_vs_base", "ETA/BASE"),
)


def _pair_value(row, name):
    if name == "slo_met_frac":
        return _slo_frac(row)
    return _num(row.get(name))


def _flag(ratio):
    if ratio is None:
        return "na"
    d = abs(ratio - 1.0)
    if d <= 0.15:
        return "green"
    if d <= 0.30:
        return "amber"
    return "red"


def calibration(rows, track) -> dict:
    """Linked official<->local pairs of the track + eta_vs_base rank agreement."""
    rows = list(rows or [])
    by_key = {r["key"]: r for r in rows if isinstance(r, dict) and r.get("key")}
    pairs = []
    for r in rows:
        if not isinstance(r, dict) or r.get("source") != "official":
            continue
        if track is not None and r.get("track") != track:
            continue
        pk = r.get("pair_key")
        if not pk:
            continue
        l = by_key.get(pk)
        if l is None or l.get("source") != "local" or l.get("status") != "done":
            continue
        metrics = []
        for name, label in _CALIB_METRICS:
            lv, ov = _pair_value(l, name), _pair_value(r, name)
            ratio = lv / ov if (lv is not None and ov not in (None, 0)) else None
            metrics.append({"name": name, "label": label, "local": lv,
                            "official": ov, "ratio": ratio, "flag": _flag(ratio)})
        pairs.append({"official_key": r["key"], "local_key": l["key"],
                      "label": r.get("label") or l.get("label"), "metrics": metrics})
    evb = [(p["local_key"], by_key[p["local_key"]]["eta_vs_base"],
            p["official_key"], by_key[p["official_key"]]["eta_vs_base"])
           for p in pairs]
    n = len(pairs)
    agree = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            li, oi = evb[i][1], evb[i][3]
            lj, oj = evb[j][1], evb[j][3]
            if None in (li, oj, lj, oi):
                continue
            total += 1
            if li == lj or (li > lj) == (oi > oj):
                agree += 1
    if n < 2:
        text = "need >= 2 linked pairs"
    else:
        text = f"{agree} of {total} pair-orderings agree"
    return {"pairs": pairs,
            "rank_agreement": {"n_pairs": n, "agree": agree, "total": total,
                               "text": text}}


def best_rows(rows, track) -> dict:
    """Highest S (None S ignored); ties -> higher eta_vs_base."""
    def best(sub):
        cands = [r for r in sub if _num(r.get("S")) is not None]
        if not cands:
            return None
        return max(cands, key=lambda r: (_num(r["S"]),
                                         _num(r.get("eta_vs_base")) if _num(r.get("eta_vs_base")) is not None else float("-inf")))

    out = {"local": None, "official": None}
    for src in ("local", "official"):
        sub = [r for r in rows or [] if isinstance(r, dict)
               and r.get("source") == src and (track is None or r.get("track") == track)]
        out[src] = best(sub)
    return out


def _f(v, nd):
    v = _num(v)
    if v is None:
        return "-"
    return f"{v:.{nd}f}"


def format_history(rows, groups=None, spread=None) -> str:
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    by_key = {r["key"]: r for r in rows if r.get("key")}

    def pair_txt(r):
        pk = r.get("pair_key")
        if not pk or pk not in by_key:
            return "-"
        o = by_key[pk]
        lab = o.get("label")
        return str(lab) if lab else str(o.get("id"))

    lines = [f"{'SRC':<8} {'LABEL':<16} {'STARTED':<19} {'S':>6} {'A':>6} {'R':>6} "
             f"{'sE':>6} {'sT':>6} {'sL':>6} {'ETA/BASE':>8} {'J/REQ':>7} "
             f"{'P95TTFT':>9} {'P95TPOT':>9} {'SLO':>8} {'ACC%':>6} {'BASE':<4} "
             f"{'PAIR':<16} CONFIG"]
    for r in rows:
        lab = str(r.get("label") or "-")
        if r.get("within_noise") is True:
            lab += "~"
        met, tot = _num(r.get("slo_met")), _num(r.get("slo_total"))
        slo = f"{int(met)}/{int(tot)}" if met is not None and tot is not None else "-"
        cfg = r.get("server_summary")
        cfg = (str(cfg)[:60] if cfg else "-")
        lines.append(
            f"{str(r.get('source') or '-'):<8} {lab:<16} {str(r.get('started') or '-'):<19} "
            f"{_f(r.get('S'), 1):>6} {_f(r.get('A'), 3):>6} {_f(r.get('R'), 3):>6} "
            f"{_f(r.get('s_E'), 3):>6} {_f(r.get('s_T'), 3):>6} {_f(r.get('s_L'), 3):>6} "
            f"{_f(r.get('eta_vs_base'), 3):>8} {_f(r.get('j_per_req'), 0):>7} "
            f"{_f(r.get('p95_ttft_s'), 2):>9} {_f(r.get('p95_tpot_ms'), 1):>9} "
            f"{slo:>8} {_f(r.get('accuracy_pct'), 1):>6} "
            f"{'*' if r.get('is_baseline') else '-':<4} {pair_txt(r):<16} {cfg}")
    for g in (groups or []):
        label = f"{g.get('track')}:{g.get('group')}"
        m, lo, hi = g.get("mean") or {}, g.get("min") or {}, g.get("max") or {}
        lines.append("")
        lines.append(f"{label} x{g.get('n')}: ETA/BASE mean {_f(m.get('eta_vs_base'), 3)} "
                     f"({_f(lo.get('eta_vs_base'), 3)}\u2013{_f(hi.get('eta_vs_base'), 3)}), "
                     f"S mean {_f(m.get('S'), 1)}, J/REQ {_f(m.get('j_per_req'), 0)}")
    if spread is not None:
        if isinstance(spread, dict):
            sp, obs = _num(spread.get("spread")), bool(spread.get("observed"))
        else:
            sp, obs = _num(spread), False
        if sp is not None:
            lines.append("")
            lines.append(f"noise spread \u00b1{sp * 100:.1f}% "
                         f"({'observed' if obs else 'default'}); "
                         f"rows marked ~ are within noise of the baseline")
    return "\n".join(lines)
