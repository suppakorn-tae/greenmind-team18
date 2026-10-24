"""Generate, freeze and load traces. Item generators are a verbatim port of greenbench.py so the same
seed yields the same prompts."""
import json, math, random, zlib
from pathlib import Path

from . import config
from .config import CLASS_MIX

ROOT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════ deterministic fake tools ═════════════════════════════════
def _h(s):
    return zlib.crc32(s.encode())


def price(sku):
    return 50 + _h(sku.strip().upper()) % 950


def stock(sku, wh):
    return _h(f"{sku.strip().upper()}|{wh.strip().upper()}") % 200


RATES = {"USD": 35.0, "EUR": 38.0, "JPY": 0.24, "THB": 1.0}  # THB per unit


def convert(amount, frm, to):
    return round(float(amount) * RATES[frm.upper()] / RATES[to.upper()], 2)


TOOLS = [
    {"type": "function", "function": {
        "name": "get_product_price", "description": "Get the current unit price (in THB) of a product by SKU.",
        "parameters": {"type": "object", "properties": {"sku": {"type": "string", "description": "e.g. SKU-1234"}},
                       "required": ["sku"]}}},
    {"type": "function", "function": {
        "name": "get_stock", "description": "Get the number of units in stock for a SKU at one warehouse.",
        "parameters": {"type": "object", "properties": {"sku": {"type": "string"},
                                                        "warehouse": {"type": "string", "description": "e.g. BKK-1"}},
                       "required": ["sku", "warehouse"]}}},
    {"type": "function", "function": {
        "name": "convert_currency", "description": "Convert an amount between currencies (THB, USD, EUR, JPY) at today's rate.",
        "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from_currency": {"type": "string"},
                                                        "to_currency": {"type": "string"}},
                       "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]


def tool_exec(name, args_json):
    try:
        a = json.loads(args_json) if args_json else {}
        if name == "get_product_price":
            return json.dumps({"sku": a["sku"].upper(), "unit_price_thb": price(a["sku"])})
        if name == "get_stock":
            return json.dumps({"sku": a["sku"].upper(), "warehouse": a["warehouse"].upper(),
                               "units_in_stock": stock(a["sku"], a["warehouse"])})
        if name == "convert_currency":
            frm = a.get("from_currency") or a.get("from")
            to = a.get("to_currency") or a.get("to")
            return json.dumps({"amount": a["amount"], "from": frm, "to": to, "result": convert(a["amount"], frm, to)})
        if name == "get_weather":
            c = a["city"]
            return json.dumps({"city": c, "temp_c": 24 + _h(c.lower()) % 12, "condition": ["sunny", "cloudy", "rain"][_h(c) % 3]})
        return json.dumps({"error": f"unknown tool {name}"})
    except Exception as e:
        return json.dumps({"error": f"bad arguments: {e}"})


# ═════════════════════════════════ item generators ═════════════════════════════════
BOX = "\n\nPut only the final numeric answer inside \\boxed{}."
CHAT = [
    "Write a polite email to a client explaining that their delivery will be two days late because of a customs inspection.",
    "Explain the difference between TCP and UDP to a junior developer, with one real-world example of each.",
    "Give me a 5-day beginner workout plan that needs no equipment.",
    "Summarize the main arguments for and against a four-day work week.",
    "Rewrite this sentence to sound more professional: 'we kinda messed up the report, will fix it soon'.",
    "What are good practices for naming variables in Python? Give examples.",
    "Draft a short product description for a reusable stainless-steel water bottle.",
    "Explain what a hash table is and why lookups are usually O(1).",
    "Suggest three names for a coffee shop that also sells plants, with a one-line rationale each.",
    "How should I prepare for a job interview for a data analyst position?",
    "Write a haiku-free, four-line poem about rain on a city at night.",
    "Explain the concept of opportunity cost with an everyday example.",
    "List the steps to safely back up a laptop before reinstalling the operating system.",
    "Compare renting versus buying a home for a 28-year-old who may move cities in 3 years.",
    "Write a friendly reminder message to my team about Friday's deadline.",
    "ช่วยเขียนข้อความขอบคุณลูกค้าที่สั่งซื้อสินค้าเป็นครั้งแรกให้หน่อย",
]
DEPTS = ["Finance", "Logistics", "Customer Care", "Procurement", "R&D", "Legal", "Marketing", "Facilities", "IT Operations", "Quality"]
CITIES = ["Bangkok", "Chiang Mai", "Khon Kaen", "Phuket", "Hat Yai", "Nakhon Ratchasima", "Udon Thani", "Pattaya", "Rayong", "Surat Thani"]
GOODS = ["solar inverters", "LED panels", "copper cable", "water pumps", "air filters", "lithium cells", "steel brackets", "circuit boards", "coolant", "sensors"]
NAMES = ["Somchai", "Nattaya", "Kittipong", "Pimchanok", "Arthit", "Ploy", "Chaiwat", "Siriporn", "Thanakorn", "Wanida"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
THAI_FACTS = [
    ("ดอยอินทนนท์ตั้งอยู่ในจังหวัดใด ตอบสั้น ๆ", ["เชียงใหม่"]),
    ("แม่น้ำสายหลักที่ไหลผ่านกรุงเทพมหานครคือแม่น้ำอะไร ตอบสั้น ๆ", ["เจ้าพระยา"]),
    ("สกุลเงินของประเทศไทยคืออะไร ตอบสั้น ๆ", ["บาท"]),
    ("เกาะที่ใหญ่ที่สุดของประเทศไทยคือเกาะอะไร ตอบสั้น ๆ", ["ภูเก็ต"]),
    ("อุทยานแห่งชาติแห่งแรกของประเทศไทยคืออุทยานแห่งชาติใด ตอบสั้น ๆ", ["เขาใหญ่"]),
    ("ราชธานีของไทยที่อยู่ระหว่างกรุงศรีอยุธยากับกรุงรัตนโกสินทร์คือกรุงอะไร ตอบสั้น ๆ", ["ธนบุรี"]),
    ("วันสงกรานต์ตรงกับวันที่เท่าใดของเดือนเมษายน (วันแรก) ตอบเป็นตัวเลข", ["13"]),
]


def _sentence(rng):
    k = rng.randrange(5)
    if k == 0:
        return f"In Q{rng.randint(1, 4)} {rng.randint(2019, 2025)}, the {rng.choice(DEPTS)} team in {rng.choice(CITIES)} processed {rng.randint(100, 9999)} support tickets."
    if k == 1:
        return f"Warehouse {rng.choice('ABCDEFGH')}{rng.randint(1, 40)} in {rng.choice(CITIES)} currently stores {rng.randint(10, 5000)} units of {rng.choice(GOODS)}."
    if k == 2:
        return f"{rng.choice(NAMES)} was appointed deputy head of {rng.choice(DEPTS)} on {rng.randint(1, 28)} {rng.choice(['March', 'June', 'August', 'November'])} {rng.randint(2018, 2025)}."
    if k == 3:
        return f"The maintenance window for server node-{rng.randint(10, 999)} is every {rng.choice(DAYS)} at {rng.randint(0, 23):02d}:00 local time."
    return f"The {rng.choice(DEPTS)} budget for {rng.choice(GOODS)} was revised to {rng.randint(1, 900)} thousand baht after the {rng.choice(CITIES)} audit."


def _rag_item(rng, target_tokens):
    vaults = []   # insertion-ordered (greenbench used a set → prompts depended on PYTHONHASHSEED)
    while len(vaults) < 6:
        v = f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(10, 99)}"
        if v not in vaults:
            vaults.append(v)
    codes = {v: f"{rng.choice('KMPRTX')}{rng.randint(1, 9)}-{rng.randint(1000, 9999)}" for v in vaults}
    sents = [_sentence(rng) for _ in range(max(30, target_tokens // 24))]
    for v in vaults:
        sents.insert(rng.randrange(len(sents)), f"The access code for vault {v} is {codes[v]}.")
    docs = [" ".join(sents[i:i + 12]) for i in range(0, len(sents), 12)]
    body = "\n\n".join(f"[Doc {i + 1}] {d}" for i, d in enumerate(docs))
    v = rng.choice(vaults)
    return {"messages": [
        {"role": "system", "content": "You answer questions using only the provided documents."},
        {"role": "user", "content": f"<documents>\n{body}\n</documents>\n\nQuestion: What is the access code for vault {v}? Answer with the code only."}],
        "check": {"type": "contains", "any": [codes[v]]}}


def gen_chat(rng):
    return {"messages": [{"role": "user", "content": rng.choice(CHAT)}], "check": {"type": "nonempty"}}


def gen_rag(rng):
    return _rag_item(rng, rng.randint(1500, 8000))


def gen_long(rng):
    return _rag_item(rng, rng.randint(14000, 28000))


def gen_reason(rng):
    k = rng.randrange(4)
    if k == 0:
        a, b, x, y = rng.randint(3, 25), rng.randint(20, 90), rng.randint(2, 9), rng.randint(1, 6)
        z = 1000 if a * x + b * y < 1000 else 2000
        q, ans = (f"A shop sells pencils at {a} baht each and notebooks at {b} baht each. Mali buys {x} pencils and "
                  f"{y} notebooks and pays with a {z} baht note. How much change does she get, in baht?"), z - a * x - b * y
    elif k == 1:
        n = rng.randint(30, 150)
        ans = sum(p for p in range(2, n) if all(p % d for d in range(2, int(p ** 0.5) + 1)))
        q = f"What is the sum of all prime numbers less than {n}?"
    elif k == 2:
        xs = [rng.randint(-20, 40) for _ in range(8)]
        ans = sum(v * v for v in xs if v % 3 == 0)
        q = f"What does this Python function return for xs = {xs}?\n\n```python\ndef f(xs):\n    return sum(v * v for v in xs if v % 3 == 0)\n```"
    else:
        a, b, m = rng.randint(2, 30), rng.randint(5, 40), rng.randint(7, 97)
        q, ans = f"What is {a}^{b} mod {m}?", pow(a, b, m)
    return {"messages": [{"role": "user", "content": q + BOX}], "check": {"type": "number", "value": ans}}


def gen_thai(rng):
    if rng.random() < 0.5:
        q, ans = rng.choice(THAI_FACTS)
        return {"messages": [{"role": "user", "content": q}], "check": {"type": "contains", "any": ans}}
    a, b, x, y = rng.randint(30, 80), rng.randint(20, 60), rng.randint(1, 5), rng.randint(1, 5)
    q = (f"แม่ค้าขายมะม่วงกิโลกรัมละ {a} บาท และขายส้มกิโลกรัมละ {b} บาท สมศรีซื้อมะม่วง {x} กิโลกรัม และส้ม {y} กิโลกรัม "
         f"แล้วจ่ายด้วยธนบัตร 1000 บาท สมศรีจะได้รับเงินทอนกี่บาท ใส่คำตอบสุดท้ายที่เป็นตัวเลขไว้ใน \\boxed{{}}")
    return {"messages": [{"role": "user", "content": q}], "check": {"type": "number", "value": 1000 - a * x - b * y}}


def gen_agent(rng):
    skus = [f"SKU-{rng.randint(1000, 9999)}" for _ in range(3)]
    k = rng.randrange(4)
    if k == 0:
        qty = [rng.randint(1, 9) for _ in skus]
        q = ("Look up the current unit prices and tell me the total cost in THB of buying "
             + ", ".join(f"{n} x {s}" for n, s in zip(qty, skus)) + ". Put the final total inside \\boxed{}.")
        chk = {"type": "number", "value": sum(n * price(s) for n, s in zip(qty, skus))}
    elif k == 1:
        whs = ["BKK-1", "CNX-2", "HKT-3"]
        q = (f"How many units of {skus[0]} are in stock in total across warehouses {', '.join(whs)}? "
             "Check each warehouse with the tools. Put the final total inside \\boxed{}.")
        chk = {"type": "number", "value": sum(stock(skus[0], w) for w in whs)}
    elif k == 2:
        cur = rng.choice(["USD", "EUR"])
        q = f"What is the unit price of {skus[0]} in {cur}? Look up the THB price and convert it. Put the final number inside \\boxed{{}}."
        chk = {"type": "number", "value": convert(price(skus[0]), "THB", cur), "rel": 0.01}
    else:
        city = rng.choice(CITIES)
        q = f"What's the weather like in {city} right now? Should I bring an umbrella?"
        chk = {"type": "toolcall", "name": "get_weather", "arg": city.lower()}
    return {"messages": [{"role": "user", "content": q}], "tools": TOOLS, "episode": True, "check": chk}


GEN = {"chat": gen_chat, "rag": gen_rag, "agent": gen_agent, "reason": gen_reason, "thai": gen_thai, "long": gen_long}



# ═════════════════════════════════ schedule + freeze ═════════════════════════════════
def phase_bounds(n):
    n_r, n_p = round(n * config.RAMP_FRAC), round(n * config.PEAK_FRAC)
    n_q = n - n_r - n_p
    Tr = 2 * n_r / config.PEAK_RATE      # linear 0→peak ramp delivers peak*Tr/2 arrivals
    Tp, Tq = n_p / config.PEAK_RATE, n_q / config.QUIET_RATE
    return (n_r, n_p, n_q), {"ramp": (0.0, Tr), "peak": (Tr, Tr + Tp), "quiet": (Tr + Tp, Tr + Tp + Tq)}


def schedule(n, arrivals="uniform", seed=0):
    """Arrival times for n items + phase bounds. `uniform` = evenly spaced (greenbench); `poisson` = seeded
    exponential gaps in each phase (ramp via time-change of the integrated rate)."""
    (n_r, n_p, n_q), bounds = phase_bounds(n)
    Tr, Tp = bounds["ramp"][1], bounds["peak"][1] - bounds["peak"][0]
    if arrivals == "uniform":
        ts = [math.sqrt(2 * Tr * (i + 0.5) / config.PEAK_RATE) for i in range(n_r)]
        ts += [Tr + (i + 0.5) / config.PEAK_RATE for i in range(n_p)]
        ts += [Tr + Tp + (i + 0.5) / config.QUIET_RATE for i in range(n_q)]
        return ts, bounds
    rng = random.Random(f"arrivals-{seed}-{n}")

    def poisson_points(k, lo, hi):  # k points of a rate-homogeneous process, conditioned on count → sorted uniforms
        return sorted(rng.uniform(lo, hi) for _ in range(k))

    # ramp: rate r(t)=peak*t/Tr, integrated rate Λ(t)=peak*t²/(2Tr); uniform points in Λ-space map back via sqrt
    ts = [math.sqrt(2 * Tr * u / config.PEAK_RATE) for u in poisson_points(n_r, 0, config.PEAK_RATE * Tr / 2)]
    ts += poisson_points(n_p, Tr, Tr + Tp)
    ts += poisson_points(n_q, bounds["quiet"][0], bounds["quiet"][1])
    return ts, bounds


def phase_of(t, bounds):
    for name, (a, b) in bounds.items():
        if a <= t < b:
            return name
    return "drain"


def build_trace(n, seed, arrivals="uniform"):
    rng = random.Random(seed)
    tot = sum(CLASS_MIX.values())
    counts = {c: max(1, round(n * v / tot)) for c, v in CLASS_MIX.items()}
    counts["chat"] += n - sum(counts.values())
    items = []
    for c, k in counts.items():
        for _ in range(k):
            it = GEN[c](rng)
            it["cls"] = c
            items.append(it)
    rng.shuffle(items)
    ts, bounds = schedule(len(items), arrivals, seed)
    out = []
    for i, (it, t) in enumerate(zip(items, ts)):
        out.append({"id": i, "t": round(t, 3), "phase": phase_of(t, bounds), "cls": it["cls"],
                    "messages": it["messages"], "tools": it.get("tools"), "episode": bool(it.get("episode")),
                    "check": it["check"]})
    return out, bounds


def trace_path(track, root=None):
    return Path(root or ROOT) / config.TRACKS[track]["file"]


def freeze(seed=0, force=False, root=None, arrivals=None):
    """Write both trace files (+ a .meta.json sidecar each). Refuses to overwrite without force."""
    arrivals = arrivals or config.SETTING_DEFAULTS["ARRIVALS"]
    paths = {t: trace_path(t, root) for t in config.TRACKS}
    existing = [p for p in paths.values() if p.exists()]
    if existing and not force:
        raise FileExistsError(f"{', '.join(str(p) for p in existing)} exist; use --force to overwrite")
    for track, spec in config.TRACKS.items():
        items, bounds = build_trace(spec["n"], seed, arrivals)
        p = paths[track]
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        p.with_suffix(".meta.json").write_text(json.dumps(
            {"track": track, "n": spec["n"], "seed": seed, "arrivals": arrivals,
             "bounds": {k: list(v) for k, v in bounds.items()}}, indent=1))
    return paths


def load(track, root=None):
    """→ (items, bounds, meta). Bounds come from the sidecar, else recomputed from the item count."""
    p = trace_path(track, root)
    items = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta_p = p.with_suffix(".meta.json")
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        bounds = {k: tuple(v) for k, v in meta["bounds"].items()}
    else:
        meta, bounds = {}, phase_bounds(len(items))[1]
    return items, bounds, meta


def stats(items):
    """Counts per class/phase, approx prompt tokens (chars/4) per class, max prompt size."""
    by_class, by_phase, toks = {}, {}, {}
    for it in items:
        by_class[it["cls"]] = by_class.get(it["cls"], 0) + 1
        by_phase[it["phase"]] = by_phase.get(it["phase"], 0) + 1
        toks.setdefault(it["cls"], []).append(len(json.dumps(it["messages"], ensure_ascii=False)) // 4)
    per_class = {c: {"mean": sum(v) / len(v), "max": max(v)} for c, v in toks.items()}
    return {"n": len(items), "by_class": by_class, "by_phase": by_phase, "prompt_tokens_approx": per_class,
            "max_prompt_tokens_approx": max((v["max"] for v in per_class.values()), default=0)}
