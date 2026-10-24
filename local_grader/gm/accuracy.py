"""Answer checkers (port of greenbench `extract_number` / `is_correct`) and per-class accuracy."""
import re


def extract_number(text):
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    s = boxed[-1] if boxed else text
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not nums:
        return None
    return float(nums[0] if boxed else nums[-1])


def check_answer(chk, text, calls):
    """Pure answer check. `text` = final-turn content, `calls` = every tool call made in the episode
    (dicts with name/arguments). Request-level failures are handled by `class_accuracy`, not here."""
    text = text or ""
    t = chk["type"]
    if t == "nonempty":
        return len(text.strip()) > 0
    if t == "contains":
        norm = lambda s: re.sub(r"\s+", "", s).lower()
        return any(norm(a) in norm(text) for a in chk["any"])
    if t == "number":
        v = extract_number(text)
        tol = max(0.01, chk.get("rel", 0) * abs(chk["value"]))
        return v is not None and abs(v - chk["value"]) <= tol
    if t == "toolcall":
        return any(c["name"] == chk["name"] and chk["arg"] in (c["arguments"] or "").lower() for c in calls)
    return False


HARD_FAILS = {"error", "no_stream", "no_usage", "cutoff"}


def class_accuracy(items, final_fails, settings):
    """items: rows of items.jsonl ({id, cls, correct, ...}); final_fails: {item_id: [fail reasons of the item's
    final turn]} evaluated under `settings`. FAILED_IN_ACCURACY=wrong → a hard-failed final turn is wrong;
    =excluded → such items are dropped from the class mean. → (class_acc, class_n)"""
    acc, n = {}, {}
    buckets = {}
    for it in items:
        failed = bool(HARD_FAILS & set(final_fails.get(it["id"], [])))
        if failed and settings["FAILED_IN_ACCURACY"] == "excluded":
            continue
        buckets.setdefault(it["cls"], []).append(bool(it["correct"]) and not failed)
    for c, xs in buckets.items():
        acc[c], n[c] = sum(xs) / len(xs), len(xs)
    return acc, n
