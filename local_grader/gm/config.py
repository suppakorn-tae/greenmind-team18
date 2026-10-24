"""Constants and calibration switches (spec section 9). Everything the official grader leaves unspecified
is a named setting here; the effective settings are stored in each run row so `rescore` can reproduce them."""

MAX_TOKENS = 8192
TIMEOUT_S = 900
TPOT_SLO = 0.050
MAX_TURNS = 15
K_ENERGY = 3.0
TTFT_BASE_S = 0.5
TTFT_PER_1K_S = 0.25

PEAK_RATE = 4.0    # req/s
QUIET_RATE = 1.0   # req/s
RAMP_FRAC = 0.2
PEAK_FRAC = 0.6

CLASS_MIX = {"chat": 90, "rag": 60, "agent": 60, "reason": 45, "thai": 30, "long": 15}  # official Final, per 300
CHECKED = sorted(c for c, n in CLASS_MIX.items() if n >= 20)  # classes subject to class_factor (not `long`)
PHASES = ("ramp", "peak", "quiet", "drain")

TRACKS = {
    "val": {"n": 100, "file": "traces/validation.jsonl"},
    "final": {"n": 300, "file": "traces/final.jsonl"},
}

SETTING_CHOICES = {
    "TTFT_FROM": ("any", "content_only"),
    "TPOT_MODE": ("per_request", "p95_over_tokens"),
    "ENERGY_WINDOW": ("first_arrival_to_last_done", "first_send_to_last_done"),
    "ARRIVALS": ("uniform", "poisson"),
    "FAILED_IN_ACCURACY": ("wrong", "excluded"),
    "EPISODE_TURNS_SCORED": ("each_turn", "episode_once"),
}
SETTING_DEFAULTS = {k: v[0] for k, v in SETTING_CHOICES.items()}


def resolve_settings(overrides=None):
    """Defaults merged with overrides (e.g. a stored settings_json); unknown keys / bad values raise."""
    s = dict(SETTING_DEFAULTS)
    for k, v in (overrides or {}).items():
        if k not in SETTING_CHOICES:
            raise ValueError(f"unknown setting {k!r}")
        if v not in SETTING_CHOICES[k]:
            raise ValueError(f"{k}={v!r} not in {SETTING_CHOICES[k]}")
        s[k] = v
    return s
