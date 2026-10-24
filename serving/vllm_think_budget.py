#!/usr/bin/env python3
"""Server-side thinking budget (backlog #12; authorized by team lead 2026-10-24 ~19:00, organizer confirmation still recommended).
Runs `vllm serve` with a server-side default `thinking_token_budget` (env THINK_BUDGET) applied to chat requests
that don't set one. Uses vLLM's built-in thinking-budget logits state (forces the reasoning end string).

Optional (env SHORT_FIRST_CAP=<tokens>, off by default; E040): while short requests (prompt <= SHORT_PROMPT_TOKENS,
default 1024) are waiting, that scheduler step uses SHORT_FIRST_CAP as the per-request long-prefill chunk cap instead
of --long-prefill-token-threshold, so in-flight long prefills yield most of the step to the short arrivals.
This module-level patch also runs in the spawned EngineCore process (spawn re-imports this file)."""
import os, sys
from vllm.entrypoints.openai.chat_completion import protocol

BUDGET = int(os.environ["THINK_BUDGET"])
_orig = protocol.ChatCompletionRequest.to_sampling_params

def _with_budget(self, *a, **k):
    if self.thinking_token_budget is None:
        self.thinking_token_budget = BUDGET
    return _orig(self, *a, **k)

protocol.ChatCompletionRequest.to_sampling_params = _with_budget

_SHORT_CAP = int(os.environ.get("SHORT_FIRST_CAP", "0"))
if _SHORT_CAP > 0:
    from vllm.v1.core.sched.scheduler import Scheduler

    _SHORT_PROMPT = int(os.environ.get("SHORT_PROMPT_TOKENS", "1024"))
    _orig_schedule = Scheduler.schedule

    def _short_first_schedule(self, *a, **k):
        cfg = self.scheduler_config
        normal = cfg.long_prefill_token_threshold
        if normal and any(r.num_prompt_tokens <= _SHORT_PROMPT for r in self.waiting):
            cfg.long_prefill_token_threshold = min(normal, _SHORT_CAP)
            try:
                return _orig_schedule(self, *a, **k)
            finally:
                cfg.long_prefill_token_threshold = normal
        return _orig_schedule(self, *a, **k)

    Scheduler.schedule = _short_first_schedule
    print(f"[E040 patch] short-first scheduling: cap {_SHORT_CAP} while prompts <= {_SHORT_PROMPT} tok wait (pid {os.getpid()})", file=sys.stderr, flush=True)

if __name__ == "__main__":  # engine core is spawned and re-imports this module
    from vllm.entrypoints.cli.main import main
    sys.argv = ["vllm", *sys.argv[1:]]
    sys.exit(main())
