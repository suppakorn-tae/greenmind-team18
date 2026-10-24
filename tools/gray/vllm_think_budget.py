#!/usr/bin/env python3
"""Server-side thinking budget (backlog #12; authorized by team lead 2026-10-24 ~19:00, organizer confirmation still recommended).
Runs `vllm serve` with a server-side default `thinking_token_budget` (env THINK_BUDGET) applied to chat requests
that don't set one. Uses vLLM's built-in thinking-budget logits state (forces the reasoning end string)."""
import os, sys
from vllm.entrypoints.openai.chat_completion import protocol

BUDGET = int(os.environ["THINK_BUDGET"])
_orig = protocol.ChatCompletionRequest.to_sampling_params

def _with_budget(self, *a, **k):
    if self.thinking_token_budget is None:
        self.thinking_token_budget = BUDGET
    return _orig(self, *a, **k)

protocol.ChatCompletionRequest.to_sampling_params = _with_budget
if __name__ == "__main__":  # engine core is spawned and re-imports this module
    from vllm.entrypoints.cli.main import main
    sys.argv = ["vllm", *sys.argv[1:]]
    sys.exit(main())
