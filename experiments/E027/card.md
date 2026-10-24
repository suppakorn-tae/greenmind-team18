# E027 — [GRAY AREA, LOCAL TEST ONLY] champion E017 + server-side thinking budget 5120
Approved by user for local measurement only (2026-10-24 ~18:05); never for champion.sh/official without organizer OK.
Parent E017. One variable: default thinking_token_budget=5120 (vLLM built-in forced reasoning end) via tools/gray/vllm_think_budget.py.
Why: in E017 runs no request finishes naturally between 5K and 8192 tokens; only the 3 cutoffs (8192, scored as failures) exceed 5K, and they extend the energy window ~25 s (drain phase ~50% of val energy). Expect: J/req -15..-25%, cutoffs -> answers (maybe correct), eta up. Risk: forced answer quality; budget-state overhead with spec decode. Start 2026-10-24T18:05:26+07:00.
