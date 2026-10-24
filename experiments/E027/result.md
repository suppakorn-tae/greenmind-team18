# E027 result — [GRAY AREA] LOCAL WIN, NOT SHIPPABLE without organizer OK
Server-side default thinking_token_budget=5120 (vLLM built-in forced reasoning end; tools/gray/vllm_think_budget.py), else = E017.
- val 20261024-180730-e027: 144/160, 84/100, eta 1.76x, 261 J/req, p95 TTFT 1.49 s, p95 TPOT 50 ms, fails ttft 8/tpot 9/**cutoff 0**, acc 80%.
- val #2 20261024-181152-e027-2: 139/160, 80/100, eta 1.58x, 280 J/req, fails ttft 10/tpot 11/cutoff 0, acc 78%.
- final 20261024-180854-e027: S 25.2 (E017 22.7), A 1.00, eta 1.38x (1.30), 200 J/req (209), 118/300, cutoff 0 (E017: 3), acc 82.3% (81.0): reason 0.956 (0.933), long 0.733 (0.60), others equal.
vs E017 val mean: eta 1.67 vs 1.36 (+23%), J/req -17%. Mechanism: the 2-4 requests that previously ran to the 8192 cutoff (always scored as failures) now stop thinking at 5120 and answer; this shortens the energy window's drain phase (~25 s at ~300 W) and turns some of them into correct answers.
In E017 runs no request finished naturally between 5K and 8192 tokens, so the budget only touches requests that were already failing.
Status: evidence for the organizer question; do NOT put in champion.sh / official runs unless organizers approve.
