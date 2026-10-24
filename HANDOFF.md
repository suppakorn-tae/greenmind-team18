# Handoff — end of session Sat 24 Oct 2026 ~16:15 +07

## State
- **Champion E017** is serving on :8000 (`configs/champion.sh`, 400 W default, no clock lock), and health passes.
  Restart it with `tools/restart.sh champion`. That script stops any vLLM, waits for the GPU to go idle, starts the config in its own session, and waits for `/health`.
  Don't `pkill -f "vllm serve"` from a shell whose own command line contains that string, because it kills the shell.
- Fallback: `configs/E008.sh` (official val S 31.5).

## Numbers (OFFICIAL vs local are labelled)
| | S | η vs base | J/req | SLO-met | p95 TTFT | p95 TPOT | acc |
|---|---|---|---|---|---|---|---|
| OFFICIAL baseline val | — | 0.89× | 467 | 64/100 | 2.06 s | 48 ms | 87.2% |
| OFFICIAL E008 val | 31.5 | 1.27× | 333 | 65/100 | 2.02 s | 38 ms | 88.9% |
| **OFFICIAL E017 final** | **59.0** | 1.78× | 247 | 215/300 | 1.91 s | 59 ms | 88.8% |
| local E017 val (3 runs) | 13–42 (A noise) | 1.30–1.44× | 311–338 | 78–80/100 | 1.5 s | 50 ms | 76–80% |
| local E017 final | 22.7 | 1.30× | 209 | 117/300 | 3.88 s | 80 ms | 81.0% |

E017 has **no official val run yet**. Worth one if the Final budget is scarce, but the official final is what counts.

## Untested ideas, ranked by expected payoff
1. **TPOT on the official final** (sT 0.62, p95 59 ms is now the weakest term). Try MTP k=3 on top of E017.
   Under the old scheduler k=3 tied k=2, but the TPOT-bound regime may favour more tokens per step. Also `--max-num-seqs` 96–128 to bound decode batch on final.
2. **Cap and budget fine-tune around 4096/1024**, e.g. 6144/1024 or 4096/768. 2048/1024, 4096/512 and 4096/2048 were all worse.
3. **Power cap ~350 W or a locked SM clock on E017.** The prefill-scheduling fix may have created TTFT headroom that 250–300 W lacked before.
4. **`--kv-cache-dtype fp8`** (Ampere, hybrid + spec decode = experimental). KV is no longer binding on val, so it only helps the final track. Check accuracy hard.
5. **Own AWQ/GPTQ quantization with tool-calling and Thai calibration data.** Only worth it if agent accuracy holds, and the provided AWQ didn't hold it.
6. **Gray area, needs organizer OK:** thinking budget (2–5 cutoffs per run only, so small gain) and SLO-aware shedding.

## Open questions for organizers
- Is a pre-measurement warmup request burst allowed? Not used. Triton JIT warnings appear on the first requests.
- Is the provided AWQ-mtp checkpoint counted as "made during the event"? Moot, since we don't use it.
- Are the thinking budget and deadline-based shedding allowed? Neither is used.

## Pitch notes
- Diagnosis first: we read the vLLM scheduler log at peak (KV%, waiting queue, prompt vs generation throughput) and per-request TTFT staircases instead of sweeping blindly.
- Hybrid-model specific insight: MTP multiplies Gated-DeltaNet recurrent state per sequence, so a bf16 state is what makes speculative decoding pay off on this model.
- Scheduling insight: one long-context request blocks every short request behind it. A per-request prefill cap fixed TTFT without any accuracy or energy trade.
- Energy: 247 J/request official vs 467 baseline (−47%), accuracy 88.8% vs 87.2%. Original weights, open-source vLLM only.
- Failures we can explain: AWQ (agent accuracy), power caps (prefill compute-bound), prefix caching (3% hits), warm-run artifacts.
