# E022 result — rejected
350 W cap. val 20261024-165748-e022: 123/146, 77/100, eta 1.33x, 294 J/req, p95 TPOT 61 ms, fails ttft 14/tpot 9.
val #2 20261024-165940-e022-2: 130/160, 70/100, eta 1.27x, 326 J/req, fails ttft 20/tpot 9/cutoff 4.
Mean eta 1.30 vs champion 1.36 (1.44/1.30/1.34), SLO 70-77 vs 78-80 -> slightly worse; avg power only 324-329 W vs 344 W.
Key finding (phase energy): drain phase = 14.5-17.6 kJ of ~30-33 kJ per val run (~50%). Drain = last ~25-50 s where 1-3 reasoning requests run to the 8192-token cutoff (they fail anyway) at low batch. A power cap can't touch that (draw already below cap) -> try a locked SM clock (memory-bound low-batch decode loses little).
