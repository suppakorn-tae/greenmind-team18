# E032 — champion E030 with MTP k=2 -> 3
Parent E030. One variable. Why: E020 (k=3 on E017) cut final TPOT fails 123 -> 23 but collapsed from memory (4 GDN state slots/seq; KV 98.7%, 51 waiting). With the 3072 thinking budget, long sequences are shorter, so peak KV may stay below the cliff. Risk: same collapse. Judge on local final (watch peak waiting/KV) + val. Start 2026-10-24T19:51:16+07:00.
