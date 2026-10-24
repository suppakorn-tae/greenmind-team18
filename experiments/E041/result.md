# E041 result — abandoned (SGLang 0.5.22 slower than vLLM on this model/GPU)
SGLang serves the model correctly (qwen3_5 + qwen3_5_mtp NEXTN, qwen3_coder tools, qwen3 reasoning; grader health PASS; local runs need grader's --force because its foreign-process check only recognises vllm).
All runs without thinking budget -> compare to vLLM E017 (val 78-80/100, eta ~1.34x, TTFT fails 9-12).
| attempt | val run | per-arrival | eta | TTFT fails | p95 TTFT |
|---|---|---|---|---|---|
| mrr 80, chunk 4096, no mixed chunk | 20261024-215711-e041 | 32/100 | 0.68x | 61 | 6.2 s |
| + --enable-mixed-chunk, decode graphs to bs 80 | 20261024-220059-e041-mix | 58/100 | 0.92x | 41 | 3.2 s |
| + chunk 1024 | 20261024-220411-e041-c1024 | 44/100 | 0.84x | 57 | 4.9 s |
Why slower: (1) GDN spec-decode keeps intermediate SSM states per slot (~96 MB/slot) -> only 80 slots fit, KV pool 169K tokens (vLLM 475K); (2) MTP draft loads its own 4.2 GB of weights (vLLM shares); (3) default decode CUDA graphs only to bs 32; (4) no per-request long-prefill cap equivalent to vLLM's --long-prefill-token-threshold, so short arrivals queue behind long prefills.
Interesting: agent class 9/20 on run 1 (vLLM 4-6/20) — tool-call parsing differs; not pursued (A already 1.0 officially).
Lesson: on 1x A100 40 GB with this hybrid GDN + MTP model, vLLM 0.32 is the better engine. TensorRT-LLM not attempted (no GDN/Qwen3.5 support expected; heavy build).
