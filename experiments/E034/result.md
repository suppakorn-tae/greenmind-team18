# E034 result — crashed (OOM)
util 0.98 (KV pool 540,374 tokens). val 20261024-201855-e034: CUDA OOM under load ~45 s in -> 37 errors, 45 no_stream, 88 no_usage; server died; final not run.
Lesson: 0.96 is the ceiling with MTP k=2 + 4096 prefill budget on 40 GB (0.96 survived 2 val + 2 final runs, no OOM).
