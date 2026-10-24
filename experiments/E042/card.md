# E042 — TensorRT-LLM 1.4.0rc1 (PyTorch backend) engine
Parent E030 (vLLM, OFFICIAL 88.0). Variable: engine -> TensorRT-LLM 1.4.0rc1 (Apache-2.0; has modeling_qwen3_5.py with MTP mapping), official BF16 weights, trtllm-serve OpenAI server.
Rules note: 1.4 no longer requires the closed  package (PyTorch backend), but depends on NVIDIA binary wheels (cutlass-dsl, matmul-heuristics, cuda-tile). Organizer confirmation needed before any official use.
Plan: bring-up (health: stream+usage, tool calls qwen3_coder, reasoning parser), then val vs vLLM E017-equivalent (no budget), then thinking budget if TRT-LLM has one. Separate venv ~/trtllm-env (vLLM untouched); SGLang env deleted (abandoned, reinstallable). Start 2026-10-24T22:15:29+07:00.
