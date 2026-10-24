# System and dependency versions (serving VM, measured 2026-10-25)

| Item | Version | Source |
|---|---|---|
| GPU | NVIDIA A100-SXM4-40GB (40960 MiB), default/max power limit 400 W | `nvidia-smi` |
| Driver / CUDA (driver) | 580.173.02 / 13.0 | `nvidia-smi`, /etc/image-manifest.txt |
| CUDA toolkit (nvcc) | 13.0 (V13.0.88) | `nvcc --version` (needed for FlashInfer JIT) |
| OS | Ubuntu 24.04.4 LTS, 16 vCPU, 47 GB RAM | `lsb_release`, `nproc`, `free` |
| Python | 3.12.3 | venv `~/vllm-env` |
| vLLM | 0.32.0 | `vllm.__version__` |
| torch | 2.13.0+cu130 | |
| transformers | 5.19.0 | |
| flashinfer-python | 0.7.0.post1 | |
| compressed-tensors | 0.17.0 | (loads the W8A8 checkpoint) |
| triton | 3.7.1 | |
| huggingface_hub | 2.3.0 | |
| Full lock | `requirements/vllm-env.lock.txt` (`pip freeze`, 198 packages) | |

The `~/vllm-env` venv came pre-installed on the organizer VM image; we did not record its install command.
To recreate: `python3.12 -m venv ~/vllm-env && ~/vllm-env/bin/pip install -r requirements/vllm-env.lock.txt` (TODO: not re-tested from scratch).
Also needed at runtime: `ninja` (in the venv) and `nvcc` on PATH for FlashInfer's sampler JIT.

## Engines evaluated but NOT used (installed in separate venvs, since deleted)
| Engine | Version | Result |
|---|---|---|
| SGLang | 0.5.22 (torch 2.14.1+cu130) | E041: best val 58/100 vs vLLM 80/100 |
| TensorRT-LLM | 1.4.0rc1 (PyTorch backend, torch 2.14.0+cu130, system libopenmpi-dev) | E042: warm 56/100, cold 24/100 |

## Quantization environment (made the W8A8 artifact; separate "quant pod", not this VM)
Recorded in the artifact's `config.json`: compressed-tensors 0.19.0, transformers 5.14.1. The toolkit's `setup_env.sh` installs
`torch>=2.10,<=2.14` (cu128), `llmcompressor>=0.14.0`, `compressed-tensors>=0.19.0`, `transformers>=5.15,<=5.17`.
TODO (team): exact llmcompressor / torch versions actually used for the uploaded artifact.
