# vLLM at TP2: NaN logprobs on some requests

A Qwen2.5-1.5B-shaped model served by vLLM at tensor parallel 2 on MI350X (gfx950) returns NaN log-probabilities
for some requests. The run stops at the first one: request 9 (a 1024-token prompt), in every run. Requests 0..8
are fine.

## The workload

vLLM's own `Qwen2ForCausalLM` with the public `Qwen/Qwen2.5-1.5B` architecture (hidden 1536, 28 layers) in bf16,
random init, served at TP2 by offline `LLM.generate()` with
vLLM's defaults (`torch.compile` and HIP graphs): greedy, prompts given as token ids, 8 generated tokens per request, their log-probabilities returned.
Requests are 256 tokens, except every 10th, which is 1024 tokens. The server's environment sets
`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT8`.

| File | Content |
| --- | --- |
| `run.sh` | builds the image, runs the serving loop on 2 GPUs (Docker, Podman or Apptainer), prints `RESULT` and a verdict |
| `Dockerfile` | the image: the public `vllm/vllm-openai-rocm:v0.33.0-rocm72` pinned by digest + the script; nothing installed |
| `qwen_vllm_serve.py` | the serving loop: vLLM `LLM` at TP2, `generate()`, logprob check, one `RESULT` line |

## Image

| | |
| --- | --- |
| Image name (built by `run.sh`) | `qwen-vllm-tp2:v0.33.0-rocm72` |
| Base (Docker Hub) | `vllm/vllm-openai-rocm:v0.33.0-rocm72` |
| Base digest | `sha256:4fe7bc121cbe2f0791ebc38dcc2f0d615612233b4bfc0c1f4b9daeab05aba182` |
| vLLM / PyTorch / HIP | 0.33.0 / 2.13.0+git733fca1 / 7.2.53211 (ROCm 7.2) |

The `Dockerfile` adds nothing but the script and clears the base's entrypoint (the OpenAI server). Apptainer
cannot build the `Dockerfile`: `run.sh` pulls the base as a SIF and runs the script from this folder.

## Run

Needs one node with 2 AMD Instinct GPUs (seen on MI350X), Docker, Podman or Apptainer, and network access to
Docker Hub for the first build (about 12 GB).

```bash
cd examples/inference/qwen_vllm_tp2_nan
./run.sh                                   # serve 40 requests, then RESULT and a verdict (~1.5 min with the image)
sbatch -p <partition> run.sh               # the same on one exclusive SLURM node
./run.sh pull                              # only build / pull the image
```

Settings: `REQUESTS=40 LONG_EVERY=10 SHORT=256 LONG=1024 NEW=8 SEED=0 GPUS=0,1`, `GEN_ARGS=` (extra script
arguments). Logs go to `out/<date>/serve.log`. Each engine logs a `Chat template warmup failed` traceback at
start-up: a warning of `skip_tokenizer_init=True` (the prompts are token ids), not an error. The output ends with
`VERDICT REPRODUCED` (the requests went non-finite), `NOT_REPRODUCED` or `INCOMPLETE`.

Without containers, on a ROCm machine with vLLM:
`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT8 python3 qwen_vllm_serve.py --tp 2`

## What we see (MI350X, 2026-11-20)

```text
Logprobs are not finite at request 9 (1024-token prompt): mean logprob nan
RESULT nonfinite=yes requests_done=10 nan_request=9 long_requests=1 ... tp=2 seed=0
VERDICT REPRODUCED
```

A clean run would end with `RESULT nonfinite=no requests_done=40 nan_request=- long_requests=4 ...`.

NAN_SIGNAL: `Logprobs are not finite`
