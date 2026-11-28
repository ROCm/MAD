# Qwen2.5-0.5B served by vLLM: NaN log-probabilities at request 49

A vLLM serving workload run with vLLM's defaults (`torch.compile` and HIP graphs) returns non-finite
log-probabilities. It happens in every run, on every GPU tried, at the same request (49 with the default
settings).

The model is vLLM's own `Qwen2ForCausalLM` with the `Qwen/Qwen2.5-0.5B` architecture in bf16, random init,
served by offline `LLM.generate()`: greedy, 256-token prompts given as token ids, 8 generated tokens per
request, and each generated token's log-probability returned. A non-finite value is the failure
(`Logprobs are not finite`).

| File | Content |
| --- | --- |
| `run.sh` | builds the image, runs the serving loop on one GPU (Docker, Podman or Apptainer), prints a verdict |
| `Dockerfile` | the image: the public `vllm/vllm-openai-rocm:v0.33.0-rocm100` pinned by digest + the script; nothing installed |
| `qwen_vllm_serve.py` | the serving loop: vLLM `LLM`, `generate()`, finiteness check, one `RESULT` line |
| `capture.spec` | a debug-numeric capture setup for this workload |

## Image

| | |
| --- | --- |
| Image name (built by `run.sh`) | `qwen-vllm-serve:v0.33.0` |
| Base (Docker Hub) | `vllm/vllm-openai-rocm:v0.33.0-rocm100` |
| Base digest | `sha256:4fcd39b758b660c4e9c345c021e2ea8f0ccdb31e3c32507d676cc3005ae83fba` |
| vLLM / PyTorch / HIP | 0.33.0 / 2.13.0+rocm10.0.0 / 7.15.26333 |

The `Dockerfile` adds nothing but the script and clears the base's entrypoint (the OpenAI server). Apptainer
cannot build the `Dockerfile`: `run.sh` pulls the base as a SIF and runs the script from this folder.

## Run

Needs one node with an AMD Instinct GPU (seen on MI350X), Docker, Podman or Apptainer, and network access to
Docker Hub for the first build (one image of about 20 GB).

```bash
cd examples/inference/qwen_vllm_serve_nan
./run.sh                                   # the serving loop on GPU 0, then a verdict
sbatch -p <partition> run.sh               # the same on one exclusive SLURM node
./run.sh pull                              # only build / pull the image
```

Settings: `REQUESTS=200 PROMPT=256 NEW=8 SEED=0 GPU=0`, `GEN_ARGS=` (extra script arguments). Logs go to
`out/<date>/serve.log`. The output ends with the `RESULT` line and `VERDICT REPRODUCED` (a non-finite
log-probability), `NOT_REPRODUCED` or `INCOMPLETE`. The engine logs a `Chat template warmup failed` traceback
at start-up: a warning of `skip_tokenizer_init=True` (the prompts are token ids), not an error.

A failing run:

```text
Logprobs are not finite at request 49: mean logprob nan
RESULT requests_done=50 nan_request=49 logprob0=... logprob_last=nan seed=0 prompt=256 eager=0
VERDICT REPRODUCED
```

A clean run would end with `RESULT requests_done=200 nan_request=- ...` and finite logprobs.

Without containers, on a machine with vLLM: `python3 qwen_vllm_serve.py --requests 200 --seed 0`.

## Seen on

MI350X (2026-11-19), one exclusive node, default settings: `nan_request=49`, about 5 minutes in all, most of
it the image pull. Requests 0..48 are finite.
