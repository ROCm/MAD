# Qwen2.5-0.5B inference with Hugging Face `generate()` returns non-finite logits on gfx950

An inference workload: `transformers`' `Qwen2ForCausalLM` (the public `Qwen/Qwen2.5-0.5B` architecture, random
init by default, or the checkpoint) runs greedy `generate()` under `torch.inference_mode()` in fp32, eager
attention, random-token prompts of 1280 tokens, 8 generated tokens per request, batch 1, 3 inference processes
per GPU on all 8 GPUs of a node. Each request's mean log-probability is read back to the host; now and then it is
not finite (`Logits are not finite`) and the process stops.

| File | Content |
| --- | --- |
| `run.sh` | builds the image, runs the workload on every GPU (Docker, Podman or Apptainer), prints a verdict |
| `Dockerfile` | the image: a public `rocm/pytorch` base pinned by digest + `transformers==5.20.2` + the script |
| `qwen_generate.py` | the inference loop: `transformers` Qwen2, `generate()`, NaN check, one `RESULT` line per process |

## Environment

| | |
| --- | --- |
| Image name (built by `run.sh`) | `qwen-hf-generate:rocm7.2.0` |
| Base (Docker Hub) | `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.10.0` |
| Base digest | `sha256:bcb2efa1d43f66cc59fa45a1975d621a50b2b06d21a2abdfa2ba821ab32763e4` |
| torch / HIP | 2.10.0+rocm7.2.0.gitb6ee5fde / 7.2.26015 |
| transformers | 5.20.2 |
| GPUs | 8 x MI350X (gfx950) |

The `Dockerfile` adds `transformers` (default 5.20.2, `TRANSFORMERS=` to change) with pip on top of the base;
torch, numpy and the ROCm libraries stay the base's. Apptainer cannot build the `Dockerfile`: `run.sh` pulls the
base as a SIF and installs `transformers` into `pydeps/<image>/` (without the packages the base already has),
which it puts on `PYTHONPATH`.

## Run

Needs one node with 8 gfx950 GPUs (MI350X / MI355X), Docker, Podman or Apptainer, and network access to Docker
Hub and PyPI for the first build.

```bash
cd examples/inference/qwen_hf_generate_nan
./run.sh                                   # build the image, run, print a verdict
sbatch -p <gfx950 partition> run.sh        # the same on one exclusive SLURM node
./run.sh pull                              # only build / pull the image
```

Settings: `GPUS=8 PROCS=3 REQUESTS=1000 PROMPT=1280 NEW=8 BS=1`, `SEED=` (random by default, printed),
`GEN_ARGS=` (extra script arguments). Logs go to `out/<date>/`, one per GPU.

Each process prints one line. A failing process:

```text
RESULT trial=<i> requests_done=<n> nan_request=<r> nan_token=<t> nan_module=<first non-finite module> ...
```

and a clean one has `nan_request=- nan_token=- nan_module=-`. `nan_request` is the first request with a
non-finite log-probability, `nan_token` the first generated token whose logits were non-finite (0 = the
prefill), `nan_module` the first `nn.Linear` or attention module whose output went non-finite. The output ends
with `processes_with_nan=<n>/<total>` and `VERDICT REPRODUCED`, `NOT_REPRODUCED` or `INCOMPLETE`.

With the pretrained checkpoint instead of random init:

```bash
mkdir -p models/qwen2.5-0.5b && cd models/qwen2.5-0.5b && for f in config.json model.safetensors; do
  curl -fsSLO "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/060db6499f32faf8b98477b0a26969ef7d8b9987/$f"; done
cd ../.. && GEN_ARGS="--model-dir models/qwen2.5-0.5b" ./run.sh
```

Without containers, in the same environment:
`python3 qwen_generate.py --prompt-len 1280 --new-tokens 8 --batch-size 1 --requests 1000 --trials 3 --seed <seed>`

## What we see (MI350X, 2026-11-19)

One job (random init, default settings, 24 processes of 1000 requests each):

```text
processes_with_nan=10/24
VERDICT REPRODUCED
```

About 4 processes in 10 fail, at requests 4 to 674 (8 of 10 within the first 250); the failing request varies
from run to run. Every failure was at `nan_token=0` (the prefill), and the first non-finite module was an
attention projection in a different layer each time (`q_proj` or `o_proj`). A clean process takes about 0.35 s
per request; the whole job, with the image build, takes about 30 min.

## For Numeric Doctor

There is no `capture.spec` here yet: deriving the capture setup for an inference loop is part of what this
example tests. The loop to wire is the request loop in `trial()` (one `cap.step()` per `generate()` call). The
NaN reaches the host through `.item()`-style reads (`float(...)` of the mean log-probability), which is what
settles a capsule in settle mode.
