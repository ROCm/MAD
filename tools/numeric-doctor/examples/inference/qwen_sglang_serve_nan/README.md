# Qwen2.5-0.5B served by SGLang: NaN logprobs with SGLang's default settings

A Qwen2.5-0.5B model served by SGLang's offline `sglang.Engine.generate()` with SGLang's defaults (no engine
argument beyond what the workload needs: the checkpoint, token-id prompts, the context length, the seed)
returns non-finite log-probabilities for some requests: `Logprobs are not finite`.

The model is SGLang's own `Qwen2ForCausalLM` with the public `Qwen/Qwen2.5-0.5B` architecture in bf16, random
init, written as a local checkpoint. Requests are greedy, 256-token prompts given as token ids, 8 generated
tokens per request, each generated token's log-probability returned. The model runs in SGLang's scheduler
process, a child of the process that sends the requests; only the result comes back to that process.

| File | Content |
| --- | --- |
| `run.sh` | builds the image, runs the workload (Docker, Podman or Apptainer), prints a verdict |
| `Dockerfile` | the image: the public `lmsysorg/sglang:v0.5.23-rocm10-mi35x` pinned by digest, SGLang's Rust radix tree prebuilt, and the script |
| `qwen_sglang_serve.py` | the serving loop: the checkpoint, `sglang.Engine`, `generate()`, logprob check, one `RESULT` line |
| `capture.spec` | Numeric Doctor's capture setup (validated 2026-11-20) |

## Environment

| | |
| --- | --- |
| GPU | AMD Instinct MI350X (gfx950) |
| Image name (built by `run.sh`) | `qwen-sglang-serve:v0.5.23` |
| Base (Docker Hub) | `lmsysorg/sglang:v0.5.23-rocm10-mi35x` (MI350X/MI355X build) |
| Base digest | `sha256:cff45eca86da10fbe140f85df196ef241e932bbae8036afa45942e633c3ed6c5` |
| SGLang / PyTorch / HIP | 0.5.23 / 2.11.0+rocm10.0.0 / 7.15.26333 |

The `Dockerfile` installs nothing. It builds SGLang's Rust radix tree, the default prefix cache: the base does not
ship it, so every container would otherwise download its crates from crates.io and compile it on the first
request. The containers run as root, because the image's aiter writes into root-owned directories (its JIT
directory, `/tmp/aiter_configs`). With rootless Podman, root in the container is the host user. The container
writes nothing to the host; `run.sh` redirects the logs. The first request of each engine takes about 30 s
(JIT compilation).

Apptainer cannot build the `Dockerfile`: `run.sh` pulls the base as a SIF and runs the script from this folder,
as the host user; SGLang then builds its radix tree on the first request (network needed). The Podman and
Apptainer paths are not validated: the validation node had Docker only.

## Run

Needs one node with an AMD Instinct MI350X/MI355X GPU, Docker, Podman or Apptainer, and network access to
Docker Hub and crates.io for the first build (an image of about 14 GB to download, 60 GB unpacked).

```bash
cd examples/inference/qwen_sglang_serve_nan
./run.sh                                   # run the workload on GPU 0, then a verdict
sbatch -p <partition> run.sh               # the same on one exclusive SLURM node
./run.sh pull                              # only build / pull the image
```

Settings: `REQUESTS=200 PROMPT=256 NEW=8 SEED=0`, `GEN_ARGS=` (extra script arguments, e.g. `--engine-args
'{"log_level":"info"}'` for SGLang's start-up log, or any other `sglang.Engine` argument as JSON). Logs go to
`out/<date>/`. Without containers, on a machine with SGLang: `python3 qwen_sglang_serve.py --seed 0`.

## Symptom

Every run fails, on every GPU tried, at the same request: request 49 (0-based) with the default settings.
Requests 0..48 return finite logprobs (`request <i> mean_logprob <x>` every 50 requests), then:

```text
Logprobs are not finite at request 49: mean logprob nan
RESULT requests_done=50 nan_request=49 logprob0=... logprob_last=nan seed=0 prompt=256 engine_args={}
```

A clean run would end with `RESULT requests_done=200 nan_request=- ...`. `run.sh` ends with
`VERDICT REPRODUCED` (a non-finite logprob), `NOT_REPRODUCED` or `INCOMPLETE`.

Seen on MI350X (Slurm + docker, 2026-11-19, `sbatch run.sh`, default settings; shown in the current `run.sh` format):

```text
serve        nan_request=49
VERDICT REPRODUCED
```
