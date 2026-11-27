# DeepSeek-V3-shaped MLA decode: wrong finite attention outputs, greedy tokens differ between identical passes

**Validated** on MI350X (gfx950, 2026-11-20): 143/2540 decode attention calls failed the fp32 parity check
(worst error 0.97) and the greedy tokens changed between identical passes. There is no NaN and no Inf: the
outputs are wrong **finite** values that differ from run to run on the same inputs.

## The workload

`mla_decode_serve.py` is a batched greedy-decode loop over a small decoder with DeepSeek-V3's attention
geometry, random init (no download):

- MLA with 128 heads, qk_nope 128, qk_rope 64, v_head 128, kv_lora_rank 512. Decode uses absorbed weights, the
  way vLLM and SGLang run MLA: Q is 576 wide (512 latent + 64 rope), the cache holds one 576-wide latent row per
  token, and the output is 512 wide and goes through W_UV. Hidden size 2048, 2 layers, SwiGLU MLP, vocab 8192.
- A paged KV cache with page_size 1. Physical slots come from a shuffled free list, as in vLLM's block table.
- 512 requests with prompts of 16-96 tokens. The prefill runs in PyTorch. Then 128 greedy decode steps follow
  (kv_len up to 224), each calling AITER's `mla_decode_fwd` once per layer.
- The generation runs `--passes 10` times on the same prompts and weights.

The signal is wrong finite values, so the workload has two explicit checks:

1. **Parity.** Every decode attention call is compared with an fp32 PyTorch reference on the same Q and KV
   cache. A call is bad if any request's output is off by more than `--tol 0.03`. Correct calls stay at or below
   about 0.01.
2. **Determinism.** The greedy tokens of every pass are compared with pass 0.

| File | Content |
| --- | --- |
| `run.sh` | builds the image and runs the workload with Docker, Podman or Apptainer, then prints a verdict |
| `Dockerfile` | the image: the public `vllm/vllm-openai-rocm:v0.33.0-rocm100` pinned by digest, plus the script. Nothing is installed. |
| `mla_decode_serve.py` | the decode loop, the parity and determinism checks, and one `RESULT` line |

## Image

| | |
| --- | --- |
| Image name (built by `run.sh`) | `deepseek-mla-decode:v0.33.0` |
| Base (Docker Hub) | `vllm/vllm-openai-rocm:v0.33.0-rocm100` |
| Base digest | `sha256:4fcd39b758b660c4e9c345c021e2ea8f0ccdb31e3c32507d676cc3005ae83fba` |
| PyTorch / HIP | 2.13.0+rocm10.0.0 / 7.15.26333 |
| amd-aiter | 0.1.26 |

Apptainer cannot build the `Dockerfile`, so `run.sh` pulls the base as a SIF and runs the script from this
folder. The containers run as the host user. Nothing is JIT compiled.

## Run

You need one gfx950 GPU (MI350X or MI355X; validated on MI350X), Docker, Podman or Apptainer, and access to
Docker Hub for the first build (one image of about 20 GB).

```bash
cd examples/inference/deepseek_mla_decode_wrong
./run.sh                                   # the workload, then a verdict (about 1 minute once the image is there)
sbatch -p <partition> run.sh               # the same on one exclusive SLURM node
./run.sh pull                              # only build / pull the image
```

The output ends with a line for the run and `VERDICT REPRODUCED` (wrong outputs seen), `NOT_REPRODUCED` or
`INCOMPLETE`. Settings: `REQUESTS=512 NEW=128 PASSES=10 LAYERS=2 TOL=0.03 SEED=0`, plus `GEN_ARGS=` for extra
script arguments. Logs go to `out/<date>/`. The counts change from run to run; in the validated run every one of
the 10 passes had bad calls (8 to 19 each), about 6% of all decode attention calls.

A failing run prints, once the tokens first diverge:

```text
Generations differ between identical passes: pass 1 vs pass 0, 5/512 requests
```

and at the end:

```text
RESULT wrong=yes bad_calls=143/2540 bad_rows=162 worst_err=0.9692 tol=0.03 passes_diverged=9/9 requests_diverged=34/512 first_bad=pass0/step24/layer1/req342(kv_len=113)
```

A clean run would print `RESULT wrong=no bad_calls=0/2540 ... passes_diverged=0/9 requests_diverged=0/512
first_bad=-`.

`bad_calls` counts the decode attention calls (pass x step x layer) with any request over `tol` against fp32,
and `bad_rows` counts the requests involved. `passes_diverged` counts the passes whose greedy tokens differ from
pass 0, and `requests_diverged` the requests whose tokens differed in any pass. `first_bad` is the first bad
call.

Without containers, on a gfx950 machine with aiter: `python3 mla_decode_serve.py`.
