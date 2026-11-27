# DeepSeek-V3.2 / GLM-5-shaped sparse-MLA decode: logits are not finite from the first decode step

A greedy decode loop of a DeepSeek-V3.2 / GLM-5-shaped model with sparse attention, running on vLLM's ROCm
image with aiter, produces non-finite logits. On MI350X (gfx950) it happens at every decode step, starting at
step 0, on every run.

## The workload

`sparse_mla_decode_serve.py` is a greedy-decode loop over one tensor-parallel rank (TP8) of a small decoder with
DeepSeek-V3.2 / GLM-5 sparse attention, random init (no download):

- MLA with 16 local heads (128 / 8), qk_nope 128, qk_rope 64, v_head 128, kv_lora_rank 512, absorbed-weight
  decode: Q is 576 wide, the cache holds one 576-wide latent row per token, the output is 512 wide. Hidden size
  2048, 4 layers, SwiGLU MLP, vocab 8192.
- A DeepSeek-V3.2-style lightning indexer (4 heads, dim 128, ReLU scores weighted per head) that selects the
  top 2048 context positions for every decode token and layer.
- A 65536-slot KV pool with page_size 1 and shuffled physical slots. One request with an 8192-token prompt;
  the prefill runs in PyTorch (dense causal). Then 128 greedy decode steps, each calling aiter's
  `mla_decode_fwd` once per layer with the 2048 selected slots as `kv_indices`, the way ATOM calls it for a
  non-segmented MLA cache.
- The logits are checked at every decode step.

| File | Content |
| --- | --- |
| `run.sh` | builds the image and runs the decode with Docker, Podman or Apptainer, then prints the result |
| `Dockerfile` | `vllm/vllm-openai-rocm:v0.33.0-rocm100` pinned by digest, aiter's `mla.py` from our aiter build, plus the script |
| `aiter_mla.py` | `aiter/mla.py` from aiter commit 95abba475, the aiter build this deployment uses |
| `sparse_mla_decode_serve.py` | the decode loop and one `RESULT` line |

## Environment

| | |
| --- | --- |
| Image name (built by `run.sh`) | `sparse-mla-decode:v1` |
| Base (Docker Hub) | `vllm/vllm-openai-rocm:v0.33.0-rocm100` |
| Base digest | `sha256:4fcd39b758b660c4e9c345c021e2ea8f0ccdb31e3c32507d676cc3005ae83fba` |
| PyTorch / HIP | 2.13.0+rocm10.0.0 / 7.15.26333 |
| amd-aiter | 0.1.26, with `aiter/mla.py` from aiter commit 95abba475 (`aiter_mla.py`, sha256 `40cef25d...baf19281f`) |
| GPU | MI350X (gfx950) |

Apptainer cannot build the `Dockerfile`: `run.sh` pulls the base as a SIF and bind-mounts `aiter_mla.py` over
`aiter/mla.py`. The containers run as the host user.

## Run

You need one node with a gfx950 GPU (MI350X or MI355X), Docker, Podman or Apptainer, and access to Docker Hub
for the first build (one image of about 20 GB).

```bash
cd examples/inference/deepseek_sparse_mla_decode_nan
./run.sh                                   # the decode on one GPU, then a verdict (about 20 s once the image is there)
sbatch -p <partition> run.sh               # the same on one exclusive SLURM node
./run.sh pull                              # only build / pull the image
```

In the image: `python3 sparse_mla_decode_serve.py` on one GPU. Settings: `PROMPT=8192 NEW=128 LAYERS=4
REQUESTS=1 SEED=0`, plus `GEN_ARGS=` for extra script arguments. Logs go to `out/<date>/decode.log`.

## Symptom

The signal line is `Logits are not finite at decode step N`. A failing run prints:

```text
Logits are not finite at decode step 0
RESULT wrong=yes steps_with_nonfinite_logits=127/127 first_nonfinite_step=0
```

and `run.sh` ends with

```text
decode   wrong=yes  steps_with_nonfinite_logits=127/127 first_nonfinite_step=0
VERDICT REPRODUCED
```

A clean run would print `RESULT wrong=no steps_with_nonfinite_logits=0/127 first_nonfinite_step=-` and
`VERDICT NOT_REPRODUCED`.
