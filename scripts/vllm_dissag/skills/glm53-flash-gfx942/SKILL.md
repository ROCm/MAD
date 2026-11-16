---
name: glm53-flash-gfx942
description: Builds, launches and verifies the GLM-5.3-Flash 1P/1D disaggregated cards on gfx942 (MI300X / MI325X), FP8 or MXFP4 checkpoint with EP8 prefill (GLM-5.3-Flash-{FP8,MXFP4}-gfx942) or hybrid TP4 x DP2 prefill (...-gfx942-hyb), and checks each leg runs the intended layout and expert format. Use before submitting a GLM-5.3-Flash card, when its load log or accuracy looks wrong, or when choosing between the checkpoints or the prefill layouts.
---

# GLM-5.3-Flash on gfx942

All four cards use one image (`docker/vllm_disagg_inference.glmv53flash.gfx942.ubuntu.amd.Dockerfile`) and the same decode
leg: EP8 (MTP, FULL_AND_PIECEWISE), MoRIIO READ, max-model-len 760000. Prefill is eager, EP8 or hybrid TP4 x DP2.

## 1. Pick the card

| Card | Checkpoint (`MODEL_WEIGHTS_NAME`) | Size | Use when |
|---|---|---|---|
| `GLM-5.3-Flash-FP8-gfx942` | `GLM-5.3-Flash-FP8` (`zai-org/GLM-5.3-Flash`) | 306 GB | baseline |
| `GLM-5.3-Flash-MXFP4-gfx942` | `GLM-5.3-Flash-MXFP4` (`amd/GLM-5.3-Flash-Quark-MXFP4`) | 173 GB | many long sessions; decode KV per GPU +17% (MI325X) / +31% (MI300X) |

Then the prefill layout: the `-hyb` cards (`GLM-5.3-Flash-FP8-gfx942-hyb`, `GLM-5.3-Flash-MXFP4-gfx942-hyb`) run prefill as
TP4 x DP2 (2 engines, attention over 4 GPUs, experts still EP over all 8) with a KDA prefix-cache state kept every 17408
tokens. Each long prompt prefills ~2x faster and a session's second request resumes from the shared prefix, so time to
first token on agentic traffic drops; but only 2 prompts prefill at once instead of 8, so bursts of concurrent cold
prompts favour the EP8 cards. The settings are listed in the README ("EP8 or hybrid prefill").

gfx942 has no FP4 matrix math. The MXFP4 card converts the experts exactly to FP8 128x128 blocks on prefill and keeps
them 4-bit on decode; accuracy equals the FP8 card. Its decode step is ~3-4% slower than FP8.

Expert all-to-all: all four cards use **MoRIIO** (KV transfer) + vLLM **`allgather_reducescatter`** (expert all-to-all);
the cards override the combo-3 `mori` wideEP default. MoRI-EP is a tested, accurate opt-in (decode-TPOT parity on MI325X,
~6-7% slower on MI300X), not the default — to enable it see `Recipe_enable_MoriEP.md` in this folder.

## 2. Build and check the image

```bash
docker build --ulimit nofile=1048576:1048576 -t <registry>/vllm-disagg:glmv53flash-gfx942 \
  -f docker/vllm_disagg_inference.glmv53flash.gfx942.ubuntu.amd.Dockerfile .
docker run --rm --entrypoint cat <registry>/vllm-disagg:glmv53flash-gfx942 /app/versions.txt
```

The build fails unless every GLM-5.3 fix marker is present (`all 29 GLM-5.3 fix markers present`).
`versions.txt` must list the vLLM, AITER, MoRI and vllm-router commits pinned in the Dockerfile.

## 3. Weights and NICs

The probe looks for `<root>/$MODEL_WEIGHTS_NAME` under `/mnt/m2m_nobackup/models_blog`, `/shared_inference/models_blog`
and `$MODEL_DIR`. The recipe key is not the directory name, so always export `MODEL_WEIGHTS_NAME`; set `MODEL_DIR` if
the checkpoint lives elsewhere (it must be under a path the container mounts). Set NICs with `moriio-nics-cx7` or
`moriio-nics-thor2`.

## 4. Submit

```bash
cd MAD/scripts/vllm_dissag
export DOCKER_IMAGE_NAME=<registry>/vllm-disagg:glmv53flash-gfx942 CONNECTOR=moriio WIDE_EP=1 xP=1 yD=1
export MODEL_NAME=GLM-5.3-Flash-MXFP4-gfx942 MODEL_WEIGHTS_NAME=GLM-5.3-Flash-MXFP4   # or the FP8 pair
export BENCHMARK_SCRIPT=niah        # or long_context, agentic, sweep
sbatch -N 2 -n 2 --nodelist=<prefill_node,decode_node> run_xPyD_models.slurm
```

The first start compiles AITER JIT kernels for ~20-25 min; later starts reuse the JIT cache.

## 5. Verify each leg (MXFP4 card)

```bash
L=$LOG_PATH/<job_id>
grep -h "converted exactly to FP8 128x128" $L/prefill_NODE0.log | head -1    # prefill: experts as FP8 blocks
grep -h "FlyDSL W4A16 with exact e2m1"     $L/decode_NODE*.log  | head -1    # decode: experts 4-bit
grep -h "Model loading took"               $L/*_NODE*.log | sort -u          # prefill 54.33 GiB, decode 38.84 GiB
grep -h "GPU KV cache size"                $L/decode_NODE*.log | head -1
```

`-hyb` cards, prefill node (vLLM logs only non-default arguments):

```bash
grep -ho "'tensor_parallel_size': [0-9]*"            $L/prefill_NODE0.log | head -1   # 4
grep -ho "'prefix_cache_retention_interval': [0-9]*" $L/prefill_NODE0.log | head -1   # 17408 (agentic / prefix caching on)
```

Expected decode KV per GPU at gpu-memory-utilization 0.6: FP8 ~6.83M (MI325X) / ~3.71M (MI300X); MXFP4 ~7.97M /
~4.85M. If the decode log shows 54.32 GiB with the MXFP4 card, the per-leg env did not apply: check that
`PREFILL_EXTRA_ENV` and `VLLM_ROCM_MXFP4_*` reached the container (`models.yaml` env).

## 6. Expected results (MI300X, 1P/1D)

| Card / workload | Result |
|---|---|
| long_context 256K/1K c1, c8 | TTFT ~34 s, ~60 s |
| long_context 512K/1K c1, c8 | TTFT ~73 s, ~126-152 s |
| niah, 2K and 8K words | 10/10 (FP8 9.7-10) |

The `niah` card's long sizes (20K, 35K words) use a 40-word repetitive filler, which is hard for linear-attention
layers: both cards occasionally return a single needle there, so judge them over several `NIAH_SEEDS` rather than
three. Seeded random-haystack NIAH up to 704K tokens is identical for the two cards.

## 7. When something is off

- **Server never gets past its first log line, or a start finds GPU memory in use with no container:** a previous
  crash left device memory stuck. Check per GPU (`torch.cuda.mem_get_info` in the image); free memory well below
  192 GB (MI300X) means the node needs a GPU reset.
- **Short prompts return garbage:** the vLLM pin predates `6a0cb9395` (KV zeroing raced the MoRIIO READ).
- **GSM8K ~94% instead of ~97% with MTP:** the vLLM pin lacks the DSA indexer causal-rows fix (marker
  `GLM53_SHORT_DECODE_CAUSAL_ROWS`).
- **`docker pull` fails with 429:** Docker Hub rate limit on the cluster egress; build locally or copy the image with
  `docker save | ssh <node> docker load`.
