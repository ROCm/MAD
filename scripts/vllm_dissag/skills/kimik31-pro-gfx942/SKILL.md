---
name: kimik31-pro-gfx942
description: Build, launch and verify the Kimi-K3.1-Pro 1P/1D MoRIIO card (Kimi-K3.1-Pro-gfx942) on gfx942 (MI325X). Use when serving Kimi-K3.1-Pro disaggregated on MI300-series GPUs.
---

# Kimi-K3.1-Pro on gfx942: 1P/1D over MoRIIO

## Shape
- TP8 attention + TP-sharded experts on both legs, DCP8 on both legs (`--decode-context-parallel-size 8`,
  `--dcp-comm-backend a2a`): the KV cache is sharded across the 8 GPUs, ~6.6M tokens per leg on MI325X.
- One 8-GPU node per leg. EP8 with DP1 is slower (all-gather), and TP4xDP2 + MoRI-EP leaves ~0.6 GiB KV per GPU:
  wide EP for this model needs two nodes per pool.
- Experts: MXFP4 requantized to int4 per-group-32 at load (`--quantization-config '{"moe":{"weight":"int4_per_group_32"}}'`),
  FlyDSL wint4 SiTUv2 kernels (gfx942 has no FP4 MMA). KV: `fp8_ds_mla`.

## Build
```bash
docker build --ulimit nofile=1048576:1048576 -t <registry>/vllm-disagg:kimik31pro-gfx942 \
  -f docker/vllm_disagg_inference.kimik31pro.gfx942.ubuntu.amd.Dockerfile .
```

## Run (MI325X with Broadcom NICs: export the `moriio-nics-thor2` settings first)
```bash
cd MAD/scripts/vllm_dissag
export DOCKER_IMAGE_NAME=<registry>/vllm-disagg:kimik31pro-gfx942 CONNECTOR=moriio WIDE_EP=0 xP=1 yD=1
export MODEL_NAME=Kimi-K3.1-Pro-gfx942 MODEL_WEIGHTS_NAME=Kimi-K3.1-Pro
export BENCHMARK_SCRIPT=niah   # or agentic, long_context, sweep
sbatch -N 2 -n 2 --nodelist=<prefill_node,decode_node> run_xPyD_models.slurm
```

## gfx942 settings the card carries, and why
| Setting | Why |
|---|---|
| `VLLM_SSM_CONV_STATE_LAYOUT=DS` | MoRIIO's KDA conv-state transfer refuses the default layout |
| `GPU_MEMORY_UTILIZATION=0.88`, prefill `--max-num-seqs 16` | KDA chunked prefill allocates per-sequence workspace that startup profiling misses; 0.9 OOMs at 32 concurrent prompts |
| `--block-size 1024` | the upstream tip picks 992; the gfx942 DSA indexer then falls back to an uncapturable torch path |
| `CUDAGRAPH_CAPTURE_SIZES=1..64` (9 sizes) | breakable graphs cost ~0.25 GiB per size on the tip |
| decode `--max-num-seqs 256` | the default 1024 exceeds the KDA state blocks and graph capture refuses |
| `MORIIO_READ_MODE=1`, router `consistent_hash` | READ mode is the validated transfer; sessions stay on one prefill for prefix reuse |

## Measured (MI325X, Thor2, AgentX conformance_256k, 900 s)
| Topology | conc | TTFT median | E2E median | tok/s/GPU |
|---|---|---|---|---|
| 1P1D, DCP8 + retention | 8 | 29.9 s | 55.4 s | 8259 |
| 2P1D, DCP8 + retention | 4 | 11.6 s | 18.4 s | 6254 |
| 2P1D, DCP8 + retention | 8 | 17.6 s | 28.6 s | 12935 |

One prefill node is the bottleneck for agentic traffic; run `xP=2 yD=1` (`sbatch -N 3`) when three nodes are available.

## DSpark speculative decoding (card Kimi-K3.1-Pro-gfx942-dspark)
Place `Inferact/Kimi-K3.1-Pro-DSpark-preview` (5.4 GB) next to the checkpoint. The draft runs on both legs, its KV stays
bf16 (`"kv_cache_dtype":"auto"`; OCP fp8 crashes gfx942 AITER MLA decode), and decode drops to `--max-num-seqs 128`
(the draft reduces KDA state blocks). Measured 1P1D, per-stream decode: 158 tok/s at c1 on reasoning text (acceptance
length 4.7), 90 on mixed text (2.7), 81 / 48 at c32, versus ~49 without the draft. GSM8K and NIAH to 750K unchanged.

## Routing for agentic traffic
Both cards route with `sticky_least_loaded` (a session's turns return to the prefill holding its KDA checkpoints). With very
long sessions (conformance_512k), strict pinning can stack several 512K sessions on one prefill; a router with bounded overflow
(`VLLM_ROUTER_SLL_OVERFLOW_DELTA=2`: re-bind a session when its prefill has 2+ more in-flight requests than the idlest one)
measured 512k c4 TTFT 20.8 s -> 13.1 s and +25% sessions. The stock upstream router ignores the variable (strict pinning).

## Verify
- Load log: `Requantizing Kimi-K3 MXFP4 experts to groupwise int4 for gfx942`, `flydsl_moe1_abf16_wint4`, `GPU KV cache size: ~6.6M tokens`.
- Prefill with prefix caching logs `Mamba cache mode is set to 'align'`.
- Through the router: GSM8K ~0.99 (thinking off: `chat_template_kwargs.enable_thinking=false`), passkey NIAH 8K-750K.
- The router answers 503 on `/v1/models` in MoRIIO discovery mode; the agentic path uses `agentic_models_shim.py`.
- A container killed during AITER JIT leaves `aiter_jit/build/lock_*` batons that hang the next start: clear them.
