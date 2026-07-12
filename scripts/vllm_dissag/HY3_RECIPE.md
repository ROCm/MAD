# Hunyuan-3.0 (Hy3-preview) — MoRI-EP Disaggregated WideEP Recipe

Enablement of **tencent/Hy3-preview** (Hunyuan-3.0 preview) on the vllm_dissag MoRI-EP
prefill/decode disaggregated WideEP stack. Hy3 is an 80-layer **GQA** MoE: 192 routed
experts (8 active/token) + 1 shared expert, 256K native context.

## Validated configurations (bf16 weights)

Measured on AMD MI300X/MI308 (gfx942) + the MoRIIO connector (`CONNECTOR=moriio WIDE_EP=1`):

| Topology | EP | experts/rank | Known-answer | NIAH (54K/128K/256K) | Verdict |
|----------|----|----|--------------|----------------------|---------|
| 1P/1D | 8  | 24 | 20/20 (100%) | 9/9 incl 256K deep | ✅ PASS |
| 2P/2D | 16 | 12 | 20/20 (100%) | 9/9 incl 256K deep | ✅ PASS |
| 2P/1D, 1P/2D | 16 | — | perf 16/0 | — | ✅ serving |
| 4P/4D | 32 | 6  | 0/20 (garbage) | — | ❌ see note |

**EP8 and EP16 are production-correct and 256K-capable.** NIAH retrieves the needle at
every depth up to 256K with fp8 KV cache + chunked prefill.

**EP32 note:** 4P/4D hits the known MoRI-EP all-to-all `>2-pod` garbage bug (silent —
HTTP 200 + "successful requests" but degenerate repeated-token output). Same class as the
DeepSeek pre-fix issue; it is in the MoRI-EP all2all compute path, not the model/router/KV.
Use EP8/EP16 until the upstream MoRI fix lands. **Accuracy MUST be gated separately** — perf
benchmarks are blind to this (see `BENCHMARK_SCRIPT=accuracy`).

## How to run

```bash
# EP16 2P/2D accuracy (known-answer + NIAH):
export DOCKER_IMAGE_NAME=<your-moriep-image>
export MODEL_NAME=Hy3-preview CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori
export xP=2 yD=2 BENCHMARK_SCRIPT=accuracy
# ... via run_xPyD_models.slurm (slurm_multi launcher, -N 4 -n 4)

# perf sanity:
export BENCHMARK_SCRIPT=long_context BENCHMARK_COMBINATIONS=1024/1024
```

Or via `models.json`: `pyt_vllm_disagg_mori_hy3-preview_1p1d` / `_2p2d`.

## Recipe knobs (models.yaml `Hy3-preview` → `_hy3_recipe_env`)

The single-home recipe lives in `models.yaml`. Deltas vs the DeepSeek (MLA) recipe, because
Hy3 is **GQA** (head_size 128):

| Knob | Value | Why |
|------|-------|-----|
| `KV_BLOCK_SIZE` | 16 | block=1 has no valid ROCm attention backend under fp8 KV |
| `KV_CACHE_DTYPE` | fp8 | KV cache in fp8 (does not degrade NIAH; 9/9 at 256K) |
| `KV_CACHE_MEMORY_BYTES` | 48e9 | GQA KV ≈ 40 GiB @256K (vs MLA's compressed KV ~20 GiB); also skips the boot profiling forward |
| `VLLM_ROCM_USE_AITER_MLA` | 0 | Hy3 is GQA (no MLA path); also avoids the fp8 MLA decode kernel GPU-fault |
| `PREFILL_CUDAGRAPH_MODE` | NONE | prefill cudagraph capture deadlocks wide-DP |
| `DECODE_CUDAGRAPH_MODE` | PIECEWISE | decode ITL win |
| `PREFILL_MORI_BACKEND` | mori_high_throughput | InterNodeV1 |
| `DECODE_MORI_BACKEND` | mori_low_latency | InterNodeV1LL |

## Accuracy tooling (`HY_Accuracy_folder/`)

`benchmark_accuracy_hy3.sh` runs three tiers (all greedy/deterministic):
1. **known-answer** (`accuracy_eval.py --known`): ~20 factual/math prompts vs ground truth
   — the gross-correctness gate that catches wideEP garbage (`0%` when the MoE path is wrong).
2. **NIAH** (`niah_probe.py`): needle retrieval at 54K/128K/256K × depths 0.1/0.5/0.9 —
   long-context KV-path correctness.
3. **equivalence** (optional): greedy exact-match vs an EP16 golden.

Note: the sanity probe uses a generous per-attempt budget (8×240s) because the FIRST
inference after "Ready" triggers last-mile AITER JIT compile (minutes); a short probe
would time out and abort the run.
