# MiniMax-M3 — MXFP4 disaggregated (P/D) ATOM benchmark (from InferenceX)

Vendored **verbatim** from [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX)
— the ATOM MXFP4 prefill/decode disaggregated benchmark for MiniMax-M3 on AMD MI355X (gfx950),
config key **`minimaxm3-fp4-mi355x-atom-disagg`** (atomesh router + mooncake RDMA KV transfer).

This is the top-throughput MiniMax-M3 disagg config on MI355X (InferenceX): ~10,829 tok/s/GPU at
8k/1k (2P1D + DP-attn). Dashboard: <https://inferencex.semianalysis.com/inference?preset=minimax-m3-launch>

## Contents (copied unchanged from InferenceX @ `a174f20`)
| Path | InferenceX source |
|------|-------------------|
| `amd_utils/server_atom.sh` | `benchmarks/multi_node/amd_utils/server_atom.sh` — ATOM disagg launcher (reads `models_atom.yaml`) |
| `amd_utils/models_atom.yaml` | `benchmarks/multi_node/amd_utils/models_atom.yaml` — per-model ATOM flags (incl. `MiniMax-M3-MXFP4`) |
| `amd_utils/server.sh` | dispatcher → `server_atom.sh` when `ENGINE=atom-disagg` |
| `amd_utils/env_atom.sh` | ATOM/mooncake env + RDMA device detection |
| `amd_utils/setup_deps.sh` | in-container dep setup |
| `amd_utils/bench.sh` | bench driver (calls the bench client against the atomesh router) |
| `amd_utils/job.slurm`, `submit.sh`, `sync.py` | SLURM multi-node plumbing |
| `utils/bench_serving/` | InferenceX benchmark client (`benchmark_serving.py` + backends) |
| `configs/minimaxm3-fp4-mi355x-atom-disagg.yaml` | the config entry extracted from `configs/amd-master.yaml` |

## The MiniMax-M3-MXFP4 ATOM config (`models_atom.yaml`)
All workers TP4 (no EP; TP4 beats TP8/TP4-EP on gfx950). Key flags:
- `env: AITER_QUICK_REDUCE_QUANTIZATION=INT4 ATOM_FORCE_ATTN_TRITON=1`
- `kv_cache_flags: --kv_cache_dtype fp8`
- `online_quant_config: ptpc_fp8` (exclude lm_head/embed/vision/moe)
- `block_size 128`, `mem_frac_static 0.8`, `max_model_len 32768`, `max_num_batched_tokens 32768`

## Topologies (`configs/minimaxm3-fp4-mi355x-atom-disagg.yaml`)
| shape | conc | layout | nodes |
|-------|------|--------|-------|
| 8192/1024 | 1–256 | 1P1D TP4 (STP, `DECODE_MTP_SIZE=0`) | 2 |
| 8192/1024 | 256–1024 | 2P1D TP4 + DP-attn | 3 |
| 1024/1024 | 1–256 | 1P1D TP4 (STP) | 2 |

## Provenance
Source commit: `a174f20d26983d499e0eac315ee39f0c9aa48262` (SemiAnalysisAI/InferenceX). Files are
unmodified. Update by re-copying from the same paths in a newer InferenceX revision.
