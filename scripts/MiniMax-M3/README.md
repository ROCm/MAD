# MiniMax-M3 — disaggregated (P/D) serving/benchmark harness

Multi-node **prefill/decode disaggregated** serving for MiniMax-M3 (MXFP4) on AMD Instinct
MI355X (gfx950), ported from InferenceX (`minimaxm3-fp4-mi355x-atom-disagg`).

## Why disagg
MiniMax-M3 is served **disaggregated**: prefill workers (kv_producer) and decode workers
(kv_consumer) on separate nodes, KV streamed over RDMA, fronted by ONE router. Bench drives the
router → a TRUE aggregate number (no ×N extrapolation).

**All workers are TP4, no EP.** The single-node M3 recipe found plain TP4 beats both TP8 and
TP4/EP4 on tok/s/GPU on gfx950, so prefill and decode both use TP4 and we tune the
prefill:decode worker ratio (xP:yD) instead of TP. STP only (no MTP): `DECODE_MTP_SIZE=0`.

## Engine & router
| Engine | Router | KV transfer | Status |
|--------|--------|-------------|--------|
| **atom-disagg** | **atomesh** | **mooncake** (RDMA) | PRIMARY — InferenceX-validated M3 MXFP4 config |

Precision: **MXFP4** (`amd/MiniMax-M3-MXFP4`). Image: `rocm/atom-dev:nightly_202607011530`.

## Files
| File | Role |
|------|------|
| `cluster.yaml` | nodes, gpus/node, reservation, router/prefill/decode/handshake ports, paths |
| `model.yaml` | M3 def + shared `defaults{}` + atom-disagg image/router/kv + `topologies{}` |
| `run_atom_disagg.sh` | thin wrapper → `lib/run_disagg.sh` |
| `lib/run_disagg.sh` | orchestrator: clean nodes → launch prefill/decode servers → start router → bench → teardown |
| `lib/topo.py` | resolve a topology (1p1d / 2p1d_dpa) → node-role placement (prefill/decode, ports) |
| `lib/cfg.py` | yaml reader |
| `lib/clean_node.sh` | pre-flight GPU clean |
| `atom_disagg/` | vendored InferenceX ATOM server path (`server_atom.sh`, `env_atom.sh`, `bench.sh`, `setup_deps.sh`) + `launch.sh` SLURM-equivalent |
| `utils/bench_serving/` | vendored InferenceX bench client |
| `docs/proxy_and_disagg.md` | per-engine proxy/router/KV mechanics + topologies |

## Topologies (from minimaxm3-fp4-mi355x-atom-disagg)
| name | nodes | layout | shapes | conc |
|------|-------|--------|--------|------|
| `1p1d` | 2 | 1 prefill TP4 + 1 decode TP4 (STP) | 8192/1024, 1024/1024 | 1–256 |
| `2p1d_dpa` | 3 | 2 prefill + 1 decode, TP4 + DP-attn | 8192/1024 | 256–1024 |

All layouts keep `xP + yD <= 3` for the 3-node MI355X disagg pool (1 node/worker at TP4).

## Usage
```bash
# ATOM disagg, 1-prefill/1-decode (2 nodes), full bench sweep through the router:
MODEL=minimaxm3-fp4 TOPO=1p1d     ./run_atom_disagg.sh
MODEL=minimaxm3-fp4 TOPO=2p1d_dpa ./run_atom_disagg.sh      # 3 nodes, high-conc 8k1k tail

# Just serve (hold the endpoint up; drive it yourself):
MODEL=minimaxm3-fp4 TOPO=1p1d ACTION=serve ./run_atom_disagg.sh
```
Bench drives `http://<router_ip>:8000` (OpenAI-compatible). Results land under `results/`.

## Reference performance (InferenceX, MI355X)
From the InferenceX `minimaxm3-fp4-mi355x-atom-disagg` sweep (PR #2000), this is the
**top-throughput** MiniMax-M3 disagg config on MI355X:

| shape | layout | peak tok/s/GPU | conc |
|-------|--------|----------------|------|
| 8k/1k | 2P1D + DPA | **~10,829** | 1024 |
| 8k/1k | 1P1D | ~7,594 | 256 |
| 1k/1k | 1P1D | ~2,703 | 256 |

Dashboard: <https://inferencex.semianalysis.com/inference?preset=minimax-m3-launch>

## Status / TODO
- **SCAFFOLDED** from the InferenceX config; **needs a live gfx950 dry-run** (RDMA/mooncake
  handshake + atomesh router) once `amd/MiniMax-M3-MXFP4` weights are staged.
- Fill in `cluster.yaml` (reservation, `models_root`, node list) before running.
- MXFP4 quick-reduce knob `AITER_QUICK_REDUCE_QUANTIZATION=INT4` is applied by the launcher;
  STP-only is enforced via `DECODE_MTP_SIZE=0`.
