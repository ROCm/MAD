# DeepSeek-V4-Pro — disaggregated (P/D) serving/benchmark harness

Multi-node **prefill/decode disaggregated** serving for DeepSeek-V4-Pro on AMD Instinct
(gfx950), ported from InferenceX.

## Why disagg
DSv4-Pro is a large MoE (61 layers, 384 experts, hidden 7168, moe_intermediate 3072; FP8 block-quant
+ FP4 experts). It's served **disaggregated**: prefill workers (kv_producer) and decode workers
(kv_consumer) on separate nodes, KV streamed over RDMA, fronted by ONE router. Bench drives the
router → a TRUE aggregate number (no ×N extrapolation). TP8 is valid here (3072/8=384 ≥ 128).

## Engines & proxy/router (see docs/proxy_and_disagg.md for detail)
| Engine | Router | KV transfer | Status |
|--------|--------|-------------|--------|
| **atom-disagg** | **atomesh** | **mooncake** (RDMA) | PRIMARY — InferenceX-validated DSv4 config |
| **sglang-disagg** | sglang_router | MoRI-IO | EXPERIMENTAL — adapted from DeepSeek-R1 recipe |

(vLLM disagg uses MoRI-IO/Nixl; not scaffolded — its `server_vllm.sh` wasn't in our snapshot.)

## Files
| File | Role |
|------|------|
| `cluster.yaml` | nodes, gpus/node, reservation, router/prefill/decode/handshake ports, paths |
| `model.yaml` | DSv4 def + shared `defaults{}` + per-engine image/router/kv + `topologies{}` |
| `run_atom_disagg.sh` / `run_sglang_disagg.sh` | thin wrappers → `lib/run_disagg.sh` |
| `lib/run_disagg.sh` | orchestrator: clean nodes → launch prefill/decode servers → start router → bench → teardown |
| `lib/topo.py` | resolve a topology (1p1d / 2p1d_dpa) → node-role placement (prefill/decode, ports) |
| `lib/cfg.py` | yaml reader (shared with qwen_v2) |
| `lib/clean_node.sh` | pre-flight GPU clean (shared with qwen_v2) |
| `utils/bench_serving/` | vendored InferenceX bench client |
| `docs/proxy_and_disagg.md` | full per-engine proxy/router/KV mechanics + topologies |

## Topologies (from dsv4-fp4-mi355x-atom-disagg)
| name | nodes | layout | shapes | conc |
|------|-------|--------|--------|------|
| `1p1d` | 2 | 1 prefill TP8 + 1 decode TP8 | 1024/1024, 8192/1024 | 4–128 |
| `2p1d_dpa` | 3 | 2 prefill + 1 decode, TP8 + DP-attn | 8192/1024 | 256–2048 |

## Usage
```bash
# ATOM disagg, 1-prefill/1-decode (2 nodes), full bench sweep through the router:
MODEL=dsv4-pro TOPO=1p1d   ./run_atom_disagg.sh
MODEL=dsv4-pro TOPO=2p1d_dpa ./run_atom_disagg.sh      # 3 nodes, high-conc

# Just serve (hold the endpoint up; drive it yourself):
MODEL=dsv4-pro TOPO=1p1d ACTION=serve ./run_atom_disagg.sh

# SGLang disagg (experimental, R1-template):
MODEL=dsv4-pro TOPO=1p1d ./run_sglang_disagg.sh
```
Bench drives `http://<router_ip>:8000` (OpenAI-compatible). Results land in `results_dsv4/`.

## Status / TODO
- ✅ Weights: DSv4-Pro 64/64 shards integrity-verified on NFS; staging to local scratch (12 nodes).
- ⏳ ATOM disagg: scaffolded faithfully from server_atom.sh; **needs a live dry-run** (RDMA/mooncake
  handshake, atomesh router) once weights are staged.
- ⏳ SGLang disagg: scaffolded from R1 template; DSv4 model support + sglang_router flags TBC.
- ATOM DSv4 env applied by launcher: `ATOM_MOE_GU_ITLV=1`, `AITER_BF16_FP8_MOE_BOUND=0` (+ TBO path
  `GPU_MAX_HW_QUEUES=5`, `ATOM_CPU_AFFINITY=1`).
