# MiniMax-M3 — MXFP4 disaggregated (P/D) ATOM benchmark

The ATOM MXFP4 prefill/decode disaggregated benchmark for MiniMax-M3 on AMD MI355X (gfx950),
config **`minimaxm3-fp4-mi355x-atom-disagg`** (atomesh router + mooncake RDMA KV) — the
top-throughput MiniMax-M3 disagg config on InferenceX (~10,829 tok/s/GPU at 8k/1k, 2P1D+DPA).
Dashboard: <https://inferencex.semianalysis.com/inference?preset=minimax-m3-launch>

## Layout
The engine code is copied **verbatim** from [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX)
@ `a174f20` (unmodified — update by re-copying from the same paths). A thin MAD wrapper
(`run.sh` + `cluster.yaml`) supplies the per-node env + mounts that InferenceX's
`job.slurm`/`submit.sh` normally provide, so it runs standalone from MAD.

| Path | Origin |
|------|--------|
| `benchmarks/multi_node/amd_utils/` | **verbatim** InferenceX (`server_atom.sh`, `models_atom.yaml`, `env_atom.sh`, `bench.sh`, `setup_deps.sh`, `server.sh`, `job.slurm`, `submit.sh`, `sync.py`) |
| `benchmarks/benchmark_lib.sh` | **verbatim** InferenceX (bench.sh sources `../../benchmark_lib.sh`) |
| `utils/bench_serving/` | **verbatim** InferenceX benchmark client |
| `configs/minimaxm3-fp4-mi355x-atom-disagg.yaml` | config entry from `amd-master.yaml` |
| `run.sh`, `cluster.yaml` | **MAD wrapper** (the only non-verbatim files) |

Model flags (all-TP4, `--kv_cache_dtype fp8`, `online_quant_config` ptpc_fp8, `ATOM_FORCE_ATTN_TRITON=1`,
`AITER_QUICK_REDUCE_QUANTIZATION=INT4`, block 128, mem-frac 0.8, max-model-len 32768) live in
`models_atom.yaml` and are read by `server_atom.sh` — the wrapper sets none of them, so runtime
behavior matches InferenceX exactly.

## Prerequisites
1. **`amdgpu_peermem` kernel module loaded on every host** — required for GPUDirect RDMA so
   mooncake can stream the KV cache GPU→NIC→GPU across nodes. Check: `lsmod | grep amdgpu_peermem`;
   load: `sudo modprobe amdgpu_peermem`. Without it, mooncake KV transfer stalls / times out.
2. **RDMA fabric** reachable between nodes (the launcher passes `--device=/dev/infiniband --ulimit memlock=-1`).
3. **Weights staged** at `<models_root>/MiniMax-M3-MXFP4` (from `amd/MiniMax-M3-MXFP4`), same path on every node.
4. **SLURM** with the nodes/reservation from `cluster.yaml`; Docker + ROCm on each node.
5. Image `rocm/atom-dev:nightly_202607011530` pullable on each node.

## Usage
Edit `cluster.yaml` (reservation, `models_root`, `nodes`), then:
```bash
# 8k/1k balanced, 2 nodes (1 prefill + 1 decode), conc 1-256:
TOPO=1p1d ISL=8192 OSL=1024 ./run.sh

# 8k/1k high-throughput, 3 nodes (2 prefill + 1 decode + DP-attn), conc 256-1024:
TOPO=2p1d_dpa ISL=8192 OSL=1024 ./run.sh

# 1k/1k, 2 nodes:
TOPO=1p1d ISL=1024 OSL=1024 ./run.sh

# pin nodes / dry-run (print commands, launch nothing):
NODES=n1,n2,n3 TOPO=2p1d_dpa ./run.sh
ACTION=dry TOPO=1p1d ./run.sh
```
Bench drives the atomesh router (`:8000`, OpenAI-compatible) → true aggregate throughput (no ×N).
Result JSONs land under `results/minimaxm3-fp4_atom_<topo>_<shape>_<stamp>/`.

## Topologies (`configs/minimaxm3-fp4-mi355x-atom-disagg.yaml`)
| TOPO | shape | conc | layout | nodes |
|------|-------|------|--------|-------|
| `1p1d` | 8192/1024, 1024/1024 | 1–256 | 1 prefill TP4 + 1 decode TP4 (STP) | 2 |
| `2p1d_dpa` | 8192/1024 | 256–1024 | 2 prefill + 1 decode, TP4 + DP-attn | 3 |

All-TP4 ⇒ 1 node/worker on 8-GPU nodes; STP only (`DECODE_MTP_SIZE=0`).

## Status
Engine code is verbatim InferenceX (byte-identical). The MAD wrapper is syntax-checked and its
topology/mount contract mirrors InferenceX `job.slurm`, but has **not** been dry-run on live
MI355X hardware — validate with `ACTION=dry` first, then a small `TOPO=1p1d` run.
