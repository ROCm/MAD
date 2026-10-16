# MiniMax-M3 — MXFP4 disaggregated (P/D) ATOM benchmark

Prefill/decode disaggregated ATOM benchmark for MiniMax-M3 MXFP4 on AMD MI355X (gfx950),
config `minimaxm3-fp4-mi355x-atom-disagg` (atomesh router + mooncake RDMA KV).

## Nodes required
- `1p1d` → **2 nodes** (1 prefill + 1 decode)
- `2p1d_dpa` → **3 nodes** (2 prefill + 1 decode)

8× MI355X per node, all workers TP4 (1 worker/node). 3 nodes covers both topologies.

## Prerequisites
1. `amdgpu_peermem` kernel module loaded on every host (GPUDirect RDMA for mooncake KV):
   `lsmod | grep amdgpu_peermem` / `sudo modprobe amdgpu_peermem`.
2. RDMA fabric between nodes.
3. Weights staged at `<models_root>/MiniMax-M3-MXFP4` (from `amd/MiniMax-M3-MXFP4`), same path on every node.
4. SLURM with the nodes/reservation in `cluster.yaml`; Docker + ROCm on each node.
5. Image `rocm/atom-dev:nightly_202607011530` available on each node.

## Usage
Edit `cluster.yaml` (reservation, `models_root`, `nodes`), then:
```bash
TOPO=1p1d     ISL=8192 OSL=1024 ./run.sh     # 2 nodes, conc 1-256
TOPO=2p1d_dpa ISL=8192 OSL=1024 ./run.sh     # 3 nodes, conc 256-1024
TOPO=1p1d     ISL=1024 OSL=1024 ./run.sh     # 2 nodes, 1k/1k

NODES=n1,n2,n3 TOPO=2p1d_dpa ./run.sh        # pin nodes
ACTION=dry     TOPO=1p1d ./run.sh            # print commands, launch nothing
```
Bench drives the atomesh router (`:8000`, OpenAI-compatible). Results land under
`results/minimaxm3-fp4_atom_<topo>_<shape>_<stamp>/`.

## Layout
Engine code (`benchmarks/`, `utils/bench_serving/`) is copied unchanged from
SemiAnalysisAI/InferenceX @ `a174f20`; all model flags live in
`benchmarks/multi_node/amd_utils/models_atom.yaml`. `run.sh` + `cluster.yaml` are the MAD
launcher (supply the env + mounts InferenceX's job.slurm normally provides).

Topologies are defined in `configs/minimaxm3-fp4-mi355x-atom-disagg.yaml` (all-TP4, STP,
`DECODE_MTP_SIZE=0`). Validate with `ACTION=dry` before a real run.
