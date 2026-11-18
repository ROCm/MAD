# Kimi-K3 (MXFP4) on MI300X / gfx942 — Wide-EP via MoRI-EP + a8w4

Serves Kimi-K3 (MXFP4) across **2 nodes** (16× MI300X) with **expert parallelism** over
true **MoRI-EP** all-to-all kernels: PP2×TP8 for weight fit (~102 GB/GPU) plus
`--enable-expert-parallel --all2all-backend mori_low_latency`, so the 896 experts are
split **8-way across each node's 8 GPUs** (112 experts/GPU, `[EP Rank x/8]`), replicated on
each of the 2 pipeline stages, and dispatched via MoRI all2all. The EP group is intra-node,
so MoRI-EP dispatch/combine runs across a node's 8 GPUs; the only cross-node traffic is the
PP activation hand-off (NCCL).
`AITER_SITUV2_A8W4=1` selects the a8w4 (fp8-activation × int4-weight) SiTU MoE path.
This is the "MoRI-EP + a8w4" path (vs the generic all2all in `../wideep_int4_allgather`).
Colocated (single instance; no prefill/decode disaggregation).

- Image: `amdsiloai/vllm:kimi-k3-mi325x-release-v2` (gfx942 K3 vLLM build; public, anonymous pull)
- Model: `moonshotai/Kimi-K3` (MXFP4), on local NVMe or NFS

## Run (worker FIRST, then head)

```bash
# on the worker node (rank 1):
MODEL_DIR=/path/to/Kimi-K3-MXFP4 ROLE=worker MASTER=<head eth0 IP> bash run.sh

# on the head node (rank 0, serves the API on :8000):
MODEL_DIR=/path/to/Kimi-K3-MXFP4 ROLE=head   MASTER=<head eth0 IP> bash run.sh
```

- `MASTER` = the head node's **eth0** IP. `MODEL_DIR` = Kimi-K3-MXFP4 weights (prefer local NVMe).
- First start compiles the MoRI-EP dispatch kernels + a8w4 MoE (a few minutes), then cached.

## Verify

```bash
curl http://<head eth0 IP>:8000/v1/models
python3 niah_probe.py --url http://<head eth0 IP>:8000 --model kimi-k3 --ctx 8500 --depths 0.1,0.5,0.9
```
The response `system_fingerprint` contains `-ep-` when expert parallelism is active.

## Key env (set in run.sh)

| Var | Value | Why |
|-----|-------|-----|
| `VLLM_ROCM_USE_AITER_MLA` | `0` | Required on gfx942 (AITER MLA is gfx950-only). |
| `AITER_SITUV2_A8W4` | `1` | Route K3 SiTU MXFP4 MoE through the a8w4 interleaved flydsl kernels. |
| `MORI_GPU_ARCHS` | `gfx942` | MoRI-EP target arch. |
| `MORI_IB_GID_INDEX` / `MORI_IB_ENABLE_RELAXED_ORDERING` / `MORI_NUM_QP_PER_PE` | `3` / `1` / `8` | MoRI RDMA fabric tuning (**cluster-specific**). |
| `NCCL_IB_HCA` / `NCCL_IB_GID_INDEX` | 8× mlx5 / `3` | NCCL RDMA fabric (**cluster-specific — override for yours**). |
| `HSA_ENABLE_IPC_MODE_LEGACY` / `PYTORCH_(HIP_)ALLOC_CONF` | `0` / `expandable_segments:False` | ROCm 7.2.x requirements. |

`--all2all-backend mori_low_latency` is used for both roles (low-latency MoRI-EP dispatch).
Overridable env: `IMAGE`, `MODEL_DIR`, `MASTER`, `PORT` (8000), `MAX_MODEL_LEN` (10240),
`MAX_NUM_SEQS` (8), `GPU_UTIL` (0.90).
