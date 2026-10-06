# Kimi-K3 (MXFP4) on MI300X / gfx942 — PP2×TP8 baseline

Serves Kimi-K3 (MXFP4) across **2 nodes** (16× MI300X): tensor-parallel 8 within each
node, pipeline-parallel 2 across nodes. Each node holds half the layers (~102 GB/GPU);
a single 8-GPU node cannot fit the model + KV. This is the simplest, lowest-latency K3
serve on MI300X, with no expert parallelism. See `../wideep_int4_moriep` for wide-EP.

- Image: `amdsiloai/vllm:kimi-k3-mi325x-release-v2` (gfx942 K3 vLLM build; public, anonymous pull)
- Model: `moonshotai/Kimi-K3` (MXFP4), on local NVMe or NFS
- Colocated (single instance; no prefill/decode disaggregation)

## Run (worker FIRST, then head)

```bash
# on the worker node (rank 1):
MODEL_DIR=/path/to/Kimi-K3-MXFP4 ROLE=worker MASTER=<head eth0 IP> bash run.sh

# on the head node (rank 0, serves the API on :8000):
MODEL_DIR=/path/to/Kimi-K3-MXFP4 ROLE=head   MASTER=<head eth0 IP> bash run.sh
```

- `MASTER` = the head node's **eth0** IP (NCCL/PP bootstrap).
- `MODEL_DIR` = path to the Kimi-K3-MXFP4 weights. Prefer **local NVMe** over NFS (much faster load).
- First start recompiles gfx942 AITER kernels (a few minutes), then cached.

## Verify

```bash
curl http://<head eth0 IP>:8000/v1/models
python3 niah_probe.py --url http://<head eth0 IP>:8000 --model kimi-k3 --ctx 8500 --depths 0.1,0.5,0.9
```

## Key env (set in run.sh)

| Var | Value | Why |
|-----|-------|-----|
| `VLLM_ROCM_USE_AITER_MLA` | `0` | Required on gfx942 — the AITER MLA kernel is gfx950-only and asserts at TP8. |
| `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | `eth0` | Control-plane NIC. |
| `NCCL_IB_HCA` | `mlx5_0,2,3,4,5,7,8,9` | 8× RDMA NICs (**cluster-specific — override for your fabric**). |
| `NCCL_IB_GID_INDEX` | `3` | RoCE GID (**cluster-specific**). |
| `HSA_ENABLE_IPC_MODE_LEGACY` | `0` | ROCm 7.2.x IPC. |
| `PYTORCH_(HIP_)ALLOC_CONF` | `expandable_segments:False` | Required on ROCm 7.2.x. |

Overridable env: `IMAGE`, `MODEL_DIR`, `MASTER`, `PORT` (8000), `MAX_MODEL_LEN` (10240),
`MAX_NUM_SEQS` (8), `GPU_UTIL` (0.90).
