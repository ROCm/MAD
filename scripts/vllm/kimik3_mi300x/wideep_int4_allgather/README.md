# Kimi-K3 (MXFP4) on MI300X / gfx942 — Wide-EP (generic all2all) + a8w4

Serves Kimi-K3 (MXFP4) across **2 nodes** (16× MI300X) with **expert parallelism**:
PP2×TP8 for weight fit (~102 GB/GPU) plus `--enable-expert-parallel`, so the 896 experts are split 8-way across each node's 8 GPUs (112/GPU), replicated per PP stage. Expert all-to-all uses the generic
`--all2all-backend allgather_reducescatter` (see `../wideep_int4_moriep` for the true
MoRI-EP kernels). `AITER_SITUV2_A8W4=1` selects the a8w4 (fp8-activation × int4-weight)
SiTU MoE path. Colocated (single instance; no prefill/decode disaggregation).

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
- First start compiles the AITER EP MoE asm + a8w4 kernels (a few minutes), then cached.

## Verify

```bash
curl http://<head eth0 IP>:8000/v1/models
python3 niah_probe.py --url http://<head eth0 IP>:8000 --model kimi-k3 --ctx 8500 --depths 0.1,0.5,0.9
```
The response `system_fingerprint` contains `-ep-` when expert parallelism is active.

## Toggles

- Drop a8w4: remove `-e AITER_SITUV2_A8W4=1` from `run.sh` → default a16w4 MoE path.
- MoE dispatch: add `-e VLLM_ROCM_AITER_MOE_DISPATCH_POLICY=2` (multi-pass; may help MoE-heavy at higher concurrency).

## Key env (set in run.sh)

| Var | Value | Why |
|-----|-------|-----|
| `VLLM_ROCM_USE_AITER_MLA` | `0` | Required on gfx942 (AITER MLA is gfx950-only). |
| `AITER_SITUV2_A8W4` | `1` | Route K3 SiTU MXFP4 MoE through the a8w4 interleaved flydsl kernels. |
| `NCCL_IB_HCA` / `NCCL_IB_GID_INDEX` | 8× mlx5 / `3` | RDMA fabric (**cluster-specific — override for yours**). |
| `HSA_ENABLE_IPC_MODE_LEGACY` / `PYTORCH_(HIP_)ALLOC_CONF` | `0` / `expandable_segments:False` | ROCm 7.2.x requirements. |

Overridable env: `IMAGE`, `MODEL_DIR`, `MASTER`, `PORT` (8000), `MAX_MODEL_LEN` (10240),
`MAX_NUM_SEQS` (8), `GPU_UTIL` (0.90).
