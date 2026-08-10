# Kimi-K3 (MXFP4) serving on AMD Instinct MI300X (gfx942)

[Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) is Moonshot AI's ~2.8T-parameter
Mixture-of-Experts model (natively MXFP4/QAT, hybrid MLA + Kimi-Delta-Attention, 896
experts). These are **colocated** (single-instance) vLLM serving recipes for **MI300X /
gfx942**, complementing MAD's existing single-node gfx950/MI355X K3 recipe.

Why MI300X needs its own recipe: MI300X has 192 GB/GPU, so the ~1.5 TB checkpoint does
**not** fit a single 8-GPU node under TP8. All recipes here shard across **2 nodes (16×
MI300X)** using PP2×TP8 (~102 GB/GPU per node). gfx942 also requires
`VLLM_ROCM_USE_AITER_MLA=0` (the AITER MLA kernel is gfx950-only).

## Image & model

- **Image:** `amdsiloai/vllm:kimi-k3-mi325x-release-v2` — a gfx942 K3 vLLM build. Public
  and anonymously pullable (`docker pull amdsiloai/vllm:kimi-k3-mi325x-release-v2`); built
  for MI325X, also runs MI300X. Override with `-e IMAGE=` if you have a newer tag.
- **Model:** `moonshotai/Kimi-K3` (MXFP4). Place on local NVMe for fast load; pass via `MODEL_DIR`.

## The recipes

| Folder | Parallelism | Expert all2all | MoE path | Use when |
|--------|-------------|----------------|----------|----------|
| [`pp2xtp8/`](pp2xtp8/) | PP2×TP8, no EP | — | a16w4 | Simplest baseline; lowest single-user latency. |
| [`wideep_int4_allgather/`](wideep_int4_allgather/) | PP2×TP8, EP8/node | `allgather_reducescatter` (generic) | a8w4 (`AITER_SITUV2_A8W4=1`) | Expert-parallel without MoRI kernels. |
| [`wideep_int4_moriep/`](wideep_int4_moriep/) | PP2×TP8, EP8/node | `mori_low_latency` (**MoRI-EP**) | a8w4 (`AITER_SITUV2_A8W4=1`) | MoRI-EP all2all expert dispatch (intra-node EP group). |
| [`wideep_disagg_2p2d/`](wideep_disagg_2p2d/) | 2P/2D disagg, TP2×DP8 per pool → EP16, no PP | `mori_low_latency` (**MoRI-EP**) + **MoRIIO** KV/state transfer | MXFP4 | Prefill/decode disaggregation across 4 nodes. Highest concurrent throughput; NIAH validated to 300K. |

The first three are colocated (single-instance, no prefill/decode split); the
fourth splits prefill and decode across two pools.

**`wideep_disagg_2p2d/` is validated** — single-needle NIAH passes deterministically
through **300K tokens** (all depths). Two root-cause fixes closed the decode-recall
bug (4-KV-group block routing + multi-chunk prefill transfer); see its
[`STATUS.md`](wideep_disagg_2p2d/STATUS.md) and
[`RESULTS.md`](wideep_disagg_2p2d/RESULTS.md). **Pick by workload:** the colocated
recipes give the lowest single-request latency (one request spans all 16 GPUs) —
best for interactive / low-QPS; disagg gives **5.7× throughput at concurrency 8**
(7.3× at 16) plus decode-latency isolation, at ~4× higher single-stream latency —
best for batch / high-QPS.

> **EP scope:** the 896 experts split **8-way across each node's 8 GPUs** (112 experts/GPU → `[EP Rank x/8]`), and that EP8 group is replicated on each of the 2 pipeline stages. The expert all2all (incl. MoRI-EP) therefore runs **intra-node**; the only cross-node traffic is the PP activation hand-off, over NCCL. ("16" is the GPU count, not the EP width.)

## Quick start

Each recipe is self-contained (`run.sh` + `README.md` + `niah_probe.py`). Launch the
**worker (rank 1) first, then the head (rank 0)**:

```bash
cd <recipe>/
MODEL_DIR=/path/to/Kimi-K3-MXFP4 ROLE=worker MASTER=<head eth0 IP> bash run.sh   # node 1
MODEL_DIR=/path/to/Kimi-K3-MXFP4 ROLE=head   MASTER=<head eth0 IP> bash run.sh   # node 0 (API :8000)

# verify
curl http://<head eth0 IP>:8000/v1/models
python3 niah_probe.py --url http://<head eth0 IP>:8000 --model kimi-k3 --ctx 8500 --depths 0.1,0.5,0.9
```

See each recipe's `README.md` for its specific flags and env.

## Common env (set in every run.sh)

- `VLLM_ROCM_USE_AITER_MLA=0` — **required** on gfx942.
- `--trust-remote-code --reasoning-parser kimi_k3 --mm-encoder-tp-mode data
  --safetensors-load-strategy prefetch`.
- ROCm 7.2.x: `HSA_ENABLE_IPC_MODE_LEGACY=0`, `PYTORCH_ALLOC_CONF` /
  `PYTORCH_HIP_ALLOC_CONF=expandable_segments:False`.
- **RDMA fabric env is cluster-specific** (`NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, and the
  `MORI_*` knobs in the moriep recipe) — adjust for your cluster's NICs/GIDs.

Overridable per recipe: `IMAGE`, `MODEL_DIR`, `MASTER`, `PORT` (8000), `MAX_MODEL_LEN`
(10240), `MAX_NUM_SEQS` (8), `GPU_UTIL` (0.90).
