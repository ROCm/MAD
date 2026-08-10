# wideep_disagg_2p2d — Kimi-K3 MI300X 2P/2D EP16 MoRIIO disagg

> **✅ VALIDATED.** Single-needle NIAH passes **deterministically through 300K
> tokens** (all depths) on this 2-prefill + 2-decode EP16 disagg serve. The
> previously-open decode-recall bug is fixed (two root causes: 4-KV-group block
> routing + multi-chunk prefill transfer). See [`STATUS.md`](STATUS.md) for the
> root-cause write-up and [`RESULTS.md`](RESULTS.md) for the full NIAH +
> latency/throughput tables.
>
> **When to use this vs. colocated:** disagg (TP2×DP8) is for **concurrent
> throughput** and decode-latency isolation — 5.7× throughput at concurrency 8,
> 7.3× at 16. A *single* request runs on one DP replica (2 GPUs), so single-stream
> latency is ~4× a colocated PP2×TP8 serve (which spreads one request across all
> 16 GPUs). For low-latency interactive / low-QPS, use the colocated recipes
> ([`../pp2xtp8`](../pp2xtp8), [`../wideep_int4_moriep`](../wideep_int4_moriep));
> for high-QPS / batch, use this.

## What this is

Prefill/decode **disaggregated** Kimi-K3 across **4 MI300X nodes**:

```
  Prefill pool                         Decode pool
  ┌───────────────┐   MoRIIO (RDMA)   ┌───────────────┐
  │ PM + PW       │  KV + KDA state   │ DM + DW       │
  │ 2 nodes = 16  │ ────────────────► │ 2 nodes = 16  │
  │ GPU, EP16     │   write + notify  │ GPU, EP16     │
  └───────────────┘                   └───────────────┘
       ▲ router (:30000) on PM fans requests P → D
```

- **Per pool**: TP2 × DP8 → **EP16** expert-parallel via **MoRI-EP** all2all.
- **No pipeline parallelism.** Disaggregation is the only cross-stage split;
  everything else is DP/EP.
- **Connector**: MoRIIO in **WRITE** mode — prefill RDMA-writes both the MLA
  attention KV *and* the Kimi-Delta-Attention (KDA/mamba) recurrent+conv state
  into the decode engine's blocks, then notifies over TCP/ZMQ.

K3 is hybrid: 24 MLA full-attention layers (paged fp8 KV) + 69 KDA layers
(recurrent + conv state). The two live in **separate** vLLM KV-cache groups, and
the connector transfers both — routing the KDA state by the *mamba* group's block
ids (see fix 1 below).

## Quick start

Runs from a control host that can `ssh` to all four nodes. Edit the node
IPs/hostnames (or export `PM_NODE/PM_IP/…`) and point `MODEL_DIR` at your
Kimi-K3-MXFP4 weights (local NVMe on every node recommended).

```bash
cd wideep_disagg_2p2d/

# 1. build or obtain the disagg image (see "Image" below), tag it kimik3-wideep-disagg:latest
# 2. edit the 4 node IPs at the top of run_2p2d_launch.sh (or export PM_NODE=… etc.)

MODEL_DIR=/path/to/Kimi-K3-MXFP4 \
PM_NODE=… PM_IP=… PW_NODE=… PW_IP=… DM_NODE=… DM_IP=… DW_NODE=… DW_IP=… \
AUTO_ROUTER=1 \
K3_GROUP_ROUTING=1 K3_EXTRA_FIXES=1 LOAD_STRATEGY=lazy \
MAX_MODEL_LEN=320000 MAX_NUM_BATCHED_TOKENS=2048 GPU_UTIL=0.85 \
bash run_2p2d_launch.sh

# watch both masters for "Application startup complete", then (AUTO_ROUTER does this
# for you) a single vllm-router comes up on PM:30000. Verify + probe:
curl http://<prefill-master-ip>:30000/v1/models
python3 niah_probe.py --url http://<prefill-master-ip>:30000 --model kimi-k3 \
    --ctx 50000 --depths 0.1,0.5,0.9      # PASS (deterministic to 300K)
```

### The winning config (flags that matter)

| Flag | Value | Why |
|------|-------|-----|
| `K3_GROUP_ROUTING` | `1` | **Fix #1** — 4-KV-group block routing (always on). |
| `K3_EXTRA_FIXES` | `1` | **Fix #2** — multi-chunk compute-progress gate + all-group accumulation. Required for recall past `max_num_batched_tokens`. |
| `LOAD_STRATEGY` | `lazy` | `prefetch` double-loads RAM when the model is on tmpfs → decode OOM. |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | Best measured throughput; raising to 8192 did **not** cut latency and hurt throughput (compute-bound prefill). |
| `MAX_MODEL_LEN` | `320000` | Needed for > 131K-token NIAH (default 131072 caps ~120K). |
| `GPU_UTIL` | `0.85` | 0.88 razor-misses KV headroom on some nodes. |
| `KV_CACHE_DTYPE` | `fp8` (default) | Transfer geometry assumes 1-byte elements; bf16 corrupts. |
| `PREFILL_BACKEND` | `mori_low_latency` | V1 high_throughput dispatch warmup crashes on this stack. |

**Model on tmpfs (recommended):** loading the 1.5 TB checkpoint from a tmpfs RAM
cache (`/mnt/rammodel/Kimi-K3-MXFP4`) with `LOAD_STRATEGY=lazy` is ~2 min vs
~20 min from NFS (whose page cache gets evicted between runs).

`run_2p2d_launch.sh` deploys the scripts + patchers + image to all four nodes,
starts **workers first, then masters**, then (with `AUTO_ROUTER=1`) waits for both
masters' `/v1/models` before launching exactly one router. `run_2p2d.sh` is the
per-node entrypoint (dispatches on `ROLE=prefill_master|prefill_worker|
decode_master|decode_worker`).

## Load-bearing env (set by the launcher)

| Var | Default | Meaning |
|-----|---------|---------|
| `TP_SIZE` / `DP_SIZE` / `DP_LOCAL` | 2 / 8 / 4 | TP2×DP8 → EP16 per pool; 4 DP ranks per node. |
| `PREFILL_BACKEND` | `mori_low_latency` | MoRI-EP all2all backend. |
| `KV_CACHE_MEMORY_BYTES` | 8e9 | Per-engine KV cache budget. |
| `PMASTER`/`DMASTER`/`PROXY_IP` | node IPs | Pool masters + router/proxy host. |
| `PREFILL_POD_HOSTS`/`DECODE_POD_HOSTS` | node IP lists | Pool membership. |
| `MODEL_DIR` | *(required)* | Kimi-K3-MXFP4 weights path (must exist on every node). |

### RDMA fabric (overridable; validated defaults ON)

`run_2p2d.sh` reads all fabric from env, so **nothing is hardcoded**, but the
defaults are the validated Broadcom **Thor2 (bnxt RoCE)** values so it works
out-of-the-box on the reference cluster:

| Var | Default (Thor2) | Override for e.g. Mellanox |
|-----|-----------------|---------------------------|
| `SOCKET_IFNAME` | `eno0` | `eth0` |
| `NCCL_IB_HCA` / `RDMA_DEVICES` | `rdma0..rdma7` | `mlx5_0,mlx5_2,…,mlx5_9` |
| `IB_GID_INDEX` | `3` | your GID |
| `THOR2_BNXT_FIX` | `1` | `0` (non-Thor2) |

`THOR2_BNXT_FIX=1` mounts the host **v34** `libibverbs`/`libbnxt_re` onto the
image's resolved soname (the image ships v59, but the Thor2 `bnxt_re` kernel
driver only accepts v34 → otherwise 0 RDMA devices). On non-bnxt fabric set
`THOR2_BNXT_FIX=0`. Example override:
```bash
SOCKET_IFNAME=eth0 NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9 \
RDMA_DEVICES=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9 \
IB_GID_INDEX=3 THOR2_BNXT_FIX=0 bash run_2p2d_launch.sh

## The connector fixes (folded into the vLLM source branch)

The two root-cause fixes are **folded into vLLM source** on branch
`kimi-k3-wideep-disagg-fullsource-v2` of `raviguptaamd/vllm` (the `-v2` = the
base `fullsource` branch + these fixes baked in). The Dockerfile builds that
branch (`VLLM_REF=kimi-k3-wideep-disagg-fullsource-v2`), so the **image has the
fixes baked in**. The same fixes also ship as runtime patchers under
[`patchers/`](patchers/); on a v2 image they detect "already applied" and no-op
(idempotent) — so the recipe also works on an older/unfolded image. To rebuild
the image from scratch see the [Image](#image) section.

**The two root-cause fixes that make NIAH pass** (full write-up in
[`STATUS.md`](STATUS.md)):

1. **4-KV-group block routing** (`apply_kimik3_moriio_group_routing.py`, always
   on, `K3_GROUP_ROUTING=1`) — K3 has **4** KV-cache groups (3 KDA/mamba + 1 MLA);
   the shipped connector hardcoded 2-group indices and sent MLA KV to mamba
   block-ids. Fix carries all groups' block-ids end-to-end and routes each layer
   by its own group. Fixes short (≤ 1 block) recall.
2. **Multi-chunk prefill transfer** (`apply_kimik3_chunk_gate_fix.py` +
   `apply_kimik3_chunked_allgrp.py`, `K3_EXTRA_FIXES=1`) — the connector detected
   the final prefill chunk by *block count*, which fires after chunk 1 when a
   prompt fits in ≤ 1 padded block → only `max_num_batched_tokens` of KV crossed.
   Fix gates on **compute progress** from `scheduler_output` (build map → entry
   defer → accumulation final-detect → post-loop sweep). Removes the razor cliff
   at `max_num_batched_tokens`; recall now scales to 300K.

Pre-existing connector fixes (still required, always on):

3. **mamba block-id routing** — transfer KDA/mamba state by the *mamba* KV-cache
   group's block ids (superseded in the general case by fix 1; kept as fallback).
4. **remote_tp_size normalize** — degenerate `remote_tp_size ≤ 1` → `world_size`,
   so KV fans out to **all** decode TP ranks (not just rank 0).
5. **mamba N−1 boundary** — producer computes through token N−1, decoder recomputes
   token N (matches vLLM's nixl/mooncake hybrid-PD handling).

[`patchers/diagnostics/`](patchers/diagnostics/) holds element-wise transport
probes (all env-gated OFF). See the Debugging section.

## Image

[`Dockerfile.kimik3_disagg`](Dockerfile.kimik3_disagg) is **fully self-contained and
builds from a public base** — no gated images required:

- **Base** (`BASE_IMAGE`): `rocm/vllm-dev:ci_base-0fcd9b99...` — the open ROCm 7.2 /
  cp312 vLLM CI base (publicly pullable). It only supplies ROCm + torch; the entire
  Wide-EP stack is built **from source** on top:
  - **MoRI** `ROCm/mori @ v1.2.2` (gfx942, `BUILD_UMBP=OFF`) — the EP all2all kernels;
  - **AITER** `0.1.19` wheel + `flydsl 0.2.4`;
  - **vLLM** compiled from `VLLM_REPO`/`VLLM_REF` = `raviguptaamd/vllm` branch
    `kimi-k3-wideep-disagg-fullsource-v2` (the folded connector fixes above);
  - **vllm-router** (DP-rank round-robin + MoRIIO KV-notify) built in;
  - with `WITH_NIXL=1` (default): UCX + RIXL + rocSHMEM + DeepEP from source too.
- **K3 AITER graft** (`PROVEN_K3_IMAGE`): `amdsiloai/vllm:kimi-k3-mi325x-release-v2`
  (public) — only the Kimi-K3 tuned MXFP4 MoE configs / conv kernels are copied from it.

```bash
docker build -f Dockerfile.kimik3_disagg \
  --build-arg GH_TOKEN=$(gh auth token) \
  -t kimik3-wideep-disagg:latest .
```

Build args (`BASE_IMAGE`, `PROVEN_K3_IMAGE`, `MORI_REF`, `VLLM_REPO`, `VLLM_REF`,
`WITH_NIXL`, `GH_TOKEN`) let you override any component. Push the result to a registry
you control and set `HUB_IMAGE` for [`load_image.sh`](load_image.sh) to pull it onto
each node. **Note:** the image builds `VLLM_REF` from GitHub, so push the vLLM fork
branch before building for a reproducible image.

## Tests

| Script | What |
|--------|------|
| [`niah_probe.py`](niah_probe.py) | Single-needle NIAH via the router (`--ctx --depths`). The deliverable metric — PASS to 300K. |
| [`benchmark_niah.py`](benchmark_niah.py) | Stricter 10-needle multi-needle stress (`NIAH_WORDS=…`). |
| [`concurrency_bench.py`](concurrency_bench.py) | Concurrent throughput (req/s, tok/s, latency percentiles). |

See [`RESULTS.md`](RESULTS.md) for the full NIAH sweep + latency/throughput tables.

## Debugging (opt-in, all default OFF)

Turn diagnostics up/down via env flags; none change default behavior:

| Flag | Effect |
|------|--------|
| `K3_CHUNK_GATE_DEBUG=1` | Log the chunk-gate decision per request (`entry`/`accum`/`sweep`, computed/scheduled/npt). |
| `K3_XFER_PROBE=1` | Producer offsets + src checksum, decode read-block + dst checksum. |
| `K3_DECODE_RECV_PROBE=1` | Decode reads its own KV slot norm on write-completion (~0 = bytes didn't land). |
| `K3_KDA_STATE_PROBE=1` | KDA recurrent/conv state norm at the decode read slot. |
| `K3_WRITE_BC=1` / `K3_HS_BC=1` / `K3_INPUTS_PROBE=1` | Write-delivery / handshake-dial / decode-inputs breadcrumbs. |
| `K3_WRITE_FENCE=delay K3_WRITE_FENCE_MS=…` | Sender write→notify delay (investigative; did not fix the residual race). |

**Verify a patch landed in-container** (`/patchers` is bind-mounted; a stale file
silently re-applies old behavior):
```bash
docker exec k3disagg_prefill_master bash -lc \
 'B=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/moriio; \
  grep -c k3-group-routing $B/moriio_connector.py; grep -c _k3_prog $B/moriio_connector.py'
```

## Status

**VALIDATED** — single-needle NIAH passes deterministically to 300K. See
[`STATUS.md`](STATUS.md) for root cause, the fix, and the known residual
(multi-needle write race at ≥ 20K, single-needle unaffected).
