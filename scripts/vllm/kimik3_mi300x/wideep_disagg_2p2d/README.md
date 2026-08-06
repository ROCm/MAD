# wideep_disagg_2p2d — Kimi-K3 MI300X 2P/2D EP16 MoRIIO disagg (WIP)

> **⚠️ WORK IN PROGRESS — read [`STATUS.md`](STATUS.md) first.** The disagg
> transport is byte-perfect and every connector-level bug is fixed, but an open
> decode-side accuracy bug means exact long-context (NIAH) recall does **not** pass
> yet. This lands as reviewable infra + an honest status write-up, **not** a
> production deployment. For serving today, use the colocated recipes
> ([`../pp2xtp8`](../pp2xtp8), [`../wideep_int4_moriep`](../wideep_int4_moriep)).

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
bash run_2p2d_launch.sh

# watch both masters for "Application startup complete", then (AUTO_ROUTER does this
# for you) a single vllm-router comes up on PM:30000. Verify + probe:
curl http://<prefill-master-ip>:30000/v1/models
python3 niah_probe.py --url http://<prefill-master-ip>:30000 --model kimi-k3 \
    --ctx 6000 --depths 0.1,0.5,0.9      # <-- currently FAILS; see STATUS.md
```

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

RDMA fabric env (`NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `MORI_*`) is
**cluster-specific** — adjust for your NICs/GIDs.

## The connector fixes (folded into the vLLM source branch)

These were developed here as runtime patchers and are now **folded into vLLM
source** on branch `kimi-k3-wideep-disagg-fullsource` of `raviguptaamd/vllm`
(one commit per fix). The image builds that branch directly; the patchers under
[`patchers/`](patchers/) remain for reference / older images.

1. **mamba block-id routing** — transfer KDA/mamba state by the *mamba* KV-cache
   group's block ids, not the attention group's. Without this, KDA state lands in
   the wrong blocks.
2. **remote_tp_size normalize** — a degenerate `remote_tp_size ≤ 1` is normalized
   to `world_size`, so prefill rank *k* writes to decode rank *k* and the KV fans
   out to **all** decode TP ranks (not just rank 0).
3. **mamba N−1 boundary** — the producer computes hidden state through token N−1
   and the decoder recomputes token N, matching vLLM's own nixl/mooncake hybrid-PD
   handling.
4. **write-fence / devsync** *(gated, default OFF: `K3_WRITE_FENCE`,
   `K3_WRITE_DEVSYNC`)* — RDMA-write ordering knobs; inert on the known recall bug.
5. **MLA single-split** *(gated, default OFF: `K3_MLA_SINGLE_SPLIT`)* — forces a
   single KV split for a deterministic reduction; inert on the known recall bug.

Fixes 1–3 are correctness and always on. 4–5 are opt-in diagnostics.
[`patchers/diagnostics/`](patchers/diagnostics/) holds element-wise transport
probes (all env-gated OFF) used to prove byte-perfect transfer.

## Image

[`Dockerfile.kimik3_disagg`](Dockerfile.kimik3_disagg) is **fully self-contained and
builds from a public base** — no gated images required:

- **Base** (`BASE_IMAGE`): `rocm/vllm-dev:ci_base-0fcd9b99...` — the open ROCm 7.2 /
  cp312 vLLM CI base (publicly pullable). It only supplies ROCm + torch; the entire
  Wide-EP stack is built **from source** on top:
  - **MoRI** `ROCm/mori @ v1.2.2` (gfx942, `BUILD_UMBP=OFF`) — the EP all2all kernels;
  - **AITER** `0.1.19` wheel + `flydsl 0.2.4`;
  - **vLLM** compiled from `VLLM_REPO`/`VLLM_REF` = `raviguptaamd/vllm` branch
    `kimi-k3-wideep-disagg-fullsource` (the folded connector fixes above);
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

## Status

See [`STATUS.md`](STATUS.md) for what is verified, the open recall bug, the exact
repro, and everything that was ruled out. **NIAH recall does not pass yet.**
