# GLM-5.2-MXFP4 — disaggregated (P/D) serving on MI355X (ATOM + mooncake TCP)

Two-node **prefill/decode disaggregated** serving for GLM-5.2-MXFP4 on AMD Instinct MI355X (gfx950):
prefill (`kv_producer`) TP8 on node A + decode (`kv_consumer`) TP8 on node B, fronted by one `atomesh`
router, with the KV cache streamed prefill → decode over the **mooncake TransferEngine (TCP)**.

## Why this stack

- **ATOM native engine.** GLM-5.2 is `GlmMoeDsa` — a DeepSeek-v4-style MoE with multi-head latent
  attention (MLA) and sparse (DSA) attention. ATOM's native engine implements this architecture directly.
  (SGLang's GLM-5.2 DSA top-k kernels are CUDA-only and not yet ported to ROCm/hipcc.)
- **mooncake KV transfer over TCP.** These nodes do not load the `amdgpu_peermem` kernel module, so
  GPU-direct RDMA memory registration is unavailable. mooncake's TCP transport moves the KV cache over
  the host RoCE interface instead — no GPU-direct RDMA hop required.

## The two settings that make it work

Both are already applied by `serve_atom.sh`; they are the non-obvious part of this recipe.

1. **MXFP4 quant config excludes `*expert*`.** `--online_quant_config` keeps the MoE experts in MXFP4
   (not re-quantized). The FP8 per-block expert re-quantization deadlocks the decode engine at
   "Weight post-processing (online quantization)".
2. **`MC_FORCE_TCP=1`.** Forces mooncake's TCP data path. Setting only `protocol: tcp` in the
   kv-transfer-config is not enough — mooncake still attempts GPU-direct RDMA writes and every transfer
   returns `block RDMA chunk error -1`.

## Prerequisites

- **2 nodes**, 8× gfx950 (MI355X) each, on the same RoCE fabric.
- Image `rocm/atom-dev:latest` (ships `atom.entrypoints.openai_server`, `atomesh`, and mooncake).
  It is public on Docker Hub (anonymously pullable).
  > **FYI — tested image.** This recipe was validated on `rocm/atom-dev@sha256:280d2fe1a4d79db51cb37ae96112cecfcc208436b29ba0880bf04039606f5157`
  > (the `:latest` tag at time of testing). `:latest` floats, so for a reproducible run pin that digest via
  > `IMG=rocm/atom-dev@sha256:280d2fe1a4d7...`.
- GLM-5.2-MXFP4 weights staged to the same path on every node (default `/models/GLM-5.2-MXFP4`;
  the host `/models` dir is mounted into the containers).
- ionic RDMA provider libs present on the host (bind-mounted automatically by the launcher).

## Files

| File | Role |
|------|------|
| `serve_atom.sh` | Per-node launcher (`ROLE=prefill\|decode\|router`); starts the ATOM engine / atomesh router. |
| `bench_pd.sh`   | Concurrency-sweep driver against the router (OpenAI-compatible `/v1/chat/completions`). |

## Usage

Pick the prefill node IP (`PROXY_IP`) and decode node IP (`DECODE_IP`). Weights must be staged to
`$MODEL` on both nodes.

```bash
# On node A (prefill):
PROXY_IP=<nodeA_ip> DECODE_IP=<nodeB_ip> ROLE=prefill ./serve_atom.sh

# On node B (decode):
PROXY_IP=<nodeA_ip> DECODE_IP=<nodeB_ip> ROLE=decode  ./serve_atom.sh

# On node A, once prefill (:8010) and decode (:8020) are serving — start the router:
PROXY_IP=<nodeA_ip> DECODE_IP=<nodeB_ip> ROLE=router  ./serve_atom.sh
```

The router exposes an OpenAI-compatible endpoint on `http://<nodeA_ip>:30000`:

```bash
curl -s http://<nodeA_ip>:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"/models/GLM-5.2-MXFP4",
  "messages":[{"role":"user","content":"What is the capital of France? One word."}],
  "max_tokens":32,"temperature":0}'
```

Throughput sweep through the router:

```bash
ROUTER_IP=<nodeA_ip> ./bench_pd.sh                          # 1k/1k @ conc 1,8,32
SHAPES="8192,1024" CONCS="8 16" ROUTER_IP=<nodeA_ip> ./bench_pd.sh
```

## Overridable env (both scripts)

| Var | Default | Meaning |
|-----|---------|---------|
| `MODEL` | `/models/GLM-5.2-MXFP4` | model path (same on every node) |
| `MOUNT` | `/models` | host dir mounted into the container (must contain `MODEL`) |
| `TP` | `8` | tensor-parallel size per engine |
| `PF_PORT` / `DC_PORT` | `8010` / `8020` | prefill / decode server ports |
| `HS_PORT` | `6301` | mooncake handshake port |
| `ROUTER_PORT` | `30000` | atomesh router port |
| `SOCKET_IFNAME` | `ens3` | host interface for mooncake/NCCL sockets |
| `CACHE_ROOT` | `$PWD/cache` | host dir for persisted JIT caches |
| `IMG` | `rocm/atom-dev:latest` | container image |
