# MiniMax-M3 disaggregated serving — proxy/router mechanics (from InferenceX)

MiniMax-M3 (MXFP4) is served **prefill/decode disaggregated (PD-disagg)** across nodes on AMD
MI355X (gfx950): prefill workers (kv_producer) and decode workers (kv_consumer) on separate
nodes, KV streamed over RDMA, fronted by ONE atomesh router. Bench drives the router → a true
aggregate throughput number.

**All workers are TP4, no EP.** The single-node M3 recipe found plain TP4 beats both TP8 and
TP4/EP4 on tok/s/GPU on gfx950. We tune the prefill:decode worker ratio (xP:yD) instead of TP.
Serving is **STP only** (no MTP): `DECODE_MTP_SIZE=0`.

## The proxy/router this recipe uses

| Engine | Router/proxy | KV-transfer backend | Ports | InferenceX source |
|--------|--------------|---------------------|-------|-------------------|
| **ATOM** | **atomesh** (`/usr/local/bin/atomesh launch --pd-disaggregation`) | **mooncake** (RDMA, `--kv-transfer-config` JSON) | router 8000, prefill 8010, decode 8020, handshake 6301 | `server_atom.sh` (`minimaxm3-fp4-mi355x-atom-disagg`) |

This folder ships the **ATOM MXFP4** path only (the top-throughput MiniMax-M3 disagg config on
MI355X). The InferenceX config family also includes vLLM disagg (MoRI-IO) and MXFP8 variants;
those are out of scope here.

## ATOM disagg mechanics

- **Topology**: `xP` prefill workers + `yD` decode workers, each TP4 (1 worker = 1 node on an
  8-GPU node, since ceil(4/8)=1).
  - `1p1d` = 2 nodes; `2p1d_dpa` = 2 prefill + 1 decode with DP-attention = 3 nodes (high conc).
- **Prefill server** (kv_producer):
  ```
  python3 -m atom.entrypoints.openai_server --model <path> --host 0.0.0.0 --server-port 8010 \
    -tp 4 [--enable-dp-attention]  --block-size 128 \
    --gpu-memory-utilization 0.8 --max-num-seqs N --max-model-len 32768 \
    --max-num-batched-tokens 32768 --no-enable_prefix_caching \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","proxy_ip":"<ip>","handshake_port":6301}'
  ```
- **Decode server** (kv_consumer): same, `--server-port 8020`, `kv_role":"kv_consumer"`.
- **Router** (on prefill node 0, after all servers healthy):
  ```
  atomesh launch --host 0.0.0.0 --port 8000 --pd-disaggregation \
    --prefill http://<pf_ip>:8010 [...] --decode http://<dec_ip>:8020 [...] \
    --policy random --backend atom --disable-health-check --disable-circuit-breaker
  ```
- **Bench**: drive `http://<router_ip>:8000` (OpenAI-compatible) at aggregate concurrency.
- **M3-specific env**: `AITER_QUICK_REDUCE_QUANTIZATION=INT4` (MXFP4 quick-reduce), applied for
  all non-DSv4 models by `server_atom.sh`. No `--enable-tbo` (that is a DSv4-only path).
- KV transfer uses mooncake's own RDMA device selection (IBDEVICES not passed as a server arg).

## Topologies to benchmark (from minimaxm3-fp4-mi355x-atom-disagg)
| shape | conc | topo | nodes |
|-------|------|------|-------|
| 8192/1024 | 1–256 | 1P1D TP4 | 2 |
| 1024/1024 | 1–256 | 1P1D TP4 | 2 |
| 8192/1024 | 256–1024 | 2P1D + DP-attn TP4 | 3 |
