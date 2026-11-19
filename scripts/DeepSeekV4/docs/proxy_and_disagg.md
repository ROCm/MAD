# DSv4-Pro disaggregated serving — proxy/router per engine (from InferenceX)

DeepSeek-V4-Pro is a large MoE (61 layers, 384 routed experts, hidden 7168, `moe_intermediate_size=3072`).
Checkpoint is **FP8 block-quant (128×128, scale ue8m0)** with FP4 MoE expert weights applied at
runtime — i.e. "FP4+FP8 mixed". It does NOT fit/serve well aggregated on one node at scale, so
serving is **prefill/decode disaggregated (PD-disagg)** across nodes.

Note: unlike Qwen3-Next FP8 (`moe_intermediate_size=512` → pure TP8 invalid), DSv4 has 3072/8=384 ≥ 128,
so **TP8 is valid** for DSv4 FP8. InferenceX runs TP8 throughout.

## The proxy/router each engine uses

| Engine | Router/proxy | KV-transfer backend | Ports | InferenceX source |
|--------|--------------|---------------------|-------|-------------------|
| **ATOM**   | **atomesh** (`/usr/local/bin/atomesh launch --pd-disaggregation`) | **mooncake** (RDMA, `--kv-transfer-config` JSON) | router 8000, prefill 8010, decode 8020, handshake 6301 | `server_atom.sh` (DSv4 disagg config exists) |
| **SGLang** | **sglang_router** / mini-lb (`ENGINE=sglang-disagg`) | **MoRI** (MoRI-IO RDMA) | router 8000 (all servers 8000) | `server_sglang.sh` + `env.sh` (only DeepSeek-**R1** disagg config exists; adapt to DSv4) |
| **vLLM**   | vLLM disagg proxy | **MoRI-IO** / Nixl (`MoRIIOConnector`) | — | `server_vllm.sh` (referenced by dispatcher `server.sh`; not in our snapshot) |

Dispatcher `server.sh` selects by `ENGINE`: `atom-disagg`→server_atom.sh, `vllm-disagg`→server_vllm.sh,
else server_sglang.sh.

## ATOM disagg mechanics (the one DSv4-validated path)

- **Topology**: `xP` prefill workers + `yD` decode workers, each TP8 (1 worker = 1 node at TP8).
  - 1P1D = 2 nodes; 2P1D + DP-attention = 3 nodes (high concurrency).
- **Prefill server** (kv_producer):
  ```
  python3 -m atom.entrypoints.openai_server --model <path> --host 0.0.0.0 --server-port 8010 \
    -tp 8 [--enable-dp-attention --enable-tbo]  --kv-cache-dtype fp8 --block-size 16 \
    --gpu-memory-utilization 0.85 --max-num-seqs N --no-enable_prefix_caching \
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
- **DSv4-specific env** (from env_atom.sh): `ATOM_MOE_GU_ITLV=1`, `AITER_BF16_FP8_MOE_BOUND=0`,
  and for the TBO path `GPU_MAX_HW_QUEUES=5`, `ATOM_CPU_AFFINITY=1`.
- KV transfer uses mooncake's own RDMA device selection (IBDEVICES not passed as a server arg).

## SGLang disagg mechanics (adapt from DeepSeek-R1 recipe — no DSv4 config yet)

- `ENGINE=sglang-disagg`; MoRI-IO env in `env.sh` (`MORI_IO_*`, `MORI_RDMA_TC/SL` QoS by NIC).
- Prefill: `--disaggregation-mode prefill`; Decode: `--disaggregation-mode decode` + DP-attention/EP
  (R1 decode uses `ep:8 dp-attn:true`, "DEP" mode). All servers on port 8000; sglang_router fronts.
- DSv4 caveat: needs an sglang image carrying the DSv4 model branch (the `-DSv4` rocm/sgl-dev tags).
  R1 disagg uses `rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-*` — DSv4 support must be confirmed.

## Topologies to benchmark (from dsv4-fp4-mi355x-atom-disagg)
| shape | conc | topo | nodes |
|-------|------|------|-------|
| 1024/1024 | 4–1024 | 1P1D TP8 | 2 |
| 8192/1024 | 4–128 | 1P1D TP8 | 2 |
| 8192/1024 | 256–2048 | 2P1D + DP-attn TP8 | 3 |
