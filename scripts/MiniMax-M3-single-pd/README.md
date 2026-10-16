# MiniMax-M3 MXFP4 — single-node vLLM + MoRIIO P/D disaggregation over XGMI

Simulates prefill/decode (P/D) disaggregation on a **single** AMD MI355X node (8× gfx950):
prefill (kv_producer) TP4 on GPU0-3 + decode (kv_consumer) TP4 on GPU4-7, fronted by
`vllm-router`, with the KV cache moving prefill→decode via **MoRIIO over the XGMI GPU fabric**
(`backend=xgmi`) — no NIC/RDMA hop for the intra-node transfer.

Validated end-to-end: router-mediated P/D, XGMI backend engaged, correct inference, and a
throughput sweep at 1k/1k and 8k/1k (see **Results** below).

> This is the **single-node XGMI** counterpart to the cross-node ATOM recipe in
> `scripts/MiniMax-M3/`. It uses the **vLLM + MoRIIO** stack (not ATOM/mooncake) — see "Why this
> stack" for the reason single-node co-location requires vLLM here.

## Why this stack (vLLM+MoRIIO, not ATOM/mooncake)
Two co-located TP groups on one node collide in the ATOM/aiter custom all-reduce (a fixed-key IPC
region → `hipIpcGetMemHandle ... invalid argument` on the 2nd engine), and ATOM has no switch to
disable it (its all-gather asserts `ca_comm != None`). ATOM's mooncake transport also has **no
XGMI path** (RDMA/TCP only). vLLM provides a supported `--disable-custom-all-reduce` **and** MoRIIO
has a native XGMI backend — so single-node co-located P/D over XGMI is only achievable on this stack.

## What was non-obvious (each was a hard failure until fixed)
1. **`--disable-custom-all-reduce` is mandatory** for two co-located TP groups (else the AITER
   custom all-reduce IPC handles collide and the 2nd engine aborts). Falls back to RCCL.
2. **Distinct MoRIIO ports per engine.** Both containers share host network; each binds
   `handshake_port` (+1/rank) and `notify_port`. Use different bases (prefill 6301/61005,
   decode 6311/61015) or they collide with `Address already in use`.
3. **`--max-model-len 32768`.** The model default is 1,048,576 (1M) → ~44 GiB KV, which won't fit.
4. **GPU memory headroom.** `--gpu-memory-utilization 0.90` needs a mostly-free node; on a shared
   node (another tenant holding VRAM) lower it (≈0.30) or you OOM at warmup / get near-zero KV.
5. **Per-container GPU isolation.** Set both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`
   (0-3 / 4-7) so MoRIIO/QuickReduce map physical GPUs correctly.
6. **Flat model dir with remote-code** — the HF weights-only download lacks
   `configuration_minimax_m3_vl.py` referenced by `config.json` `auto_map` (see `prep_model.sh`).
7. **XGMI backend** via `kv_connector_extra_config: {"backend": "xgmi"}` (default `"rdma"`).
   Confirmed by `Using MoRIIO backend: XGMI` in every worker log.

## Files
| File | Purpose |
|------|---------|
| `run_pd_singlenode.sh` | launch router + prefill + decode, wait for registration, smoke inference |
| `prep_model.sh` | stage `amd/MiniMax-M3-MXFP4` as a flat dir (weights symlinks + remote-code .py) |
| `bench_pd.sh` | `vllm bench serve` throughput sweep through the router (shapes/concurrency via env) |

## Prerequisites
- AMD MI355X node, ROCm; Docker with `--device=/dev/kfd,/dev/dri,/dev/infiniband`.
- Images: `rocm/vllm-dev:vllm-0.23.1-rocm723-mi35x-mori-0625` (engine) and
  `vllm/vllm-router:nightly-<date>` (router; pin a currently-available nightly — Docker Hub keeps
  only ~16). Override via `IMG` / `ROUTER_IMG`.
- Model staged flat (see `prep_model.sh`), reachable at `$MODEL` inside the container mount `$MOUNT`.
- Enough free VRAM (mostly-idle node for `GPU_MEM_UTIL=0.90`; otherwise lower it).

## Run
```bash
# stage the model once (adjust paths):
SNAPSHOT=/path/to/models--amd--MiniMax-M3-MXFP4/snapshots/<hash> \
PYSRC=/path/to/minimax-m3-remote-code DEST=/models/MiniMax-M3-MXFP4 ./prep_model.sh

# launch P/D over XGMI + smoke inference (run detached so containers persist):
MODEL=/models/MiniMax-M3-MXFP4 MOUNT=/models BACKEND=xgmi nohup ./run_pd_singlenode.sh &

# throughput benchmark through the router (1k/1k and 8k/1k):
SHAPES="1024,1024 8192,1024" CONCS="1 8 32" ./bench_pd.sh
```
`BACKEND=rdma` switches KV transfer to RDMA-over-NIC (the upstream default) for comparison.

## Verifying it worked
- Router log shows `🔵Add Prefill [...]` and `🔵Add Decode [...]` (both registered).
- A chat request to `http://<host>:30000/v1/chat/completions` returns a correct answer, and the
  router log maps it to **both** `:2584` (prefill) and `:2585` (decode).
- Decode/prefill logs show `Using MoRIIO backend: XGMI`.

## Results (validation, MI355X, backend=xgmi)
Functional/scaling smoke test through the router (`vllm bench serve`), not a tuned peak sweep:

| ISL/OSL | Concurrency | Output tok/s | Total tok/s | Mean TPOT (ms) |
|---------|-------------|--------------|-------------|----------------|
| 1024/1024 | 1  | 72   | 157   | 9.5  |
| 1024/1024 | 8  | 564  | 1,225 | 13.8 |
| 1024/1024 | 32 | 1,822| 3,957 | 16.6 |
| 8192/1024 | 8  | 502  | 4,603 | 14.3 |

Accuracy (temperature 0, through the router): GSM8K-style prompts answered correctly
(e.g. Natalia=72, robe=3, 3×24−17=55, "17×23"=391). `lm_eval` was not in the image; validation
used direct known-answer probes rather than the full harness.

## Provenance
Derived from the InferenceX `minimaxm3-fp4-mi355x-vllm-disagg` config (which runs MoRIIO
**cross-node over RDMA**); adapted here to **single-node co-located P/D over XGMI**.
Related upstream fixes: [ROCm/mori #481](https://github.com/ROCm/mori/pull/481) (dma-buf MR
registration for peermem-less MI355X + Ionic clusters),
[vllm-project/router #181](https://github.com/vllm-project/router/pull/181) (discovery DP-rank
round-robin).
