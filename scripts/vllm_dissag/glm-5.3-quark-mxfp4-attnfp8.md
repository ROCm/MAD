# GLM-5.3-Quark-MXFP4-AttnFP8 on MI355X: P/D serving recipe

Everything needed to build, serve, benchmark and check
[`amd/GLM-5.3-Quark-MXFP4-AttnFP8`](https://huggingface.co/amd/GLM-5.3-Quark-MXFP4-AttnFP8/tree/bcd1ec31dc1fd474b9997dd2476c92687790b83d)
with prefill/decode disaggregation (P/D) over MoRI-IO and MoRI-EP on AMD Instinct MI355X.

| | |
|---|---|
| Model | `amd/GLM-5.3-Quark-MXFP4-AttnFP8`, revision `bcd1ec31dc1fd474b9997dd2476c92687790b83d` |
| Architecture | `GlmMoeDsaForCausalLM` (MLA + DeepSeek sparse attention), base `zai-org/GLM-5.3`, 744B total / 40B active, plus one MTP layer |
| Quantization | MLP and expert weights MXFP4, attention (including the sparse indexer) FP8 (AMD Quark) |
| Served model name (tested) | `amd/GLM-5.3-Quark-MXFP4-AttnFP8` |
| `MODEL_NAME` in this launcher | `GLM-5.3-Quark-MXFP4-AttnFP8` (`models.yaml`, slurm allowlists) |
| KV per token | 93,912 bytes: 79 layers x 1,152 B BF16 MLA latent + 22 layers x 132 B FP8 indexer keys with scales, MTP layer included |

The tested deployment used llm-d on Kubernetes. The llm-d guide for this recipe (to be
published) has the manifests; this document gives every value they need. The same image and
serving flags also run through this directory's slurm launcher (see [Slurm launcher](#slurm-launcher)),
which has not been run on GPUs with this model yet.

## Contents

- [Prerequisites](#prerequisites)
- [Image](#image)
- [Serving configuration](#serving-configuration)
- [llm-d routing](#llm-d-routing)
- [Slurm launcher](#slurm-launcher)
- [Benchmarks](#benchmarks)
- [Accuracy and KV-transfer gates](#accuracy-and-kv-transfer-gates)
- [Reference results](#reference-results)
- [Known caveats](#known-caveats)

## Prerequisites

- **GPUs:** AMD Instinct MI355X (gfx950), 8 per node. The tested host driver was amdgpu 6.16.13.
- **NICs:** AMD Pensando RDMA NICs, one rail per GPU (8 per node).
  - Firmware `1.117.5-a-196`; the image's AINIC userspace must match it (`AINIC_VERSION`).
  - Netdev MTU 9000 on every rail, RDMA path MTU 4096 (`MORI_IB_PATH_MTU=4096`). Check the
    active MTU on every rail before a run.
- **Weights:** the pinned revision. Expect 141 safetensors shards with index
  `total_size=409520061992`; serve with `HF_HUB_OFFLINE=1`.
- **Nodes per layout:** P/D 1P1D uses 2 nodes, 2P1D uses 3, a colocated control uses 1 node
  per measurement.
- **llm-d deployments:**
  - Kubernetes with LeaderWorkerSet 1.0.0, the DisaggregatedSet CRD and Gateway API Inference
    Extension v1.6.2.
  - Each engine pod gets 8 `amd.com/gpu`, 8 RDMA virtual NICs and `IPC_LOCK`.
- **Slurm deployments:** see [Slurm launcher](#slurm-launcher) and [README.MD](README.MD).

## Image

```bash
cd MAD
docker build -f docker/vllm_disagg_inference.glmv5.3_quark_mxfp4_attnfp8.ubuntu.amd.Dockerfile \
  -t <your-registry>/vllm-disagg:glmv5.3_quark_mxfp4_attnfp8 .
```

The build context is the repo root. The Dockerfile rebuilds, from public sources, the pins of the
tested engine image:

| Build arg | Default | Notes |
|---|---|---|
| `BASE_DOCKER` | `vllm/vllm-openai-rocm:base-nightly-40941f2c…@sha256:3f24c91d…` | ROCm SDK 10.0.0, torch 2.13.0+rocm10.0.0, triton 3.8.0+git669b31ac, AITER v0.1.27. Built from the same `docker/Dockerfile.rocm{,_base}` and requirements as vLLM `cd1effa2e7` |
| `GFX_COMPILATION_ARCH` | `gfx950` | vLLM csrc, MoRI, and AITER if rebuilt; also the final `PYTORCH_ROCM_ARCH` / `AITER_ROCM_ARCH` |
| `AINIC_VERSION` | `1.117.5-a-196` | AMD Pensando userspace; must match host NIC firmware |
| `MORI_REF` | `a5ac2bc9c7a5d933c720c83060c4e454adbfce5b` | ROCm/mori main, includes the atomics-free path behind `MORI_EP_DISABLE_RDMA_ATOMICS` (ROCm/mori#736) |
| `AITER_REBUILD` | `0` | `1` rebuilds AITER v0.1.27 for gfx950 only, as in the tested image |
| `VLLM_REF` | `cd1effa2e7d204b22527f14739734751f38ed737` | public vLLM; patches below are applied on top |
| `VLLM_VERSION` | `0.32.0rc1.dev1492+g8673602b8` | the version the tested image reported |
| `MAX_JOBS` | `$(nproc)` | compile parallelism |

The image builds the MoRI transport only (no UCX, RIXL or DeepEP), because the recipe uses MoRI-IO
for KV transfer and MoRI-EP for expert parallel. It sets no serving env; every run-time setting
below is passed at launch.

### vLLM patches (temporary overlay)

`docker/patches/vllm_disagg_inference.glmv5.3_quark_mxfp4_attnfp8/` holds one file per vLLM change.
Each file starts with a header: what the change does, its upstream PR, and when to remove it.
`SHA256SUMS` pins each patch body. The build applies the patches in filename order with
`git apply --check` and stops on any patch that does not apply cleanly.

The overlay is temporary: each fix belongs in vLLM. Delete a patch, and its `SHA256SUMS` line,
once `VLLM_REF` contains the fix.

| Patch | Change | Upstream | Remove when |
|---|---|---|---|
| 0000 | MoRIIO WRITE waits for the decode-rank handshake. This is the tested commit `8673602b` (public `cd1effa2e7` + this change); READ does not use this path | [vllm#62552](https://github.com/vllm-project/vllm/pull/62552), open draft (earlier revision) | `VLLM_REF` contains #62552 |
| 0001 | MoRIIO accepts kernel-block-split MLA/indexer KV views (needed at block sizes other than 1 or multiples of 16) | [vllm#60783](https://github.com/vllm-project/vllm/pull/60783), open | `VLLM_REF` contains #60783 |
| 0002 | ROCm native `fused_add_rms_norm` for remote-prefilled DSA decode under CUDA graphs | [vllm#58722](https://github.com/vllm-project/vllm/pull/58722), open draft | `VLLM_REF` contains #58722 |
| 0003 | MoRIIO hybrid READ skips locally cached prefix pages (hybrid models; no-op here) | [vllm#62190](https://github.com/vllm-project/vllm/pull/62190), open | `VLLM_REF` contains #62190; rebase 0007 |
| 0004 | MoRI-EP V1 IntraNode: bound the receive rows by the DP token counts | local; overlaps [vllm#63412](https://github.com/vllm-project/vllm/pull/63412) | an equivalent bound is upstream and the AITER small-M fault is fixed |
| 0006 | `VLLM_MORI_RECV_ROW_ALIGN` (receive-row rounding, default 32) and the opt-in `VLLM_MORI_RECV_BOUND_CHECK` diagnostic | local configuration for 0004 | 0004 is removed |
| 0007 | MoRIIO non-hybrid READ skips the decode-side prefix-hit blocks, so a warm turn reads only its new tokens | local, to be upstreamed | an equivalent fix is upstream |

## Serving configuration

### Environment (every layout with MoRI)

These settings come from the tested manifests. P/D prefill, P/D decode and colocated
expert-parallel use the same set.

| Variable | Value | Why |
|---|---|---|
| `VLLM_MORI_EP_V2` | `0` | MoRI-EP V1 IntraNode kernel (patch 0004 applies to it) |
| `VLLM_MORI_RECV_ROW_ALIGN` | `4096` | required with patch 0004: at the default 32, AITER's small-M fused quant+sort path faulted on DEP8 P/D |
| `MORI_EP_DISABLE_RDMA_ATOMICS` | `1` | atomics-free MoRI-EP path |
| `MORI_ENABLE_HOST_PROXY`, `MORI_ENABLE_RAIL_ONLY`, `MORI_ENABLE_DMABUF_REG`, `MORI_IO_RAIL_AFFINITY` | `1` | AMD Pensando NICs: proxy posting, one rail per GPU, dma-buf registration, MoRI-IO peers on the same rail |
| `MORI_RDMA_SL`, `MORI_RDMA_TC`, `MORI_IB_PATH_MTU` | `3`, `106`, `4096` | tested fabric QoS and path MTU; set for your fabric |
| `MORI_GPU_ARCHS` | `gfx950` | |
| `MORI_SHMEM_HEAP_SIZE` | `17179869184` | 16 GiB MoRI-EP heap |
| `MORI_DISABLE_TOPO`, `MORI_IO_CP_TIMEOUT_MS` | `1`, `60000` | |
| `VLLM_ROCM_USE_AITER`, `_MOE`, `_MLA`, `_RMSNORM` | `1` | AITER MoE (MXFP4), MLA and RMSNorm kernels |
| `VLLM_ROCM_USE_AITER_FP8BMM`, `_FUSION_SHARED_EXPERTS`, `_LINEAR` | `0` | |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | `4096` | |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `1` | |
| `TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER` | `0` | |
| `PYTORCH_ALLOC_CONF`, `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:False` | KV must not be VMM-backed for RDMA registration |
| `HSA_ENABLE_SDMA`, `HSA_FORCE_FINE_GRAIN_PCIE`, `HSA_NO_SCRATCH_RECLAIM`, `HIP_FORCE_DEV_KERNARG` | `1` | |
| `ROCPROFILER_QUEUE_INTERPOSITION` | `1` | as tested (the base image sets 0) |
| `NCCL_IB_DISABLE` | `1` | NCCL carries only in-node control traffic (one node per instance) |
| `VLLM_SERVER_DEV_MODE` | `1` | `/reset_prefix_cache`, used by benchmarks |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`, `VLLM_ENGINE_READY_TIMEOUT_S`, `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `120`, `7200`, `3600` | |
| `HF_HUB_OFFLINE` | `1` | |

The colocated tensor-parallel control uses only the AITER, sparse-indexer and breakable-cudagraph
settings above.

### Layouts

| Layout | Instances | Per instance | KV bytes per GPU | Max batched tokens | Notes |
|---|---|---|---|---|---|
| P/D 1P1D | 1 prefill + 1 decode, one node each | DP8 / EP8 / TP1 | 80e9 | prefill 16,384, decode 256 | headline layout |
| P/D 2P1D | 2 prefill + 1 decode, one node each | DP8 / EP8 / TP1 | prefill 80e9, decode 55e9 | prefill 16,384, decode 256 | cold prompts; the prefill instances are independent DP8 instances |
| Colocated 2x expert parallel | 2 replicas per node | DP4 / EP4 / TP1 | 100e9 | 2,048 (also measured: 1,024) | control |
| Colocated 2x tensor parallel | 2 replicas per node | TP4 | 149e9 | 2,048 | control |

Common to all: `--block-size 16 --kv-cache-dtype auto` (BF16 MLA latent + FP8 indexer keys),
`--max-model-len 560000 --max-num-seqs 64 --enable-prefix-caching`, MTP-3
(`--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`), and
`--trust-remote-code`. Decode and colocated engines use
`--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+quant_fp8"]}'`
with `--cudagraph-capture-sizes 4 8 16 32 48 64 96 128 160 192 224 256`; P/D prefill runs
`--enforce-eager`. At 80e9 bytes, a P/D GPU holds about 852k tokens of KV: one 512k session.

**P/D prefill (per instance):**

```bash
vllm serve <model-dir> --served-model-name amd/GLM-5.3-Quark-MXFP4-AttnFP8 --port 8000 \
  --api-server-count 8 --tensor-parallel-size 1 --data-parallel-size 8 --data-parallel-size-local 8 \
  --data-parallel-address <pod-ip> --data-parallel-rpc-port 29550 \
  --enable-expert-parallel --all2all-backend mori_high_throughput \
  --block-size 16 --kv-cache-dtype auto --gpu-memory-utilization 0.6 --kv-cache-memory-bytes 80000000000 \
  --max-model-len 560000 --max-num-batched-tokens 16384 --max-num-seqs 64 --enable-prefix-caching \
  --trust-remote-code --distributed-timeout-seconds 7200 --skip-mm-profiling --shutdown-timeout 45 \
  --tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice --chat-template-content-format string \
  --disable-access-log-for-endpoints=/health,/metrics,/v1/models \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --enforce-eager \
  --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_connector_extra_config":{"http_port":"8000","handshake_port":"6301","notify_port":"61005","read_mode":"true","defer_timeout":"1800","recv_abort_timeout":"600","qp_per_transfer":"2","num_workers":"4"}}'
```

**P/D decode:** same as prefill except:
- `--port 8200 --data-parallel-rpc-port 29560 --all2all-backend mori_low_latency`
- `--max-num-batched-tokens 256` (and, on 2P1D, `--kv-cache-memory-bytes 55000000000`)
- the decode `--compilation-config` and capture sizes above instead of `--enforce-eager`
- `"kv_role":"kv_consumer"` and `"http_port":"8200"`

On llm-d, the routing sidecar listens on 8000 in front of decode.

**Colocated expert parallel (per replica, 4 GPUs):**

```bash
vllm serve <model-dir> --served-model-name amd/GLM-5.3-Quark-MXFP4-AttnFP8 --port 8000 \
  --data-parallel-multi-port-external-lb --data-parallel-supervisor-port 8008 \
  --tensor-parallel-size 1 --data-parallel-size 4 --data-parallel-size-local 4 \
  --enable-expert-parallel --all2all-backend mori \
  --block-size 16 --kv-cache-dtype auto --kv-cache-memory-bytes 100000000000 \
  --max-model-len 560000 --max-num-batched-tokens 2048 --max-num-seqs 64 --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+quant_fp8"]}' \
  --cudagraph-capture-sizes 4 8 16 32 48 64 96 128 160 192 224 256 --trust-remote-code
```

Each DP rank is its own endpoint (ports 8000-8003). The 1,024-token-chunk variant changes only
`--max-num-batched-tokens 1024`.

**Colocated tensor parallel (per replica, 4 GPUs):** `--tensor-parallel-size 4
--data-parallel-size 1 --kv-cache-memory-bytes 149000000000`, with the other flags as for the
expert-parallel replica minus the expert-parallel, data-parallel-LB and all2all flags. The tested
TP controls ran the same image without patches 0006 and 0007, which only affect MoRI-EP and the P/D
READ path respectively, so neither is active in a TP-only layout.

## llm-d routing

The `InferencePool` spans both roles on port 8000. Prefill vLLM serves it directly; on decode
pods it is the routing sidecar. Envoy calls the Endpoint Picker (EPP) through `ext_proc`.

**Images (tested):**
- EPP `ghcr.io/llm-d/llm-d-router-endpoint-picker@sha256:fadd5d9ea21a6c9d0a1515b93f0f442a8a7d73f02f394e6bc0c5bf9accc54b9c`
- Sidecar `ghcr.io/llm-d/llm-d-router-disagg-sidecar@sha256:ed2f80ed776720fa70331105cf67be32e01344d241f329ba9d83096a1dab49d5`
- Envoy `envoyproxy/envoy:distroless-v1.40.2@sha256:09ca6f2140a61ad8df11779a4a30d52b48433d444466328494f9188c9c35eded`

**Sidecar flags:** `--port=8000 --model-server-port=8200 --kv-connector=nixlv2` (the sidecar's
generic P/D protocol) `--secure-proxy=false --metrics-port=9095`. Leave out
`--moriio-write-mode` (READ) and `--moriio-parallel-dispatch`: READ needs prefill's block IDs
before decode can pull, so dispatch is serial.

**EPP flags:** `--allow-experimental-plugins=true --health-checking=true`.

**EPP P/D configuration:**

```yaml
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: disaggregatedset-rollout-screener
  parameters:
    scope: {labelSelector: "disaggregatedset.x-k8s.io/name=<deployment>"}
    revisionGating: {mode: max-role, requiredRoles: [prefill, decode]}
- type: prefix-based-pd-decider
  parameters: {nonCachedTokens: 2048}
- type: disagg-profile-handler
  parameters: {deciders: {prefill: prefix-based-pd-decider}}
- type: prefill-filter
- type: decode-filter
- type: approx-prefix-cache-producer
  parameters: {maxPrefixTokensToMatch: 131072, autoTune: false, blockSizeTokens: 64, lruCapacityPerServer: 106484}
- type: inflight-load-producer
- type: prefix-cache-affinity-filter
  parameters: {peakPrefillThroughput: 24000}
- type: token-load-scorer
- type: active-request-scorer
- type: max-score-picker
schedulingProfiles:
- name: prefill
  plugins: [{pluginRef: prefill-filter}, {pluginRef: prefix-cache-affinity-filter}, {pluginRef: token-load-scorer}, {pluginRef: max-score-picker}]
- name: decode
  plugins: [{pluginRef: decode-filter}, {pluginRef: active-request-scorer}, {pluginRef: max-score-picker}]
```

- **P/D decision:** disaggregate when the EPP's prefix estimate leaves at least 2,048 input
  tokens uncached. The estimate matches at most 131,072 tokens. At these context lengths every
  measured request was disaggregated.
- **Accuracy runs:** swap in `always-disagg-pd-decider`, so every request exercises KV transfer.
- **Prefix-index units:** `blockSizeTokens=64`. `lruCapacityPerServer` counts router blocks per
  endpoint, not vLLM blocks or tokens: engine KV tokens / 64, times 8 for a P/D pod (one endpoint
  per DP8 pod).
- **Endpoint headers:** the EPP returns the decode endpoint in `x-gateway-destination-endpoint`
  and the prefill endpoint in `x-llm-d-prefiller-host-port`. Envoy routes to the decode sidecar,
  which sends the one-token prefill leg, forwards `kv_transfer_params` to decode and streams the
  decode response.
- **Health:** require Envoy's `ext_proc` membership to be healthy before any gate or benchmark.
  Prefer `failure_mode_allow: false`; otherwise poll membership and refuse traffic while it is
  unhealthy. The tested health-check intervals were 5 s (`no_traffic_interval`,
  `unhealthy_interval`).

**Session rank pinning (prefix locality).** Keeping every turn of a session on one DP rank, on both
roles, keeps its retained prefix where the next turn needs it. MoRI-IO does not need matching
ranks: decode reads from the prefill rank named in `kv_transfer_params`. In the tested deployment
an Envoy Lua filter, placed before `ext_proc`, sets `x-data-parallel-rank`:

```lua
function envoy_on_request(h)
  local hd = h:headers()
  if hd:get("x-data-parallel-rank") ~= nil then return end
  local s = hd:get("x-session-id") or hd:get("x-correlation-id")
  if s == nil then
    -- no session: one random rank, used by both legs
    hd:add("x-data-parallel-rank", tostring(math.random(0, 7)))
    return
  end
  local x = 0
  for i = 1, #s do x = (x * 31 + string.byte(s, i)) % 2147483647 end
  hd:add("x-data-parallel-rank", tostring(x % 8))
end
```

`benchmark_multiturn_retained.py --sticky-mod 8` balances session IDs over the eight ranks with this
same hash. Leave the sidecar's `--moriio-dp-size` unset: it replaces the prefill rank with a
per-request hash, which breaks per-session prefix locality on prefill.

**Colocated controls:** expose each DP rank (expert parallel) or replica (tensor parallel) as an
endpoint, and use prefix affinity plus in-flight and token-load scoring instead of a session pin.

## Slurm launcher

`MODEL_NAME=GLM-5.3-Quark-MXFP4-AttnFP8 CONNECTOR=moriio WIDE_EP=1` (`RUN_MORI=1`) with `xP=1 yD=1`
emits the 1P1D flags above from `models.yaml`; `tests/multiturn_assert.sh` checks them. Cards
(`models.json`, all `-N 2 -n 2`):

| Card `pyt_vllm_disagg_mori_glm-5.3-quark-mxfp4-attnfp8_…` | Runs |
|---|---|
| `mt-r128-10`, `mt-r128-30`, `mt-r128-70`, `mt-h256-30`, `mt-h256-70`, `mt-m512-30` | warm multi-turn presets at the tested 1P1D concurrency |
| `mt-m512-30-c4`, `mt-m512-30-c2` | 512k, +30k at 4 and 2 concurrent sessions |
| `cold-512k` | cold 512k prompt, 1P1D, concurrency 4, 12 prompts |
| `gates` | KV-transfer probe, GSM8K, NIAH, multi-turn needle |

Differences from the tested llm-d deployment. None of these has been run on GPUs yet:
- The proxy is vllm-router or the MoRIIO toy proxy. READ through vllm-router is unverified; the
  toy proxy supports READ with serial dispatch.
- Prefill runs `cudagraph_mode NONE` + `+quant_fp8` instead of `--enforce-eager` (the connector
  never emits bare `--enforce-eager`).
- `xP=2` builds one DP16/EP16 prefill instance across two nodes (inter-node MoRI-EP), not two
  independent DP8 instances, so the tested 2P1D layout is not reproduced; there is no 2P1D card.
- `connectors/moriio.env` is forwarded at container start and wins over `models.yaml`, so the
  following must be set at submit time:
  - `MORI_GPU_ARCHS=gfx950` (the cards set it);
  - your fabric's `MORI_RDMA_TC`, `MORI_RDMA_SL` and `MORI_IB_GID_INDEX`;
  - `MORI_IB_PATH_MTU`.
  The tested deployment also ran with the image default `HSA_ENABLE_IPC_MODE_LEGACY=1`, where
  `moriio.env` sets 0.

## Benchmarks

Every request generates exactly 1,000 output tokens. Warm sessions use Project Gutenberg #1184:
fetched at run time, SHA256-checked, never vendored, and repeated cyclically to reach the longest
contexts.

### Warm multi-turn

Each session runs one warm-up turn, then two measured turns, for six rounds. Session IDs are
balanced over the eight ranks.

| Preset | Cached (avg) | New per turn | ISL of the measured turns | 1P1D sessions | Colocated sessions per node | TTFT target |
|---|---:|---:|---|---:|---:|---:|
| `R128-10` | 128k | 10k | 132.5k, 143.5k | 32 | 16 | 12.8 s |
| `R128-30` | 128k | 30k | 142.5k, 173.5k | 32 | 16 | 12.8 s |
| `R128-70` | 128k | 70k | 162.5k, 233.5k | 16 | 8 | 12.8 s |
| `H256-30` (headline) | 256k | 30k | 270.5k, 301.5k | 16 | 8 | 25.6 s |
| `H256-70` | 256k | 70k | 290.5k, 361.5k | 16 | 8 | 25.6 s |
| `M512-30` | 512k | 30k | 526.5k, 557.5k | 8 (also 4 and 2) | 4 (also 2 and 1) | 51.2 s |

The 4-session and 2-session 512k points ran 8 and 12 rounds. In each round, turn 1 is the same as
in earlier rounds, so only round 1 prefills it cold.

```bash
# slurm card
madengine run --tags pyt_vllm_disagg_mori_glm-5.3-quark-mxfp4-attnfp8_mt-h256-30
# standalone against an llm-d gateway (1P1D: 16 GPUs)
MT_PRESET=H256-30 MT_BASE_URL=http://<gateway> SERVED_MODEL_NAME=amd/GLM-5.3-Quark-MXFP4-AttnFP8 \
TOKENIZER=<model-dir> MT_GPUS=16 METRICS_ENDPOINTS="prefill=<p-pod>:8000 decode=<d-pod>:8200" \
MT_RUN_DIR=<out> bash scripts/vllm_dissag/benchmark_multiturn_retained.sh
# colocated control, one node (repeat on a second node; do not double one measurement)
MT_PRESET=H256-30 BENCHMARK_CON=8 MT_GPUS=8 MT_LAYOUT=coloc MT_BASE_URL=http://<coloc-gateway> \
METRICS_ENDPOINTS="coloc=<node>:8000 coloc=<node>:8001 ..." ... bash scripts/vllm_dissag/benchmark_multiturn_retained.sh
```

### Cold prompts

Cold prompts use exact-length random tokens (`vllm bench serve --dataset-name random
--random-range-ratio 0 --random-prefix-len 0 --ignore-eos`).

| Prompt | Layout | Concurrency | Prompts | TTFT target |
|---|---|---:|---:|---:|
| 256k | 2P1D | 12 | 36 | 25.6 s |
| 256k | colocated, per node | 4 | 12 | 25.6 s |
| 512k | 1P1D | 4 | 12 | 51.2 s |
| 512k | 2P1D | 6 | 18 | 51.2 s |
| 512k | colocated, per node | 2 | 6 | 51.2 s |

Through `benchmark_long_context.sh`, for example:
`BENCHMARK_COMBINATIONS=512000/1000 BENCHMARK_CON=4 NUM_PROMPTS=12 WARMUPS=8 LC_DETAILED=1`
(card `cold-512k`).

### Metrics

`multiturn_metrics.py` reduces each run; `parse_to_csv.py --result-json` writes the perf.csv rows.

- **Throughput:** output tok/s per GPU, counting every GPU of every role (16 for 1P1D, 24 for
  2P1D, 8 per colocated node). Tokens are counted over the window in which all sessions are
  active.
- **Engine ITL:** decode-step gaps from the `vllm:inter_token_latency_seconds` histogram of the
  decode role (or the colocated engines), snapshotted around each round over rounds 2-6.
  Percentiles are interpolated within buckets (edges at 50, 75, 100, 150, 200, 300 and 400 ms in
  this range). The 150 ms verdict needs no interpolation, because 150 ms is a bucket edge.
- **Client ITL:** exact per-chunk gaps from the streamed responses, plus TTFT and E2E percentiles.
- **Goodput** at an ITL target T (150, 250, 500 ms; 100 ms also reported): tok/s/GPU when engine
  ITL p99 <= T and client TTFT p90 meets the TTFT target above, otherwise 0. The TTFT target is
  `max(10 s, retained context / 10k tokens/s)`; cold prompts use the prompt length.
- **Also reported:** prefix-cache hit rate, the prompt-token source split (local compute, local
  cache, external KV transfer), MTP acceptance, preemptions, and `osl_exact_frac` (reject the run
  unless it is 1.0).

## Accuracy and KV-transfer gates

`benchmark_gates.sh` (card `gates`, or standalone with `GATES_BASE_URL` and `METRICS_ENDPOINTS`)
runs the four gates in the tested order. Every request must be disaggregated: on llm-d use
`always-disagg-pd-decider`; the slurm proxies always disaggregate.

| Gate | How | Pass |
|---|---|---|
| Every-rank KV-transfer probe | `benchmark_read_probe.py`: 2 rounds x 16 session IDs balanced over 8 ranks, `/metrics` deltas on every engine | all 200; prefill 1 generated token per request; decode 1 locally computed prompt token per request, the rest by external KV transfer; every decode engine served. Also check, outside the script, at least 20 GiB free GPU memory per rank after the first transfer |
| GSM8K | lm-eval 0.4.13 `local-completions`, gsm8k, 5-shot, all 1,319, `num_concurrent=64`, `max_gen_toks=512`; `exact_match,strict-match` | reference 91.28% |
| NIAH | `benchmark_niah_corpus.py single`: 64k / 256k / 512k x depth 0 / 0.5 / 1 x 2 needles, 6 in flight | reference 18/18 |
| Multi-turn needle | `benchmark_niah_corpus.py multi` at 246k and 490k: 4 sessions, 3 needles, 3 turns, +10k new tokens per turn | reference 7/8 and 8/8 |

Also check the MTP counters on decode: acceptance rate 0.45-0.60 and accepted length 2.35-2.75,
with no degenerate (repeating) output.

## Reference results

> Reference results on the tested image (`8673602b` + the patches above, built with vLLM's own
> bake recipe) on llm-d, November 2026. They are functional-enablement results; performance
> tuning is ongoing, and performance on your hardware, software and workload may vary. An image
> built from this Dockerfile has not been measured yet.

| Workload | P/D tok/s/GPU | Colocated control tok/s/GPU | P/D TTFT p90 | P/D engine ITL p99 / p99.9 | Verdict at 150 ms ITL + TTFT target |
|---|---:|---:|---:|---:|---|
| 128k, +10k | 70.4 | 92.5 (TP) | 8.9 s | 311 / 599 ms | colocated ahead |
| 128k, +30k | 55.0 | 60.7 (EP, one node) | 19.9 s | 129 / 180 ms | colocated ahead; P/D TTFT above 12.8 s |
| 128k, +70k | 23.9 | 29.6 (EP) | 24.0 s | 50 / 287 ms | colocated ahead; P/D TTFT above 12.8 s |
| **256k, +30k** | 31.8 | 37.6 (EP) | 14.0-14.1 s | 71-73 / 145-149 ms | **only P/D meets 150 ms** |
| 256k, +70k | 21.1 | 29.0 (EP) | 27.6 s | 65 / 287 ms | colocated ahead; P/D TTFT above 25.6 s |
| 512k, +30k | 16.6 | 18.1 (EP) | 12.3 s | 50 / 185 ms | both meet every target; colocated ahead |
| Cold 256k, 2P1D | 5.5 | 6.7 (TP, one node) | 91.8 s | 50 / 1,516 ms | neither meets 25.6 s TTFT |
| Cold 512k, 1P1D | 1.61 | 2.43 (EP) | 172.4 s | 50 / 50 ms | neither meets 51.2 s TTFT |
| Cold 512k, 2P1D | 1.65 | 2.44 (EP, three nodes) | 146.7 s | 50 / 1,666 ms | neither meets 51.2 s TTFT |

- **Controls:** colocated controls used 2,048-token chunks, measured on two independent nodes
  (three for 2P1D) unless noted.
- **1,024-token chunks** (colocated EP, one node): 256k +30k gave ITL p99 190 ms, TTFT p90 7.2 s,
  37.7 tok/s/GPU; 256k +70k gave 213 ms, 16.0 s, 28.2 tok/s/GPU. Both meet 250 ms, not 150 ms.
- **512k, +30k by concurrent sessions** (P/D vs colocated tok/s/GPU): 2: 4.54 / 5.11, 4: 8.07 /
  10.10, 8: 16.6 / 18.1.
- **Accuracy**, on the deployment that passed: GSM8K 91.28%, NIAH 18/18, multi-turn needle 7/8
  (~246k) and 8/8 (~490k), KV-transfer probe 32/32 with at least 22 GiB free after registration.
  NIAH across three deployments was 51/54.
- **KV transfer:** decode NIC bytes matched new tokens x 93,912 on the warm +30k/+70k workloads
  and on the cold 256k prompt.

## Known caveats

- **`VLLM_MORI_RECV_ROW_ALIGN=4096` is required with patch 0004.** At the default alignment, the
  bounded receive rows select AITER's small-M fused quant+sort path, which faulted on DEP8 P/D.
  4096 avoids that path; the kernel fault itself is not fixed.
- **RDMA registration of the KV cache reduces free GPU memory as reported by the driver** by about
  the KV size, without a matching allocation. Use explicit KV bytes (`--kv-cache-memory-bytes`)
  and keep at least 20 GiB free per rank after the first transfer.
- **The session rank pin is for prefix locality, not correctness.** MoRI-IO reads from whichever
  prefill rank computed the KV. READ with parallel dispatch is not supported.
- **NIAH varied between deployments of the same software** (18/18, 17/18, 16/18). The misses
  were on requests with no decode prefix hit, which patch 0007 does not touch; the cause is under
  evaluation.
- **512k at 16 concurrent sessions does not fit P/D:** two 512k sessions per GPU need about
  1.117M tokens against an 851,856-token KV budget.
- **One colocated expert-parallel 512k run stopped once** with an out-of-resources error on one
  rank. Its retry, and the same workload on the other nodes and on P/D, completed.
- **The engine ITL p99 of colocated runs is sensitive to the share of decode steps that overlap
  another session's prefill:** p99 reads about 50-90 ms or 310-360 ms depending on whether
  those stalls are under or over 1% of gaps.
