# Kimi-K3 MXFP4 — vLLM 2P/2D Disaggregated EP16 (Kimik3D_DI)

Prefill/decode **disaggregated** serving of **Kimi-K3-MXFP4** across **4 MI300X/MI325X
nodes** on the latest upstream stack, using **MoRI-EP** all2all + the **MoRIIO** KV
connector. Self-contained: one Dockerfile builds the image, one launcher brings up all
4 roles + the router, and probe scripts validate accuracy/perf.

```
   PREFILL POOL (2 nodes, 16 GPU)        DECODE POOL (2 nodes, 16 GPU)
  ┌──────────────────────────┐  MoRIIO  ┌──────────────────────────┐
  │ TP2 x DP8 -> EP16          │  KV +    │ TP2 x DP8 -> EP16          │
  │ mori_high_throughput       │  KDA     │ mori_low_latency           │
  │ cudagraph = NONE (eager)   │ ──RDMA─► │ cudagraph = FULL_AND_PIECE │
  │ kv_producer                │ write+   │ kv_consumer                │
  └──────────────────────────┘  notify   └──────────────────────────┘
                         router (vllm-router, PD-disagg discovery) on prefill master
```

## Component pins (all in `Dockerfile.kimik3_disagg`)

| Component | Pin |
|---|---|
| Base image | `rocm/vllm-dev:ci_base-dedbf6be8b1afa17a6220473b9c8c98242ac1c03` (ROCm 7.2, cp312) |
| vLLM | `raviguptaamd/vllm@206fffe` (branch `v4-disagg-situ-restore-mambafix`) = int4-SiTU base `184535a92` + 4 con>1 accuracy fixes (see **Accuracy** below) |
| MoRI | `624002c897a3` (built from source, `WITH_MORI_BUILD=1`) |
| AITER | `0.1.19` (prebuilt rocm7.2 wheel) + **flydsl 0.2.4** |
| vllm-router | pinned `ROUTER_REF` (built from source) |
| NIXL | disabled (`WITH_NIXL=0`) |

All K3 source fixes are **baked into the vLLM branch** — the recipe carries **zero
runtime patchers**. `/app/versions.txt` in the built image records the resolved shas.

## 1. Build the image (on one node, ~30–60 min)

```bash
cd Kimik3D_DI
docker build -f Dockerfile.kimik3_disagg -t kimik3-wideep-disagg:v4 .
# Reproducible: /app/versions.txt in the image records the resolved base/vLLM/MoRI/AITER shas.
```
Distribute to all 4 nodes (registry push/pull, or `docker save|load`). `load_image.sh`
pulls from a hub tag if you set `HUB_IMAGE`/`DOCKER_USER`/`DOCKER_PAT`.

## 2. Configure the 4-node allocation

Edit the node IPs at the top of `run_2p2d_launch.sh`:
```
PM_NODE/PM_IP  prefill master (+ router/proxy)
PW_NODE/PW_IP  prefill worker
DM_NODE/DM_IP  decode master
DW_NODE/DW_IP  decode worker
```
Set `MODEL_DIR` to the Kimi-K3-MXFP4 weights path (must exist on every node).
Fabric env (Broadcom Thor2 / bnxt shown; override for your NIC): the launcher already
sets GID index, DSCP/SL, and mounts the host RDMA libs — see the `BNXT_MOUNTS` /
`MORI_*` block in `run_2p2d.sh`.

## 3. Launch

```bash
AUTO_ROUTER=1 \
IMAGE=kimik3-wideep-disagg:v4 \
MODEL_DIR=/path/to/Kimi-K3-MXFP4 \
K3_WRITE_READBACK=1 \
bash run_2p2d_launch.sh
```
This deploys scripts + image to all 4 nodes, starts **workers → masters → router** in the
discovery window, and (AUTO_ROUTER=1) auto-starts `vllm-router` once both masters serve
`/v1/models`. Model load is ~15–20 min (WekaFS). Watch for `Application startup complete`
on both masters, then `Add Prefill`/`Add Decode` in the router log.

### Recommended serve settings (defaults in `run_2p2d.sh`)
| Knob | Value | Why |
|---|---|---|
| `K3_WRITE_READBACK` | **1** | RDMA read-after-write barrier — fixes concurrency KV write-race. |
| per-role backend/cudagraph | prefill mori_HT/eager, decode mori_LL/FULL_AND_PIECEWISE | the working contract |
| int4-SiTU MoE | `--quantization-config '{"moe":{"weight":"int4_per_group_32"}}'` | AITER_MXFP4_BF16 fast MoE (else ~8× slower MXFP4 emulation) |
| `MAX_MODEL_LEN` | ≤ 320000 | >320K blows the compile-vs-handshake window |
| `MAX_NUM_SEQS` | 32 | decode batch |
| `MAX_NUM_BATCHED_TOKENS` | ≤ 4096 | larger blows the MoE profiling-compile shape |
| MoRIIO `qp_per_transfer`/`num_workers`/`post_batch_size` | 1 / 1 / -1 (defaults) | non-default values destabilize the WRITE notify path |
| `THINKING_DEFAULT` | `false` for benchmarks | K3 is a reasoning model; off puts the answer in `content` |

## 4. Validate

```bash
export ROUTER_URL=http://<PM_IP>:30000     # default 127.0.0.1:30000

# Accuracy (single-needle) — depth × ctx grid
python3 niah_sweep.py            # 50K/100K/200K/280K x depths 0.1/0.5/0.9

# Accuracy under CONCURRENCY (the con>1 fix) — distinct needle per concurrent request,
# count how many recall their OWN needle. Correct serve = N/N at every concurrency.
python3 niah_con_sweep.py        # con=1,8,16,32 @ 50K  -> expect 57/57 = 100%
python3 niah_con_sweep.py 6000 1,8,16,32   # quick 6K variant

# Throughput / latency — streaming TTFT, e2e, tok/s at concurrency 16 & 32
python3 perf_sweep.py            # input 8K/16K, output 1k
```

## Status & known items (full log in `docs/`)

- **Accuracy: solid, including under concurrency.** Two independent fixes, both baked into
  the pinned vLLM branch:
  1. **KV write-race (RDMA readback, `K3_WRITE_READBACK=1`)** — an RDMA read-after-write
     barrier forces written KV globally visible before the request is released to decode.
  2. **con>1 KDA state-recycle (issue #35219)** — KDA (mamba) recurrent/conv state blocks
     were never zeroed when recycled, so a new request inherited a finished request's
     finite-but-wrong state → sustained con=32 recall fell to ~65%. `needs_kv_cache_zeroing`
     is True for mamba but two `isinstance(AttentionSpec)` gates excluded `MambaSpec`. The
     branch adds a pure-torch zero-on-recycle channel for mamba blocks (+ a de-alias clone
     of the KDA decode state-index view, a MoRI-EP dispatch trim, and a MoRIIO all-group
     KV-completion gate). Runs eager, outside cudagraph capture; int4-SiTU + decode
     `FULL_AND_PIECEWISE` intact.

  **Validated (MI325X 2P2D EP16):** NIAH @50K con=1/8/16/32 = **57/57 = 100%** (via
  `niah_con_sweep.py`); con=32 @6K **9 consecutive runs @100%**; con=1 permanence after
  sustained con=32 hammering = **12/12 = 100%** (no residual poison). Pre-fix con=32 swung
  42–83% (~65% avg).

- **Perf — open:** a ~150 s per-decode-wave latency **floor** (amortizes across
  concurrency: con16 ≈ con1 wall). Prime suspect: MoRI `624002c8` InterNodeV1LL decode
  all2all warmup. See `docs/OPT_ROADMAP.md` (all2all A/B, rocprof, MoRI bisect; plus
  prefill/decode context-parallelism and chunked-prefill tuning).

- **Known-separate (not this fix): 100K+ MoRIIO transfer stall.** Very-long-context (≥100K)
  requests can stall on a MoRIIO `BatchRead()` argument mismatch in the KV transfer path
  (`moriio_engine.py`); the serve recovers for normal-length contexts afterward. Tracked
  separately from the accuracy fixes above (50K exercises the same con=32 recycle path and
  passes 100%).

- **Cosmetic:** kimi_k3 reasoning parser can leak `<|open|>response<|sep|>` into `content`
  on `thinking:false` (needle still recalled; use `max_tokens>=256`).

`docs/PERF_DIAGNOSIS.md` — the full root-cause investigation.
`docs/OPT_ROADMAP.md` — pending optimizations, prioritized.
