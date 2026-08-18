# GLM-5.2 on MI355X (gfx950) + AMD AI NIC — 1P/1D EP8

Recipe notes for `GLM-5.2-FP8` and `GLM-5.2-MXFP4` in `models.yaml`, disaggregated
prefill/decode over MoRI-IO. Sibling of the GLM-5.1 wideEP recipe; this file records only
what is specific to GLM-5.2 and to gfx950 + ionic NICs.

Measured on AAC, 2 nodes x 8 x MI355X (gpu-44 / gpu-45), ROCm 7.2.3.

---

## 1. Invocation

```bash
export DOCKER_IMAGE_NAME=<your-registry>/vllm-disagg:glm52-gfx950-ionic
export MODEL_DIR=/shared/data/amd_int/models          # holds the HF snapshots
export MODEL_NAME=GLM-5.2-FP8                         # or GLM-5.2-MXFP4
export CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori
export CONNECTOR_ENV_FILE=$PWD/connectors/moriio.env.aac   # AAC/ionic platform env
export xP=1 yD=1
export BENCHMARK_COMBINATIONS=28672/1024
export BENCHMARK_CON="16 32 64"

srun --jobid $SLURM_JOB_ID --overlap -N2 -w "$NODES" --ntasks-per-node=1 \
     bash ./run_xPyD_models.slurm
```

`CONNECTOR_ENV_FILE` is the only cluster-specific knob. On a different cluster, copy
`connectors/moriio.env.aac`, then re-check the three things that are physically per-site:
the RDMA device list, `MORI_SOCKET_IFNAME`, and `FABRIC_SUBNET` (§4).

---

## 2. Measured

FP8, ISL/OSL 28,672/1,024, EP8, 1P/1D. Two independent runs on different node
pairs (gpu-44/45, then gpu-41/45), 0 failed requests in either:

| concurrency | TPOT (ms) | TPOT rerun (ms) | TTFT mean (s) | total tok/s |
|---|---|---|---|---|
| 16 | 33.7 | 33.2 | 11.6 | 9,978 |
| 32 | 36.4 | 34.7 | 16.3 | 16,559 |
| 64 | 40.2 | 38.6 | 27.5 | 24,298 |

TPOT target for AIMODELS-1198 is 50 ms avg — **met with margin at all three
points, and reproducible across node pairs.** TTFT is the axis that misses (§5);
it is drain time under concurrency, so it responds to xP and not to decode tuning.

Prefill throughput measures **6,135–8,359 tok/s/rank** against a 34,000 tok/s/rank target,
i.e. **~4–5x short**, and that is the binding constraint. See §5.

`GLM-5.2-MXFP4` is **config-only and has never been booted.** Its recipe is a copy of the
FP8 one with the two values that provably do not transfer re-derived (§3). Treat its first
run as a bring-up, not a benchmark.

---

## 3. The two values that carry the recipe

**`DECODE_CUDAGRAPH_MODE: FULL_AND_PIECEWISE`** — the single change that fixed TPOT,
104 ms -> 34–40 ms. `PIECEWISE` splits the decode graph on the 3 DSA ops x 78 layers,
~234 launch boundaries per step. That cost is *batch-independent*, so it presents as a
latency floor that does not move when you change concurrency, batch size, or the fabric —
which is exactly why it survived so much tuning before being found.

> **Hard co-requisite:** `use_inductor_graph_partition` must stay ON, and never pass a bare
> `--enforce-eager`. With the partitioner off, boot and warmup both PASS and the first real
> MLA decode dies.

**`prefill.dp: --gpu-memory-utilization 0.72`** — prefill only. The MLA chunked-prefill
workspace is sized from a hardcoded 64k-token clamp
(`determine_chunked_prefill_workspace_size`, `mla_attention.py:1935`), *not* from
`max_num_batched_tokens`. It therefore allocates `65536*64*(192+256)*2` = **3.50 GiB** on
top of the KV pool regardless of how small the scheduler batch is — and lazily, on the
first long prefill, i.e. after boot and warmup have both reported healthy. At 0.80 this
OOM'd in `mla_attention.py:739`.

MXFP4 inherits the same 0.72 clamp. Its weights are ~42% smaller and it runs
`GPU_MEMORY_UTILIZATION 0.90`, so it has *less* free HBM at the moment those 3.50 GiB land,
not more. That value is inferred from the FP8 failure, not measured.

---

## 4. Cluster-specific: what will bite you elsewhere

**`FABRIC_SUBNET`.** The launcher defaults to `10.158.` (the MI300X cluster). AAC is
`10.2.80.`, set in `moriio.env.aac`. Wrong value => `MASTER_ADDR` is advertised as an
address the peer cannot reach => the run hangs at `Waiting for nodes` forever, with no
error. This one key is host-side, so the `.env` loader has an explicit host-export
allow-list for it (`run_xPyD_models.slurm:255-263`); every other key in that file is
container-only.

**Pick the interface by testing TCP, not ICMP.** On AAC the public/default-route NIC
answers ping but has TCP firewalled node-to-node. `/dev/tcp/<peer>/22` is the honest test.

**8 ionic rails**, `rocep{9,25,105,121,137,153,233,249}s0`, with `MORI_IB_GID_INDEX=1`.
Control/bootstrap traffic rides `MORI_SOCKET_IFNAME=enp193s0f1np1` (mgmt); bulk EP and KV
traffic ride the rails.

**Two patchers are opt-in OFF on this image** (`GLM_PERSIST_GATE`, `GLM_DSA_SENTINEL_FIX`)
because both are actively harmful here: a C++ abort at `asm_mla.cu:945`, and a
`hipErrorIllegalAddress` from the sentinel 0 -> -1 change respectively. Both are correct on
older images — the gate is by image, not by model.

**Sharing a node with another tenant: use a `models.yaml` overlay, don't edit the tracked
file.** `MODEL_CONFIG_PREFILL`/`MODEL_CONFIG_DECODE` are composed from `models.yaml` alone
(`vllm_disagg.sh:200-223`), so per-role flags like `--gpu-memory-utilization` and
`--max-model-len` cannot be injected at submit time — only the protect-listed env keys can.
Copy `models.yaml` to e.g. `models.tenant.yaml` beside it, edit the two `dp:` lines, and
point `MODELS_YAML` at the **container-side** path: `$NIXL_REPO_DIR` is bind-mounted at
`/opt/nixl-vllm-cookbook` (`run_xPyD_models.slurm:706`), so
`MODELS_YAML=/opt/nixl-vllm-cookbook/models.tenant.yaml` resolves in-container while the
tracked recipe stays clean. A host path silently fails. Verify from the flags echo, not the
CLI you typed.

**`--gpu-memory-utilization` is a fraction of TOTAL card capacity (287.98 GiB), not of free
memory, and it cannot shrink the weights floor** (~107.8 GiB/rank here). Two different util
values printing an identical `... GiB is allocated by PyTorch` means you are reading the
floor, not a budget. With a co-tenant, vLLM's profiler counts the tenant's bytes as
consumed, so `Available KV cache memory` can go **negative** while the startup check still
demands `free >= util * total` — those two constraints become unsatisfiable and no util
value boots. Wait for the node rather than tuning. Probe every card
(`/sys/class/drm/card*/device/mem_info_vram_{used,total}`; note `card0` does not exist) in
the same breath as the launch — occupancy here moved 300 GB -> 82 GB within minutes, and
per-GPU.

**Budget the first boot for AITER JIT.** `module_gemm_a8w8_blockscale_cktile` takes ~5 min
of `hipcc` while every other worker prints `waiting for baton release`; a killed run leaves
a 0-byte lock with an empty build dir and no compiler alive, which looks identical. Check
`ps -eo etime,comm | grep hipcc` and the lock mtime. Do **not** delete the build dir under
live workers — they race past the lock into `No module named
'module_gemm_a8w8_blockscale_cktile'`. The cache is keyed by hostname
(`~/.cache/vllm_jit/<host>/<image-hash>/aiter_jit/`), so check both nodes; the `.so` is
portable between identical nodes and can just be copied. A peer dying with gloo
`Connection closed by peer` while the other compiles is collateral, not the bug.

---

## 5. What this recipe does **not** fix

Prefill. Profiled at 8,192 tokens (~1,314 ms/step):

| component | share |
|---|---|
| MLA sparse attention (indexer + core + kv gather) | **~82%** |
| all GEMM | 17.8% |
| — of which MoE | **2.9%** |

Two structural facts, each verified two independent ways:

1. **Attention is linear in context, not quadratic.** `index_topk=2048` caps keys per
   query, so the attention core is *flat* at 0.791 ms across L = 7,168 / 14,336 / 28,672 /
   57,344. DSA works as designed — good news for the 256K/1M context targets.
2. **Only the DSA indexer grows** (it scores all L keys to pick 2,048): 1.00 / 1.66 / 3.07
   / 5.78x over that same 8x range. End-to-end this reads as
   `TTFT ~= 200ms + ~160us/token`.

**Levers closed by measurement — do not re-litigate these without new evidence:** MoE/AITER
tuning (2.9% of prefill), indexer sub-chunking (<=2.5%), `max_num_batched_tokens` (<=4%),
RDMA QP/worker count (-1.9%, within +-0.4% noise), FP8BMM (~0.7%), `block_size` 1 -> 64
(<2%).

**Consequence for the SLO.** At 80K ISL the fitted single-request TTFT is **~13.7 s on one
rank, before any concurrency**, against a <7 s target. TTFT at concurrency is drain time —
`458,752 tok / 39,340 tok/s = 11.66 s`, an identity that reproduces the measurement — so
adding prefill nodes (xP) fixes *concurrent* TTFT but does **nothing** for the
single-request number. The only structural fixes are (a) fused/tuned MLA sparse-attention
kernels for gfx950 or (b) intra-request sharding.

**PCP is blocked three ways**, so it is not the near-term answer: `config/parallel.py:528`
is incompatible with DP; `rocm_aiter_mla_sparse.py` has zero PCP plumbing (uniquely among
the attention backends); and `platforms/rocm.py:894` force-sets `cudagraph_mode=PIECEWISE`,
which would silently undo the TPOT fix in §3.
