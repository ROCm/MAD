# GLM-5.2 on MI355X (gfx950) + AMD AI NIC — 1P/1D EP8

Recipe notes for `GLM-5.2-FP8` and `GLM-5.2-MXFP4` in `models.yaml`, disaggregated
prefill/decode over MoRI-IO. Sibling of the GLM-5.1 wideEP recipe; this file records only
what is specific to GLM-5.2 and to gfx950 + ionic NICs.

Measured on AAC, 2 nodes x 8 x MI355X, ROCm 7.2.3. The §2 sweep ran twice: first on
gpu-44/45, then on gpu-41/45. Both node pairs are quoted there.

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

### 1a. Getting the weights

On AAC these are already staged at `$MODEL_DIR` and `MODEL_DIR/$MODEL_NAME` resolves
without any download — do NOT re-pull them there. This section is for a fresh cluster.

| `MODEL_NAME` | HuggingFace repo | on disk | notes |
|---|---|---|---|
| `GLM-5.2-FP8` | [`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8) | ~704G, 141 shards | vendor FP8; the production path |
| `GLM-5.2-MXFP4` | [`amd/GLM-5.2-MXFP4`](https://huggingface.co/amd/GLM-5.2-MXFP4) | ~408G, 282 shards | Quark MXFP4 of `zai-org/GLM-5.2`; MI355X-only (gfx950 has the native MXFP4 path, gfx942 does not) |

```bash
# directory name MUST equal $MODEL_NAME -- run_xPyD_models.slurm resolves $MODEL_DIR/$MODEL_NAME
hf download zai-org/GLM-5.2-FP8   --local-dir "$MODEL_DIR/GLM-5.2-FP8"
hf download amd/GLM-5.2-MXFP4     --local-dir "$MODEL_DIR/GLM-5.2-MXFP4"
```

Both are MIT-licensed and ungated, so no `HF_TOKEN` is required. The shard counts above
are the integrity check — a short download reads on this stack as a missing-weights load
error, not as a clean failure. `amd/GLM-5.2-MXFP4` quantizes MoE weights only; the KV
cache stays fp8 (see the `GLM-5.2-MXFP4` entry in `models.yaml`), so do not set
`kv-cache-dtype` to fp4.

---

## 2. Measured

FP8, ISL/OSL 28,672/1,024, EP8, 1P/1D. Two independent runs on different node
pairs (gpu-44/45, then gpu-41/45), 0 failed requests in either:

| concurrency | TPOT (ms) | TPOT rerun (ms) | TTFT mean (s) | total tok/s |
|---|---|---|---|---|
| 16 | 33.7 | 33.2 | 11.6 | 9,978 |
| 32 | 36.4 | 34.7 | 16.3 | 16,559 |
| 64 | 40.2 | 38.6 | 27.5 | 24,298 |

TPOT target is **60 ms avg** (the figure being tracked for AIMODELS-1198). The
benchmark harness still encodes `SLO_TPOT_MS=50` as its goodput threshold, which is
the tighter of the two — so a cell can miss goodput while meeting the target. Read
the raw `mean_tpot_ms`, not just the goodput count, and reconcile the two numbers
before quoting either upstream.

Against 60 ms: **met with margin at all three points, and reproducible across node
pairs.** TTFT is the axis that misses (§5); it is drain time under concurrency, so
it responds to xP and not to decode tuning.

Prefill throughput measures **6,135–8,359 tok/s/rank** against a 34,000 tok/s/rank target,
i.e. **~4–5x short**, and that is the binding constraint. See §5.

`GLM-5.2-MXFP4` **now BOOTS and serves on gfx950** (2026-08-27, Crusoe MI355X). An earlier
attempt hit Triton **Code-209** ("no kernel image for gfx950") in the aiter MXFP4 quant
kernel; that is **resolved on the current aiter/vLLM/flydsl stack** — the MXFP4 MoE kernels
compile for gfx950. Verified standalone TP8:

```bash
vllm serve $MODEL_DIR/GLM-5.2-MXFP4 -tp 8 --quantization quark --trust-remote-code \
    --max-model-len 8192 --gpu-memory-utilization 0.85
```

`Application startup complete`, 0 kernel errors; smoke test returns coherent output
("The capital of France is" -> "Paris. Bordering countries are Belgium, Luxembourg,
Germany, ..."). Model load **51.7 GiB/rank** (vs FP8 ~89 GiB — the ~42% smaller Quark
MoE-weights-only) so KV headroom is larger (175.8 GiB avail, 2.03M tokens, 248x concurrency
at 8K). Note `--quantization quark` (NOT fp8); KV stays fp8 (no `kv-cache-scheme` in the
config — do not set `kv-cache-dtype` to fp4). Disaggregated 1P/1D MXFP4 and a perf sweep are
the remaining bring-up; the "never booted / Code-209 blocker" status is now **superseded**.

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

### 3a. The util values actually in force may not be these — check before quoting

`models.yaml` (committed) is not always what a given run used. Long-context work on AAC
has been carried on a cluster-local `models.tenant.yaml` overlay, and it diverges:

| role | `models.yaml` | tenant overlay (job 5968) |
|---|---|---|
| prefill | `--gpu-memory-utilization 0.72` | `0.80 --max-model-len 270336` |
| decode  | (global 0.80) | `0.50 --max-model-len 270336` |

Two things follow, both load-bearing:

1. **Prefill at 0.80 did NOT reproduce the OOM above**, even with a *larger*
   `--max-model-len`. Either something else in the overlay offsets the 3.50 GiB, or the
   0.72 finding is narrower than stated. Do not delete the 0.72 rationale on the strength
   of one clean run — but do not treat 0.80 as known-fatal either. Unresolved.
2. **Decode at 0.50 leaves 19.06 GiB of KV per GPU** — `GPU KV cache size: 429,124 tokens`,
   `Maximum concurrency for 270,336 tokens per request: 1.59x`. At 1.6 full-window requests
   per rank, a long-tail workload queues rather than runs. This is the direct cause of the
   80K/256K TTFT blowup in §5a. The §2 sweep at 28,672 ISL is short enough not to feel it;
   the 80K row is not.

Anything measured under the overlay is a property of the overlay. Record which config a
number came from, or the number is not reportable.

---

### 3b. Accuracy: NIaH to 256K, and a reproducibility caveat

Ladder 8,192 / 32,768 / 65,536 / 131,072 / 262,144, three seeds per rung, warmup on,
`NIAH_MIN_SCORE=8.0`. Run **twice on the same job (5968)** across an engine restart:

| tokens | run 1 | run 2 |
|---|---|---|
| 8,191 | 9.7 | 10.0 |
| 32,767 | 8.3 | 8.7 |
| 65,518 | 9.0 | 8.7 |
| 131,015 | 9.7 | 10.0 |
| 262,139 | **7.3 → FAIL** | **9.3 → PASS** |

**The stack is not reproducible across engine boots at temperature 0.** The seed drives
filler text and needle offsets through a plain `random.Random(seed)`, so the prompt bytes
are byte-identical between the two runs — and 32,767 seed-0 still scored 9/10 then 10/10.
This is engine-side, not a harness artifact. Suspects, none confirmed: chunked-prefill
splitting differently under different batch composition; MoE expert routing varying with
batch; FP8 accumulation order changing with the CUDA-graph capture bucket.

**How to read a NIaH result here:** ~8.5–10 per rung with **±1 needle of run-to-run
jitter**, and **no length trend through 262K**. Both runs agree on that. A single run at
7.3 is not evidence that 262K is broken, and a single run at 10.0 is not evidence that it
is fixed. Score a rung only against repeated boots.

Prerequisite: `--max-model-len` must admit 262,144 + `NIAH_MAXTOK`. The tenant overlay's
270,336 does; the earlier 131,072 turned every rung above it into an HTTP 400 that reads
exactly like broken retrieval.

**Timeout model is wrong (harmless, but fix it).** `benchmark_niah_256k.sh:76-81` sizes
timeouts on a quadratic prefill model. Measured 262K TTFT is ~50 s against the 401 s that
model predicts; doubling ratios are 1.92 / 2.04 / 2.14, i.e. exponent ~1.05 — linear, as
§5 predicts for DSA. The whole ladder runs in ~8 min against a ~55 min budget, so the
wrapper over-budgets by ~8x.

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

---

### 5a. First 80K / 256K data point — capacity-limited, not model-limited

Job 5968, `benchmark_avg_80K_ten.sh`, c16 iteration 1 (the `CELL_WARMUP` rehearsal,
discarded by `slo_report.py --min-iter 2`; recorded here because it is the first
observation of this row at all). Lognormal ISL, realised mean 79,201, p99 262,137.

| metric | mean | p50 | p95 | p99 | SLO |
|---|---|---|---|---|---|
| TTFT (ms) | 49,108 | 27,191 | 174,584 | 181,822 | 7,000 |
| TPOT (ms) | 57.6 | **36.5** | 188.6 | 189.3 | 50 (target 60) |

64/64 completed, 0 failed, `request_goodput 0.0`, total 10,312 tok/s.

**Read this as a capacity result, not a model result.** Decode was running at
`--gpu-memory-utilization 0.50` (§3a) = 19.06 GiB KV/GPU = **1.59 full-window requests per
rank**. c16 across 8 DP ranks is 2 requests/rank nominal, and the lognormal tail puts two
long prompts on the same rank routinely. The p95/p99 TTFT of ~175–182 s is queueing behind
a full KV pool; the p95 TPOT of 189 ms is the same event seen from the decode side. Median
TPOT of 36.5 ms — which is what a request sees once it is actually resident — matches the
§2 sweep exactly, and is inside the 60 ms target.

`benchmark_avg_80K_ten.sh:42-45` already drops c64 on this arithmetic ("214 GiB of KV
resident at once … fits only at decode util 0.80"); c16/c32 were sized against 0.80 as
well, so the same objection applies to them at 0.50.

**Before this row is reported, decode util has to be resolved.** Either re-run at 0.80 and
quote that, or quote 0.50 explicitly as a capacity-constrained configuration. What is not
defensible is presenting a 0.50 number as the model's 80K behaviour. Note also that §5's
fitted single-request TTFT at 80K is ~13.7 s — so even with the KV pool fixed, this row
misses the 7 s target on prefill alone. The two effects are separate and both real.

### 5b. The other three cells did not fail on capacity — MoRI-IO lost KV transfers

The row above is the ONLY one of four cells that produced a result. This has to be stated
next to it, because "1 of 4 cells completed" changes what the number means.

| cell | config | tqdm reached | wall | outcome |
|---|---|---|---|---|
| 1 | c16 iter1 | 64/64 | 535 s | result above |
| 2 | c16 iter2 | 56/64 @ 8:47 | 2,709 s | `[STALL]`, no JSON |
| 3 | c32 iter1 | 99/128 @ 10:40 | 4,322 s | `[STALL]`, no JSON |
| 4 | c32 iter2 | 99/128 @ 10:31 | — | allocation expired first |

`[STALL]` is `benchmark_customer_slo.sh:315` — the `timeout $tmo` wrapper returning 124.
It is the harness giving up, so a stalled cell writes NOTHING: no result block, no JSON.
**A missing JSON is therefore not "still running", and the tqdm bar is not a liveness
signal** — it is the last line the client managed to print before it wedged, and it stays
on screen indefinitely. Judge liveness from `_stalls.log` and from whether the file is
still GROWING (`stat -c %s` twice, seconds apart), never from the last progress line.

Cell 2 is the diagnostic one: same JSONL, same seed, same concurrency as cell 1. It reached
56/64 in 527 s — tracking cell 1 almost exactly — then emitted nothing for 36 more minutes.
Identical work that completes once and hangs once is not a capacity limit. Capacity makes
requests slow; it does not make them never return.

The engine logs name the mechanism:

```
prefill  ERROR moriio_engine.py:228     Deferred write task ... expired after 60.0s
                                        (remote blocks never arrived), marking done      x66
prefill  WARNING moriio_connector.py:1044  Reaped 1 deferred sends with no finished_sending
                                        notification after 60s. This indicates lost async
                                        KV completion notifications from the KV connector. x45
decode   WARNING moriio_connector.py:473    MoRI-IO unmap MISS: rid=...                    x30
```

Both sides agree and the direction is consistent: prefill finishes a block, hands it to
MoRI-IO, and the completion notification never comes back; decode is later asked to unmap
a request it has no record of. `marking done` is the giveaway — prefill abandons the
transfer and reports success, so the request is neither completed nor failed. The client
counts 0 failures and simply waits forever. That is exactly the observed signature:
`Failed requests: 0` on the cell that finished, and a frozen tqdm on the ones that did not.

Onset is time-correlated, not load-correlated: 8 errors in the 23:00 hour, then 29 and 29
in the two hours after, while c16 (the *lighter* cell) hung and c32 got further. It
degrades with uptime, which points at leaked/exhausted transfer state in the connector
rather than at KV pool pressure.

Consequences for how this row gets re-run:

1. **Re-running at decode util 0.80 will not by itself fix this.** §5a's capacity argument
   stands for the *shape* of cell 1's tail, but capacity is not what killed cells 2–4.
   Expect the re-run to stall too unless the connector issue is addressed.
2. **Restart the engine between cells**, or at least between concurrency points, until the
   leak is understood — the failure accumulates with uptime.
3. **`VLLM_MORIIO_*` env vars are being silently ignored.** Decode logs
   `moriio_common.py:236` three times: `VLLM_MORIIO_QP_PER_TRANSFER`,
   `VLLM_MORIIO_POST_BATCH_SIZE`, `VLLM_MORIIO_NUM_WORKERS` are all deprecated and must now
   be set inside `kv_transfer_config.kv_connector_extra_config`. Any tuning we believe we
   applied through those variables was not applied. Check this before blaming the tuning.

Also present and so far unexplained: `coordinator.py:404` "Received stats for out-of-order
step" x69 on prefill. Possibly benign DP stats reordering; not yet ruled out as related.
