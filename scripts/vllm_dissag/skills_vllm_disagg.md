# skills_vllm_disagg.md

Hard-won operational knowledge for **vLLM PD-disaggregated WideEP serving on AMD MI300X**
(MoRI-EP all-to-all + MoRI-IO RDMA KV transfer), learned while bringing GLM-5.1-FP8
(MLA + DeepSeek Sparse Attention) onto vLLM v0.27.

Everything below is *measured*, not theorised. Where a belief turned out to be wrong,
the wrong belief is kept alongside the correction — those are the expensive lessons.

---

## 1. Benchmarking methodology (read this first — it invalidated three of my conclusions)

### 1.1 ALWAYS discard the first bench run after a boot
The first real request after startup pays **cold Triton JIT** for the sparse/indexer
kernels (`_indexer_k_quant_and_cache_kernel`, `generate_sparse_seqlen_kernel`,
`_convert_req_index_to_global_index_kernel`). Measured on 1P/1D EP8, identical bench:

| run | TTFT | TPOT |
|---|---|---|
| 1st after boot (cold) | 13,451 ms | 88.7 ms |
| 2nd (warm) | **906 ms** | 88.0 ms |

TTFT moved **14.9x**; TPOT barely moved. A single cold run made me invent (and act on)
a false "scheduler admission" theory twice. Discard it, or warm up explicitly.

### 1.2 Why cold JIT lands on TTFT and not TPOT
With the standard recipe **prefill = eager (`CUDAGraphMode.NONE`)**, decode = PIECEWISE:
- prefill has no graph capture at boot -> its kernels compile lazily on the **first real
  request** -> the whole compile cost is inside TTFT.
- decode captured graphs at boot ("Graph capturing finished") -> already warm -> TPOT is
  correct even on the cold run.
This asymmetry is diagnostic: *cold-JIT symptoms show up in TTFT only*.

### 1.3 The harness warmup is not a warmup
`benchmark_xPyD.sh` warms at `isl=32 osl=32 con=1` — which never exercises a 1024/8192/28672
prefill path, nor the decode cudagraph batch sizes. The first *measured* cell therefore
absorbs residual JIT. Fix applied: a per-shape warmup at the real ISL/OSL before each
shape's cells (writes to a separate `_SHAPEWARMUP.log` so it can't pollute the CSV).

### 1.4 Client timeouts read as server failures
A 50-80s curl timeout against a cold server returns an **empty body**, which looks exactly
like a crash. Use >=300s on the first request. This produced a false "total failure"
verdict during EP32 debugging.

### 1.5 Sanity-check against physics before blaming a kernel
GLM-5.1-FP8 activates ~37.7B params/token. At 5.3 TB/s HBM, fp8:
- full model on one rank: **7.1 ms/step**
- 1/8 of experts per rank (EP8): **0.9 ms/step**

Measured 290-300 ms => ~320x the bound. That immediately rules out "compute" or "bandwidth"
and says *stall / oversized transfer*. Do this arithmetic early; it saves hours.

---

## 2. The big perf trap: `max_num_batched_tokens` sizes the MoRI EP buffer

### 2.1 The chain
```
vllm/model_executor/layers/fused_moe/layer.py:349
    max_num_tokens = max_num_batched_tokens            # 8192 default (SCHEDULER knob)
vllm/model_executor/layers/fused_moe/all2all_utils.py:181
    max_num_tokens_per_dp_rank = moe.max_num_tokens
vllm/distributed/device_communicators/all2all.py  (MoriAll2AllManager)
    max_num_inp_token_per_rank = <that 8192>
```
`max_num_batched_tokens` is a **chunked-prefill scheduler** setting. Using it to size the
EP dispatch/combine buffer means a **decode** instance runs an 8192-token-wide all-to-all
**every step, per layer, x78 layers**, while decoding a handful of tokens.

### 2.2 Signature of this bug
- fixed per-step cost: TPOT identical at concurrency 1, 4, 8, 16
- independent of KV length: 292 ms at isl=128, 297 ms at isl=1024
- batching still scales perfectly (con=1 -> 8 gave 8.3x throughput, TPOT flat)
- orders of magnitude above the bandwidth bound
=> "constant oversized transfer", not compute.

### 2.3 The fix that works today (no code change)
Lower `--max-num-batched-tokens` **on the decode role only** (`models.yaml` `decode.dp:`):

| decode mnbt | TPOT | TTFT (warm) | out tok/s |
|---|---|---|---|
| 8192 (default) | 302.5 ms | 2431 ms | 24.9 |
| **2048** | **88.0 ms** | **906 ms** | **78.8** |

3.4x faster decode, and it also *improved* TTFT and throughput. Prefill keeps 8192 (it
genuinely dispatches wide batches).

### 2.4 What does NOT work (dead ends — do not repeat)
- **`VLLM_MORI_MAX_TOKENS_PER_RANK` alone** (512 or 2048): device assert at boot
  ```
  mori .../dispatch_combine/intranode.hpp:134
    `destTokId < config.MaxNumTokensToRecv() &&
     "Total recv token overflow: increase maxTotalRecvTokens"'
  ```
  because vLLM's **profiling/warmup dummy run deliberately pushes
  `max_num_batched_tokens` (8192) tokens** through the model. The EP buffer must survive
  that even though steady-state decode never needs it.
- **`max_total_recv_tokens` to decouple recv from send: IMPOSSIBLE in current mori.**
  ```
  MaxNumTokensToRecvPerRank():
    if maxTotalRecvTokens > 0:
        perRank = ceil(maxTotalRecvTokens / worldSize)
        return perRank < maxNumInpTokenPerRank ? perRank : maxNumInpTokenPerRank   # min()
    return maxNumInpTokenPerRank
  ```
  It returns **min(perRank, send_width)** — it can only *lower* recv capacity, never raise
  it above the send width. `send=1024, recv=65536` behaves identically to leaving it unset.
  Recv capacity is structurally bounded by send width.

### 2.5 The proper upstream fix (not yet done)
Bound the **profiling/dummy run on a decode instance by `max_num_seqs`** instead of
`max_num_batched_tokens`. Then the EP buffer can be narrow and profiling never exceeds it.
This lives entirely in vLLM. (Alternative: a mori change allowing recv > send.)

### 2.6 MoRI has no tuning knobs (parity gap vs DeepEP)
DeepEP exposes `VLLM_DEEPEP_BUFFER_SIZE_MB`; MoRI hardcodes `warp_num_per_block`,
`block_num`, `rdma_block_num` and inherits its token width from an unrelated scheduler
default. Added `VLLM_MORI_*` knobs for parity (all defaulting to current values).

---

## 3. Accuracy: the DSA sparse-index sentinel landmine

**Never set the invalid/OOB sentinel to `-1`** in
`_convert_req_index_to_global_index_kernel`
(`vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`). It must be `0`.

Why: aiter's `mla_decode_fwd` **dereferences** `paged_kv_indices`, so `-1` becomes
`kv_cache + (-1)*stride` -> page-aligned GPU memory access fault -> worker dies -> gloo
DP all-reduce collapse. `0` is masked out by `paged_kv_indptr`/`last_page_len`.

**Why it hides:** it only fires at **disaggregated long context**. Decode inherits the
prefill's long `seq_lens` -> `generate_sparse_seqlen` widens `paged_kv_indptr` -> `-1`
padding entries land *inside* a live indptr range and get dereferenced. Short prompts and
non-disagg decode never emit an in-range `-1`, so it passes every quick test.
`HIP_LAUNCH_BLOCKING=1` does **not** help (data-dependent OOB, not an async race).

---

## 4. Debugging playbook — isolate the layer before optimising

The ladder that found the perf bug, in order of cheapness:

| Suspect | Test | What it showed |
|---|---|---|
| router / KV transfer | bench **prefill-direct** (`:20005`) vs via router (`:30000`) | 341 ms vs 303 ms TPOT -> router + MoRI-IO exonerated (router *does* add TTFT) |
| JIT warmup | re-run on a 30-min-warm server | identical -> not warmup |
| cudagraph config | grep `cudagraph_mode`, `Graph capturing finished`, GiB captured | config correct |
| KV length | sweep isl 128 / 1024 / 8192 at con=1 | flat -> fixed floor, not KV |
| batching | con=1 vs 8 | TPOT flat, throughput 8.3x -> batching fine |
| all2all backend | force `PREFILL_MORI_BACKEND=mori_low_latency` | still broken -> not an HT-kernel bug |

**Also:** compare against a *known-good* build. Diffing our stack against the passing
v0.25 blog stack showed MoRI, aiter and the recipe were **byte-identical** — only the base
image and vLLM differed. That single comparison exonerated two whole components.

---

## 5. Operational gotchas

### 5.1 Three caches, three different rules
| cache | keyed by | bake into image? |
|---|---|---|
| `aiter_jit` | (aiter commit, gfx arch) | **yes** — topology-independent, and it is the cold-boot long pole |
| `vllm` (torch.compile/inductor), `triton` | model cfg + batch sizes + cudagraph mode + **topology** (EP8/16/32) | **no** — a stale graph is a real trap; wipe when changing shapes |
| `comgr` | ROCm code objects | harmless |

Host persistence: `VLLM_CACHE_PERSIST=1` ->
`/mnt/m2m_nobackup/$USER/vllm_jit_cache/a256<imageid>` -> `/opt/vllm_cache`.
**Keyed by image ID**, so every new image = full cold rebuild.

If baking `aiter_jit` into an image: **scrub `lock_module_*`, `.ninja_log`, `/root/.mori`,
`/tmp/mori_jit_*` first**, or a fresh container waits on a baton nobody holds -> boot hang.
Verify: a fresh container must start with **zero** ninja/hipcc/clang processes.

### 5.2 Measured boot times (8 nodes, model on local NVMe)
| scenario | time |
|---|---|
| cold aiter, 2 nodes | ~25 min |
| cold aiter, 4 nodes (2 cold decode) | ~41 min |
| cold aiter, 8 nodes (4 cold decode) | ~106 min |
| **warm cache, 8 nodes** | **~11 min** |

Changing `max_num_batched_tokens` invalidates the torch.compile cache -> full recompile.

### 5.3 The aiter baton lock looks like a hang but isn't
`[aiter] waiting for baton release at /opt/vllm_cache/aiter_jit/build/lock_<module>` —
one worker compiles, the rest block. Diagnose by counting build procs *inside the
container*: 100-160 = actively building; **0 on all nodes** = genuinely wedged.

### 5.4 Readiness: don't count "Application startup complete"
Only nodes running an API server print it. For 2P/2D and 4P/4D the non-master DP ranks
never will. Judge readiness by: router `All servers healthy` + `Graph capturing finished`
+ `GPU KV cache size` per node.

### 5.5 Verify teardown on EVERY node
`docker rm -f` can leave a container alive on one node; a stale container then collides
with the new run (observed: 2 containers on one node -> prefill shut down mid-boot).
Always re-check `docker ps -q | wc -l == 0` everywhere before relaunching.

### 5.6 Per-role env: `models.yaml env:` applies to BOTH roles
Prefill and decode often need **opposite** values (e.g. EP buffer width). The pattern is
`PREFILL_*` / `DECODE_*` keys in `env:`, split inside `connectors/moriio.sh` (mirrors the
existing `PREFILL_MORI_BACKEND` / `DECODE_MORI_BACKEND`). Verify it landed by reading
`/proc/<vllm-serve-pid>/environ` **inside the container** — the value is exported into the
server process, not the container shell, so `docker exec env` shows nothing.

### 5.7 RDMA fabric
- GID index **3** = RoCEv2 IPv4 (check `show_gids` / `sysfs .../gid_attrs/types`).
- Restrict `MORI_RDMA_DEVICES` / `NCCL_IB_HCA` to the 8 GPU-local NICs; leave the mgmt NICs
  out or QPs try to form over a non-routable fabric -> `ibverbs.cpp:189 Connection timed out`.
- NCCL/GLOO control sockets on `eth0` (mgmt); KV data on the RDMA NICs.
- **Verify the fabric before blaming code**: `ping -I rdma0` matrix, then `ib_write_bw`
  (healthy pair measured 386 Gb/s). A node can be `alloc` in SLURM with a **dead** fabric —
  SLURM does not detect this. One dead node cost a whole 2P/2D campaign.
- `ib_write_bw` needs both endpoints co-alive: run it as a single 2-task srun step, not two
  separate sruns (a backgrounded server dies when its srun returns).

### 5.8 Images
Push to a registry (`docker push`) rather than relying on `docker save | ssh | docker load`
serially — parallel `docker pull` across 7 nodes is far faster and removes the
"image missing on one node" failure that silently stalls the launcher barrier.

---

## 6. Open items (state as of 2026-08-16)

- **4P/4D EP32 silent output corruption.** Cluster boots healthy (memfault=0), but output is
  garbage. MoRI + aiter + recipe are byte-identical to the passing v0.25 stack; only base +
  vLLM differ => **v0.27 regression**. Both `mori_high_throughput` and `mori_low_latency`
  corrupt at EP32 while both are clean at EP8/EP16 => an EP-width (>16 ranks) issue, not a
  kernel-specific one. Next probe: 3P/3D (EP24) to test "any EP>16" vs "exactly 32".
- **Proper upstream fix for §2.5** (bound decode profiling by `max_num_seqs`).
- **Prewarmed image** (§5.1) to kill the cold-boot and first-request-JIT costs.

---

## 7. One-line summary of the two bugs found

1. **Accuracy:** a `-1` sentinel that aiter dereferences -> GPU fault, but only at disagg
   long context. Use `0`.
2. **Perf:** the MoE all-to-all buffer is sized from a chunked-prefill *scheduler* knob, so
   decode moves an 8192-wide buffer every step. Lower `--max-num-batched-tokens` on the
   decode role: **302 ms -> 88 ms TPOT**.
