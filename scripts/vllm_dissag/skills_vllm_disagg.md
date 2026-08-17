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
- GID index is **site-dependent — enumerate it, do not copy it**. `show_gids` /
  `sysfs .../gid_attrs/types`. Index 3 = RoCEv2 IPv4 on the clusters the stock
  `moriio.env` was written for, but **on AAC (Pensando/Ionic) index 3 does NOT exist**
  (`type=none`) and index 0 is link-local `fe80::` (not routable cross-node). AAC must
  run `MORI_IB_GID_INDEX=1`; the stock `GID_INDEX=3` selects a nonexistent entry.
  See `run_niah_glm52.sh:47-49`.
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

---

## 8. Perf tuning: the decode TPOT budget (MI355X gfx950 + Pensando/Ionic AINIC)

Added 2026-08-16 from job 5642 (GLM-5.2-FP8, 1P/1D EP8, AAC mi355-gpu-45/46). This
section is the standing perf-tuning reference: **measure the budget before tuning a
kernel**. Everything here is measured on the real deployment shape, not theorised.

### 8.1 Measure where the step actually goes BEFORE tuning anything

Microbenchmarks on idle MI355X GPUs, in the serving image, at the real EP8 decode
shape (78 layers, decode batch ~32, 28k context):

| component (x78 layers) | ms/step |
|---|---|
| fused MoE (untuned, real EP shape) | 24-38 |
| dense GEMMs (q_proj, kv_a, kv_b, o_proj, indexer_q) | ~9.7 |
| DSA sparse indexer scoring (28k ctx, top-2048) | ~6.4 |
| **compute subtotal** | **~40** |
| **measured TPOT** | **~104** |
| **unaccounted: all2all + per-layer launch overhead** | **~64 (62%)** |

Per-op detail (fused_moe MD=6144 ID=2048 E=32 TOPK=8; x78 layers):

    tokens=  8   0.4873 ms/call -> 38.01 ms      q_proj    6144->16384   0.0398 -> 3.10
    tokens= 16   0.3203 ms/call -> 24.98 ms      kv_a      6144->576     0.0133 -> 1.04
    tokens= 32   0.3039 ms/call -> 23.71 ms      kv_b      512->28672    0.0131 -> 1.02
    tokens= 64   0.4646 ms/call -> 36.24 ms      o_proj    16384->6144   0.0425 -> 3.32
                                                 indexer_q 6144->4096    0.0162 -> 1.26
    DSA indexer tok=32 ctx=28672: 0.0818 ms/call -> 6.38 ms

**Consequence: MoE kernel tuning cannot reach a 60 ms TPOT SLO.** Even a *perfect*
MoE kernel saves <=30 ms; realistic 10-20% gains yield 3-7 ms against a 44 ms deficit.
Do not spend allocation time on AITER MoE tuning until the non-compute term is closed.

Three independent signatures agree the bottleneck is fixed-cost dispatch, not batch work:
- TPOT flat at 103.8 / 104.3 / 104.9 ms across con=16/32/64 (**4x** concurrency).
- `max_num_batched_tokens` curve flat below 2048 (512 -> 87.9 ms, 2048 -> 88.0 ms).
- The ROCm MI300X + **CX7** blog hits the same ~89 ms on completely different NICs.

### 8.2 Read AITER's real fused-MoE lookup key — never guess it from source

AITER logs its own lookup decision. One grep gives the exact runtime key:

    grep -o "\[fused_moe\] using .* for (.*)" decode_NODE1.log | sort -u

Live GLM-5.2-FP8 EP8 decode printed exactly one key, always "default" (= untuned):

    [fused_moe] using 1stage default for
      ('gfx950', 256, 16384, 6144, 2048, 32, 7, Silu, 'torch.bfloat16',
       'torch.float8_e4m3fn', 'torch.float8_e4m3fn', 'QuantType.per_1x128', True, False)
    key order: (gfx, cu_num, token, model_dim, inter_dim, expert, topk, ...)

**EP and TP shard the MoE in opposite dimensions** — this is the trap:
- **EP**: experts shard (256 routed / 8 ranks = **32**); `inter_dim` does **NOT** shard (**2048**).
- **TP**: `inter_dim` shards (2048/8 = **256**); expert count stays whole (256 + 1 fake = **257**).

`a8w8_blockscale_tuned_fmoe_glm5.csv` ships `(6144, 256, 257, topk 9)` — a **TP** table.
Our EP shape `(6144, 2048, 32)` has **zero** tuned rows in *any* shipped table. All
`model_dim=6144` coverage that exists at all:

    a8w8_blockscale_tuned_fmoe_glm5.csv   inter 256        expert 257   topk 9
    minimax_m3_fp4 / mxfp8                inter 384/768    expert 129   topk 5
    tuned_fmoe.csv                        inter 4096       expert 8     topk 2

On a miss the kernel shape genuinely changes: `block_m = 64 if token > 32 else 16`,
whereas the tuned table specifies `block_m=32` at tokens 32/64. Two env handles for
zero-patch differential tests: `AITER_BYPASS_TUNE_CONFIG=1` (force default path) and
`AITER_ONLINE_TUNE=1` (gate the online tuner; only this appends to `untuned_fmoe.csv`).

### 8.3 Upstream bug: `topk -= int(is_ep)` makes every EP row unreachable

`aiter/fused_moe.py:1261` decrements `topk` before the table lookup, but every shipped
table stores the EP row with the fake expert slot in **both** `expert` and `topk`
(kimik2 384/8 + 385/9; qwen3.5 512/10 + 513/11; dsv3 257/9). So the runtime probes
e.g. `(257, 8)` — a row that exists for no model. Proven by AITER's own log line:
`topk=9` -> HIT (`kernelName1=_ZN5aiter49fmoe_bf16_blockscaleFp8_g1u1_novs_silu_1tg_32x256E`);
`topk=8` and `topk=7` -> `"using 1stage default"` (MISS).

**Real bug, but NOT our bottleneck** — our key misses on `expert` and `inter_dim` too,
and per §8.1 MoE is only ~25% of the step. Report it upstream; don't chase it for TPOT.

### 8.4 CUDA graph mode is a first-class TPOT knob (top current candidate)

`PIECEWISE` cuts the graph at every op in `splitting_ops`. For DSA that list contains
**three** hot ops — `vllm::unified_mla_attention_with_output`, `vllm::sparse_attn_indexer`,
`vllm::rocm_aiter_sparse_attn_indexer` — so a 78-layer decode step pays on the order of
**~234 launch boundaries**. That is a fixed per-step cost independent of batch size,
which is exactly the §8.1 signature.

Verified preconditions for full-decode graphs on this stack:

| check | result | where |
|---|---|---|
| `ROCMAiterMLASparseBackend` CG support | `AttentionCGSupport.UNIFORM_BATCH` | `rocm_aiter_mla_sparse.py:364` |
| DSA indexer backend CG support | `AttentionCGSupport.UNIFORM_BATCH` | `mla/indexer.py:462-467` |
| mori-specific CG guard? | **none** (only `deepep_high_throughput` is force-disabled) | `config/compilation.py` |
| host syncs in mori hot path? | **none** — the `.item()`/`.cpu()`/`synchronize()` calls are all in `get_dispatch_src_token_pos`, a debug helper, not `dispatch`/`combine` | `mori/ops/dispatch_combine.py:1380` |

`UNIFORM_BATCH` is precisely the level that permits `FULL_DECODE_ONLY` /
`FULL_AND_PIECEWISE` (vLLM's own v1 default). We were overriding *down* to `PIECEWISE`.
The knob is already plumbed per-role end-to-end:
`models.yaml DECODE_CUDAGRAPH_MODE` -> `run_xPyD_models.slurm:725` -> `connectors/moriio.sh:307`.
No code change needed.

**Simulate the resolution before spending a boot.** Call the real resolver in-image
instead of relaunching and hoping — `set_splitting_ops_for_v1(all2all_backend, dp)` then
`resolve_cudagraph_mode_and_sizes(min_cg_support=..., min_cg_attn_backend=..., max_num_reqs=...)`:

    FULL_DECODE_ONLY     mori_low_latency -> FULL_DECODE_ONLY    decode=FULL      mixed=NONE
    FULL_AND_PIECEWISE   mori_low_latency -> FULL_AND_PIECEWISE  decode=FULL      mixed=PIECEWISE
    PIECEWISE            mori_low_latency -> PIECEWISE           decode=PIECEWISE mixed=PIECEWISE

Both full modes survive unchanged under mori. **Prefer `FULL_AND_PIECEWISE`**: identical
`decode=FULL`, but `FULL_DECODE_ONLY` sets `mixed=NONE` (mixed prefill-decode batches drop
to eager) whereas `FULL_AND_PIECEWISE` keeps the piecewise graphs those batches already
have — a strict superset of current behaviour.

Two constraints a careless cudagraph change breaks:
- `use_inductor_graph_partition` must stay **on** — it keeps the STABLE-ABI
  `concat_and_cache_mla` out of the compiled graph. Without it the first *real* MLA decode
  dies with `RuntimeError: unknown parameter type` (boot and warmup pass via the fake path,
  so it looks healthy until the first request).
- Never express "no graphs" as bare `--enforce-eager` — it drops `+quant_fp8` and hits an
  AITER `dynamic_per_token_scaled_quant` signature mismatch at engine init. Use
  `cudagraph_mode: NONE` **with** `+quant_fp8`.

**Verification hook:** grep the decode log for `Capturing CUDA graphs (decode` — the
`"decode"` label at `gpu_model_runner.py:7057`. Under PIECEWISE only the
`(mixed prefill-decode, PIECEWISE)` line ever appears.

### 8.5 Block size: what the source actually supports

| backend | `get_supported_kernel_block_sizes()` | file:line |
|---|---|---|
| `ROCMAiterMLASparseBackend` (**loaded**) | **`[1, 64]`** | `rocm_aiter_mla_sparse.py:282` |
| `DeepseekV32IndexerBackend` (ROCm) | **`[1, 64]`** | `indexer.py:136` |
| `DeepseekV4IndexerBackend` | `[256]` | `indexer.py:176` |
| `AiterMLABackend` | `[MultipleOf(1)]` | `rocm_aiter_mla.py:140` |

**16 and 32 are NOT supported; 64 IS.** We run `block_size=1`, which makes the block
table enormous: `max_len/block_size` int32 entries per request row, so
`1048576 / 1 = 1,048,576` entries/row — vs 16,384 at `bs=64`, and 1,024 at `bs=64` +
64k context. Two knobs (`block_size` 1->64, `max_model_len` 1M->64k) compound to a
**1024x** smaller block table.

**Unresolved contradiction, resolve before changing:** `models.yaml:204` asserts DSA
*requires* block-size 1, while `tests/argv_assert.sh:10` asserts `--block-size 16` for
the prefill role, and the source above allows 64. Do not change this blind.

### 8.6 Perf-tuning method rules (learned the hard way)

1. **Budget first, tune second.** §8.1 saved a whole restart-and-tune cycle chasing a
   <=7 ms prize against a 44 ms deficit.
2. **Batch-independent TPOT means fixed per-step cost.** If 4x concurrency doesn't move
   TPOT, stop looking at kernels and look at launches/collectives.
3. **Re-baseline inside the same engine boot.** Job 5642 produced 104.94 ms TPOT at
   con=64 and then **146.30 ms** for the identical config ~25 min later (13,133 vs
   10,177 tok/s total). Runs were sequential (log names are **UTC**, mtimes **CDT** —
   convert before concluding overlap), watchers held, zero throttle/preempt/recompute
   markers, GPUs idle at 45-50 C junction on `auto`. **Cause still unknown.** Never A/B
   against a number from a previous boot.
4. **Grep the engine's own decisions** rather than reading source and inferring — AITER,
   vLLM compilation config, and cudagraph capture all log what they actually chose.

### 8.7 Head-to-head results (ISL/OSL 28k/1k, 1P/1D EP8, GLM-5.2-FP8)

| run | c16 tok/s | c32 tok/s | c64 tok/s | c64 TPOT |
|---|---|---|---|---|
| 171621 (run 1) | 3,978 | 7,423 | 13,133 | 104.94 ms |
| 174152 (run 2) | 3,001 | 5,739 | 10,177 | **146.30 ms** |

NIAH accuracy **PASSED** in both runs. Ticket SLO is TPOT **50 ms** avg / TTFT < 7 s.

### 8.8 Ranked lever list (current)

1. **Decode CUDA graph mode** `PIECEWISE` -> `FULL_AND_PIECEWISE` (§8.4) — holds the ~64 ms.
2. **`block_size` 1 -> 64** (§8.5) — legal per source; resolve the `models.yaml:204` claim first.
3. **`max_model_len` 1M -> 64k** — 16x smaller block-table row; the existing "do not cap"
   note is an *accuracy* argument only, not a perf one.
4. ~~AITER MoE tuning~~ — **deprioritised by measurement** (<=7 ms realistic).
5. ~~decode `max_num_batched_tokens` below 2048~~ — **exhausted**, curve is flat.
6. ~~`MORI_IB_GID_INDEX` 1 -> 3~~ — **closed, not a lever**. On AAC index 3 does not
   exist (§5.7); 1 is correct. And it could not have mattered: in 1P/1D the decode EP8 is
   8 GPUs on ONE node over Infinity Fabric, so the NICs carry only the prefill->decode KV
   handoff — GID choice moves **TTFT, not TPOT**.
7. AITER / MoRI / vLLM commit deltas vs the blog pins (`AITER e03fa6040`, `MoRI 42e895472b08`).

### 8.9 Suspects eliminated — do NOT re-chase

MoRI-EP kernel mode (correct: high_throughput prefill / low_latency decode) · absent
`FLYDSL_GPU_ARCH=gfx000` · DP5 empty-CUDA-graph asymmetry (on all 8 workers) · untuned
a8w8 GEMM as a *differential* vs the blog · CX7-vs-Ionic transport (not in the decode
loop) · graph capture failure (PIECEWISE=9 on all 8 workers, 3.69 GiB) · mnbt below
2048 · the glm5 `topk` off-by-one as *our* miss cause · thermal throttling and
double-launch as the run-2 regression cause.
