# GLM-5.3-Flash disagg — test & benchmark plan

Reproduce and gate the recipe on **two gold-pair nodes** (8×MI355X gfx950 + ionic
each). Image `rocmshared/vllm-glm53-flash:glm53-flash-disagg-v3` (self-contained).
Launch env (host pieces): `GLIBC_SWAP=1 HOSTLIBS=<glibc-2.39 closure> AITER_KSPLIT=1`
(or a warm aiter cache). `OVERLAYS=0` (image carries the fixes).

## Prerequisites
- **Gold pair** — both nodes' 8 ionic rails on identical `/64` subnets (all-rail, no
  IBDEV pinning). GID-map to find one:
  `for i in 0..7; do cat /sys/class/infiniband/ionic_$i/ports/1/gids/1; done` and
  compare the 3rd hextet rail-by-rail. Non-gold → CQE-status-12 flush.
- Model weights on both nodes; the glibc-2.39 closure dir.

---

## Part A — Correctness / boot gates (pass/fail)

**Gate 0 — image self-containment.** `docker inspect vllm_prefill` shows the image
is `glm53-flash-disagg-v3`, arch gfx950, **no** `patches/*.py` bind-mounts and **no**
`vllm-router`/`libmori_*.so` host mounts. In-image checks:
`strings $(python3 -c 'import mori,os;print(os.path.dirname(mori.__file__))')/libmori_application.so | grep MORI_IO_DISABLE_ATOMIC_MR` = present; `which vllm-router` = `/usr/local/bin/vllm-router`; `grep CHUNKFIX …/moriio_connector.py`.

**Gate 1 — TP4 bring-up + short recall.** `run_flash_disagg_tp4.sh` on the gold pair.
Both legs reach "Application startup complete"; a tiny prompt serves; built-in recall
smoke prints `DELTA-9931` at 500w **and** 8000w (8000w > one attention group block →
exercises the MLA block-mapping fix). This is the single most valuable check.

**Gate 2 — TP4 long-context needle.** Chunked prefill
(`--enable-chunked-prefill --max-num-batched-tokens 16384`, `MAXLEN=940000`), needle
at depths 0.1/0.5/0.9. Expect `DELTA-9931` at every depth from 30K → **433K+**.
Prompts > `max_model_len` → clean HTTP-400 (not mis-recall). Exercises the
cross-chunk KV accumulation (fix #2).

**Gate 3 — EP8 bring-up + recall.** `orch_ep8.sh` (`MODE=ep`, `ROUTER_DP_LOCAL=8`,
util 0.40, `SPARSE_IDX_MB=4096`) on the 2nd gold pair. Expect `DELTA-9931` at the
same envelope; watch for MoE all2all crashes (should be none with
allgather/reducescatter) and indexer OOM-resources.

**Gate 4 — negative control (optional).** Same image but ionic atomic strip OFF
(`MORI_IO_DISABLE_ATOMIC_MR=0`): expect `RegisterRdmaMemoryRegion` EINVAL and no
serve — proves the mori fix is load-bearing.

### NIAH recall matrix (the correctness deliverable)
Needle `DELTA-9931` in prose filler, `/v1/completions`, temp 0. Report a grid:

| context (tok) | depths | pass criterion |
|---|---|---|
| 8K, 30K, 60K, 100K, 200K, 400K, (640K→871K) | 0.1 / 0.5 / 0.9 | exact `DELTA-9931` at every cell |

Run TP4 and EP8. A cell fails if the code is wrong or absent.

---

## Part B — Performance benchmarks

**Metrics:** TTFT (time-to-first-token), TPOT (per-output-token latency),
end-to-end throughput (output tok/s), prefill throughput (input tok/s). Report
p50/p99 where a distribution exists. Temp 0, fixed output length (e.g. 128 tok) so
TPOT is comparable across runs. Warm the JIT + handshake first (discard the first
2–3 requests). Pin the gold pair; record image digest + launch env in each result.

**Sweep — context × concurrency (both TP4 and EP8):**

| input context | output tok | concurrency (in-flight reqs) |
|---|---|---|
| 1K | 128 | 1, 8, 32 |
| 8K | 128 | 1, 8, 32 |
| 60K | 128 | 1, 8 |
| 200K | 128 | 1, 4 |

For each cell: TTFT (prefill-bound), TPOT (decode-bound), throughput. Expect TTFT to
scale ~linearly with input length (matches the recall-grid TTFT: 8K≈1s, 60K≈3–6s,
400K≈20s); TPOT roughly flat vs context but rising with concurrency; throughput to
climb with concurrency until KV/compute-bound.

**Driver:** vLLM's `benchmark_serving` against the router `/v1/completions`
(`--dataset-name random`, fixed input/output lens, `--max-concurrency`), or a small
async harness firing N concurrent fixed-length requests and recording per-request
TTFT/TPOT. Router runs `--policy round_robin`.

**Disagg-specific comparisons (the interesting numbers):**
1. **TP4 vs EP8** at each context/concurrency — where does EP8's DP8+expert-parallel
   win on throughput, and what TTFT/TPOT does it cost?
2. **Disagg (1P/1D) vs colocated** at 8K & 60K — the prefill/decode split's TTFT and
   throughput trade vs a single-node colocated baseline.
3. **WRITE vs READ mode** (mori) at 8K & 200K — WRITE should show lower TTFT (prefill
   pushes KV, overlaps compute) vs READ (decode pulls, adds a round-trip).
4. **Concurrency scaling** — throughput vs in-flight requests at 8K to find the
   saturation knee.

**What I will actually run first** (minimal, high-signal): TP4 + EP8, contexts
{1K, 8K, 60K, 200K} × concurrency {1, 8}, output 128 — TTFT/TPOT/throughput table +
the TP4-vs-EP8 and WRITE-vs-READ deltas at 8K/200K. Extend to concurrency 32 and the
colocated baseline if the numbers warrant.

**Output:** a results table appended to `RESULTS.md` (image digest, env, gold pair,
date), one row per cell.
