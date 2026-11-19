# GLM-5.3-Flash-FP8 disaggregated — verified recall results (MI355X gfx950 + ionic)

Needle-in-haystack recall, disaggregated 1P/1D over MoRIIO. Needle =
`The special access code is DELTA-9931.` inserted at a given depth into varied
prose filler; the model is asked to recall it. `DELTA-9931` = exact code returned.
Greedy (temperature 0), `/v1/completions`, MoRI **WRITE** mode. Recipe = base
image `rocmshared/vllm-glm53-flash:ionic-aiter-tip-clrfix` + the 11 `../patches/`
overlays (connector MLA KV block-mapping + per-group WRITE routing + cross-chunk
KV accumulation, and the model/aiter fixes) mounted by `../vllm_pd_launch.sh`,
with **chunked prefill** (`--enable-chunked-prefill --max-num-batched-tokens
16384`, `--max-model-len 940000`, `--max-num-seqs 256`). Re-verified live on gold
pair 014↔021 (2026-09-22).

## TP4 1P/1D  (prefill/decode = 4-way TP each, MoRIIO KV, GPU util 0.5)

Chunked prefill (`--enable-chunked-prefill --max-num-batched-tokens 16384`),
`max-model-len 940000`. Verified live on a gold ionic rail pair (014↔021).

| context (words) | prompt tokens | needle depth      | TTFT   | result |
|-----------------|---------------|-------------------|--------|--------|
| 8,000           | 10,435        | 0.9               | 1.0s   | DELTA-9931 |
| 30,000          | 39,743        | 0.05 / 0.5 / 0.95 | 1.9s   | DELTA-9931 (all) |
| 60,000          | 80,650        | 0.1 / 0.5 / 0.9   | 3.3s   | DELTA-9931 (all) |
| 100,000         | 136,045       | 0.1 / 0.9         | 5.4s   | DELTA-9931 (all) |
| 200,000         | 273,703       | 0.1 / 0.9         | 11.1s  | DELTA-9931 (all) |
| 400,000         | 540,968       | 0.1 / 0.9         | 24.7s  | DELTA-9931 (all) |
| 500,000         | 678,463       | 0.1 / 0.9         | 33.6s  | DELTA-9931 (all) |
| 640,000         | 871,315       | 0.1 / 0.9         | 49.5s  | DELTA-9931 (all) |

**Exact recall to 871,315 tokens, all needle depths.** The only ceiling is
`max_model_len` (940,000): a prompt above it is cleanly rejected (HTTP 400 from the
prefill leg, surfaced by the router), **not** mis-recalled. Chunked prefill both
keeps every chunk's per-group KV (fix #2) and avoids the single-chunk Triton
compile wall (<~780K) — so it supersedes the earlier single-chunk recipe (which
was correct but capped ~100K on VRAM/compile).

## EP8 1P/1D  (DP8 + expert-parallel, MoRIIO KV + allgather/reducescatter MoE, util 0.40)

Chunked prefill (`--enable-chunked-prefill --max-num-batched-tokens 16384`),
`max-model-len 940000`, util 0.40. Verified live on a gold ionic rail pair
(030↔038). The cross-chunk-accumulation fix is config-agnostic (it is in the
connector, not the parallelism), so EP8 reaches the same envelope as TP4.

| context (words) | prompt tokens | needle depth | TTFT   | result |
|-----------------|---------------|--------------|--------|--------|
| 8,000           | 10,435        | 0.9          | 3.3s   | DELTA-9931 |
| 30,000          | 39,743        | 0.1 / 0.9    | 5.9s   | DELTA-9931 (all) |
| 60,000          | 80,650        | 0.1 / 0.9    | 5.8s   | DELTA-9931 (all) |
| 100,000         | 136,045       | 0.1 / 0.9    | 13.3s  | DELTA-9931 (all) |
| 200,000         | 273,703       | 0.1 / 0.9    | 22.9s  | DELTA-9931 (all) |
| 400,000         | 540,968       | 0.1 / 0.9    | 37.2s  | DELTA-9931 (all) |
| 500,000         | 678,463       | 0.1 / 0.9    | 45.6s  | DELTA-9931 (all) |
| 640,000         | 871,315       | 0.1 / 0.9    | 60.8s  | DELTA-9931 (all) |

**Exact recall to 871,315 tokens, all needle depths** — same as TP4. The prior
single-chunk EP8 recipe capped at 64K (`max-model-len`); chunked prefill lifts it
to the model max. The only ceiling is `max_model_len` (over → clean HTTP-400).

## Self-contained baked image (OVERLAYS=0) — re-verified 2026-09-23

The baked image `rocmshared/vllm-glm53-flash:glm53-flash-disagg-overlays-v1`
(`docker/…glm53flash.overlay.amd.Dockerfile` — the 11 overlays COPYed in-source
onto the proven base) was verified with **no runtime overlays** (`OVERLAYS=0`) on
gold pair 014↔021, MoRI WRITE mode, TP4 1P/1D:

| context (words) | prompt tokens | needle depth      | TTFT   | result |
|-----------------|---------------|-------------------|--------|--------|
| 8,000           | 8,684         | 0.1 / 0.5 / 0.9   | ~1s    | DELTA-9931 (all) |
| 60,000          | 65,026        | 0.1 / 0.9         | 2.8–6.2s | DELTA-9931 (all) |
| 400,000         | 433,355       | 0.1 / 0.9         | 18.7–21.6s | DELTA-9931 (all) |

Byte-identical recall to the base-image + overlays recipe (the baked files are
sha256-identical to `../patches/`).

**`glm53-flash-disagg-v2` (mori also baked) — re-verified 2026-09-23, NO MORI_PATCHED.**
The v2 image additionally rebuilds mori from the ionic-fix fork
(`raviguptaamd/mori:ionic-atomic-mr-strip`), so the atomic-MR strip is in-source and
`MORI_PATCHED`/the shared-`.so` mount is no longer needed. Verified `OVERLAYS=0`,
**no MORI_PATCHED**, WRITE mode, on 014↔021:

| context (words) | prompt tokens | needle depth      | TTFT   | result |
|-----------------|---------------|-------------------|--------|--------|
| 8,000           | 8,684         | 0.1 / 0.5 / 0.9   | 0.9s   | DELTA-9931 (all) |
| 60,000          | 65,026        | 0.1 / 0.9         | 2.8–6.0s | DELTA-9931 (all) |
| 400,000         | 433,355       | 0.1 / 0.9         | 18.8–21.6s | DELTA-9931 (all) |

The baked mori is confirmed stripped (`MORI_IO_DISABLE_ATOMIC_MR` string present in
`libmori_application.so`/`libmori_cco.so`; the base image's mori has 0). v2 still
needs the two host pieces (`GLIBC_SWAP`, `AITER_KSPLIT=1`/warm cache) documented in
`README.md`.

**`glm53-flash-disagg-v3` (mori + router both baked) — re-verified 2026-09-23.**
v3 additionally builds `vllm-router` from source into `/usr/local/bin`, so **no
`ROUTER_BIN` host binary is needed** either. Verified `OVERLAYS=0`, **no
MORI_PATCHED, no ROUTER_BIN**, WRITE mode, on 014↔021 — router confirmed running
from the in-image binary (no mount):

| context (words) | prompt tokens | needle depth    | TTFT   | result |
|-----------------|---------------|-----------------|--------|--------|
| 8,000           | 8,684         | 0.1 / 0.5 / 0.9 | 0.8s   | DELTA-9931 (all) |
| 60,000          | 65,026        | 0.1 / 0.9       | 2.8–5.9s | DELTA-9931 (all) |

v3 is the fully self-contained artifact: vLLM/aiter overlays + mori (ionic strip) +
vllm-router all in-image. Only `GLIBC_SWAP` + `AITER_KSPLIT=1`/warm cache remain as
node-infra launch pieces (see `README.md`). Full reproduction gates + the
TTFT/TPOT/throughput benchmark plan are in `TEST_PLAN.md`.

### EP8 1P/1D on the v3 self-contained image — re-verified 2026-09-23 (gold pair 030↔038)
EP8 (DP8 + expert-parallel, `allgather_reducescatter` MoE dispatch + MoRIIO KV,
util 0.40) on the **v3 image** (`OVERLAYS=0`, no external mounts), **decode CUDA
graphs** on, single-chunk prefill (`--max-num-batched-tokens = max_model_len`).
Recall `DELTA-9931`:

| context (words) | prompt tokens | depths | TTFT | result |
|---|---|---|---|---|
| 8,000  | 8,684  | 0.1/0.5/0.9 | 1.7–3.4s | PASS (all) |
| 20,000 | 21,684 | 0.1/0.5/0.9 | 2.4–3.8s | PASS (all) |
| 28,000 | 30,355 | 0.1/0.9     | 2.0–4.3s | PASS (all) |

Ceiling = `max_model_len` (32768, EP8 single-chunk). Confirms EP8 disagg on the
self-contained v3 image (previously only the base-image + overlays recipe was
verified for EP8). Notes: EP8 needs the gold aiter JIT seed on cold nodes
(`module_quant.so` won't compile from source; seed
`jitcache_*/aiter/` from a warm set) and **single-chunk** prefill (chunked-prefill
auto-default 400s the disagg-prefill request under DSA+EP).

## Performance (v3, TP4 1P/1D, gold pair 014↔021, WRITE mode)

### NIAH recall ladder (correctness) — all depths recall `DELTA-9931`
| context (words) | prompt tokens | depths | TTFT | result |
|---|---|---|---|---|
| 8,000   | 8,684   | 0.1/0.5/0.9 | 0.9s  | PASS (all) |
| 30,000  | 32,526  | 0.1/0.5/0.9 | 1.7s  | PASS (all) |
| 60,000  | 65,026  | 0.1/0.5/0.9 | 2.8s  | PASS (all) |
| 100,000 | 108,355 | 0.1/0.5/0.9 | 4.4s  | PASS (all) |
| 200,000 | 216,684 | 0.1/0.5/0.9 | 8.7s  | PASS (all) |
| 236,000 | 255,684 | 0.1/0.5/0.9 | 10–12s | PASS (all) |

TTFT scales ~linearly with prompt tokens (~43 tok/ms prefill; slight upward bend at
the top from MLA O(n²) attention).

### Throughput / latency — 256K in / 1024 out, concurrency 8 (eager decode)
TTFT p50 44.9s / p99 130.5s, TPOT p50 140ms, agg out 40 tok/s. The single-request
256K TTFT is ~10s → the C=8 TTFT growth is prefill-**capacity contention** (8 prefills
through 4 prefill GPUs), not per-request cost. Levers to flatten: prefix caching
(shared-prefix workloads), context/sequence parallelism, or more prefill capacity
(xPyD / larger prefill TP). Decode has headroom (flat TPOT).

### Decode CUDA graphs vs eager — 8K in / 200 out, concurrency 8 (matched)
Decode `cudagraph_mode=FULL_AND_PIECEWISE` (prefill stays eager) vs the shipped
`--enforce-eager` decode. **Recall unchanged** (DELTA-9931 correct at 8K + 60K, both
depths under graphs). Graph capture: 9s, 1.84 GiB.

| metric | eager (current recipe) | CUDA graphs | improvement |
|---|---|---|---|
| TPOT p50 | 83.1 ms | **14.0 ms** | **5.9×** |
| TPOT p99 | 84.8 ms | 14.2 ms | 6.0× |
| out throughput | 72.6 tok/s | **243.7 tok/s** | **3.4×** |
| TTFT p50 | 3.9 s | 2.0 s | 1.95× |

→ decode CUDA graphs are a large, correctness-safe win (task #223). Enable with
`EAGER=0 DECODE_CUDAGRAPH_MODE=FULL_AND_PIECEWISE` and drop `--enforce-eager` from the
decode leg's EXTRA_ARGS. (The launcher's `EAGER=0` path was fixed to make this work:
a `local role=...` same-line unbound-var bug that mis-wrote the compilation-config,
and inline-JSON quoting through the docker heredoc.)

## Before the fix (for reference)

Two "before" states, both fixed in-source:
- **Without the MLA KV block-mapping fix (fix #1):** disagg served coherent local
  continuation but recall was garbage at every length (e.g. "the password is
  banana. the password is" → "is is is is"). Colocated (non-disagg) inference on
  the same image recalled correctly, localizing the bug to the MoRIIO KV transfer.
- **With fix #1 but not the cross-chunk accumulation (fix #2):** recall was exact
  within a single prefill chunk but broke at *every* needle depth once a prompt
  spanned ≥2 chunks (only chunk-1 KV reached decode). Fixed by accumulating each
  chunk's per-group blocks across the whole prompt.

## Environment

- Image + overlays: base `rocmshared/vllm-glm53-flash:ionic-aiter-tip-clrfix`
  (carries the ionic mori atomic-MR strip + aiter-tip + clr fix) + the 11
  `../patches/` overlays (vLLM connector/model + aiter fixes) bind-mounted by
  `../vllm_pd_launch.sh`. gfx950 (MI355X). (A self-contained in-source image is a
  follow-up.)
- Nodes: 8-GPU MI355X (gfx950), AMD AI NIC (ionic) RoCE, one leg per node.
  Verified on GID-mapped "gold" ionic rail pairs (all 8 rails routed, no IBDEV
  pinning) — TP4 on one pair, EP8 on another.
- Model: `GLM-5.3-Flash` (`Glm5NextForConditionalGeneration`).
