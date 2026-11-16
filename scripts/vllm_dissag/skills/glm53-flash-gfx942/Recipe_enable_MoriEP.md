# Enabling MoRI-EP for GLM-5.3-Flash on gfx942 (optional expert all-to-all)

The shipped GLM-5.3-Flash cards use **MoRIIO** for prefill→decode KV transfer and vLLM
**`allgather_reducescatter`** for the expert all-to-all. This is the validated, SLO-meeting
recipe and the default you should use.

This note documents how to switch the **expert all-to-all** from all-gather/reduce-scatter
to **MoRI-EP** (`mori_high_throughput` prefill / `mori_low_latency` decode), for experiments,
multi-node expert-parallel deployments, or future work. **It is opt-in and not the default.**
MoRIIO (the KV connector) is unchanged either way.

> **Recommendation (Oct 2026): keep the default (all-gather).** MoRI-EP is correct and
> accurate on gfx942, but on a single node per leg it is decode-TPOT **parity on MI325X and
> ~6–7% slower on MI300X** — no performance win. See "Measured" below. Enable it only where
> its architecture pays off (multi-node EP) or to reproduce these numbers.

## What changes (per card, ~4 knobs)

Each GLM-5.3-Flash card sets, in its `env:` block:

```yaml
    VLLM_ALL2ALL_BACKEND: "allgather_reducescatter"   # prefill expert all-to-all
    DECODE_MORI_BACKEND:  "allgather_reducescatter"   # decode expert all-to-all
```

To enable MoRI-EP, change those two lines, cap the decode MoRI buffer via the per-leg env
passthrough, and lower the **decode** `max-num-batched-tokens` to match (see the landmine below):

```yaml
    VLLM_ALL2ALL_BACKEND: "mori_high_throughput"   # prefill: MoRI-EP high-throughput
    DECODE_MORI_BACKEND:  "mori_low_latency"       # decode:  MoRI-EP low-latency
    # Decode MoRI dispatch/combine buffer width, via the connector's per-leg env passthrough
    # (moriio.sh reads DECODE_EXTRA_ENV = "KEY=VAL ..." for the decode leg only):
    DECODE_EXTRA_ENV: "VLLM_MORI_MAX_TOKENS_PER_RANK=256"
```

and set the **decode** leg's `max-num-batched-tokens` to 256 in its serve args (the decode
`dp:`/model-config block the connector emits — the same place the decode `--max-model-len` /
`--max-num-seqs` come from). It must be ≤ the buffer above. For an MXFP4 card that already uses
`DECODE_EXTRA_ENV` (e.g. for `VLLM_ROCM_MXFP4_DENSE_FP8BLOCK=1`), append the MoRI var to the
same string: `DECODE_EXTRA_ENV: "VLLM_ROCM_MXFP4_DENSE_FP8BLOCK=1 VLLM_MORI_MAX_TOKENS_PER_RANK=256"`.

One node per leg → MoRI's `IntraNode` kernel is used for both; `mori_high_throughput` vs
`mori_low_latency` only differ across nodes. `MORI_SHMEM_HEAP_SIZE` 8 GiB is sufficient (16 GiB
is **not** required — it did not change the result).

> If the harness has no per-leg knob to lower **only** the decode `max-num-batched-tokens`,
> raise `VLLM_MORI_MAX_TOKENS_PER_RANK` to ≥ the decode `max-num-batched-tokens` instead (e.g.
> 16384): bring-up then succeeds, but decode TPOT regresses ~10% (wide buffer). Lowering decode
> MNBT to 256 is the fast path; the buffer just has to be ≥ whatever the profiling batch is.

## The one landmine: the decode buffer must cover the profiling forward

**This is the non-obvious part.** `VLLM_MORI_MAX_TOKENS_PER_RANK` sizes the MoRI dispatch/combine
buffer. vLLM's startup memory-profiling runs a dummy forward at `max_num_batched_tokens`
(default 16384), and MoRI's `prepare()` does **not** clamp the batch to the buffer width. So if
the buffer (e.g. 128) is smaller than `max_num_batched_tokens` (16384), the first forward writes
past the buffer and the decode engine dies at bring-up with:

```
Memory access fault by GPU node-N ... Reason: Write access to a read-only page.
...
RuntimeError: Engine core initialization failed.
```

on every rank (seen with **both** FP8 and MXFP4 — it is not checkpoint-specific).

**Fix:** keep the buffer ≥ the profiling batch. Decode genuinely only needs ~128–256 tokens per
step (`max_num_seqs` × (1 + MTP)), so set **decode** `max_num_batched_tokens` down to 256 and the
buffer to the same 256. 256 covers profiling and matches the real decode batch, so it is both
safe and fast. Do **not** lower the **prefill** `max_num_batched_tokens` — prefill genuinely
processes wide batches for long context, and its MoRI buffer is sized for them.

Buffer-width trade-off (decode, measured): 128 crashes profiling; **256 = parity**; 2048 brings
up but costs ~10% TPOT (16× the real decode batch moved per layer per step).

## Verify it is active

On each leg's load log (`disagg_prefill.log`, `disagg_decode_mtp.log`):

- `Using MoriAll2AllManager all2all manager`
- `[mori-jit] Compiling ... ep_intranode for gfx942`
- `Using MoriPrepareAndFinalize` (prefill `fp8.py`/`mxfp4.py` AND decode)
- Decode `non-default args` shows `all2all_backend: 'mori_low_latency'`, `max_num_batched_tokens: 256`
- **No** `Memory access fault` in the decode log; all legs report `/health` 200

MXFP4 load lines are unchanged (prefill "converted exactly to FP8 128×128 blocks", decode
"FlyDSL W4A16 exact e2m1 decode") — MoRI-EP coexists with the 4-bit decode path.

## Measured (MI325X and MI300X, 1 prefill + 1 decode node, MXFP4)

Accuracy (both checkpoints, MoRI-EP active): seeded NIAH 100k 20/20 (MXFP4) and 40/40 (FP8);
GSM8K 250 = 99.2% (MXFP4). 0 errors.

Decode TPOT, 8K/1K, `vllm bench serve`, median (all-gather | MoRI-EP buffer 256):

| Platform | Conc | all-gather | MoRI-EP | Δ |
| --- | --- | --- | --- | --- |
| MI325X | 32 | 16.74 ms | 16.75 ms | ~0% |
| MI325X | 8  | 13.93 ms | 14.20 ms | +1.9% |
| MI300X | 32 | 18.86 ms | 19.99 ms | +6.0% |
| MI300X | 8  | 14.29 ms | 15.29 ms | +7.0% |

MoRI-EP is decode-TPOT parity on MI325X and ~6–7% slower on MI300X; TTFT is comparable. It does
**not** close the MXFP4-vs-FP8 decode gap on MI300X. The default (MoRIIO + all-gather, with the
hybrid TP4×DP2 prefill on the `-hyb` cards) remains the performance recipe.

## Not yet validated: sustained agentic load

MoRI-EP was validated for bring-up, accuracy and decode speed, not for long agentic runs: in one long AgentX run a
decode engine exited, and the cause was not established. Run a long AgentX test with the full decode logs kept before
relying on it; the default all-gather recipe ran the full AgentX matrix on MI325X and MI300X with 0 errors.
