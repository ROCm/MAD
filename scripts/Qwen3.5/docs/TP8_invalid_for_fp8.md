# Why pure TP8 is INVALID for Qwen3-Next-80B-A3B-Instruct-FP8 (and how EP fixes it)

**Short version:** the FP8 checkpoint uses **block quantization with a 128×128 block size**.
Pure tensor-parallel sharding must split weight dimensions into chunks that are still divisible
by the quant block size. At TP8, the MoE expert intermediate dimension (512) splits to 64 per GPU,
which is **< 128** and **not divisible by 128** → the server fails to load.

Valid **pure-TP** for this FP8 model: **1, 2, 4** (set as `valid_tp: [1,2,4]` in `model.yaml`).

> ⚠️ **This limit is specific to TENSOR-sharding the experts.** With **Expert Parallelism
> (`--ep-size 8` / `--enable-expert-parallel`)** each GPU holds **whole experts** — the expert
> weight is never tensor-split, so the 128-block constraint is never violated and **TP8+EP8 is
> valid**. This is exactly the shape InferenceX and the customer run in production (`tp:8, ep:8`).
> So "TP8 invalid" means *pure* TP8, NOT TP8+EP. See `docs/native_DP_broken_fp8.md` and the
> README "EP next-step" note. EP boot on this model is still to be validated empirically here.

---

## The exact error
`vllm serve ... --tensor-parallel-size 8` (or ATOM `-tp 8`) fails at weight creation with:
```
ValueError: Weight input_size_per_partition = 64 is not divisible by weight quantization
block_k = 128.
```
(raised in the FP8 block-quant `create_weights` path).

## The arithmetic
From the model `config.json`:
- `moe_intermediate_size = 512`
- FP8 block quant: `weight_block_size = [128, 128]`

Per-GPU expert intermediate size after TP sharding = `512 / TP`:

| TP | 512 / TP | ≥128 and divisible by 128? | Valid? |
|----|----------|----------------------------|--------|
| 1  | 512      | yes (512 = 4×128)          | ✅ |
| 2  | 256      | yes (256 = 2×128)          | ✅ |
| 4  | 128      | yes (128 = 1×128)          | ✅ |
| 8  | **64**   | **no (64 < 128)**          | ❌ |

At TP8 each shard would hold a 64-wide slice of a tensor whose quant scales are defined per
128-element block — the block can't be split across two GPUs, so the loader rejects it.

## Why this is fundamental, not a flag/config issue
- It is a property of **how the checkpoint was quantized** (128-block FP8), combined with the
  model's small `moe_intermediate_size` (512). No serving flag changes the block size.
- Confirmed empirically: vLLM TP8 FP8 fails at `create_weights` with the error above
  (WorkerProc failed to start). Not an OOM, not an env issue — a hard divisibility constraint.

## What WOULD allow TP8
- A different quantization with a **smaller block size** (e.g. MXFP4 uses `group_size = 32`:
  512/8 = 64, and 64 is divisible by 32 → TP8 *may* be valid for MXFP4). This is why
  `model.yaml` lists `valid_tp: [1,2,4,8]` for the MXFP4 model (note: MXFP4 is currently PARKED
  for a separate engine-correctness reason — see README).
- A model with a larger `moe_intermediate_size` (so 512/8=64 wouldn't be the limiter).

## Practical implication for the sweep
For FP8 full-node (8-GPU) packing, the valid (TP, DP) combos that use all 8 GPUs are:
- **TP1 × DP8**, **TP2 × DP4**, **TP4 × DP2**.
- **TP8 × DP1 is NOT runnable for FP8** — omit it from FP8 sweeps.
