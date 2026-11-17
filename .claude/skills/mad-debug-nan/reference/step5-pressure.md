# Step 5, pressure: when replay copies are not the load the fault needs

Read when a group's 5b ends `NOT_SET_UP`, or `NOT_REPRODUCED` while the model
reproduces reliably, or when step 1 ledgered a load trigger (`TRIGGER` hint). The
5b ladder's only load is copies of the replay. A fault gated on what the *model*
runs alongside (HBM pressure, a kernel mix) never appears under copies of a small
cache-resident op, at any concurrency -- millions of clean replays, then a quick
reproduction beside a training-step stand-in. Each rung below is one short
job (via `mad-run.sh`); stop at the first that reproduces.

Step 3 (oversubscription) is interactive-only; this file's co-tenant rungs are the
automated route to the same load.

## 5c-kernel. Same kernel? (first, ~1 min)

hipBLASLt GEMMs only (any other op: go to 5c). The ladder writes
`replay/<G>/kernel.*.txt` on every completed ladder. With a model hipBLASLt log (`step4-capture.md`: `HIPBLASLT_LOG_MASK=32` on one capture run):

```bash
madengine nan kernel-match <ARTIFACT_DIR>/replay/<G>/kernel.*.txt <model hipBLASLt log>
```

`KERNEL_MISMATCH=<r>/<m>`: the replay runs another solution; pin the model's
(`--extra-args "--solution-index <m>"`) and re-run 5b before anything below.
`KERNEL_MATCH`: the kernel is right, the load is not -- go on.
`KERNEL_UNKNOWN` with no model log: run the model ~20 steps with the log on, on one
GPU (it is chosen at the first call).

## 5c. The model as co-tenant (one GPU, minutes)

One replay of the failing op beside the model itself, restarted whenever it stops
(a training run exits at its first NaN), at the capture's processes per GPU minus
the replay:

```bash
bash .claude/skills/mad-common/mad-run.sh start pressure-5c -- \
  madengine nan replay <CAPSULE> --manifest build_manifest.json \
    --out-dir <ARTIFACT_DIR>/replay/<G>-5c --mode op --iters 1000000 --ngpus 1 \
    --cotenant "bash \$NAN_DOCTOR_RT/bin/cotenant_loop.sh <OUT_C>/model.log <model command, e.g. --trials 2>"
```

(`_C` = in-container paths; inner quotes double, never single.) `[verdict]
KERNEL_BUG` with `n/N > 0`: the fault needs real load -- go to 5d for a reproducer
without the model. The real model is a **weak, noisy** trigger: a clean 5c
does **not** end the search -- go to 5d anyway.

## 5d. Synthetic co-tenant sweep (one GPU per variant, ~10 min each)

One co-tenant, same capsule, one `madengine nan replay --mode op --cotenant '<variant>'`
per variant, run sequentially through `mad-run.sh` (the sweep is a manual loop; compare
each variant's `n/N` and launch rate -- a replay its co-tenant starved to a few
launches/s means nothing at 0).
**Order matters: the training-step stand-in first, simple loads only to narrow it
down** -- HBM streaming, big GEMMs and the captured op over fresh operands have run
clean where the stand-in reproduced in seconds. A clean load bounds a rate,
not a load: a co-tenant one step lighter can fail 10-100x less often.

1. `transformer_hog` with the model's sizes (`--hidden --inter --heads
   --kv-heads --head-dim --layers --vocab --seq --batch` ints, `--qkv-bias 0|1`,
   `--tf32 0|1`, `--dtype fp32|bf16|fp16`, from the model's config -- the defaults
   are one 0.5B decoder's, always override them). The model line
   (`cotenant_loop.sh`) needs **every argument its launcher passes** -- a trainer
   with a required `--seed` exits at once, and so does every other variant with a
   bad argument: the replay then shows the co-tenant exiting before loading. Check
   the co-tenant's log before counting a zero. A first sweep's variants (the
   `--cotenant` value, each run in turn):

   ```
   python3 -m nan_doctor_rt.transformer_hog --mode full --procs 2 <sizes>
   python3 -m nan_doctor_rt.transformer_hog --mode fwdbwd --procs 2 <sizes>
   python3 -m nan_doctor_rt.transformer_hog --mode fwd --procs 2 <sizes>
   python3 -m nan_doctor_rt.transformer_hog --mode adamw --procs 2 <sizes>
   python3 -m nan_doctor_rt.transformer_hog --mode full --procs 3 <sizes>
   python3 -m nan_doctor_rt.transformer_hog --mode full --procs 1 <sizes>
   bash $NAN_DOCTOR_RT/bin/cotenant_loop.sh <OUT_C>/g7_model.log <model command>
   ```

2. Nothing reproduces: a second sweep with `pressure_hog` modes (`copy`, `triad`,
   `gemm`, `mix`, `bust`) and higher `--procs` (at `--procs 2`, `triad`/`gemm` can
   starve the replay to ~40 launches/s and `bust` can exit at start -- check the
   launch rate and the co-tenant's log before counting their zeros), or the model's
   real step if its architecture is not a decoder transformer (write a stand-in on
   the same recipe: the model's op types and sizes, random weights, a synchronize
   per step and no host checks, runs until killed).
3. Still nothing: ledger `NOT_REPRODUCED` with every rung's `n/N` and load; say the
   model reproduced and under what. G1 there: **step 5N** (`step5n-native.md`) --
   these loads are its starting point.

Screening runs may cut `--iters` (e.g. 100000) when the expected rate is high; the
reproducer's verification run keeps 1M.

**Ledger**: each variant's verdict, `n/N`, gate line, host; which load is minimal
(fewest processes, fewest modes) and reproduces.

## Package it (step 6, with a co-tenant)

Package from the attempt that ran with the co-tenant (`step6-build.md`):

```bash
madengine nan package --group <G> --artifact-dir <ARTIFACT_DIR> --image <IMAGE> \
  --capsule <CAPSULE> --attempt <co-tenant attempt dir> --groups <ARTIFACT_DIR>/groups.json
```

Open the packaged `run.sh` and README: both must start the co-tenant beside the replay
and say the replay alone does not reproduce. If `run.sh` does not carry the co-tenant,
that is a manual edit; say so in the README and name the load.
