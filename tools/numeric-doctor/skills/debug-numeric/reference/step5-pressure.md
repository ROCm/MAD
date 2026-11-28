# Step 5, pressure: when replay copies are not the load the fault needs

Read when a group's 5b ends `NOT_SET_UP`, or `NOT_REPRODUCED` while the model
reproduces reliably, or when step 1 ledgered a load trigger (`TRIGGER` hint). The
5b ladder's only load is copies of the replay. A fault gated on what the *model*
runs alongside (HBM pressure, a kernel mix) never appears under copies of a small
cache-resident op, at any concurrency -- millions of clean replays, then a quick
reproduction beside a training-step stand-in. Each rung below is one short
job; stop at the first that reproduces.

**Deterministic model, clean replay: leftover memory first, not load.** A model that NaNs on
every run at the same place while its capsule replays clean is more often reading memory
nobody wrote than missing a load: a fresh replay process hands back other garbage. Run the
model twice with `scripts/alloc_poison.py --empty 0` and `--empty nan` before 5c/5d (one
short job; `step5n-native.md` *Leftover memory?*); a rate that follows the value goes to
step 5N with that recipe.

## 5c-kernel. Same kernel? (first, ~1 min, no extra job)

hipBLASLt GEMMs only (any other op: go to 5c). The ladder writes
`replay/<G>/kernel.*.txt` on every completed ladder. With a model hipBLASLt log (`step4-capture.md`: `HIPBLASLT_LOG_MASK=32` on one capture trial):

```bash
python3 <REPO_ROOT>/scripts/kernel_match.py <ARTIFACT_DIR>/replay/<G>/kernel.*.txt <model hipBLASLt log>
```

`KERNEL_MISMATCH=<r>/<m>`: the replay runs another solution; pin the model's
(`CAPSULE_EXTRA_ARGS="--solution-index <m>"`) and re-run 5b before anything below.
`KERNEL_MATCH`: the kernel is right, the load is not -- go on.
`KERNEL_UNKNOWN` with no model log: run the model ~20 steps with the log on, on one
GPU (it is chosen at the first call).

## 5c. The model as co-tenant (one GPU, minutes)

One replay of the failing op beside the model itself, restarted whenever it stops
(a training run exits at its first NaN), at the capture's processes per GPU minus
the replay:

```bash
nd submit --name p5c --gpus 1 --exclusive --time 00:30:00 -- exec.sh --gpus 0 -- \
  'python3 -m replay_nan_op <capsule as $ND_ART/...> --iters 1000000 --restore-env --contend-with "bash $ND_REPO/scripts/cotenant_loop.sh $ND_ART/pressure/model.log <model command, e.g. --trials 2>" --progress-file $ND_ART/pressure/progress'
```

(Inner quotes double, never single.) `[verdict]
KERNEL_BUG` with `n/N > 0`: the fault needs real load -- go to 5d for a reproducer
without the model. The real model is a **weak, noisy** trigger: a clean 5c
does **not** end the search -- go to 5d anyway.

## 5d. Synthetic co-tenant sweep (one node, ~10 min)

One co-tenant per GPU, same capsule, `scripts/jobs/pressure_sweep.sh` (result table,
`STARVED` flag for a GPU whose replay its co-tenant starved -- its 0 means nothing).
**Order matters: the training-step stand-in first, simple loads only to narrow it
down** -- HBM streaming, big GEMMs and the captured op over fresh operands have run
clean where the stand-in reproduced in seconds. A clean load bounds a rate,
not a load: a co-tenant one step lighter can fail 10-100x less often. A VARIANTS
line `solo` is the no-load arm on its own GPU (a `sleep` co-tenant is refused:
`CONTENTION_INVALID`).

1. `transformer_hog.py` with the model's sizes (`--hidden --inter --heads
   --kv-heads --head-dim --layers --vocab --seq --batch` ints, `--qkv-bias 0|1`,
   `--tf32 0|1`, `--dtype fp32|bf16|fp16`, from the model's config -- the defaults
   are one 0.5B decoder's, always override them). The model line
   (`cotenant_loop.sh`) needs **every argument its launcher passes** -- a trainer
   with a required `--seed` exits at once, and so does every other variant with a
   bad argument: the results table then shows `exited before loading` plus the
   co-tenant's last output line (`co-tenant said:`; full output
   `g<i>.cotenant.log`). A whole sweep lost this way on 2026-11-06. A first sweep's
   VARIANTS file:

   ```
   python3 -m transformer_hog --mode full --procs 2 <sizes>
   python3 -m transformer_hog --mode full --procs 2 <sizes>
   python3 -m transformer_hog --mode fwdbwd --procs 2 <sizes>
   python3 -m transformer_hog --mode fwd --procs 2 <sizes>
   python3 -m transformer_hog --mode adamw --procs 2 <sizes>
   python3 -m transformer_hog --mode full --procs 3 <sizes>
   python3 -m transformer_hog --mode full --procs 1 <sizes>
   bash $ND_REPO/scripts/cotenant_loop.sh $ND_ART/pressure/g7_model.log <model command>
   ```

   ```bash
   CAPSULE=<CAPSULE_FILE> OUT_DIR=<ARTIFACT_DIR>/pressure/<G> VARIANTS=<file> \
     nd submit --name sweep_<G> --gpus 8 --exclusive -- pressure_sweep.sh
   ```

2. Nothing reproduces: a second sweep with `pressure_hog.py` modes (`copy`, `triad`,
   `gemm`, `mix`, `bust`) and higher `--procs` (at `--procs 2`, `triad`/`gemm` starved
   the replay to ~40 launches/s and `bust` exited at start on 2026-11-07 -- read the
   `STARVED` / `co-tenant said:` columns before counting their zeros), or the model's real step if its
   architecture is not a decoder transformer (write a stand-in on the same recipe:
   the model's op types and sizes, random weights, a synchronize per step and no
   host checks, runs until killed).
3. Still nothing: ledger `NOT_REPRODUCED` with every rung's `n/N` and load; say the
   model reproduced and under what. G1 there: **step 5N** (`step5n-native.md`) --
   these loads are its starting point.

Screening sweeps may cut `ITERS` (e.g. 100000) when the expected rate is high; the
reproducer's verification run keeps 1M.

**Ledger**: each variant's verdict, `n/N`, gate line, host; which load is minimal
(fewest processes, fewest modes) and reproduces.

## Package it (step 6, GROUP-only capture-to-reproducer job with a co-tenant)

```bash
GROUP=<G> CAPSULE_MODE=op [CAPSULE_EXTRA_ARGS=--synthesize] \
  CAPSULE_COTENANT='python3 -m transformer_hog --mode full --procs 2 <sizes>' \
  nd submit --name repro_<G> --gpus 8 --exclusive --time 02:00:00 --step 'step5b[<G>]' -- capture_to_repro.sh
```

`[...]` = only when it applies: `--synthesize` for a shapes-only capsule (the capture
default).
(Move an earlier `repro-<G>`/`verify-<G>` aside first: the job refuses to overwrite.)
The package ships the stand-in and `run.sh` starts it beside one replay per GPU;
the README says the replay alone does not reproduce. A custom stand-in: put it in
`COTENANT_FILES` and name it `@REPO@/cotenant/<file>`.
