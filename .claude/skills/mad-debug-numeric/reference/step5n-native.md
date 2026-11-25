# Step 5N: native reproducer (the fallback when no capsule reproduces)

The capsule path saves *one op's values*. When it yields nothing -- or cannot run on
this model at all -- write the
reproducer yourself: a standalone script that runs the failing computation with
synthetic data, at the model's shapes, dtypes and load, and NaNs without the model.
Then go to step 6 (`step6-finish.md`, *Native reproducer*). Not optional, no user
decision: entering it is automatic; only the step-1 budget bounds it.

## When (exactly these exits)

| from | exit | first |
|---|---|---|
| 1/4 | `CAPTURE_UNREACHABLE`: no capture can see the ops where the NaN is born, so steps 4-5 never run -- the capture preflight calls the setup unsupported (vLLM pipeline parallel > 1 or several nodes, SGLang tp / dp > 1: the model runs in worker processes outside the capture; vLLM TP > 1 on one node IS captured -- `step1-ask.md` *vLLM*) and the supported variant (size 1) does not NaN; or any other setup step 4 demonstrably cannot wire (ledger why) | `<NAN_RATE>` measured at the original setup -- an explicit step 2 (`step2-rate.md`); the model must NaN there. Then ledger step 4 and step 5 as skipped (`capture unreachable: <why>`, `no capture`) and the size-1 run's rate as the first Ruled-out row |
| 5 | no capsule (empty `<CAPSULE_DIR>`) | the `step5-read.md` *Empty `<CAPSULE_DIR>`* list: fix and re-run a broken capture once (unwritable dir, missing `sync_stop()`, a filter hiding a NaN outside its op class). Fall back when the capture demonstrably ran and the model NaN'd (`<NAN_SIGNAL>` in a capture or control log) yet no capsule settled -- or the re-run also came back empty |
| 4/5a | no replayable group | -- (every group is an allocator/view false positive or `PROPAGATED` at capture) |
| 6 | G1 (the largest group) `NOT_REPRODUCED` / `NOT_SET_UP` | `step5-pressure.md` through 5d step 3 -- the fallback starts where that file says "ledger `NOT_REPRODUCED`" |
| 6 | G1 `PROPAGATED` | -- the bug is upstream of the capsule; the native script chases the producer |
| 4 | graphs-off capture runs clean, graphs-on controls NaN (vLLM, `step4-run.md`) | `step5n-compiled.md` *Graph-only NaN*: the graph-check run's `GRAPH_CHECK` and `nan_doctor_rt.replay_graph_piece` |

**G1 decides, under either `<GROUPS_TO_REPLAY>`.** `largest`: ledger the other
groups as N1 evidence. `all`: their 5b runs continue; one that reproduces still goes to
step 6 and step 5N continues (both reproducers ship).

**Not a trigger:** `FP32_OVERFLOW?` (a finding, reported plainly), G1 reproducing,
and the model itself not NaN'ing at its original setup (that is a step-3 problem: re-read
`<LEDGER>`, re-enter step 3; a clean size-1 variant is part of `CAPTURE_UNREACHABLE`, not this).

**World size > 1, a multi-rank native reproducer** (`step4-distributed.md`): when the
manifest names no origin rank (every NaN arrived through a collective) or the fault is
in the combination, not an op (a dropped or doubled reduction, wrong placements:
every op correct, only a cross-rank invariant broken), or `CAPTURE_UNREACHABLE` at a
size > 1 whose size-1 run is clean. Write a minimal `torchrun`
program at the smallest world size and mesh that still shows it (2 ranks, one mesh
dimension first), with the model's collectives and shapes -- run it with `RANKS=<world size>`
(`native_run.sh`, N4: one `torchrun` per GPU group, rank 0 prints `NATIVE_RESULT` counting a hit
on any rank; `native_variants.sh`, N5, the same per variant); the verdict compares it
against the same program at world size 1 (or the invariant that must hold: replicas
equal, shard sums equal). N3-N6 apply with "NaNs" read as "breaks the invariant".

```bash
$LG start step5n;  $LG note step5n NATIVE_TRIGGER=<exit> <one-line why>
mkdir -p "$ARTIFACT_DIR"/native && chmod 777 "$ARTIFACT_DIR"/native
madengine nan handoff "$ARTIFACT_DIR"   # summarises what the capsule path tried -> native/handoff.md
```

## N1. Start from the capsule path's failures (no GPU job)

**The capsule path's failures are the first evidence, not a dead end.**
`native/handoff.md` collects them: **Facts** (the image actually used, precision,
the model's command, when the model NaNs per arm, groups sharing one op and shape);
the capture **census** over runs; a **Contention** table -- the capture's load
(other processes on the GPU, what they run, memory) beside every replay's, flagging
a clean load as **LIGHTER** when it had fewer processes or (training) no backward pass; every
replay attempt, a COLLECTIVE group's torchrun replay included; a **Ruled out** list where each null states the smallest rate it
excludes (0/1M excludes only >= 3e-6); and **PROPOSED implications**, each with its
source (e.g. NOT_SET_UP -> "the values are not the trigger"; every clean load
LIGHTER -> "the sweep never applied the capture's load"). Read it instead of the
raw artifacts, then:

1. **Confirm or strike every implication** against its source -- the script proposes,
   it does not decide. Ledger `KEEP <n>: <one line>` / `STRIKE <n>: <why>`.
2. Add a proposal the rules missed when the attempts show one (ledger `KEEP +: ...`).
3. Fill the gaps below from `<LEDGER>` and digests -- never a model log whole
   (SKILL.md rule 9):

* `<FULL_MODEL_RATE>`: the rate to beat;
* the NaN step and where the model's own check saw it (`madengine nan waves`);
* what the capture *did* see (`madengine nan digest` on a capture log): the MASKED
  sites, the census, the last op it watched -- an empty capsule dir still names
  the region;
* any capsule's op, operand shapes/strides/dtypes and chain (`madengine nan exec --image <IMAGE> -- python3 -m nan_doctor_rt.capsule_info <CAPSULE>`), and
  5b's `first op to go non-finite` tally -- the strongest suspect when present;
  `PROPAGATED`: the dirty input's producer (chain) is the suspect, not the op;
* `<CONTENTION>` (5a-load, else step 1's containers per GPU) and the loads 5c/5d
  ran, clean or not -- they bound which load to start from;
* environment: image, the env amplifiers, `OMP_NUM_THREADS`,
  `float32_matmul_precision` / TF32, autocast dtype;
* model sizes from its config; the source windows `madengine nan capture-points` names.

Ledger one line: suspect (op, or region if none), shapes, contention, env.

**`CAPTURE_UNREACHABLE`: no capture ran**, so `handoff.md` has the trigger and the step-2
facts only -- no capsule, census or attempts. The suspects are what exists at the original
setup and not in the clean size-1 run: cross-rank collectives (the framework's own
all-reduce backends too, not only `c10d`), sharded layers, parallel-only code paths and the
env switches that select them (read them from the model source and the run's start-up log).
Expect a **multi-rank native reproducer** (*When*, above), top-down (N3) unless one component is
already the obvious suspect. N2's hooks must run where the model runs -- in the worker
processes (vLLM: `LLM.collective_rpc` / `apply_model`), not the driver -- else skip N2.

**Prior work, before the first candidate** (skip with `<PRIORS>` = none, `step1-ask.md`
question 7) -- one GPU-free grep for the suspect op,
fused-kernel name or Triton/library op across earlier investigations kept beside
`<ARTIFACT_DIR>` (`grep -lE '<op>|<kernel/function>'` over their root-cause notes and
ledgers). A hit is a hypothesis (`KEEP +: ... (prior: <path>)`), never a carried-over
rate; it orders the candidates.

**Leftover memory?** The NaN may come from bytes nobody wrote -- a `torch.empty` buffer a
kernel reads but no kernel for this arch fills, an out-buffer only partly written. Signs: the
model NaNs deterministically (every run, the same layer and step) yet the capsule replay is
`NOT_SET_UP` / clean; the first call of a process is clean; layers before the failing one are
clean in the same step; the capsule's chain feeds the terminal op from `empty*` rows, or
`capsule_info` shows an `uninit` operand (its kept value on the args line). Then the first
candidate is the poison test, before any load sweep (load cannot supply missing bytes):
`nan_doctor_rt.alloc_poison` runs the model's own command, or the candidate, with every
`torch.empty*` result filled with a value, one run per value -- the model: its Python command
prefixed with `python3 -m nan_doctor_rt.alloc_poison --empty <v> --` in a working copy of its
run script (ledger the path; step 9 restores); a candidate: `from nan_doctor_rt import
alloc_poison; print(alloc_poison.from_env())` at its top and one N5 variant per value
(`3|ALLOC_POISON_EMPTY=<v>||`). Values: `0`, `-1`, `nan`, `bits:0x7f7f7f7f`, and the value the
code means to put there when you can read it (a count: the split count). **The rate follows
the value** (0 -> every run, the meant value -> none) = a read of uninitialized memory,
whatever kernel reads it; ledger `KEEP +: uninitialized read (poison <v>: n/N, <w>: m/N)`.
Same rate under every value: not this. Allocations made in C++ (`at::empty` inside an
extension) are not wrapped: add `--free-blocks <v>`, which fills the caching allocator's free
pool first. The candidate (N3) then calls the suspect wrapper after poisoning, and its README
names the poison value that hits; step 7 names the buffer (`step7-rootcause.md` *Read of
uninitialized memory*).

**Born in compiled code?** The capsule is `PROPAGATED` with its chain stopped at
`outside_window` right after a torch.compile'd module, or there is no capsule and most of
the step is compiled: read `step5n-compiled.md` -- its table decides between that track
(compile-off control, Inductor `nan_asserts`, then candidates from the named kernel) and
the plain N2-N3 below.

## N2. No suspect op? Locate before writing (one wave)

Only when N1 names a region but no op (typically no capsule). A coarse check
perturbs a race less than the per-op capture: **module-level** forward and
backward hooks on the suspect region (one combined max-abs reduction per module output,
one host read per step, never per op), recording the first module to go
non-finite and the step. Same geometry and controls as step 4's wave (`step4-run.md`);
stop at the first hit. Write it as a capture *variant file* (never edit
`<RUN_SCRIPT>` or the model in place; ledger its path -- step 9 restores). Capture
arm 0 while controls NaN: the hook suppresses -- go to N3 top-down instead.

## N3. Write a candidate

`<ARTIFACT_DIR>/native/repro_<k>.py`, numbered, never overwritten.

**Every candidate is a hypothesis about what the capsule path lacked**, stated
before it runs: which kept implication it acts on and what it adds or changes
(the load, the neighbouring ops, the kernel pin, in-place/aliasing, the step
around the op). Ledger `cand <k>: [KEEP <n>] <what it adds> -> ...`. **A candidate
equivalent to a Ruled-out row is not run** -- same op scope with the same or a
lighter load and no more runs repeats a clean experiment. A clean candidate joins
the Ruled-out list (its ledger row is enough).

* imports torch (and only the library owning the suspect kernel); no dataset, model
  checkout or repo import;
* **same shapes, strides (transposes, non-contiguous views), dtypes, layouts,
  precision flags and env** as N1 recorded -- a contiguous copy of a transposed
  operand is a different kernel;
* seeded random data, regenerated per iteration *only* if the model's values
  changed per step (fresh operands once ran clean beside a reproducing stand-in --
  `step5-pressure.md`);
* loops `--iters` (default 1000000, the floor) or until killed;
  finiteness of the **suspect output only**, counted per iteration on the device and
  read once per ~1000 iterations (a host read per launch adds stalls under
  co-tenancy); **a finite signal** (`step4-explode.md`: the model reports wrong values,
  no NaN) counts an iteration whose suspect output is off from a higher-precision
  reference of the same call beyond the op's correct-call error (measure that on a control
  shape first; `math_refs.value_rel_err` where the op has a `VALUE_CHECKS` entry) -- or, for
  a race, differs bit-wise from the previous iteration on the same inputs;
* on a hit: check the inputs were finite, re-run the op in fp64 on the same inputs
  (CPU when it fits), save inputs + output to `native/hit_<k>_<copy>.pt`, print
  `HIT iter=<i> inputs_finite=<b> fp64_ref=<finite|OUT_OF_RANGE|NON-FINITE|unavailable>`
  -- at the positions the GPU output is non-finite: `finite` = finite and within the
  output dtype's range, `OUT_OF_RANGE` = finite in fp64 but beyond it (overflow),
  `NON-FINITE` = non-finite in fp64 too (undefined arithmetic: `0*inf`, `0/0`,
  `inf-inf`, `sqrt`/`log` of a negative; the capture's `math` origin); for a finite signal
  `fp64_ref=WRONG_VALUE rel_err=<e>` (the reference finite, the output finite and off from it);
* on the first hit also the failure signature -- how the output is wrong: `from
  nan_doctor_rt.failure_signature import signature; print(signature(out, ref64, prev)["line"])`
  (`ref64` the fp64 reference, `prev` the previous iteration's good output when the loop keeps
  one); its `SIGNATURE` line goes in the ledger with the hit;
* last line `NATIVE_RESULT=<hits>/<iters> copy=<c> gpu=<g>`;
* **the hit path runs only on a hit**, so its bugs surface after the GPU work: keep
  the fp64 / save / report code to plain tensor ops on what the loop already computed
  (no second model call under a different autocast or device), and dry-run it once on
  a forced non-finite value (CPU is enough) before running.
* **`NATIVE_DISCOVER=1` in the environment: run the suspect call once and exit** -- no reference, no
  checks, no loop. `nan_doctor_rt.native_sweep --discover` (step 7's sweep of a native reproducer)
  reads the kernels loaded and dispatched in that one run; a reference computed beside it would
  name its own GEMMs as suspects.

Two directions -- pick by N1:

| N1 has | start | then |
|---|---|---|
| a suspect op | **bottom-up**: the op alone in a loop | add the model's neighbours (its chain, producer, the step around it) only while it stays clean |
| a region, no op | **top-down**: the model's step (or region) with random weights at full size | reduce (N5) |

## N4. Run it under the capture's load

Contention travels with every rate (SKILL.md rule 6): a solo `0/N` means nothing.
`bin/native_run.sh` starts `COPIES` copies of the candidate per GPU
(default 3 = capture's processes per GPU + 1; `HIP_VISIBLE_DEVICES=<g>`, `--copy`,
`--gpu`) on every GPU (`RANKS=k`: per group of k GPUs, each copy one `torchrun`), plus `COTENANT` once per GPU when 5c/5d or the handoff's
Contention table say the copies are not the load (the model via `cotenant_loop.sh`,
else `transformer_hog` with the model's sizes), and prints `NATIVE_TOTAL=`. Run it
detached through `mad-run.sh` (`env-madengine.md`):

```bash
bash .claude/skills/mad-common/mad-run.sh start native-<k> -- \
  madengine nan exec --image <IMAGE> --gpus all --mount <ARTIFACT_DIR> \
    --env COPIES=3 [--env COTENANT='<co-tenant command>'] [--env K=V ...] -- \
    bash \$NAN_DOCTOR_RT/bin/native_run.sh <ARTIFACT_DIR>/native/repro_<k>.py
$LG job step5n <run-id> candidate <k> <what changed>
```

(Escape `\$NAN_DOCTOR_RT` so the host shell does not expand it. `--env` carries the env
from N1: amplifiers, precision flags, cache dirs.) Every copy dying at start-up with no
Python traceback is the environment, not the candidate: ledger the run invalid, compare
the container's env with the model run's, and re-run.

Wait once (rule 9, `mad-run.sh wait`); read `NATIVE_TOTAL=`, `HIT`, `NO_RESULT` and
`TRACEBACK` lines only.

## N5. Reduce while it reproduces

**One run tests up to eight changes**: `bin/native_variants.sh
<candidate> <variants file>` (via `madengine nan exec --gpus all -- bash \$NAN_DOCTOR_RT/bin/native_variants.sh ...`, mounted and detached as in N4) runs line *i* of the file on GPU *i*
(`--env RANKS=k`, a multi-rank candidate: on GPU group *i*, each copy one `torchrun`, at most 8/k lines),
`<copies>|<env>|<candidate args>|<optional co-tenant>`, and prints one
`VARIANT_TOTAL` per GPU (group). Put the unchanged candidate on GPU 0 (this run's own
reference rate), then controls and reductions -- a toggle (TF32, library), solo,
fewer co-tenants, a lighter co-tenant kind, fewer layers. A candidate whose copies
double as the load takes a mode flag instead (e.g. copy 0 runs the op, the others
the training step).

Keep a change only if it still hits (`n/N > 0` at the same load): drop components
(the `step8-fix.md` component table first), halve layers/steps/sizes, remove or lighten the co-tenant,
then lower copies per GPU. A clean variant bounds a rate, not a load -- run it at
least `3 / <the reference rate>` launches before calling it clean. Stop at the first of:

* a single op (or the op + its producer) reproduces -- done;
* every remaining removal kills it -- the boundary is the finding (what it needs);
* the step-1 budget.

A deterministic candidate (hits every run, also solo): one run per change.
**Ledger every candidate**: `cand <k>: <delta> -> n/N, copies x GPUs, load, host`.

## N6. Verdict

| verdict | when |
|---|---|
| `NATIVE_KERNEL_BUG` | hits, inputs finite, fp64 ref on the saved inputs `finite` -- or `WRONG_VALUE` (a finite signal: the op's output is off from a finite reference) |
| `NATIVE_UNVERIFIED` | hits, no fp64 reference (`unavailable`), or inputs not checked |
| `NATIVE_PROPAGATED` | hits, inputs already non-finite -- move the suspect upstream and go back to N3 |
| `NATIVE_FP32_OVERFLOW` | hits, fp64 ref `OUT_OF_RANGE` -- the true result does not fit the dtype; not a kernel bug |
| `NATIVE_MATH` | hits, fp64 ref `NON-FINITE` -- undefined arithmetic (name it: `0*inf`, `0/0`, ...), a model/numerics bug, not a kernel bug and not an overflow |
| `NATIVE_NOT_REPRODUCED` | budget spent, no candidate hit -- a bound: every candidate's `n/N`, load and host |

A candidate older than the `OUT_OF_RANGE` status printed `finite` for an overflow: before
calling it `NATIVE_KERNEL_BUG`, check the fp64 values at the non-finite positions against
the dtype's max (`torch.finfo`).

Ledger it with the minimal candidate's path, `n/N`, copies x GPUs, load, host,
and **the difference from the closest Ruled-out row** -- what the native candidate
had that the capsule replay did not is the trigger's best description (e.g.
"fails beside <the load the candidate added>; the closest capsule replay, <its
load> for <N> runs, never did"); `$LG end step5n`. Every hit goes to step 6 at once --
the reproducer is the deliverable whatever the verdict: `NATIVE_KERNEL_BUG` /
`NATIVE_UNVERIFIED` as a kernel reproducer; `NATIVE_MATH` / `NATIVE_FP32_OVERFLOW` as a
model-numerics reproducer whose README says plainly it is not a kernel bug (step 6
*Native reproducer*). Steps 7-8 follow for every verdict: after a `NATIVE_MATH` /
`NATIVE_FP32_OVERFLOW` package the arithmetic and the candidate's clean controls are the
mechanism's first evidence; step 7 traces where in the model the values come from
(`step7-rootcause.md` *Model / numerics NaN*). `NATIVE_NOT_REPRODUCED` goes
to step 9.
