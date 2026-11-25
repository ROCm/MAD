# Step 6: finish the reproducer tarball (not optional)

**"Finish" means reproducer generation is finished -- not that the NaN is fixed.**
Step 6 ends with a packaged reproducer; the NaN is still unexplained. Root-causing
(step 7) and fixing (step 8) are separate, budgeted steps -- on by default -- that start
from this tarball; the hand-over (*Then*) says so.

The one artifact a vendor can act on: capsule, scripts, one command -- no model,
dataset or checkout. **It is bound to the GPU arch it was captured on** (hipBLASLt
solutions and aiter/CK code objects are built per arch): the README names it, and a
`0/N` on another arch (MI300X vs MI355X) says nothing about the bug. **For every group that reproduced** (`KERNEL_BUG`,
`KERNEL_BUG_UNVERIFIED`), as soon as its 5b result lands -- and for every step-5N
hit: `NATIVE_KERNEL_BUG` / `NATIVE_UNVERIFIED`, and `NATIVE_MATH` / `NATIVE_FP32_OVERFLOW`
as a model-numerics reproducer (*Native reproducer*, below). G1 did
not reproduce: step 5N first (`step5n-native.md`); after a miss there, ledger that, go to step 9.
Build, verify and pack with `madengine nan package`: `step6-build.md` first.

After `madengine nan package --pack`, `repro-<G>/` has been built, verified from an
extracted copy (`step6-build.md`), given a `README.md` and packed; the ledger has the
tarball's path and `sha256`. Left to you:

1. **Confirm the README's `<!-- CHECK -->` line** against the `[C1]` walk and the
   verdict table (`step5-read.md`); the script proposes, it does not decide. Fix the
   sentence and drop the marker. If it flags a deterministic walk, resolve that first
   (`step5-replay-guardrail.md`) -- do not ship. `FP32_OVERFLOW?` (the `MODEL_NUMERICS`
   line; a `madengine nan readme` that prints `DO_NOT_SHIP` for it predates this rule):
   not a kernel bug, but a reproducer all the same -- ship it as a model-numerics one, the
   README's first sentence saying so (*Then*).
2. Add what only you know, briefly: where the capsule came from (model, run), groups
   folded into this one (same kernel, on what basis), a held-back loader path that
   `run.sh` must export (`env:` banner `held back:`), and for a Tensile
   `KERNEL_BUG` the `--bench` one-liner -- already in the README's `## Kernel` section
   when the ladder identified the kernel (`kernel.*.txt`); add it by hand only
   if that section is missing.
   **`SYNTHETIC-INCONSISTENT` in the `operands:` banner**: the README must say the
   tarball shows the NaN but cannot validate a fix, and name the consistent-operand
   check that can (`step5-read.md`).
3. Edited anything? Re-pack without rewriting the README (seconds with `pigz`; a real-values
   package without it takes minutes -- give the call a long timeout):
   `madengine nan readme --repro <ARTIFACT_DIR>/repro-<G> --repack --ledger <LEDGER> --step 'step6[<G>]'`
   (tars, prints and ledgers the new sha256; the report takes the latest line).
4. Tell the user path, size and the one command -- as *the reproducer is done*, not
   *the NaN is solved*. A no-chain capsule
   (`--chain-fallback`): the README must say the operands were never validated.
5. **A co-tenant reproducer** (`step5-pressure.md`): the README's contention paragraph
   says the replay alone does not reproduce and names the load. Add which loads ran
   clean (the 5d results: e.g. forward-only, one co-tenant process, no co-tenant) --
   they bound the trigger for the vendor -- and that the co-tenant itself runs the same
   kernel family if it does (a TF32 training step calls the F32X kernels too).

## Native reproducer (from step 5N)

No capsule and no `madengine nan package` -- you build it by hand (a manual step). Tag
`step6[native]`.

```
repro-native/
  run.sh  README.md  repro.py      # the minimal candidate, renamed
  failure_signature.py             # nan_doctor_rt/failure_signature.py (the hit's SIGNATURE line)
  cotenant/<file>                  # only if the reproducing load needs one
```

1. **Ship exactly what reproduced**: the minimal candidate as `repro.py`, and as
   `run.sh` a short script that starts `IMAGE` with docker on the extracted directory
   and runs a copy of `bin/native_run.sh` (read it from the runtime:
   `madengine nan exec --image <IMAGE> -- cat \$NAN_DOCTOR_RT/bin/native_run.sh`) with
   the reproducing defaults baked in (candidate `repro.py`, `COPIES`, `COTENANT` --
   shipped in `cotenant/`, not a repo path -- and env). Mirror the docker flags of the
   capsule reproducer's `run.sh` if one exists. A multi-rank candidate (a `torchrun`
   program) ships the same way: `RANKS=<world size>` makes `native_run.sh` run each copy
   as one `torchrun` per GPU group -- no runner of your own; `run.sh` keeps `RANKS=`
   overridable beside `COPIES=` and `NGPUS=`.
2. **Verify from an extracted copy** (`step6-build.md` *Verify from an extracted copy*):
   tar, extract under `<ARTIFACT_DIR>/verify-native/`, run *that* `run.sh` through
   `mad-run.sh` (from its own directory, in an empty environment: a vendor's `bash run.sh`).
   Pass = `NATIVE_TOTAL` `n/N > 0` near the ledger rate.
3. **README**, short: what it reproduces (op or region, shapes, dtypes, precision
   flags) and the verdict with the fp64 check -- for a finite signal (`fp64_ref=WRONG_VALUE`
   hits) the first sentence says it reproduces **wrong values, not a NaN**, and names the
   reference and the tolerance a hit exceeds; for `NATIVE_MATH` /
   `NATIVE_FP32_OVERFLOW` the first sentence says it is **not a kernel bug**: the
   arithmetic (`0 * rsqrt(0) = 0 * inf` in an eps-less norm on all-zero rows) or the
   overflow, and the clean control that removes it (e.g. the eps); an `* Image:` line with the image in
   backticks (the report reads it); the one command; the rate **with the load** (copies x GPUs,
   co-tenant) and every clean load and removal that killed it (N5 ledger rows -- they
   bound the trigger); why the capsule path failed and how the native script differs
   from it (`NATIVE_TRIGGER`, `native/handoff.md`'s Ruled-out rows, the N6 difference); that a single solo run alone means nothing if the load matters.
4. Pack and ledger:
   `madengine nan readme --repro <ARTIFACT_DIR>/repro-native --repack --ledger <LEDGER> --step 'step6[native]'`.
5. Tell the user path, size and the one command (reproducer done, NaN not yet
   root-caused -- or, for `NATIVE_MATH` / `NATIVE_FP32_OVERFLOW`, explained by the
   arithmetic, not fixed).

## Kernel capsule reproducer (from 5N's compiled track)

A kernel capsule that replays (`step5n-compiled.md` *Kernel capsule*: hits, inputs finite) is shipped
like a native reproducer -- it needs no model, and its inputs are the model's real values. Tag
`step6[kernel]`; when a hand-written candidate also hit, ship both (two packages). Built by hand:

```
repro-kernel/
  run.sh  README.md
  capsule/inductor_<pid>_<kernel>.pt     # the capsule
  source_graph.txt                       # Inductor's FX graph of the kernel's source nodes (C1's output code)
  replay_inductor_kernel.py  inductor_ref.py   # from nan_doctor_rt; import torch only
```

`run.sh` starts `IMAGE` on one GPU (the kernel replays alone; a contention-gated fault gets the
native package's `COPIES` and load instead) and runs `python3 replay_inductor_kernel.py
capsule/<capsule> --iters 1000`. The README states the reference verdict (the replay's `fp64_ref=`,
the kernel's fp64 twin; `unavailable`: eager / fp64 of the source nodes by hand,
`step5n-compiled.md`: `NATIVE_MATH` or the kernel). Pack, verify from an extracted copy under
`verify-kernel/` and report as items 2, 4 and 5 of *Native reproducer*. The kernel is recompiled
from its Triton source by **the image's** torch / triton.

## Then

Every reproducing group packed: **announce the hand-over** -- always, as its own block
(SKILL.md rule 12), combined with the path/size/command -- then go on to step 7 in the
same turn. Classify `<NAN_KIND>` first (`step7-rootcause.md` *Entry*) so the budgets quoted
fit it. **Print the block before any step-7 call** -- before `$LG start step7[<G>]`, the
`rootcause/` dir or the first experiment: it marks the boundary. **Ledger it** in the same
call as `$LG end step6[<G>]` (`$LG note step6[<G>] "HANDOVER announced: <tarball> -> step 7
(RC <RC_BUDGET>, FIX <FIX_BUDGET>)"`): text printed mid-turn can scroll off or fold under the
next tool output, so step 9 repeats it from this line (`step9-report.md` *Always report*).

```markdown
> **✅ Reproducer generated** -- `<tarball>` (<size>), run it with `<one command>`.
> It reproduces the NaN without the model; the NaN is **not** root-caused or fixed yet.
>
> **Next: root-causing, then fixing** (experimental) -- within <RC_BUDGET> for the root
> cause and <FIX_BUDGET> for the fix, on the reproducer only. Say *stop at the reproducer*
> to end here, or give other budgets.
```

Steps 7-8 turned off (step-1 Q4 `no`): the first two lines only, then step 9. Q4 "ask me
after 6" (ask-me mode): end the block with the question instead -- "root-cause and fix it
now?" with the budgets as defaults -- so a "yes" is a complete answer.
**Every verdict goes on to step 7**, a system bug and a model bug alike: a `KERNEL_BUG*`
reproducer, and a model / numerics one -- 5b `FP32_OVERFLOW?` (overflow, or undefined
arithmetic such as `0*inf`), 5N `NATIVE_MATH` / `NATIVE_FP32_OVERFLOW`. For a model / numerics
one the block's first line says so ("not a kernel bug: <the arithmetic>"); step 7 then finds
*why the model hands the op those values* (config, data, code) and step 8 fixes it in the
model (`step7-rootcause.md` *Model / numerics NaN*).
