# Step 6: finish the reproducer tarball (not optional)

**"Finish" means reproducer generation is finished -- not that the NaN is fixed.**
Step 6 ends with a packaged reproducer; the NaN is still unexplained. Root-causing
(step 7) and fixing (step 8) are separate, optional, budgeted steps that start from
this tarball; say so when you hand it over.

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
   sentence and drop the marker. If it flags a deterministic walk or `FP32_OVERFLOW?`,
   resolve that first (`step5-replay-guardrail.md`) -- do not ship.
2. Add what only you know, briefly: where the capsule came from (model, run), groups
   folded into this one (same kernel, on what basis), a held-back loader path that
   `run.sh` must export (`env:` banner `held back:`), and for a Tensile
   `KERNEL_BUG` the `--bench` one-liner -- already in the README's `## Kernel` section
   when the ladder identified the kernel (`kernel.*.txt`); add it by hand only
   if that section is missing.
   **`SYNTHETIC-INCONSISTENT` in the `operands:` banner**: the README must say the
   tarball shows the NaN but cannot validate a fix, and name the consistent-operand
   check that can (`step5-read.md`).
3. Edited anything? Re-pack without rewriting the README:
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
  cotenant/<file>                  # only if the reproducing load needs one
```

1. **Ship exactly what reproduced**: the minimal candidate as `repro.py`, and as
   `run.sh` a short script that starts `IMAGE` with docker on the extracted directory
   and runs a copy of `bin/native_run.sh` (read it from the runtime:
   `madengine nan exec --image <IMAGE> -- cat \$NAN_DOCTOR_RT/bin/native_run.sh`) with
   the reproducing defaults baked in (candidate `repro.py`, `COPIES`, `COTENANT` --
   shipped in `cotenant/`, not a repo path -- and env). Mirror the docker flags of the
   capsule reproducer's `run.sh` if one exists.
2. **Verify from an extracted copy** (`step6-build.md` *Verify from an extracted copy*):
   tar, extract under `<ARTIFACT_DIR>/verify-native/`, run *that* `run.sh` through
   `mad-run.sh`. Pass = `NATIVE_TOTAL` `n/N > 0` near the ledger rate.
3. **README**, short: what it reproduces (op or region, shapes, dtypes, precision
   flags) and the verdict with the fp64 check -- for `NATIVE_MATH` /
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

## Then

Every reproducing group packed: go to step 7 if the user wanted root-causing (step-1
Q4; ask now -- for 7 and 8 both -- if they said "ask me after 6"), else step 9. Offer
the step-7 and step-8 budget defaults in the same question (`step7-rootcause.md`
*Entry* item 2), so a "yes" is a complete answer.
A `NATIVE_MATH` / `NATIVE_FP32_OVERFLOW` package goes to step 9 directly: 7-8 hunt kernel
faults, and 5N's controls already show the mechanism (say so instead of asking).
