# Step 8 (optional, budgeted): fixing -- a loop over `<ROOT_CAUSE>`

> Experimental, interactive-only under MAD: the automated workflow never fixes anything; fixes are proposed, never applied to MAD or the image.

**Entered only with a valid `<ROOT_CAUSE>`** (`rootcause/ROOT_CAUSE.md`, all the
validity items of `step7-rootcause.md`). `NO_ROOT_CAUSE`, or step 7 skipped: no
step 8 -- ledger `step8 SKIP no root cause`, go to step 9. Ends with `<FIX>` (the
fix artifact) or `NO_FIX` once `<FIX_BUDGET>` is spent. Tag `step8[<G>]`.
Tooling: `nan_doctor_rt` modules `kernel_patch`, `fix_overlay`, `fix_values`, `attn_consistent_check`, `grad_trajectory`, `replay_nan_op`, run via `madengine nan exec --image <IMAGE> -- python3 -m nan_doctor_rt.<module> ...`, GPU jobs started with `mad-run.sh`.

## Entry

1. **Wanted?** Step-1 Q4 (or its answer after step 6). No -> ledger `step8 SKIP
   <why>`, go to step 9.
2. **Ask for `<FIX_BUDGET>`** -- skip this if step 7's entry already asked it (both
   budgets in one question, `step7-rootcause.md` *Entry*) -- wall clock (default: 1 h for an
   env-setting workaround on a deterministic reproducer; 4 h for a kernel patch) and
   GPU-hours; ledger it. The budget counts the loop's own time (`$LG` `other` +
   `compute`), not time spent waiting for the user. Quote "remaining" that way when
   offering an extension; a user away overnight must not appear to have spent it.
3. `mkdir -p <ARTIFACT_DIR>/fix`; `$LG start step8[<G>]`.

## The loop body (one iteration = `fix/iter-<k>.md`)

**1. Gather.** Re-read `<LEDGER>`'s `step8` lines, `rootcause/ROOT_CAUSE.md`,
`fix/candidates.md` and the last `iter-<k-1>.md`. Open `iter-<k>.md` with:

* **Status** -- elapsed vs `<FIX_BUDGET>`, jobs in flight.
* **Learned last iteration** -- what the last candidate did (still NaNs, NaN gone
  but wrong values, too slow, didn't build or didn't apply).
* **Excluded** -- fix approaches ruled out and why.
* **This iteration** -- the candidate to try and why it should work.

`fix/candidates.md`: `id | change | where | result (NAN / CLEAN / WRONG / ERROR /
UNDERPOWERED) | evidence path`.

**2. Propose a fix (native LLM).** Fix the mechanism `<ROOT_CAUSE>` names, at the
place it names -- the smallest change that removes the cause, e.g. the missing
`s_barrier` / `s_waitcnt` in the kernel (a CLEAN `sbarrier_only_*` row is already a
fix's shape), a guard or wider accumulator in the source, a different solution or
dispatch for that problem as a workaround (say it is one). **Never** a symptom mask
(`nan_to_num`, clamping the output, skipping the step): that is not a fix. Write
it as a patch file plus whatever applies it (a patched `.co` / library overlay via
`kernel_patch`, a source diff, an env setting). It is a proposal: do not apply it to
MAD or the image.

**Model-level workaround: turn the component off.** Cheap, needs no kernel
change, and often what the user can ship today -- but a workaround, not a fix:
say so in `fix/README.md`. Candidate when `<ROOT_CAUSE>` sits in a component below
(a list on purpose: add a row to extend it):

| component | off | on |
|---|---|---|
| HipBlasLt | `TORCH_BLAS_PREFER_HIPBLASLT=0` | `TORCH_BLAS_PREFER_HIPBLASLT=1` |
| hipBLASLt stream-K XCD remap (unwritten tiles) | `TENSILE_FIXED_WGMXCC=1` (small cost on the affected GEMM: measure it) | unset |
| hipBLASLt stream-K dynamic grid | `TENSILE_STREAMK_DYNAMIC_GRID=0` (measure the slowdown) | unset |
| CK / AITER flash attention (SDPA) | `TORCH_ROCM_FA_PREFER_CK=0` | `TORCH_ROCM_FA_PREFER_CK=1` |
| a Triton op Inductor emits (e.g. `tl.sqrt_rn` NaN on fp32 subnormals) | manual: override that `TritonOverrides` method (any method; private API) in a module on `PYTHONPATH`, e.g. replace `tl.sqrt_rn(` with `libdevice.sqrt(` | unset |
| the compiled module as a whole | the model's compile switch for that module, `TORCHDYNAMO_DISABLE=1` | the stock setting |

`TORCH_BLAS_PREFER_HIPBLASLT=0` sends plain GEMMs to rocBLAS, which for bf16/fp16 runs
the same Tensile kernels: test it like any candidate, don't assume it avoids the
kernel (`preferred_blas_library("cublas")` fails identically). A step-7 causal knob
is the first candidate: it is already the intervention that made the reproducer
clean.

**Torch.compile arms: one inductor cache per arm, and read the generated code.**
Processes sharing a `TORCHINDUCTOR_CACHE_DIR` reuse each other's compiled kernels: a
stock run after a patched one in the same container can read the patched kernel and
report clean where the stock control NaNs every time. Give every arm its own cache
dir (a separate `madengine run`, or `--env TORCHINDUCTOR_CACHE_DIR=<dir>` per
`madengine nan exec`), and grep `TORCH_LOGS=output_code` of one patched trial for
the new call and the absence of the old -- a `[fix]` marker alone shows the patch
loaded, not that the kernel used it.

**An env workaround on the reproducer: never `--restore-env`.** It re-applies the
capture's environment, which holds the very variable the workaround flips. Run
both arms with the env set explicitly (no `--restore-env`), the control first, in
one job: `$NAN_DOCTOR_RT/bin/env_workaround_validate.sh <capsule> "VAR=VALUE ..."
[iters]` inside the container via `madengine nan exec` (stock + each arm in its own
process, prints the env each saw; `PERF_CMD` for a timing per arm).

**Model A/B** (one run fills the machine, so arms cannot share it): the model runs only
through `madengine run`, one job per invocation. Interleave stock and each arm as
sequential `mad-run.sh` runs, `ROUNDS` rounds, passing the arm's env or patched
source through `--additional-context`/the model's own settings, one line per run
(arm, env, last step, NaN signal, spikes, exit code).

Validate it on the reproducer like any candidate (step 3 below), then on the
model: `<FULL_MODEL_RUN_SCRIPT>` `N ~ 3m/k` times on, `N` off, from
`<FULL_MODEL_RATE>` (state `N` first; *unmeasured* = run the on arm as the
control), counted by `<NAN_SIGNAL>`, arms interleaved. **Verify the toggle landed**
in one trial per arm (`env-madengine.md`) -- a dropped var reads as a clean pass.
Report both counts (7/10 vs 0/10 concludes; 1/10 vs 0/10 means raise `N`) and the
off arm's step time.

**3. Validate on the step-6 reproducer, with the fix applied**, via `madengine nan
exec` started by `mad-run.sh` (`$LG job`), at the reproducer's own load (same
copies, co-tenant, `--contend`, `--restore-env`):

* **A reproducer with `operands: SYNTHETIC-INCONSISTENT` cannot validate a fix**: it
  NaNs on a correct kernel too. Validate on consistent operands instead
  (`attn_consistent_check --save stock.pt`, then with the fix `--compare
  stock.pt`: all finite and `IDENTICAL-on-stock-finite`), and ship that check as the
  fix's `validate.sh`.

* **Library-overlay fixes: `--restore-env` silently removes the fix.** The capsule's
  env namespace covers `HIPBLAS*` and `TENSILE_*`, so `restore_env_dict()` UNSETS a
  `HIPBLASLT_TENSILE_LIBPATH` the capture did not have -- a replay with the overlay in
  `--env`, or any `replay_nan_op --restore-env` under it, would run the stock kernel
  and "the fix still NaNs". The replay holds both overlay variables back (the `env:`
  banner lists them as `held back`, and a `libpath:` line names the overlay
  hipBLASLt was given) -- **check that line in every patched run**; no line or
  `stock` = the fix was not in the process.
* **Binary kernel fix (a CLEAN mutation row)** -- the tools, in one job:
  1. `nan_doctor_rt.fix_overlay --out <ARTIFACT_DIR>/fix-overlay-build
     --kernel-file <kernel.*.txt> --selector <sel> --payload <payload> [--addr ...]
     --expect-sites <n>`: the same selector/payload registries as the sweep (so the
     fix has the CLEAN row's bytes), built on a path visible to later runs (`--mount`;
     not a container-local temp dir), and read back -- `READBACK=OK` only if exactly
     the selected sites changed. It needs no GPU and takes about a minute. **Build it
     outside `fix/`**: the overlay is the whole library dir (~400 MB, thousands of
     symlinks into the image) and `--repack` would tar it; ship `overlay.json`, the
     `.diff`, the kernel name and the rebuild command in `fix/patch/` instead.
  2. `nan_doctor_rt.fix_values <capsule> --overlay <OVERLAY> [--synthesize] --json
     fix/values.json`: stock vs patched in fresh processes, solo -- fp64 error, bitwise
     equality with stock (a wait/barrier-only fix should be bit-identical), cost per
     launch, and the libpath the patched child saw. `VALUES=OK`. **`--synthesize` for
     a shapes-only capsule** (the capture default; without it the tool refuses one).
     The cost comparison is one solo run each, stock first -- report it as "no
     regression seen", not as a speed-up.
  **aiter / CK** (`fix_overlay` and `fix_values` are hipBLASLt-only): the fix
  is the step-7 sweep's `--mutate-keep-fix` copy, `fix/kept/<row>/env.sh` (no copy:
  re-run that row's sweep with `--mutate-keep-fix`). Read-back = the row's `.diff`.
  Values: the 5b chain command, **solo**, stock with `--chain-save-outputs
  fix/stock.pt`, then under the fix (`source env.sh` in the same command) with
  `--chain-compare-outputs fix/stock.pt` -- `[values] IDENTICAL` is the pass for a
  wait/barrier-only fix. Then the same with `--contend <C-1>
  --chain-repeat-check`: under the fix every op should repeat (co-tenants inherit
  the env, so all processes are patched -- the strong case, beside item 3's
  conservative one). **Loaded** = the shim log (`NAN_DOCTOR_HSA_LOG`) has a
  `PATCHED` line (same-size splice) **or `REPLACED`** (a grown kernel, e.g. any
  `sbarrier_*`); for aiter, the `[aiter] hipModuleLoad: <overlay path>` line.
  Neither = the run used the stock kernel: void, not a pass.
  3. The NaN rate under load: the step-7 sweep command (same mapping), its
     `--mutate-only` the fix's mutator and a fresh `--mutate-report` under `fix/` --
     its baseline is the control, its row the patched rate, at the sweep's load. The
     co-tenants there run the **stock** kernel (only the measured process is
     patched): the conservative case; say so. Size `--chain-iters` for >= 3 expected
     hits (aim ~30+).

* **Control in the same job**: the unpatched reproducer still `NAN` (else the run
  could not have caught it -- the result is void, not a pass).
* **Patched: CLEAN with power** -- >= 3 expected hits at the reproducer's rate;
  fewer is `UNDERPOWERED`, raise the count, never a pass.
* **Correct values**: patched outputs match the fp64 reference (or the unpatched
  kernel's finite runs) within tolerance -- `WRONG` otherwise. Report the error over
  normal-range reference values and over subnormal ones separately: an fp32 value
  that is itself subnormal carries only a few significant bits, so a correct op on
  it differs from fp64 by up to ~1e-2 relative -- not `WRONG`.
* Note the cost (per-launch time vs unpatched); a large regression is reported, not
  a failure.
* `.diff` read back for a binary patch (identical = never landed).

**3b. Validate on the model, every candidate -- not only workarounds.** A clean
reproducer proves the mechanism is fixed. It does not prove the model is NaN-free.
Run the model-level A/B of the workaround paragraph above (fix vs stock
interleaved, both counted by `<NAN_SIGNAL>`, the toggle's load verified per
attempt: a print marker in the patched copy, grepped from the run's logs). Arm the
fix attempts with `NAN_DOCTOR_GRADLOG` (`nan_doctor_rt.grad_trajectory`), so a
NaN under the fix comes with its trajectory. A NaN under the fix is not a failed fix
if the reproducer passes: capture it (`NAN_CAPTURE_*` plus the fix) to see which op
now goes non-finite. If that is a correct reduction over exploding finite values
(`capsule_info` prints `overflow expected`), it is a separate, upstream
divergence: report it as such. Read `grad_trajectory <logs>` (which module
grows, from which step, weights moving or not) before offering the user more
budget for it.

**4. Decide.** Ledger one line: `step8[<G>] iter <k>: <result>; <path>`.

* **All checks pass** (reproducer and 3b) -> stop. Build the `<FIX>` artifact:

  ```
  fix/
    README.md        # root cause (one paragraph, link ROOT_CAUSE.md), the change,
                     # how to apply, validation: control n/N, patched 0/N with
                     # expected hits, load, value check, cost
    patch/           # the diff / env setting; for an overlay: overlay.json, the
                     # .diff, kernel_name.txt -- not the built overlay
    scripts/         # the tool versions validate.sh needs (nan_doctor_rt modules)
    validate.sh      # one command: reproducer with and without the fix
    iter-*.md  candidates.md
  ```

  Pack it like a reproducer:
  `madengine nan readme --repro <ARTIFACT_DIR>/fix --repack --ledger <LEDGER> --step 'step8[<G>]'`.
* **Budget left** -> next iteration.
* **Budget spent** -> `<FIX>` = `NO_FIX`: `fix/README.md` still lists every
  candidate tried and why each failed.

## Exit

`$LG end step8[<G>]`; ledger `<FIX>` (tarball path + sha256, or `NO_FIX`). Tell the
user in two lines, then step 9.
