# Step 8 (on by default, budgeted): fixing -- a loop over `<ROOT_CAUSE>`

> Experimental, interactive-only under MAD: the automated workflow never fixes anything; fixes are proposed, never applied to MAD or the image.

**Entered only with a valid `<ROOT_CAUSE>`** (`rootcause/ROOT_CAUSE.md`, all the
validity items of `step7-rootcause.md`). `NO_ROOT_CAUSE`, or step 7 skipped: no
step 8 -- `$LG skip step8 "no root cause"`, go to step 9. Ends with `<FIX>` (the
fix artifact) or `NO_FIX` once `<FIX_BUDGET>` is spent. Tag `step8[<G>]`.
Tooling: `nan_doctor_rt` modules `kernel_patch`, `fix_overlay`, `fix_values`, `attn_consistent_check`, `grad_trajectory`, `replay_nan_op`, run via `madengine nan exec --image <IMAGE> -- python3 -m nan_doctor_rt.<module> ...`, GPU jobs started with `mad-run.sh`.

## Entry

1. **Wanted?** Step-1 Q4, default **yes** (or its answer after step 6). Turned off ->
   `$LG skip step8 "<why>"`, go to step 9.
2. **`<FIX_BUDGET>`** -- already set at step 7's entry (default mode: the default, taken
   without asking; `step7-rootcause.md` *Entry*); set it here only if step 7's entry did
   not -- wall clock (default: 1 h for an env-setting workaround or a model / config fix on
   a deterministic reproducer; 4 h for a kernel patch) and
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
dispatch for that problem as a workaround (say it is one). **A setting that only routes
around the faulty code is a workaround, never the fix**, even when it is the library's own
switch: check which path the patched run takes (does the root cause's kernel / branch still
run?) and label it by that. **Never** a symptom mask
(`nan_to_num`, clamping the output, skipping the step): that is not a fix. Write
it as a patch file plus whatever applies it (a patched `.co` / library overlay via
`kernel_patch`, a source diff, an env setting). It is a proposal: do not apply it to
MAD or the image.

**A model / numerics root cause** (`<NAN_ORIGIN>` = `model`): the fix is in the model --
its source, config or input handling -- at the line(s) `ROOT_CAUSE.md` names, one change per
necessary condition it found; each alone is a fix, and the report says which is the real
one and which is defence in depth. E.g. a norm's `eps` restored to the architecture's
value; ids at or above the tokenizer size rejected at the input boundary; padded embedding
rows initialised (or excluded) instead of left zero; a scale or dtype that keeps the
reduction in range. Still **never** a symptom mask (`nan_to_num` on the output, skipping
the step). Write it as a source / config diff against a copy (`<ARTIFACT_DIR>/fix/patch/`),
never the user's tree. Validate it on the reproducer the way step 7's per-factor experiment
did (the same change applied to the capsule's operands or 5N's candidate: `NAN` without it,
`CLEAN` with it, outputs finite and matching the fp64 reference of the fixed computation),
then on the model (3b: the patched copy as its own run, stock beside it).

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
process, prints the env each saw; `PERF_CMD` for a timing per arm) -- `[iters]` from
`madengine nan power` (`step7-rootcause.md`).

**Model A/B** (one run fills the machine, so arms cannot share it): the model runs only
through `madengine run`, one job per invocation. Interleave stock and each arm as
sequential `mad-run.sh` runs, `ROUNDS` rounds, passing the arm's env or patched
source through `--additional-context`/the model's own settings, one line per run
(arm, env, last step, NaN signal, spikes, exit code). **Check the fix landed in every fix
run**: its marker line (`[fix] inductor sqrt: ...`) and, for a compiled fix, the output code
(`TORCH_LOGS=output_code`); `inductor_override` stops the process on `[fix] ... NOT applied`
(the run is an error, not a stock verdict). An env value reaches the container as given in
`docker_env_vars` -- no shell quoting inside the JSON value. Keep run logs and outputs of the
A/B outside `fix/` (`<ARTIFACT_DIR>/fix_ab/`), so the package's tarball holds none. An aiter
`.co` fix sources the kept row's relocatable `env.sh` before the model command, with
`AITER_LOG_LEVEL=INFO` in every arm, so each run proves which kernel it ran: `grep -l
'LoadKernel.*fix_work/kept/<row>/overlay'` (older aiter: `hipModuleLoad: <overlay>`) must list
every fix run -- one without it ran stock.

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
     symlinks into the image) and `--repack` would tar it; ship `overlay.json`,
     `kernel.diff` (it writes it from the read-back: `DIFF=`), `kernel_name.txt` and the
     rebuild command in `fix/patch/` instead.
  2. `nan_doctor_rt.fix_values <capsule> --overlay <OVERLAY> [--synthesize] --json
     fix/values.json`: stock vs patched in fresh processes, solo -- fp64 error, bitwise
     equality with stock (a wait/barrier-only fix should be bit-identical), cost per
     launch, and the libpath the patched child saw. `VALUES=OK`. **`--synthesize` for
     a shapes-only capsule** (the capture default; without it the tool refuses one).
     The cost comparison is one solo run each, stock first -- report it as "no
     regression seen", not as a speed-up.
  **aiter / CK** (`fix_overlay` and `fix_values` are hipBLASLt-only): the fix
  is the step-7 sweep's `--mutate-keep-fix` copy, `fix_work/kept/<row>/env.sh` (no copy:
  re-run that row's sweep with `--mutate-keep-fix`). Read-back = the row's `.diff`.
  **aiter, the package**: copy the row's `patched/` into `fix/patch/<name>/` -- a short name, no
  `@` (sweep rows are `vmcnt0_each@0x..@<call>`; e.g. `F1-vmcnt18`) -- and the arms are
  `bin/aiter_fix_validate.sh <capsule> <rel .co path> <NAME>=<patched .co> ... -- <5b load
  flags>` (stock + one GPU per fix, each fix served by `aiter_overlay.sh` from the image's own
  aiter, its load checked from aiter's log; `FIX_VALIDATE=PASS|FAIL`) -- also as
  `make_fix_validate --cmd` (its docstring). A fix not proposed by a mutator (a hand edit of
  the kept `.co`) goes the same way; `fix.changes` spells it `<instruction edit>@<addr>`.
  Values: the 5b chain command, **solo**, stock with `--chain-save-outputs
  fix_work/stock.pt`, then under the fix (`source env.sh` in the same command) with
  `--chain-compare-outputs fix_work/stock.pt` -- `[values] IDENTICAL` is the pass for a
  wait/barrier-only fix (a value-checked op is compared on what its consumer reads).
  **A non-deterministic fault can make the saved stock run itself a bad one**, and a correct
  fix then reads `DIFFER`: save stock with `--chain-iters 1` and keep it only if that run's
  walk and value check came back clean (no `WRONG VALUE`, finite) -- else save again; a
  `DIFFER` under the fix is a finding only against such a run. Keep `fix/` to what ships:
  work files (outputs, logs, overlays) go to `fix_work/`. Then the same with `--contend <C-1>
  --chain-repeat-check`: under the fix every op should repeat (co-tenants inherit
  the env, so all processes are patched -- the strong case, beside item 3's
  conservative one). **Loaded** = the shim log (`NAN_DOCTOR_HSA_LOG`) has a
  `PATCHED` line (same-size splice) **or `REPLACED`** (a grown kernel, e.g. any
  `sbarrier_*`); for aiter, the `[aiter] hipModuleLoad: <overlay path>` line (newer aiter:
  `[aiter] LoadKernel: <sym> hsaco: <overlay path>`).
  Neither = the run used the stock kernel: void, not a pass.
  3. The NaN rate under load: the step-7 sweep command (same mapping), its
     `--mutate-only` the fix's mutator and a fresh `--mutate-report` under `fix_work/` --
     its baseline is the control, its row the patched rate, at the sweep's load. The
     co-tenants there run the **stock** kernel (only the measured process is
     patched): the conservative case; say so. Size `--chain-iters` for >= 3 expected
     hits (aim ~30+).

* **A COLLECTIVE reproducer** (`repro-<G>/` replays member-rank capsules, `step4-distributed.md`
  *Synchronous collectives*): every env candidate is an arm of `bin/collective_arms.sh
  <capsule dir> <arms file>`, stock first and last. Its `ARM ... path=` column is the check that
  a fix keeps the implementation it claims to fix -- an arm whose path changed is a workaround
  (a library setting can send the call to another implementation of the same op, which says
  nothing about the one that failed).
* **Control in the same job**: the unpatched reproducer still `NAN` (else the run
  could not have caught it -- the result is void, not a pass). **A clean control off the 5b
  host is a node-bound reproducer**, not a fixed bug: a library GEMM's bad shapes can move from
  node to node (stream-K: a captured M reproduced only on the node 5b ran on; other nodes
  computed it correctly). Re-run on that host (`<LEDGER>`); unavailable: validate with a check
  that does not depend on the node -- a property of the fix rather than one shape, e.g.
  `gemm_shape_probe sweep` over the band (bad sizes in stock, none under the fix) -- ship it in
  `validate.sh` (`--tool`), and say in `fix/README.md` and the report that the capsule arm is
  node-bound.
* **Patched: CLEAN with power** -- >= 3 expected hits at the reproducer's rate;
  fewer is `UNDERPOWERED`, raise the count, never a pass.
* **Correct values**: patched outputs match the fp64 reference (or the unpatched
  kernel's finite runs) within tolerance -- `WRONG` otherwise. Report the error over
  normal-range reference values and over subnormal ones separately: an fp32 value
  that is itself subnormal carries only a few significant bits, so a correct op on
  it differs from fp64 by up to ~1e-2 relative -- not `WRONG`. A hand-written
  check prints `n/a (none sampled)` for a class with no samples, never `nan` (it reads
  as a failure), and never takes `.max()` of an empty selection.
* Note the cost (per-launch time vs unpatched); a large regression is reported, not
  a failure.
* `.diff` read back for a binary patch (identical = never landed).

**3b. Validate on the model, every candidate -- not only workarounds.** A clean
reproducer proves the mechanism is fixed. It does not prove the model is NaN-free.
Run the model-level A/B of the workaround paragraph above (fix vs stock
interleaved, both counted by `<NAN_SIGNAL>`, the toggle's load verified per
attempt: a print marker in the patched copy, grepped from the run's logs). Arm the
fix attempts with `NAN_DOCTOR_GRADLOG` (`nan_doctor_rt.grad_trajectory`) **when the
trainer already calls `after_backward`** (step 1 wired it into a working copy), so a NaN
under the fix comes with its trajectory. Not wired (a stock trainer, the torchtitan
wrapper): skip it and say so -- don't edit the model for it up front; only if a fix arm
does NaN, wire it into a working copy for that re-run. A NaN under the fix is not a failed fix
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

  **`validate.sh`: generate it, don't copy an earlier run's** (one had a hard-coded capsule
  name and a co-tenant that did not apply): `madengine nan exec --image <IMAGE> --mount
  <ARTIFACT_DIR> -- python3 -m nan_doctor_rt.make_fix_validate --fix <ARTIFACT_DIR>/fix --repro
  <ARTIFACT_DIR>/repro-<G> --cmd '<the arms>' [--gpus all]` writes it with the reproducer's own
  runtime lines and ships `fix/scripts/` (the reproducer's tools, `fix_overlay`, `fix_values`,
  `env_workaround_validate.sh`, `capsule_operands`, `inductor_ref`; anything else a `values.py`
  imports: `--tool <file>`); the reproducer is found as the sibling `../repro-<G>` (both
  tarballs extracted side by side) or `REPRO=`. Its docstring has a `--cmd` per fix kind.

  Pack it like a reproducer:
  `madengine nan readme --repro <ARTIFACT_DIR>/fix --repack --ledger <LEDGER> --step 'step8[<G>]'`.
  **Then run it as shipped**, like step 6's reproducer: `bash -n validate.sh`, extract
  the tarball under `<ARTIFACT_DIR>/verify-fix/` beside an extracted reproducer (or set
  `REPRO=`) and run its `validate.sh` there, in an empty environment from its own
  directory: `bash .claude/skills/mad-common/mad-run.sh start verify-fix -- bash
  <ARTIFACT_DIR>/verify-fix/fix/validate.sh`. Ledger its result beside the sha256. A reused
  `validate.sh` too: an earlier investigation's had a syntax error nobody ran.
* **Budget left** -> next iteration.
* **Budget spent** -> `<FIX>` = `NO_FIX`: `fix/README.md` still lists every
  candidate tried and why each failed.

## Exit

`$LG end step8[<G>]`; ledger `<FIX>` (tarball path + sha256, or `NO_FIX`). Tell the
user in two lines, then step 9.
