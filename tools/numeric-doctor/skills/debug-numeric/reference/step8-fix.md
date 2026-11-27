# Step 8 (on by default, budgeted): fixing -- a loop over `<ROOT_CAUSE>`

**Entered only with a valid `<ROOT_CAUSE>`** (`rootcause/ROOT_CAUSE.md`, all the
validity items of `step7-rootcause.md`). `NO_ROOT_CAUSE`, or step 7 skipped: no
step 8 -- `$LG skip step8 "no root cause"`, go to step 9. Ends with `<FIX>` (the
fix package: **every** fix and workaround found, each labelled, *Three kinds of fix*) or
`NO_FIX` once `<FIX_BUDGET>` is spent. Tag `step8[<G>]`.

## Three kinds of fix -- look for all three, ship all that exist

A user wants the change with the **least reach and the least cost** they can apply today,
and the real fix for whoever owns the faulty code. One working workaround is not the end
of step 8. Keep one id per candidate in `fix/candidates.md` and aim for one of each kind:

| kind | id | what | e.g. (2026-11-26: TunableOp's mixed-fp8 `_scaled_mm` decoded each operand in the other's format) |
|---|---|---|---|
| **real fix** | `R<n>` | the change to the faulty code `ROOT_CAUSE.md` names (its `location.source` / kernel / sites), in that code's own repo and language | the swapped template arguments in PyTorch's `ScaledBlas.cpp` TunableOp dispatch, as a diff against the image's torch commit |
| **targeted workaround** | `T<n>` | routes **only the affected calls** around the faulty code: one problem key (shape / dtype / layout), one call site or module, one tuned entry -- everything else keeps its stock path and speed | the tuned-results rows of the affected GEMMs (every mixed-fp8 `ScaledGemmTunableOp` row the bug reaches, not only the one that NaN'd) set to `Default`; or the library's own per-candidate validation at tuning time (`PYTORCH_TUNABLEOP_NUMERICAL_CHECK=<atol>_<rtol>`) |
| **broad workaround** | `B<n>` | switches the component off for the whole process | `PYTORCH_TUNABLEOP_ENABLED=0`; the component table below |

* **Narrowest first, but the broad one early.** A broad workaround is usually the step-7 causal
  knob and validates in one job: take it first, so the user has *something* safe, then spend the
  rest of `<FIX_BUDGET>` on T and R. Never stop at B with budget left.
* **Scope a targeted workaround by the root cause's own conditions** (`ROOT_CAUSE.md`
  *Conditions*): the set of calls it reaches is what the workaround must cover -- list them
  (`rootcause/` scope experiment: which shapes / entries / modules compute wrong, not only which
  NaN'd) and cover all of them, nothing more. Say what it leaves untouched and what it does not
  cover (a shape tuned later, a new module): a frozen tuned file needs re-editing after a re-tune;
  a tuning-time check does not.
* **Measure the cost of every workaround** on the model (step time vs stock, from the A/B's
  `s/step`) and on the reproducer (per-launch time); quote it beside its scope ("4 backward GEMM
  shapes on `Default`, step time +0%" vs "every GEMM untuned, +X%"). A workaround whose scope you
  cannot state is reported as broad.
* **The real fix ships even when it cannot be applied here** -- a C++ change in PyTorch / a
  library that needs a rebuild the budget or image cannot give, a closed vendor kernel. Write it
  as a minimal diff against the **exact version the image runs** (`torch.version.git_version`,
  the library's commit or package version; fetch that file from the project's repo with
  `WebFetch` / `git` when it is not in the image), with the file, function and why the change
  removes the mechanism, and how to build and apply it. Then validate what you can, in this
  order, and label it by the strongest that ran:
  1. `validated` -- built and run on the reproducer and the model (a pure-Python change, a
     patched `.co`, a library overlay: no rebuild of the world);
  2. `checked by equivalence` -- the patched logic exercised another way on the reproducer's
     operands (the op called with the arguments the fixed code would pass; the corrected branch
     taken through a public API; a numerical emulation of the fixed kernel) -- say what that does
     and does not prove;
  3. `not validated here -- please try` -- shipped with the exact command that would validate it
     (`validate.sh`'s arm, the build line), never claimed as tested.
  A real fix is never replaced by a workaround in the report, and a workaround is never called
  the fix (*2. Propose* below).
* **No real fix possible** (closed kernel, no source): say so in its row and give the vendor
  what they need to fix it (the reproducer, kernel name, the bench line).

## Entry

In the step-7/8 fork (`step6-finish.md` *Then*) every stop for the user ends the fork with a
`NEEDS_USER:` line, as in step 7 (`step7-rootcause.md`, *Running as the step-7/8 fork*).

1. **Wanted?** Step-1 Q4, default **yes** (or its answer after step 6). Turned off ->
   `$LG skip step8 "<why>"`, go to step 9.
2. **`<FIX_BUDGET>`** -- already set at step 7's entry (default mode: the default, taken
   without asking; `step7-rootcause.md` *Entry*); set it here only if step 7's entry did
   not -- wall clock (default: 1 h for an env-setting workaround or a model / config fix
   on a deterministic reproducer -- 2026-11-09: 14 min incl. a 9-min model A/B;
   4 h for a kernel patch) and node-hours; ledger it.
   The budget counts the loop's own time (`$LG` `other` + `compute`), not time spent
   waiting for the user. Quote "remaining" that way when offering an extension; a
   user away overnight must not appear to have spent it.
3. `mkdir -p <ARTIFACT_DIR>/fix`; `$LG start step8[<G>]`.

## The loop body (one iteration = `fix/iter-<k>.md`)

**1. Gather.** Re-read `<LEDGER>`'s `step8` lines, `rootcause/ROOT_CAUSE.md`,
`fix/candidates.md` and the last `iter-<k-1>.md`. Open `iter-<k>.md` with:

* **Status** -- elapsed vs `<FIX_BUDGET>`, jobs in flight.
* **Learned last iteration** -- what the last candidate did (still NaNs, NaN gone
  but wrong values, too slow, didn't build or didn't apply).
* **Excluded** -- fix approaches ruled out and why.
* **This iteration** -- the candidate to try and why it should work.

`fix/candidates.md`: `id (R/T/B<n>) | kind | change | where | scope (what it reaches) | result
(NAN / CLEAN / WRONG / ERROR / UNDERPOWERED / NOT_VALIDATED) | cost | evidence path`.

**2. Propose a fix (native LLM).** Fix the mechanism `<ROOT_CAUSE>` names, at the
place it names -- the smallest change that removes the cause, e.g. the missing
`s_barrier` / `s_waitcnt` in the kernel (a CLEAN `sbarrier_only_*` row is already a
fix's shape), a guard or wider accumulator in the source, a different solution or
dispatch for that problem as a workaround (say it is one). **A setting that only routes
around the faulty code is a workaround, never the fix**, even when it is the library's own
switch: check which path the patched run takes (does the root cause's kernel / branch still
run?) and label it by that -- a library setting shipped as "the fix" can simply route the
call around the faulty kernel. **Never** a symptom mask
(`nan_to_num`, clamping the output, skipping the step): that is not a fix. Write
it as a patch file plus whatever applies it (a patched `.co` / library overlay via
`kernel_patch.py`, a source diff, an env setting).

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
then on the model (3b: the patched copy as its own spec line or arm, stock beside it).

**Targeted workaround: narrow the scope to the affected calls.** Before (or right after)
the broad switch below, find the narrowest handle the component offers, in this order: a
**per-problem** entry (a tuned-results row, a solution override for one GEMM key, an
exclusion list), a **per-call-site** change in a working copy of the model (one module
excluded from a conversion -- e.g. a `module_filter_fn` --, a context manager around one
call, a dtype at one site), then a **per-op-class** switch (one op family, one dtype
pair). Each is a candidate like any other: validated on the reproducer (stock arm still
`NAN`, the workaround arm clean, values correct) and on the model, cost measured.

**Broad workaround: turn the component off.** Cheap, needs no kernel
change, and often what the user can ship today -- but a workaround, not a fix, and the
widest one: it changes every call of the component, so name what else it slows or changes
and its cost. Say so in `fix/README.md`. Candidate when `<ROOT_CAUSE>` sits in a component below
(a list on purpose: add a row to extend it):

| component | off | on |
|---|---|---|
| HipBlasLt | `TORCH_BLAS_PREFER_HIPBLASLT=0` | `TORCH_BLAS_PREFER_HIPBLASLT=1` |
| hipBLASLt stream-K XCD remap (unwritten tiles,) | `TENSILE_FIXED_WGMXCC=1` (~1% on the affected GEMM) | unset |
| hipBLASLt stream-K dynamic grid () | `TENSILE_STREAMK_DYNAMIC_GRID=0` (+22% there) | unset |
| CK / AITER flash attention (SDPA) | `TORCH_ROCM_FA_PREFER_CK=0` | `TORCH_ROCM_FA_PREFER_CK=1` |
| a Triton op Inductor emits (e.g. `tl.sqrt_rn` NaN on fp32 subnormals) | `PYTHONPATH=<REPO_ROOT>/scripts/inductor_override NAN_FIX_INDUCTOR_OVERRIDES='sqrt:tl.sqrt_rn(=libdevice.sqrt('` (any `TritonOverrides` method; private API) | unset |
| the compiled module as a whole | the model's compile switch for that module (`--compile-layers`), `TORCHDYNAMO_DISABLE=1` | the stock setting |
| PyTorch TunableOp (a tuned GEMM solution) | `PYTORCH_TUNABLEOP_ENABLED=0`; targeted: the affected rows of the results file -> `Default`, or `PYTORCH_TUNABLEOP_NUMERICAL_CHECK=<atol>_<rtol>` at tuning time (torch 2.10 refuses `=1`) | `PYTORCH_TUNABLEOP_ENABLED=1` |

`TORCH_BLAS_PREFER_HIPBLASLT=0` sends plain GEMMs to rocBLAS, which for bf16/fp16 runs
the same Tensile kernels: test it like any candidate, don't assume it avoids the
kernel (2026-11-09: `preferred_blas_library("cublas")` failed identically). A
step-7 causal knob is the first candidate: it is already the intervention that made
the reproducer clean.

**Torch.compile arms: one inductor cache per arm, and read the generated code.**
Processes sharing a `TORCHINDUCTOR_CACHE_DIR` reuse each other's compiled kernels: a
stock run after a patched one in the same container read the patched kernel and
went 0/100 where the stock control is 100/100 (2026-11-11). Give every arm its
own cache dir (`nd run` gives every run its own; inside one `exec.sh` command that
runs several arms, set it per arm), and grep `TORCH_LOGS=output_code`
of one patched trial for the new call and the absence of the old -- the `[fix]`
marker alone shows the patch loaded, not that the kernel used it.

**An env workaround on the reproducer: never `--restore-env`.** It re-applies the
capture's environment, which holds the very variable the workaround flips. Run
both arms with the env set explicitly (no `--restore-env`), the control first, in
one job: `scripts/env_workaround_validate.sh <capsule> "VAR=VALUE ..."
[iters]` through `exec.sh` (stock + each arm in its own process, prints the env each
saw; `PERF_CMD` for a timing per arm) -- `[iters]` from `power.py`, `--prefer-node` the node 5b reproduced on
(`step7-rootcause.md`).

**Model A/B for a multi-rank job** (one run fills the node, so arms cannot share it):
`CMD='<the model once>' NAN_SIGNAL=... ARMS='VAR=VALUE ...' nd submit --name ab --gpus 8 --exclusive
-- model_ab.sh` -- `CMD` runs through `nd run --env <arm>`, so the arm reaches the model on any
runtime; stock and each arm interleaved, `ROUNDS=auto` (round 2 only if round 1 is not
decisive: `AB_DECISIVE` = stock >= 3 hits, every arm 0 over as many steps), one `AB <tag> env=.. last_step=..
signal=.. spikes=.. rc=..` line per run (an old `LAUNCHER=` script still works).
**On `<CAPTURE_NODES>` > 1 nodes** (the node-count ladder's result, `step4-distributed.md`): add
`--nodes <CAPTURE_NODES>` and write `CMD` like the capture spec's `all` line -- `torchrun
--nnodes=$ND_NNODES --node_rank=$ND_NODE_RANK --master_addr=$ND_MASTER_ADDR ...` and the
data-parallel degree from `$ND_NNODES`; each run is then one job over all those nodes (one
`nd run` per node), so the A/B compares at the node count the bug was captured at, never fewer.
Validated 2026-11-12 (2 nodes, qwen3 MoE `run.sh` command): stock 100 steps, no NaN
but 86 `grad_norm` spikes; `TENSILE_FIXED_WGMXCC=1` 100 steps, 0 spikes -- at 2 nodes the bug
shows as spikes, so judge the arms on `spikes=`, not `signal=`.
**Single-process trainers**: all arms in one `wave.sh` spec (e.g. 2 stock + 3 per fix on 8
GPUs; a model-level fix as a patched copy of the source in its own line, an env/`PYTHONPATH`
fix in the env column: shell-quoted or bare, `V='a b'` and `V=a` both reach the trial as the bare value; every
word must be `VAR=VALUE` or the wave stops). **Check the fix landed in every fix trial**: its marker line
(`[fix] inductor sqrt: ...`) and, for a compiled fix, the output code (`TORCH_LOGS=output_code`) -- `wave_summary.py`
alerts on `[fix] ... NOT applied`, and `inductor_override` now stops the process on it (the trial scores ERROR, not
a stock verdict; 2026-11-23: a quoted value, five "fixed" trials ran stock). Its `WAVE_DIR` (and `CAPSULE_DIR`) go **outside `fix/`** --
`<ART>/fix_ab/wave` -- so the package's tarball holds no trial logs; `wave_summary.<tag>.txt`
lands beside `WAVE_DIR` (`<ART>/fix_ab/`).
An aiter `.co` fix in a spec line sources the kept row's relocatable `env.sh` in the command
column, with the load log on so each trial proves which kernel it ran:

```
2|fix|AITER_LOG_LEVEL=INFO|source $ND_ART/fix_work/kept/<row>/env.sh && python3 $ND_REPO/<model command>
6|ctl|AITER_LOG_LEVEL=INFO|python3 $ND_REPO/<model command>
```

then `grep -l 'LoadKernel.*fix_work/kept/<row>/overlay' <WAVE_DIR>/*.fix.*.log` (0.1.26+; older
aiter: `hipModuleLoad: <overlay>`) must list every fix trial -- one without it ran stock.

Validate it on the reproducer like any candidate (step 3 below), then on the
model: `<FULL_MODEL_RUN_SCRIPT>` `N ~ 3m/k` times on, `N` off, from
`<FULL_MODEL_RATE>` (state `N` first; *unmeasured* = run the on arm as the
control), counted by `<NAN_SIGNAL>`, **both arms in one wave** if GPUs allow (mind
the fixed-container-name trap). **Verify the toggle landed** in one trial per arm
(`nd run` prints nothing of it: echo the variable in the command) -- a dropped var reads as a clean pass. Report both counts (7/10 vs
0/10 concludes; 1/10 vs 0/10 means raise `N`) and the off arm's step time.

**3. Validate on the step-6 reproducer, with the fix applied**, as an `nd submit` job
(`--step 'step8[<G>]'`), at the reproducer's own load (same copies, co-tenant,
`--contend`, `--restore-env`):

* **A reproducer with `operands: SYNTHETIC-INCONSISTENT` cannot validate a fix**: it
  NaNs on a correct kernel too. Validate on consistent operands instead
  (`attn_consistent_check.py --save stock.pt`, then with the fix `--compare
  stock.pt`: all finite and `IDENTICAL-on-stock-finite`), and ship that check as the
  fix's `validate.sh`.

* **Library-overlay fixes: `--restore-env` silently removes the fix.** The capsule's
  env namespace covers `HIPBLAS*` and `TENSILE_*`, so `restore_env_dict()` UNSETS a
  `HIPBLASLT_TENSILE_LIBPATH` the capture did not have -- the node ladder with the
  overlay in `EXTRA_ENV`, or any `replay_nan_op.py --restore-env` under it, then
  runs the stock kernel and "the fix still NaNs". The replay now holds both overlay
  variables back (the `env:` banner lists them as `held back`, and a `libpath:` line
  names the overlay hipBLASLt was given) -- **check that line in every patched run**;
  no line or `stock` = the fix was not in the process.
* **Binary kernel fix (a CLEAN mutation row)** -- the tools, in one job:
  1. `scripts/fix_overlay.py --out <ARTIFACT_DIR>/fix-overlay-build
     --kernel-file <kernel.*.txt> --selector <sel> --payload <payload> [--addr ...]
     --expect-sites <n>`: the same selector/payload registries as the sweep (so the
     fix has the CLEAN row's bytes), built on Lustre (not node-local `/tmp`), and
     read back -- `READBACK=OK` only if exactly the selected sites changed. CPU-only:
     it runs in the image on the login node in ~1 min where the login node has the runtime
     (e.g. apptainer); else (no docker on login) a one-minute `exec.sh --gpus 0` job. **Build it outside `fix/`**:
     the overlay is the whole library dir (~400 MB, thousands of symlinks into the
     image) and `--repack` would tar it; ship `overlay.json`, `kernel.diff` (it writes
     it from the read-back: `DIFF=`), `kernel_name.txt` and the rebuild command in
     `fix/patch/` instead.
  2. `scripts/fix_values.py <capsule> --overlay <OVERLAY> [--synthesize] --json
     fix/values.json`: stock vs patched in fresh processes, solo -- fp64 error, bitwise
     equality with stock (a wait/barrier-only fix should be bit-identical), cost per
     launch, and the libpath the patched child saw. `VALUES=OK`. **`--synthesize` for
     a shapes-only capsule** (the capture default; without it the tool refuses one).
     The cost comparison is one solo run each, stock first -- report it as "no
     regression seen", not as a speed-up.
  **aiter / CK** (`fix_overlay.py` and `fix_values.py` are hipBLASLt-only): the fix
  is the step-7 sweep's `--mutate-keep-fix` copy, `fix_work/kept/<row>/env.sh` (no copy:
  re-run that row's sweep with `--mutate-keep-fix`). Read-back = the row's `.diff`.
  **aiter, the package**: copy the row's `patched/` into `fix/patch/<name>/` -- a short name, no
  `@` (sweep rows are `vmcnt0_each@0x..@<call>`; e.g. `F1-vmcnt18`) -- and the arms are
  `scripts/aiter_fix_validate.sh <capsule> <rel .co path> <NAME>=<patched .co> ... -- <5b load
  flags>` (stock + one GPU per fix, each fix served by `aiter_overlay.sh` from the image's own
  aiter, its load checked from aiter's log; `FIX_VALIDATE=PASS|FAIL`) -- also as
  `make_fix_validate.py --cmd` (its docstring). A fix not proposed by a mutator (a hand edit of
  the kept `.co`) goes the same way; `fix.changes` spells it `<instruction edit>@<addr>`.
  Values: the 5b chain command, **solo**, stock with `--chain-save-outputs
  fix_work/stock.pt`, then under the fix (`source env.sh` in the same CMD) with
  `--chain-compare-outputs fix_work/stock.pt` -- `[values] IDENTICAL` is the pass for a
  wait/barrier-only fix (a value-checked op is compared on what its consumer reads).
  **A non-deterministic fault can make the saved stock run itself a bad one** (a fault that hits a solo run often enough), and a correct fix then reads `DIFFER`: save stock with
  `--chain-iters 1` and keep it only if that run's walk and value check came back clean (no
  `WRONG VALUE`, finite) -- else save again; a `DIFFER` under the fix is a finding only against
  such a run. Keep `fix/` to what ships: work files (outputs, logs, overlays) go to `fix_work/`.
  Then the same with `--contend <C-1>
  --chain-repeat-check`: under the fix every op should repeat (co-tenants inherit
  the env, so all processes are patched -- the strong case, beside item 3's
  conservative one). **Loaded** = the shim log (`NUMERIC_DOCTOR_HSA_LOG`) has a
  `PATCHED` line (same-size splice) **or `REPLACED`** (a grown kernel, e.g. any
  `sbarrier_*`); for aiter, the `[aiter] hipModuleLoad: <overlay path>` line (0.1.26+:
  `[aiter] LoadKernel: <sym> hsaco: <overlay path>`).
  Neither = the run used the stock kernel: void, not a pass.
  3. The NaN rate under load: the step-7 sweep command (same mapping), its
     `--mutate-only` the fix's mutator and a fresh `--mutate-report` under `fix_work/` --
     its baseline is the control, its row the patched rate, at the sweep's load. The
     co-tenants there run the **stock** kernel (only the measured process is
     patched): the conservative case; say so. Size `--chain-iters` for >= 3 expected
     hits (aim ~30+).

* **A COLLECTIVE reproducer** (`repro-G<i>/` replays member-rank capsules, `step4-distributed.md`
  *Synchronous collectives*): every env candidate is an arm of `scripts/collective_arms.sh
  <capsule dir> <arms file>` (shipped in the package: `$ND_ART/scripts/collective_arms.sh
  $ND_ART/capsule $ND_REPO/patch/arms.txt` as `make_fix_validate.py --cmd`, `--gpus 0,1`), stock
  first and last. Its `ARM ... path=` column is the check that a fix keeps the implementation it
  claims to fix -- an arm whose path changed is a workaround (a library setting can send the call
  to another implementation of the same op, which says nothing about the one that failed).
* **Control in the same job**: the unpatched reproducer still `NAN` (else the run
  could not have caught it -- the result is void, not a pass). **A clean control off the 5b node is a
  node-bound reproducer**, not a fixed bug: a library GEMM's bad shapes can move from node to node (2026-11-23,
  stream-K: the captured M reproduced only on the node 5b ran on; three other nodes computed it correctly).
  Re-run on that node (`--node <5b host>`, `<LEDGER>`); held for hours by someone else: validate with a check
  that does not depend on the node -- a property of the fix rather than one shape, e.g. `gemm_shape_probe.py
  sweep` over the band (bad sizes in stock, none under the fix) -- ship it in `validate.sh` (`--tool`), and say
  in `fix/README.md` and the report that the capsule arm is node-bound.
* **Patched: CLEAN with power** -- >= 3 expected hits at the reproducer's rate
 ; fewer is `UNDERPOWERED`, raise the count, never a pass.
* **Correct values**: patched outputs match the fp64 reference (or the unpatched
  kernel's finite runs) within tolerance -- `WRONG` otherwise. Report the error over
  normal-range reference values and over subnormal ones separately: an fp32 value
  that is itself subnormal carries only a few significant bits, so a correct op on
  it differs from fp64 by up to ~1e-2 relative (2.9e-3 for a correct `sqrt` of
  subnormal softmax probabilities, 2026-11-11) -- not `WRONG`. A hand-written
  check prints `n/a (none sampled)` for a class with no samples, never `nan` (it reads
  as a failure; 2026-11-14), and never takes `.max()` of an empty selection.
* Note the cost (per-launch time vs unpatched); a large regression is reported, not
  a failure.
* `.diff` read back for a binary patch (identical = never landed).

**3b. Validate on the model, every candidate -- not only workarounds.** A clean
reproducer proves the mechanism is fixed. It does not prove the model is NaN-free
. Run the model-level A/B of the workaround paragraph above (fix vs stock
interleaved, both counted by `<NAN_SIGNAL>`, the toggle's load verified per
attempt: a print marker in the patched copy, grepped from the run's logs). Arm the
fix attempts with `NUMERIC_DOCTOR_GRADLOG` (`scripts/grad_trajectory.py`) **when the trainer
already calls `after_backward`** (step 1 wired it into a working copy), so a NaN under the
fix comes with its trajectory. Not wired (a stock trainer, the torchtitan wrapper): skip it
and say so -- don't edit the model for it up front; only if a fix arm does NaN, wire it into
a working copy for that re-run. A NaN under the fix is not a failed fix
if the reproducer passes: capture it (`NAN_CAPTURE_*` plus the fix) to see which op
now goes non-finite. If that is a correct reduction over exploding finite values
(`capsule_info.py` prints `overflow expected`), it is a separate, upstream
divergence: report it as such. Read `grad_trajectory.py <logs>` (which module
grows, from which step, weights moving or not) before offering the user more
budget for it.

**4. Decide.** Ledger one line: `step8[<G>] iter <k>: <id> <kind> <result>; <path>`.

* **A candidate passes all checks** (reproducer and 3b) -> keep it, and go on while budget
  is left and a kind is still missing (*Three kinds of fix*): no real fix yet, or no
  targeted workaround yet (only a broad one), or a narrower scope is plausible. Stop when
  each kind has a passing candidate (the real fix: or a `checked by equivalence` / `not
  validated here` one written), or is shown not to exist, or the budget is spent.
* **Then build the `<FIX>` package with every kept candidate** -- one subdir per fix, one
  `validate.sh` that runs stock plus every validated fix as arms:

  ```
  fix/
    README.md        # the summary table first (below), then per fix: the change, how to
                     # apply, validation (control n/N, patched 0/N with expected hits, load,
                     # value check, model A/B), cost, what it does not cover;
                     # root cause in one paragraph, link ROOT_CAUSE.md
    R1-<name>/       # the real fix: patch/diff against the named version, BUILD.md (how to
                     # build and apply), its validation level
    T1-<name>/       # a targeted workaround: the edited file / env / model diff, apply.sh
    B1-<name>/       # a broad workaround: the env setting
                     # (an overlay: overlay.json, the .diff, kernel_name.txt -- not the built overlay)
    scripts/         # the tool versions validate.sh needs (fix_overlay.py,
                     # fix_values.py, replay_nan_op.py and its imports)
    validate.sh      # one command: reproducer stock + one arm per validated fix
    iter-*.md  candidates.md
  ```

  `fix/README.md` opens with one row per fix, **recommended first** -- the validated real
  fix if there is one, else the narrowest validated workaround -- and the reason:

  | id | kind | change | reaches | reproducer | model | cost | status |
  |---|---|---|---|---|---|---|---|
  | T1 | targeted workaround | rows `<keys>` -> `Default` in the tuned file | 4 backward GEMM shapes | stock 100/100 -> 0/100, values = fp64 | 2/2 -> 0/4 | +0% step | validated |
  | R1 | real fix | `ScaledBlas.cpp`: ... (diff vs torch `<commit>`) | the faulty dispatch | -- | -- | -- | not validated here: please try (`BUILD.md`) |
  | B1 | broad workaround | `PYTORCH_TUNABLEOP_ENABLED=0` | every GEMM untuned | 100/100 -> 0/100 | 2/2 -> 0/3 | +X% step | validated |

  **`validate.sh`: generate it, don't copy an earlier run's** (one had a hard-coded capsule name and a
  co-tenant that did not apply, 2026-11-20): `python3 <REPO_ROOT>/scripts/make_fix_validate.py --fix
  <ARTIFACT_DIR>/fix --repro <ARTIFACT_DIR>/repro-<G> --cmd '<the arms>' [--gpus all]` writes it with
  the reproducer's own runtime lines and ships `fix/scripts/` (nd, the reproducer's tools,
  fix_overlay.py, fix_values.py, env_workaround_validate.sh, capsule_operands.py, inductor_ref.py; anything else a
  `values.py` imports: `--tool <file>`); in `--cmd`, `$ND_REPO` is `fix/` and
  `$ND_ART` the reproducer, found as the sibling `../repro-<G>` (both tarballs extracted side by
  side) or `REPRO=`. Its docstring has a `--cmd` per fix kind.

  Pack it like a reproducer:
  `python3 <REPO_ROOT>/scripts/make_repro_readme.py --repro <ARTIFACT_DIR>/fix --repack --ledger <LEDGER> --step 'step8[<G>]'`.
  **Then run it as shipped**, like step 6's reproducer: `bash -n validate.sh`, extract the
  tarball under `<ARTIFACT_DIR>/verify-fix/` beside an extracted reproducer (or set `REPRO=`) and run
  its `validate.sh` there in one job, as shipped and in an empty environment: `PACKAGE=<ARTIFACT_DIR>/verify-fix/fix
  SCRIPT=validate.sh nd submit --name verifyfix --gpus 8 --exclusive [--prefer-node <host>] --step 'step8[<G>]' --
  package_run.sh`, then `nd wait verifyfix --max 240`, never a `sleep` loop (a plain `sbatch validate.sh` + `nd adopt`
  is the same test where the session may submit outside nd). Its Slurm log goes to the jobs dir, not into `fix/`
  (a `slurm-*.out` beside validate.sh once got into the repacked tarball). Ledger
  its result beside the sha256. A reused `validate.sh` too: an earlier investigation's had a syntax
  error nobody ran (2026-11-14).
* **Budget left** -> next iteration.
* **Budget spent** -> the package with whatever passed (and a real fix written but not
  validated, labelled so); nothing passed and no real fix written: `<FIX>` = `NO_FIX`,
  `fix/README.md` still lists every candidate tried and why each failed.

## Exit

`$LG end step8[<G>]`; ledger `<FIX>` (tarball path + sha256, or `NO_FIX`) and one
`FIX=<id> <kind> <status>` line per fix shipped. Tell the user one line per fix
(recommended first), then step 9.
