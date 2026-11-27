# Step 9: close and report

Once **every replayed group** (G1 alone under `<GROUPS_TO_REPLAY>=largest`) has a
5b verdict (and reproducing groups have been through 7, and 8-9 if run) -- or, when
G1 did not reproduce, step 5N has a verdict (and been through 7, 8-9 if it hit). The
report lists groups not replayed, with op and capsule count, as `not replayed
(largest only)`. Step 7's recipe track ran: also `step9-report-sweep.md`.

## Close

* **Restore the model source** from the path in `<LEDGER>` (a capture variant
  written as a new file: delete or set aside). A capture left wired in raises on
  the next unrelated non-finite. Say you restored it; diff if the original was
  edited. Wired with `wire_capture.py`: `wire_capture.py restore <model.py>
  --backup-dir <ARTIFACT_DIR>` (verifies the sha256).
* **Hand over `<ARTIFACT_DIR>` by path** (capsules, `<OUT_JSON_PATH>`,
  `<REPORT_JSON>`, `<ASM_DIR>`, `<LEDGER>`, tarballs, `rootcause/`, `fix/`). `artifacts/` is gitignored and
  unbacked; tell the user to keep the capsules, by name.
* **Write the report from `<LEDGER>`, not memory, in two calls.** (1) In the call that
  restores the source: `python3 <REPO_ROOT>/scripts/make_report.py <ARTIFACT_DIR>` drafts
  `report.md` with every fact on disk (time table with skipped steps and why, rate
  source, wave summaries, image injected-or-stock, kernel per group, each 5b attempt
  with copies x GPUs / runs / hits / env / build / proposed verdict / first
  non-finite op / capture contention, latest tarball sha256s) and prints its `TODO(agent)`
  lines numbered, each with its section -- the judgement left to you. **Don't read the
  report back.** (2) Answer every TODO from that list and `<LEDGER>`, using the checklist
  below (don't retype its numbers), in an answers file -- `## TODO <n>` headings, each
  followed by the text that replaces TODO n -- and, in ONE call: `$LG end step9;
  python3 <REPO_ROOT>/scripts/make_report.py <ARTIFACT_DIR> --fill <ARTIFACT_DIR>/report_answers.md;
  $LG table`. `--fill` puts the answers in, refreshes the time table (so the report-writing
  time counts) and lists any TODO left (renumbered: answer those the same way).
  Give the table to the user.
* **Fill the result card** -- `<ARTIFACT_DIR>/result.json`, drafted with the report (see
  *Result card*); edit its TODO fields before the `--fill` call, which ends with
  `result_card.py check` (`RESULT_CARD_CHECK=OK`: done; else it lists what is open).

## Result card

`report.md` is prose: two runs of one example (two versions of this skill, PRIORS=use vs
none) could not be compared from it -- hosts, counts and the call site G1 happened to
capture differ every run, and the root cause and fix sit in free text. `result.json` holds
the result as data; `compare_results.py <ART_A> <ART_B>` judges bug / root cause / fix as
SAME, REFINED (one site or change set a strict subset of the other's) or DIFFERENT, and
prints the rest as notes. The draft fills the facts (verdict, op, shape, kernel, NaN kind,
groups, rates); you fill, in these spellings -- the compare matches strings:

| field | value |
|---|---|
| `bug.verdict_confirmed` | `true` once `bug.verdict` is the verdict you report (correct it first if not) |
| `root_cause.status` | `FOUND` / `NO_ROOT_CAUSE` / `SKIPPED` |
| `root_cause.component` | `hipblaslt` `rocblas` `triton` `inductor` `aiter` `ck` `miopen` `rccl` `pytorch` `framework` (vLLM, SGLang, torchtitan) `model_code` `model_config` `data` `hardware` `unknown` |
| `root_cause.mechanism` | one line (shown, not judged) |
| `root_cause.defence` | one line from `ROOT_CAUSE.md` *Defence* (step7-rootcause.md *Why was it not prevented?*): `none` (the condition is unhandled), or the guard and why it did not apply (`<guard> compiled for <arch> only`, `<check> off unless <flag>`); `-` when status is not FOUND |
| `root_cause.location.kernel` | `hipblaslt solution <index>`, the Inductor kernel name, an aiter asm `.co` as its path under `hsa/` (`gfx950/<dir>/<name>.co`, as `bug.kernel.name` drafts it), or the library op (`aiter rmsnorm2d_fwd`) |
| `root_cause.location.sites` | the instruction addresses the causal experiment named, hex (`0x2A8D9D0`) -- the minimal set, not every candidate |
| `root_cause.location.env` / `.config` / `.source` | `VAR=VALUE` / `key=value` (the faulty setting, e.g. `rms_norm_eps=0.0`) / `file.py:Class.function` -- **no line numbers** |
| `root_cause.location.source` -- which site | **the faulty code**, as `file:Class.function` (aiter `mla.py:mla_decode_fwd`, vLLM's `quick_reduce_impl.cuh:AllReduceTwoshot.run`). When that code is not Python you could name -- a compiler's lowering, a closed library -- **the nearest Python that selects it** (Inductor's `torch/_inductor/codegen/triton.py:TritonOverrides.sqrt`, which emits the faulty `tl.sqrt_rn`). **Never the model's call site** that merely triggers it: that goes in `conditions`. A model / config bug's faulty code IS the model (`model.py:Class.function`, or `.config`). 2026-11-23: one run named the model's router, another the Inductor emitter -- compared DIFFERENT on the same root cause |
| `root_cause.conditions` | the necessary conditions, short (shown, not judged) |
| `fix.status` / `fix.kind` | `FIX` `WORKAROUND` `NO_FIX` `SKIPPED` / `binary_patch` `library_env` `compiler_override` `source_patch` `config_change` `input_validation` `none` |
| `fix.changes` | the primary fix as a set: `lgkmcnt0@0x2A8D9D0` per patched site (a mutator's name; a hand edit or `vmcnt_minus1`-style tightening as the instruction it becomes, `lgkmcnt(1)@0x1A2C`), `VAR=VALUE`, `key=value`, `inductor:sqrt=libdevice.sqrt`, `file.py:Class.function` |
| `fix.extra` / `fix.alternatives` | defence in depth / other changes that also passed (shown, not judged) |
| `fix.validated.reproducer` / `.model` | `stock n/N -> fixed 0/N` as measured |

Re-running `result_card.py draft` refreshes the facts and keeps your answers.

## Always report

* **the step-6 hand-over, first** -- the reply's opening lines repeat the step-6 block
  (`step6-finish.md` *Then*) in the past tense, from the `HANDOVER announced` ledger line:
  "**✅ Reproducer generated** at <time> -- `<tarball>`; then moved on to root-causing
  (<RC_BUDGET>) and fixing (<FIX_BUDGET>)" (steps 7-8 off: "stopped at the reproducer";
  a model / numerics verdict -- `FP32_OVERFLOW?`, `NATIVE_MATH`, `NATIVE_FP32_OVERFLOW`: add "not a
  kernel bug: <the arithmetic>"). The
  mid-turn block can scroll away unseen; this one ends the turn and stays visible. No
  reproducer: say so in its place;
* **three outcomes, in this order, each on its own line**:
  * **Reproducer** (step 6): tarball path + sha256 and the one command, or none;
  * **Root cause** (step 7): `<ROOT_CAUSE>` -- where, mechanism, trigger, the causal
    experiment (`rootcause/ROOT_CAUSE.md`); or `NO_ROOT_CAUSE` with the budget
    spent, iterations, excluded and still-open hypotheses; or *skipped* and why.
    Which track found it -- (a) native or (b) recipe -- and `<NAN_KIND>`;
  * **Fix** (step 8): `<FIX>` -- the change, tarball path + sha256, validation
    (control `n/N`, patched `0/N` with expected hits, load, value check, cost), and
    whether it is a fix or a workaround; or `NO_FIX` with the budget and candidates
    tried; or *skipped* and why (incl. "no root cause").
  Never let "reproducer done" read as "NaN fixed";
* the per-step time table;
* **where the rate came from**: step 2 (with environment), or none -- step 2
  skipped, the step-4 controls were the only measurement; **and step 8's model A/B stock arm** when
  it ran -- the same model, stock, more trials: quote both counts and take the larger denominator
  as the rate (step-4 controls are 2-6 trials: 0/6 there and 3/12 in step 8 on 2026-11-20); world
  size > 1: step 4 had no control arm, so step 8's stock arm is the only rate, or none;
* **priors**: `<PRIORS>` (`use` / `none`) and, for a root cause re-tested from an earlier
  `ROOT_CAUSE.md`, "re-confirmed from <path>" -- a confirmation, not a discovery
  (`make_report.py` drafts the line under *Root cause*);
* `<NODES>` > 1: **the node-count ladder** -- one row per rung from the `node-count ladder n=`
  ledger lines (nodes, attempts, verdict), `<CAPTURE_NODES>` and why it stopped (floor, or the
  rung that did not reproduce in n attempts -- with an unmeasured rate, possibly bad luck), and
  that only the DP degree (so the global batch) shrank;
* the image/build, and whether NaN-injected per the step-7 recipe enumeration,
  else unknown;
* **the GPU** (`gpu_check.py` line from `<LEDGER>`; `make_report.py` drafts it under `## GPU`): name and arch; `UNTESTED` (below
  MI300X) -> repeat the untested-GPU warning (`step0-gates.md`) at the top of the report;
* **replay environment and build**: the `env:` verdict (or differences and why
  acceptable) and the `captured build:` line (ROCm, the `libhipblaslt` mapped). A
  differing held-back loader path = a different image -- caveat every number;
* **the contention behind every replay rate**: per 5b attempt, copies per GPU x
  GPUs x `<CAPSULE_ITER_COUNT>`, beside the capture's `contention:` line ("0/16M at
  2 copies on each of 8 GPUs, where the capture saw 2 processes", not "0/100"). A
  one-copy `0/N` on a contention-gated fault is not a clean result;
* **anything capped, sampled, skipped or folded**, and on what basis.

| stopped at | report |
|---|---|
| 5b `PROPAGATED` | op and operand; the bug is upstream. A finding |
| 5b `FP32_OVERFLOW?` | not a kernel bug, plainly: overflow (fp64 finite, beyond the dtype) or undefined arithmetic (fp64 non-finite too: `0*inf`, `0/0`) -- the walk's `fp64 ref` reason line says which; name the right one; no recipe sweep |
| 5b `KERNEL_BUG` | rate (`n/N`, concurrency, GPUs, iters), the op, fp64 ref finite and no upstream divergence -- a kernel bug |
| 5c/5d `KERNEL_BUG` beside a co-tenant | the rate **with the load named** (model, or stand-in mode and processes), and every clean load beside it (copies only, fewer processes, partial modes): the trigger's bounds are the finding |
| 5b `NOT_SET_UP` | never a final verdict on its own: report what `step5-pressure.md` found after it |
| 5b `KERNEL_BUG_UNVERIFIED` | rate and op, and **nothing established the operands were correct** (no reference, or a `DIVERGED` ancestor). The op is where the NaN first *appears*; say whether `--chain-repeat-check` put the fault upstream of it |
| 5b `NOT_REPRODUCED` | the attempts (`c<C> x 1M` on N GPUs: capture + 1, less any OOM step-downs) as a bound; whether the model reproduced |
| 5b deterministic walk | `[C1]`'s op as *unverified attribution*, how you know it is deterministic (non-finite walk, no `DIVERGED`), unchecked ops as *unverifiable* |
| 5N `NATIVE_KERNEL_BUG` / `NATIVE_UNVERIFIED` | **say it is a native reproducer, not a capsule** and why the capsule path failed (`NATIVE_TRIGGER`) and what the native script had that the closest clean capsule replay lacked (N6); the op or region, rate with the load, the fp64 check (or its absence), and the removals and loads that ran clean |
| 5N `NATIVE_MATH` / `NATIVE_FP32_OVERFLOW` | **not a kernel bug**, plainly: the arithmetic (`0*inf`, `0/0`, ...) or the overflow, where in the model, the minimal candidate and the clean controls that bound it (they are the mechanism); the native tarball; why the capsule path could not attribute it (`NATIVE_TRIGGER`, e.g. the producer ran inside a suspended compiled region); then the step-7 root cause in the model and the step-8 model fix |
| 5N `NATIVE_NOT_REPRODUCED` | every candidate's `n/N`, load and host as a bound, beside the capsule path's attempts; whether the model reproduced |
| 7 root-causing | the root-cause outcome above, with its iteration count and `rootcause/` path |
| 7 recipe sweep | `step9-report-sweep.md` |
| 8 fixing | the fix outcome above, with its iteration count and `fix/` path |

## Identify the kernel

| source | quote |
|---|---|
| Tensile | kernel name, dispatched `solution_index`, arch, library dir, **GEMM problem key** |
| aiter | `.co` path relative to `<asm_root>/<arch>/` (e.g. `fmha_v3_bwd/bwd_hd128_bf16_a32_psskddv.co`), exported symbol, arch |
| CK | mangled symbol as the HSA shim saw it, arch |

**Already done by the 5b ladder** (every runtime): it writes `kernel.*.txt` -- in `replay/<G>/` for
a ladder you submitted, in `verify-<G>/repro-<G>/out/` for a capture-to-reproducer job (solution
index, kernel, arch, injected-or-stock, pinned bench line) and `make_report.py`
puts it in the report's Kernel section, flagging groups that share a kernel. Run
anything below only when that file is missing (`CAPSULE_KERNEL_ID=0`, or an old docker ladder).

**Tensile: don't transcribe the key by hand** (5b prints `hipBLASLt key: M=...`);
emit a runnable command, through `nd run`:

```bash
nd run --gpus 0 -- 'python3 -m replay_nan_op <capsule as $ND_ART/...> --bench --solution-index <idx>'
```

It prints a complete `hipblaslt-bench` line (compute type from the capsule's
`float32_matmul_precision`; `--solution-index` pins the solution that NaN'd) that
reproduces the GEMM with no capsule, model or repo.

**A native reproducer (5N)** has no capsule and no ladder `kernel.*.txt`. aiter / CK / Triton:
`scripts/native_sweep.py --discover --cmd '<the candidate>'` names the `.co` it loaded and the
kernels it dispatched (step7-recipe-sweep.md *A native reproducer*). For a hipBLASLt
op, take the bench line from hipBLASLt's own call log: one short run of the package's
`repro.py` with the log on --
`nd run --gpus 0 -- 'HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE=$ND_ART/native_hipblaslt.log python3 $ND_ART/repro-native/repro.py <short: e.g. --iters 100>'`
-- or the model's step-4 log (`$ND_ART/model_hipblaslt.<trial>.log`, when the spec turned it
on). Quote the line whose `-m -n -k`/transposes/types match the reproducer's op, with its
`--solution_index`: that pins the kernel (a `hipblaslt-bench` run of it reproduces the GEMM
without the repo). Say the index came from the call log, not from a replay; if the
reproducer's line and the model's disagree on the solution index, report both.

**aiter / CK** have no key or bench: attach the step-6 tarball with its `sha256sum`,
saying one copy alone reports `0/N` and means nothing.
