# Step 9: close and report

Once **every replayed group** (G1 alone under `<GROUPS_TO_REPLAY>=largest`) has a
5b verdict (and reproducing groups have been through 7, and 8-9 if run) -- or, when
G1 did not reproduce, step 5N has a verdict (and been through 7, 8-9 if it hit). The
report lists groups not replayed, with op and capsule count, as `not replayed
(largest only)`. Step 7's recipe track ran: also `step9-report-sweep.md`. Steps 7-8
are interactive-only under MAD: in the automated workflow they are reported as
*skipped (automated)*.

## Close

* **Restore the model source** from the path in `<LEDGER>` (a capture variant
  written as a new file: delete or set aside). A capture left wired in raises on
  the next unrelated non-finite. Say you restored it; diff if the original was
  edited. Wired with `madengine nan wire`: use its restore command with the backup
  dir `<ARTIFACT_DIR>` (it verifies the sha256; see `madengine nan wire --help`).
* **Hand over `<ARTIFACT_DIR>` by path** (capsules, `<OUT_JSON_PATH>`,
  `<REPORT_JSON>`, `<ASM_DIR>`, `<LEDGER>`, tarballs, `rootcause/`, `fix/`). Tell the
  user to keep the capsules, by name.
* **Timing:** `$LG end step9`, then `$LG table` (SKILL.md rule 7); add rows by hand
  for steps not run, marked *skipped* with why. Give it to the user and put it in
  the report.
* **Write the report from `<LEDGER>`, not memory**: the report is drafted by
  `madengine nan report <ARTIFACT_DIR>`, which writes `report.md` with every fact on
  disk (time table with skipped steps and why, rate source, wave summaries, image
  injected-or-stock, kernel per group, each 5b attempt with copies x GPUs / runs /
  hits / env / build / proposed verdict / first non-finite op / capture contention,
  latest tarball sha256s). Fill its remaining `TODO(agent)` lines -- the judgement --
  using the checklist below; edit, don't rewrite, and do not retype its numbers.

## Always report

* **three outcomes, in this order, each on its own line**:
  * **Reproducer** (step 6): tarball path + sha256 and the one command, or none;
  * **Root cause** (step 7): `<ROOT_CAUSE>` -- where, mechanism, trigger, the causal
    experiment (`rootcause/ROOT_CAUSE.md`); or `NO_ROOT_CAUSE` with the budget
    spent, iterations, excluded and still-open hypotheses; or *skipped* and why.
    Which track found it -- (a) native or (b) recipe -- and `<NAN_KIND>`;
  * **Fix** (step 8): `<FIX>` -- the change, tarball path + sha256, validation
    (control `n/N`, patched `0/N` with expected hits, load, value check, cost), and
    whether it is a fix or a workaround; or `NO_FIX` with the budget and candidates
    tried; or *skipped* and why (incl. "no root cause"). A fix is a proposal, never
    applied to MAD or the image.
  Never let "reproducer done" read as "NaN fixed";
* the per-step time table;
* **where the rate came from**: step 2 (with environment), or none -- step 2
  skipped, the step-4 controls were the only measurement;
* the image/build, and whether NaN-injected per the step-7 recipe enumeration,
  else unknown;
* **the GPU** (`madengine nan gpu-check` line from `<LEDGER>`): name and arch; `UNTESTED` (below
  MI300X) -> repeat the SKILL.md *GPU gate* warning at the top of the report;
* **replay environment and build**: the `env:` verdict (or differences and why
  acceptable) and the `captured build:` line (ROCm, the `libhipblaslt` mapped). A
  differing held-back loader path = a different image -- caveat every number;
* **the contention behind every replay rate**: per 5b attempt, copies per GPU x
  GPUs x `--iters`, beside the capture's `contention:` line ("0/16M at
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
| 5N `NATIVE_MATH` / `NATIVE_FP32_OVERFLOW` | **not a kernel bug**, plainly: the arithmetic (`0*inf`, `0/0`, ...) or the overflow, where in the model, the minimal candidate and the clean controls that bound it (they are the mechanism); the native tarball; why the capsule path could not attribute it (`NATIVE_TRIGGER`, e.g. the producer ran inside a suspended compiled region); 7-8 skipped as not applicable |
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

**Already done by `madengine nan replay`:** it writes `replay/<G>/kernel.*.txt`
(solution index, kernel, arch, injected-or-stock, pinned bench line) and
`madengine nan report` puts it in the report's Kernel section, flagging groups that
share a kernel. Run anything below only when that file is missing.

**Tensile: don't transcribe the key by hand** (5b prints `hipBLASLt key: M=...`);
emit a runnable command:

```bash
madengine nan exec --image <IMAGE> --gpus 0 -- python3 -m nan_doctor_rt.replay_nan_op <CAPSULE_FILE> --bench --solution-index <idx>
```

It prints a complete `hipblaslt-bench` line (compute type from the capsule's
`float32_matmul_precision`; `--solution-index` pins the solution that NaN'd) that
reproduces the GEMM with no capsule, model or repo.

**aiter / CK** have no key or bench: attach the step-6 tarball with its `sha256sum`,
saying one copy alone reports `0/N` and means nothing.
