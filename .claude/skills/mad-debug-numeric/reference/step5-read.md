# Step 5: read the grouping and the replays (always read)

How to *run* 5a/5b (extra groups, long GPU commands) is `env-madengine.md`. Conditional:
`step5-replay-env.md` (an `env:` banner not `matches capture`),
`step5-replay-guardrail.md` (deterministic case), `contention-replay.md` (a
single-process replay outside 5b).

**After 5a (`madengine nan group`) and G1's 5b (`madengine nan replay`)**: show the user
the grouping (`groups.txt`; `groups_priority.txt` = launch order, `FALSE_POSITIVE?` =
allocator/view op, not replayed; a Triton / Inductor kernel's group is keyed on its source hash
and operand shapes, not its stack, so every run's capsules of one kernel are one group), read G1's
`summary.txt` as below, and do 5a-load. A **`COLLECTIVE`** G1 (a synchronous collective made
the NaN on every member rank, `step4-distributed.md`) has no ladder: its 5b is the collective
replay over the member ranks' capsules, `madengine nan exec --image <IMAGE> --gpus <one per rank>
-- torchrun --nproc_per_node=<ranks> -m nan_doctor_rt.collective_r3 replay <capsule dir> --backend
vllm` through `mad-run.sh` (read its `VERDICT` / `NARROW_INTERMEDIATE?` lines and `LADDER_RESULT=`,
not a `summary.txt`); concurrency and 5a-load do not apply. Then by
`<GROUPS_TO_REPLAY>` (step 1):

* **`largest`** (default): G1 -- the replayable group with the most capsules -- is
  the only one replayed. Ledger the other groups as
  `not replayed (GROUPS_TO_REPLAY=largest)` with their op and count.
* **`all`**: replay the remaining groups one after another within the step-1 budget:
  `madengine nan replay <CAPSULE> ... --out-dir <ARTIFACT_DIR>/replay/<G<i>>` through
  `mad-run.sh`, then `madengine nan package` for each that reproduces (ledger each with
  `$LG job step5b[G<i>]`).

A group that needed `step5-pressure.md` (G1 under `largest`, any group under `all`)
escalates the same way.

## Empty `<CAPSULE_DIR>` (`madengine nan group` finds no capsule): diagnose, do not go to 5b

Likeliest first:

* capture never ran -- `AssertionError` from `_rank_path` = dir not writable in the
  container (exits 0, looks clean); fix the mode, re-run;
* NaN outside the op filter's class (e.g. `"gemm"`), so the filter hid it -- drop the filter;
* `sync_stop()` missing (world size > 1);
* no NaN this time -- check `<NAN_SIGNAL>`, re-read `<FULL_MODEL_RATE>`; if step 4
  shortened the run, lengthen it first;
* **settle mode never fired** -- `madengine nan waves` scores capture runs
  `NAN_NO_CAPSULE` (the model printed its NaN line, nothing on disk), and
  `madengine nan digest` shows `pending:` / `MASKED` lines for the op: the NaN never
  outlived its step (fresh tensors per step, the model stops on it). Re-run the capture
  with `NAN_CAPTURE_SETTLE=0` in the run's environment (`env-madengine.md`); one capture run.

Capture ran and the model NaN'd, yet still no capsule (or the fixed re-run is empty
too): **step 5N** (`step5n-native.md`). So does a grouping with no replayable group.

## 5a-load: `<CONTENTION>` (measures nothing; ledger only)

Processes per GPU, their sources, env amplifiers. The yardstick for 5b's
`--concurrency` (starts at capture + 1) and the step-7 recipe sweep's `--contend <C-1>`. The knob lives in
one of two places -- read both:

* **in the model** (`--workers 6` -> `mp.Process`): same PID namespace, `gpu_procs`
  counts it;
* **in the launcher** (2 single-process containers per GPU): invisible to the capsule
  (`gpu_procs=1`) unless Declare-contention set `containers_per_gpu`; otherwise only
  the launcher's arithmetic and `held by others:` show it.

0. Do not re-read the launcher or source: step 1's ledger line has the launcher's
   containers per GPU, and `madengine nan capture-points` (step 4) already printed the
   `FANOUT` / `DIST` / `LAUNCH` hits -- re-run it if they are not in context.
1. Capsule first: the replay's `contention:` banner; `captured under:` is the command
   line (inside `mp.spawn`, read the **parent's** line; the stub prints as `worker
   argv:`). `not recorded` = old capsule, go to 2.
2. `<FULL_MODEL_RUN_SCRIPT>` and model source, for: `mp.Process`/`Pool`/`spawn`/
   `--workers N`; more containers than GPUs; `torchrun --nproc-per-node`,
   `--ntasks-per-node`; `DataLoader(num_workers=)`; the `hogger` module or a co-scheduled
   job; extra `torch.cuda.Stream`s, graph capture; `HSA_DISABLE_CACHE`,
   `TORCH_ROCM_FA_PREFER_CK`, `GPU_MAX_HW_QUEUES`; `OMP_NUM_THREADS` vs cores.
3. Ledger `<CONTENTION>`.

**The banner.** `_gpu_procs()` scans `/proc` for `/dev/kfd` holders -- per PID
namespace, blind to sibling containers; `containers_per_gpu` multiplies it.

| banner | reading |
|---|---|
| `gpu_procs` > 1 | in-process fan-out; `--contend-auto` correct |
| `gpu_procs` = 1, `held by others:` ~1 GB | solo at capture (~1 GB is the floor: own HIP context) |
| `gpu_procs` = 1, `held by others:` large | **uncounted co-tenant** (`!!` line); `--contend-auto` refuses -- use `--contend N` or `--contend-with` |
| `containers_per_gpu` present | declared; `--contend-auto` correct |

Any `0/N` from `--contend-auto` on an under-counted capsule (older builds print a
rate after `-> 0 co-tenant(s)`) is invalid, not negative.

## Reading a 5b result

Read **`summary.txt`** in each attempt dir (`madengine nan summary <attempt_dir>`: per-copy
`NAN`/`CLEAN`/`OOM`/`ERROR` with `n/N`, one `[C1]` table, `env:` banners, totals,
the loop's `first op to go non-finite (all copies)` tally, a `proposed verdict` line),
not copy logs; the kernel is in `kernel.*.txt` beside the attempts. The
proposed verdict applies the table below mechanically -- **confirm it**, don't
re-derive it. In order:

1. **Banners:** every copy `matches capture`, plus `captured build:` (else
   `step5-replay-env.md`) -- a wrong environment gives a clean `0/N`. With
   `--synthesize`, the `operands:` banner must not read `SYNTHETIC-INCONSISTENT`:
   derived operands (an attention backward's `out`/`softmax_lse`) drawn independently
   of q/k/v overflow in a correct kernel too. That replay can show a NaN but is no
   verdict and no fix validator: re-check on consistent operands
   (`madengine nan exec --image <IMAGE> -- python3 -m nan_doctor_rt.attn_consistent_check <capsule>`
   for a varlen backward), and say so in the README.
2. **`[C1]` walk** -- what the op was handed:
   * capture inputs not finite (`inputs_finite=False`, a dirty **input**) ->
     `PROPAGATED`: bug upstream; report op and operand. A dirty **written** operand
     (`dq`/`dk`/`dv`, `out=`: `inputs_finite=None`) is not one -- `empty_like`
     memory is routinely NaN; the repair test (`--written-operands`,
     `step5-replay-guardrail.md`) settles whether it matters;
   * `terminal fp64 ref on replay operands: finite` with the terminal op non-finite
     -> `KERNEL_BUG`, whatever the ancestors did: on exactly those operands no
     correct kernel returns non-finite;
   * a **wrong-value capsule** (origin `wrong_value`: `NAN_CAPTURE_VERIFY_EVERY`, a finite
     signal) counts a run when the terminal op fails the capsule's value check again; for a
     fused op with a `math_refs.VALUE_CHECKS` entry that check is
     an fp32 recompute of what its consumer reads, so a reproducing loop with every ancestor
     `match` is `KERNEL_BUG` even where the walk prints no fp64 reference for the op; the
     check also owns the op's raw outputs (unused scratch slots are never scored);
   * a **twin capsule** (`verify` kind `twin`, `NAN_CAPTURE_TWIN_EVERY`): a run counts when the op disagrees
     with its second implementation (`twin_source`) beyond the capsule's threshold again; the walk's terminal
     line says `twin check`. Its strength is the twin's independence (`twin_kind`: `native` > `alt_kernel` >
     `fp64_twin`) -- say which in the report. A twin of an in-place op needs pre-call inputs the replay does not
     keep: there the replay judges finiteness only (re-check by hand with `twins.check(..., pre=...)`);
   * a **rerun capsule** (`verify` kind `rerun`, `NAN_CAPTURE_RERUN_EVERY`): each run executes the
     terminal op twice, the second time on identical copies of its operands, and counts when they
     differ; the walk's terminal line reads `rerun`, and a reproducing loop is `KERNEL_BUG` whatever
     the ancestors did -- different answers on identical operands are the kernel's own;
   * a GEMM row `NO` / `WRONG VALUE` with every ancestor `match`, trailing views
     after it (`transpose`) -> it IS the terminal: `KERNEL_BUG` (deterministic on
     these operands). Its own `DIVERGED` is the op failing, not an ancestor;
   * terminal `fp64 ref` `NON-FINITE` (in the column or on replay operands) ->
     `FP32_OVERFLOW?`: not a kernel bug. The status covers two causes -- read the
     reason beside it: `exceeds <dtype> max` is an overflow; `true result non-finite in
     fp64` is undefined arithmetic (`0*inf`, `0/0`; the capture's `math` origin), not
     an overflow -- report it as that. `unavailable` / `-` is **no reference**, never
     an overflow;
   * an ancestor `DIVERGED` -> finite-but-wrong operands possible -- a flag
     (allocator rows -- `empty*`, `new_empty*` -- are never compared: uninitialised);
   * walk itself non-finite, no `DIVERGED`, no terminal line -> deterministic:
     `step5-replay-guardrail.md`.
   * a **`triton.<kernel>.<variant>`** op is a Triton kernel launched from Python
     (`kernel[grid](...)`, never in the dispatcher; `NAN_CAPTURE_TRITON=0` turns its capture
     off): `nan_doctor_rt.capsule_info`'s `triton` line names the module and source file, the
     args line names each parameter. Every tensor of up to 512 MiB is taken as written (Triton
     declares nothing), so its inputs read `inputs_finite=None`; no fp64 reference --
     `KERNEL_BUG_UNVERIFIED` when it reproduces, and step 7 reads the kernel's source.
   * an **`inductor.<kernel>.s<hash>`** op is a torch.compile kernel (`NAN_CAPTURE_INDUCTOR=0`
     turns its capture off): `nan_doctor_rt.capsule_info`'s `inductor` line names the kernel,
     its generated file and what it writes (`out_ptr*` / `in_out_ptr*`, exact); the ATen ops fused
     into it are in its name (`triton_poi_fused__softmax_mul_...`) and in the wrapper's `# Topologically
     Sorted Source Nodes` comment (`TORCH_LOGS=output_code`). Replay recompiles the recorded source (run it in the
     model's image). **Its fp64 reference is the kernel's own source promoted to fp64** (`nan_doctor_rt.inductor_ref`:
     every float type -> fp64, `tl.sqrt_rn` -> ocml's fp64 sqrt, inputs upcast exactly, run on the GPU): the
     replay's `[2] fp64 GPU reference` (op mode) or the walk's terminal line, so the proposed verdict is
     `KERNEL_BUG` (finite) or `FP32_OVERFLOW?` (non-finite in fp64 too: a fused `x / std` of a constant row
     is 0/0) as for any op -- confirm it. Same algorithm in a wider type, not an independent
     implementation. No reference (`unavailable`: an `in_out_ptr` kernel, `tl.dot`, inline asm, another
     fp32-only `tl.*` call): `KERNEL_BUG_UNVERIFIED` -- run the source nodes eagerly and in fp64 on the
     capsule's inputs by hand (`nan_doctor_rt.capsule_operands`) before calling it a kernel bug
     (`step5n-compiled.md` *Kernel capsule*, the same reading).
   * an operand marked `uninit` kept its bytes (<= 1 MiB, `NAN_CAPTURE_UNINIT_KEEP_BYTES`):
     allocator memory nobody wrote before the op (an op that declared it written but left it
     bit for bit as it was does not count), restored before the terminal op on every
     run (`kept uninitialised operand(s) restored`). If the op READS it -- a count, an index
     -- that value may be the trigger: `step5n-native.md` *Leftover memory?* tests it.
   * a capture log line `not captured: <op>'s <arg> holds the non-finite values it held
     before the call` is an op that declares a write it did not do (its out-buffer kept
     earlier garbage): not that op's NaN; the op that later writes the buffer is judged.
3. **Loop:** total `n/N` at the attempt's concurrency; tally of first non-finite op.

| verdict (ledger with `n/N`, concurrency, iters) | when |
|---|---|
| `KERNEL_BUG` | loop reproduced, and the terminal fp64 ref on the replay operands is finite -- or terminal `fp64 ref` finite with no ancestor `DIVERGED` |
| `KERNEL_BUG_UNVERIFIED` | loop reproduced, but the terminal op has no reference (`unavailable` / `-`: `_scaled_mm`, a multi-op kernel without a math reference, an unmodelled variant) or an ancestor `DIVERGED` |
| `PROPAGATED` / `FP32_OVERFLOW?` | from `[C1]`, reproduced or not (`FP32_OVERFLOW?` also covers undefined arithmetic: see above) |
| `NOT_REPRODUCED` | the completed attempt clean -- a bound (~1 in concurrency x GPUs x iters), not proof; a neighbour's kernel mix may be the trigger; say if the model reproduced. Model reproduces reliably: `step5-pressure.md` before reporting this |
| `NOT_SET_UP` | the futility stop: 0 in `--futility` x the capsule's op call index runs -- the model needed far fewer. **Not a negative**: the replay lacks the model's trigger. `step5-pressure.md` (kernel match, then co-tenants) |

* **Step 4 implicated `HipBlasLt`:** re-run the reproducing group with
  `--concurrency <C> --extra-args "--no-hipblaslt"`; zero without it
  confirms only if the run with it expected several hits.
* **`DIVERGED` = not reproducible, not proven wrong**: the named op may be
  downstream, not a bystander; the loop's first-non-finite tally localises. Separating
  the two needs an independent implementation (SDPA: `TORCH_ROCM_FA_PREFER_CK=0`).
  `MUTATED AFTER PRODUCTION` rows: the stored value may not be what the consumer read.
  **A small `DIVERGED` can be the node, not the op**: the walk compares one run with values
  recorded on the CAPTURE node, and a GEMM may dispatch another solution elsewhere (an `aten.mm`
  DIVERGED 0.0156 on one node, `match` on the capture node, bit-identical in every run of
  `--chain-repeat-check`). Before an ancestor `DIVERGED` changes the culprit or the sweep order,
  check it: an op that repeats bit for bit and is not the repeat-check's "first op to CHANGE" is
  a node difference -- name the repeat-check's op instead (`--mutate-chain-ops` / the culprit seed).
* **Stopped early** (`=== stopped early` in `summary.txt`, the default): the rate is
  non-finite / runs actually done, from ~10 hits -- roughly +-30%, slightly high.
  Quote it as such; it is enough to size step 7's experiments. A tighter rate (e.g. the
  `--no-hipblaslt` comparison): `--stop-hits` higher, or 0. `STOPPED` copies
  had no hit when the target was reached -- not clean.
* Report each group as it lands -- rates compare across hosts only as
  presence/absence. A reproducing group goes to step 6 at once.
* **G1 did not reproduce** (`NOT_REPRODUCED` / `NOT_SET_UP` after
  `step5-pressure.md`, or `PROPAGATED`): **step 5N** (`step5n-native.md`), then 7 --
  under `all` too, without waiting for the other groups. `FP32_OVERFLOW?` is a
  finding, not a trigger. **A COLLECTIVE G1 whose collective replay is `NOT_REPRODUCED`** goes straight to 5N
  too: `step5-pressure.md`'s co-tenant and leftover-memory tests replay single-op capsules and cannot reach a
  framework's spawned TP workers -- ledger them as not applicable, and check whether the captured call is the
  model's NaN at all (its request / step against the clean runs' NaN point: a request-0 all-reduce
  capture once came from a run whose clean runs NaN'd at request 9).
