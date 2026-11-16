---
name: mad-debug-nan
description: Debug a NaN, inf, or loss/gradient blow-up in a MAD PyTorch model — capture the op that first produced it with madengine's nan_doctor tool, replay it without the model, and package a standalone reproducer; optionally root-cause it and propose a fix. Use when a MAD model NaNs and the user wants to find out why.
argument-hint: "<tag> [--intake FILE] [-c '<context json>']"
disable-model-invocation: true
context: fork
agent: mad-nan-doctor
allowed-tools: Bash(madengine discover *) Bash(madengine nan *) Bash(rocm-smi *) Bash(amd-smi *) Bash(docker images *) Bash(bash .claude/skills/mad-common/preflight.sh) Bash(bash .claude/skills/mad-common/mad-run.sh *) Read Edit Write Grep Glob
---

# Debug a NaN

A ladder, each rung narrowing the search and able to end it: learn how the model runs,
measure the NaN, capture a capsule from the live model, replay it without the model,
package what reproduces (step 6: the reproducer is finished -- the NaN is **not** fixed
yet), and -- optional, budgeted, most expensive -- root-cause the NaN from that reproducer
(step 7) and fix it (step 8). Every investigation closes with step 9.

Debug `$ARGUMENTS`. The model tag is the first whitespace-delimited token. `--intake FILE`
answers step 1 from a file (`step1-intake.md`; the `mad-nan-hunt` workflow writes one);
`-c` is a madengine additional context for every model run (it must not carry `tools`).

## Pre-flight
Claude Code runs this before the skill starts. Other agents: run it first from the repo
root and stop if it fails.
```!
bash .claude/skills/mad-common/preflight.sh
```

**First output of every investigation, right after step 0:** print the block below to the
user verbatim -- the notice, then the overview table. Once per investigation (not again
after a compaction or on resume). Run step 0 silently, then **end that turn with the block
plus step 1's question 1 (the model tag, if `$ARGUMENTS` has none), with no tool call after
it**. Given an intake file: print the block, then go straight on with `step1-intake.md`.

```markdown
> **⚠️ Scope of this version:** mad-debug-nan focuses on generating a **mini-reproducer**
> -- a standalone package that triggers the NaN without the model (steps 0-6).
> **Root-causing (step 7) and bug-fixing (step 8) are experimental** and therefore
> optional: they run only if you ask for them, within a time budget you set, and a fix is
> proposed, never applied. **PyTorch only:** a JAX, TensorFlow or other non-PyTorch model
> is detected and the skill stops there. **AMD Instinct MI300X or newer only:** the skill
> is tested on MI300X-class (gfx942) and MI350X/MI355X (gfx950) GPUs; on any other GPU you
> get a warning, because the skill is likely not to work there.

| step | what it does | optional? |
|---|---|---|
| 1 | **Ask** how the model runs (run command, NaN signal) and your preferences; create the artifact dir and ledger | required |
| 2 | **Run** the model once to see that it NaNs in this environment; a few runs also measure the NaN rate. You CAN **skip** it and go to the capture (4) | optional (default: skip) |
| 3 | If the NaN rate is too low, **increase** it (e.g. by oversubscribing the GPUs) | interactive only; skipped if step 2 is |
| 4 | **Capture** the chain of ops that produced the NaN as a capsule from the model run | required |
| 5 | **Replay** the capsules without the model to confirm they reproduce | required |
| 5N | **Fallback**: write a **standalone repro** if no capsule exists or none replays | automatic, only if needed |
| 6 | **Package** the reproducer (directory + tarball + README). Ends reproducer generation; the NaN is not fixed yet | required |
| 7 | **Root-cause** the NaN from the reproducer (experimental) | optional, budgeted |
| 8 | **Fix** it and validate the fix on the reproducer, or find a model-level **workaround** (experimental) | optional, budgeted; needs a root cause from 7 |
| 9 | **Restore** sources, hand over the artifacts, **report** with a timing table | required |
```

**GPUs.** AMD Instinct GFX9: MI300X (gfx942) and MI350X/MI355X (gfx950). The tools read the
arch from the device and treat both alike; memory caps, tile/CU arithmetic and timings are
re-derived from the device in use, never carried over from another.

**This file is the map. On entering a step, read its file** (under `reference/`); read a
*conditional* file only when its trigger fires; never read a skipped step's file.

| step | read | conditional: only when |
|---|---|---|
| 1 | `step1-ask.md` and `env-madengine.md`; **given an intake file** `step1-intake.md` instead of `step1-ask.md` | `step1-contention.md`: the workload runs several processes per GPU of its own |
| 2 | `step2-rate.md` | |
| first model run (2, 3, 4) | `waves.md` | |
| 3 | `step3-raise.md` | |
| 4 | `step4-run.md` | `step4-capture.md`: the nan_doctor tool cannot see the loop (no optimizer step, a framework that owns the loop) or its knobs change. `step4-capture-memory.md`: an OOM, snapshot-skip lines to judge. `step4-capture-speed.md`: setup dominates time to capsule. `step4-distributed.md`: ONE job over several ranks (torchrun, FSDP/TP/EP/PP, any node count). `step4-explode.md`: the signal is a magnitude (grad norm / loss spike), or finite spikes precede a rare NaN |
| 5 | `step5-read.md` | `step5-run.md`: running 5a/5b. `step5-replay-env.md`: an `env:` banner not `matches capture`. `step5-replay-guardrail.md`: walk non-finite with no `DIVERGED`, or reproduces at concurrency 1. `contention-replay.md`: a single-process replay outside 5b. `step5-pressure.md`: 5b `NOT_SET_UP`, or `NOT_REPRODUCED` while the model reproduces, or a step-1 `TRIGGER` hint. `step4-distributed.md` (*Reading*, *Replay ladder*): world size > 1 |
| 5N | `step5n-native.md` | entered only from the exits it lists: no capsule (4), or G1 (the largest group) did not reproduce (5). `step5n-compiled.md`: N1 finds the NaN may be born inside torch.compile'd code |
| 6 | `step6-finish.md` | `step6-build.md`: building a package |
| 7 | `step7-rootcause.md` | `step7-recipe-sweep.md`: the reproducer's NaN is non-deterministic (recipe track (b)) |
| 8 | `step8-fix.md` | entered only with a valid `<ROOT_CAUSE>` from step 7 |
| 9 | `step9-report.md` | `step9-report-sweep.md`: step 7's recipe track ran |

## Rules at every step

1. **Commands.** The deterministic tools are `madengine nan <command>` on the host
   (`madengine nan --help`). Whatever needs torch runs in the model's image through
   `<GPU_EXEC>` = `madengine nan exec --image <IMAGE> [--gpus ...] --`, with the
   `nan_doctor_rt` package on `PYTHONPATH`: `<GPU_EXEC> python3 -m nan_doctor_rt.<module> ...`.
   `<IMAGE>` is the model's `docker_image` in `build_manifest.json` -- never asked.
2. **Long commands run detached through MAD's run helper.** The model
   (`madengine run`), and `madengine nan exec|group|replay|package`, start with
   `bash .claude/skills/mad-common/mad-run.sh start <run-id> -- <command>`; then
   `mad-run.sh wait <run-id>` until the first line says `MAD_RUN_STATUS=succeeded` or
   `failed`. A tool call ends after 10 minutes; the helper survives it and holds the host
   GPU lock, so no other GPU job shares the devices. One command per Bash call; never join
   with `&&`, `;` or a newline. Fresh run-id per run. `env-madengine.md`.
3. **Every file written goes under `<ARTIFACT_DIR>`** (`nan_hunts/<hunt>/` in the MAD repo
   root, gitignored; the intake names it). Never write to or edit a tracked MAD file; a
   source-wiring working copy lives in `<ARTIFACT_DIR>` too.

   ```bash
   mkdir -p "$ARTIFACT_DIR"/capsules "$ARTIFACT_DIR"/asm && chmod -R 777 "$ARTIFACT_DIR"
   ```

4. **The ledger** (`<LEDGER>` = `<ARTIFACT_DIR>/ledger.md`, append-only) survives
   compaction; memory does not. **Append the moment a rung yields a number or verdict,
   before reporting it; re-read it before any choice that depends on an earlier rung** and
   for every number in the report. One line each, **<= ~200 characters** -- put detail in a
   file and give its path: `<NAN_RATE>`/`<FULL_MODEL_RATE>` with run command and
   environment (or the step-2 skip); image; saved model-source path; `<CONTENTION>`; each
   5b verdict **with `n/N`, concurrency, iterations, host**; each tarball path + sha256;
   each step-7/8 budget, iteration and outcome; each recipe-sweep row.
5. **Rates are environment-bound** -- (run command, image, host and GPU arch, step count,
   containers per GPU, host threads per process) -- and so are decisions cached from them
   (step-3 skip, step-4 length and geometry). **Environment changed or a re-measurement
   disagrees: re-read `<LEDGER>`, re-enter the lowest dependent rung** (usually 3). A low
   rate is a step-3 problem before a hardware one. Never carry a rate over from a previous
   run: unmeasured is unmeasured. **The step a NaN or spike appears on is
   non-deterministic** (MoE: always) unless proven for this case -- bit-identical per-step
   values from >= 2 runs on different nodes, same flags (`step4-explode.md`).
6. **Contention travels with every rate**: a contention-gated capsule replayed alone gives
   a meaningless clean `0/N` (`step1-ask.md`).
7. **Time steps with `madengine nan ledger`** (it writes ledger lines and does the
   arithmetic):

   ```bash
   LG="madengine nan ledger <LEDGER>"
   $LG start step4;  $LG note step4 "<text>";  $LG job step4 <run-id> "<what>"   # quote: ( ) break the shell
   $LG wait step4 start <question>;  $LG wait step4 end
   $LG end step4     # compute / waiting-user / other
   $LG table         # step 9
   ```

   Record **every** long run with `job`; relay the `TELL USER:` line `end` prints. Tag
   steps 5b-8 per group (`step6[G1]`); a re-entered step gets a new `start`.
8. **No skipping ahead, per group**: a group may go to 6-8 while others are in 5b; step 9
   waits for every replayed group (`<GROUPS_TO_REPLAY>`) -- and, if G1 did not reproduce,
   for step 5N's verdict. Step 7 only from a finished step-6 reproducer, after lower rungs
   ruled out an upstream NaN and fp32 overflow; step 8 only with a valid `<ROOT_CAUSE>`.
9. **Spend tokens on decisions, not output.**
   * Read each step's file once; after a compaction re-read `<LEDGER>` first, then only the
     current step's file.
   * **One call per question, not per grep**: batch independent reads into one command
     (every call re-reads the whole context). A capsule:
     `<GPU_EXEC> python3 -m nan_doctor_rt.capsule_info <pt>` (op, dirty operands, each
     operand's dtype/shape/stride and small integer values, chain, origin/settle, model
     frames, CPU+fp64 re-run of a pointwise op -- the 0/0 false-positive check) instead of
     ad-hoc `torch.load` scripts.
   * **Never read a model source whole**: map it with `madengine nan capture-points`
     (step 4) and read the windows it names.
   * **Never `cat` a model log**, run output or capsule dump. What the capture did (armed,
     pending/MASKED, captured op, settle, census): `madengine nan digest <log or run dir>`.
     Several runs: `madengine nan waves <log dir|files> --signal '<NAN_SIGNAL>'
     [--capsules <CAPSULE_DIR>]`. Exploding (finite) values: the per-step trajectory,
     `madengine nan exec ... -- python3 -m nan_doctor_rt.grad_trajectory <grad.jsonl...>`
     (`NAN_DOCTOR_GRADLOG`, `step8-fix.md`). 5b: the attempt's `summary.txt` /
     `LADDER_RESULT=`. Else `grep -m` / `tail -n` for the line the decision needs.
   * **Wait once, not by polling**: `mad-run.sh wait` blocks up to 9 minutes; call it again
     only when it returns `running`. Then read only the run's last line and the files it
     names. A status question from the user: `tail -n 5 <LEDGER>`.
   * Quiet flags (`git --no-pager`, `-q`); quote the line that matters and give the path
     rather than echoing files or JSON.
10. **Report, then keep climbing.** When a rung yields a result, report it in a line or two
    and continue *in the same turn* to the next rung that needs no user decision (a
    reproducing 5b -> step 6 -> step 9 once every replayed group has a verdict, unless
    steps 7-8 were wanted; no capsule, or G1 clean -> step 5N -> 6). Never end a turn on
    "next I'll ...". Stop only for a user decision (beyond the step-1 budget, the step-7/8
    time budgets, the sweep plan's cost, an unresolvable CHECK) or to wait on a run.
    Inside a step-7/8 loop, keep iterating until it ends.
11. **Talk to the user in their terms, not the skill's.** Say "the capture run" for the
    madengine run with the `nan_doctor` tool, and, the first time, what it does: it runs
    the model until a NaN outlives its step, and saves the op that made it.
12. **Fixes are proposals.** Never edit a tracked MAD file, the model's registry entry, or
    the image to fix a NaN; step 8 produces a proposal and its validation on the
    reproducer.

## Placeholders

| name | what | step |
|---|---|---|
| `<MAD_ROOT>` | the MAD repository root, where every command runs | 0 |
| `<ARTIFACT_DIR>`, `<LEDGER>` | rule 3, rule 4 | 1 |
| `<RUN_SCRIPT>`, `<NAN_SIGNAL>`, `<GPU_EXEC>`, `<IMAGE>` | the model's `madengine run` command; how its output shows a NaN (or a threshold regex on a logged magnitude, `step4-explode.md`); rule 1; the model's image | 1 |
| `<WORLD_SIZE>` | ranks of ONE job; > 1 -> `step4-distributed.md` | 1 |
| `<ORIGIN_RANKS>` | the job manifest's origin rank(s): whose capsule 5b replays | 5a (world size > 1) |
| `<NAN_RATE>` | e.g. "3 of 8 runs" **with its environment**; or *unmeasured* (step 2 skipped) | 2 |
| `<MORE_NANS_RUN_SCRIPT>`, `<MORE_NANS_RATE>` | rate-raising variant and its rate | 3 |
| `<FULL_MODEL_RUN_SCRIPT>`, `<FULL_MODEL_RATE>` | step-3 versions if it ran, else the originals -- the rate step 5N and step 8's model-level workaround compare against | 3 |
| `<CAPSULE_DIR>` | `<ARTIFACT_DIR>/capsules/`, empty before capture | 4 |
| `<CONTENTION>` | processes per GPU at capture, sources, env amplifiers | 5a-load |
| `<OUT_JSON_PATH>` | `<ARTIFACT_DIR>/groups.json` | 5a |
| `<GROUPS_TO_REPLAY>` | `largest` (default: only the replayable group with the most capsules, G1) or `all` | 1 |
| `<CAPSULE_FILE>` | a group's representative `.pt` | 5b |
| `<CAPSULE_ITER_COUNT>`, `<CAPSULE_CONCURRENCY>` | chain runs per copy (1000000); copies per GPU (capture's processes per GPU + 1; one less per OOM, down to 1) | 5b |
| `<RC_BUDGET>`, `<FIX_BUDGET>` | wall-clock budgets the user gave for the step-7 / step-8 loops | 7, 8 |
| `<NAN_KIND>` | `deterministic` or `non-deterministic`, from the step-6 reproducer | 7 |
| `<REPORT_JSON>`, `<ASM_DIR>` | `<ARTIFACT_DIR>/mutations.json`; `<ARTIFACT_DIR>/asm/` | 7 (recipe) |
| `<MUTATE_ONLY>` | `--mutate-only` flags of the current sweep tier | 7 (recipe) |
| `<ROOT_CAUSE>` | the step-7 outcome: a validated root cause, or `NO_ROOT_CAUSE` | 7 |
| `<FIX>` | the step-8 outcome: the fix proposal's path, or `NO_FIX` | 8 |

## The ladder (one line each; the step files hold the rules)

**0. Gates, from the MAD repo root.** No artifact dir or ledger yet: step 1 creates both
once the run command is known.

**PyTorch gate -- the first thing done with the model, before anything is created.** Every
rung hooks PyTorch (the capture is a `__torch_dispatch__` mode, a capsule is saved ATen
ops, the replay re-dispatches them); a non-PyTorch model yields no capsule and no replay.
Run, on the host:

```bash
madengine nan framework-check <RUN_SCRIPT or the model's run script> [--source <model.py>]
```

Static (no torch, no GPU); last line `FRAMEWORK=<f> VERDICT=<v>`:

* `RUN` (`pytorch`) -- continue; step 1 ledgers the line.
* `SKIP` (`jax`, `tensorflow`, ...) -- **do not apply the skill**: no artifact dir, ledger
  or run. End the turn with this warning, filled in, and nothing else of the skill:

  ```markdown
  > **⚠️ mad-debug-nan skipped: the model is not a PyTorch program** -- it runs
  > <framework> (<the evidence line, e.g. `entry: jax (jax, flax.linen, optax)`>).
  > mad-debug-nan works only on PyTorch models: it captures the failing op through
  > PyTorch's dispatcher (`__torch_dispatch__`), saves it as PyTorch tensors and replays
  > it through PyTorch, so on <framework> it would capture nothing and could build no
  > reproducer. Nothing was created or run.
  ```

  If the user insists, repeat why it cannot work; never wire the capture into a
  non-PyTorch model.
* `ASK` (`mixed(...)` -- e.g. torch only loads the data -- or `unknown`) -- one question:
  "Is the model's compute (forward/backward) written in PyTorch?" with the evidence lines.
  Yes -> continue and ledger the answer; no -> the `SKIP` warning above.

**GPU gate -- warn on anything below MI300X.** The tooling assumes ROCm and the GFX9-family
ISA of AMD Instinct MI300X and newer; other GPUs are untested. On the host:

```bash
madengine nan gpu-check     # no torch needed
```

Last line `GPU=<name> ARCH=<arch> VERDICT=<v>`:

* `SUPPORTED` (gfx942 MI300X/MI300A/MI325X, gfx950 MI350X/MI355X, newer Instinct MI<n>,
  n >= 300) -- nothing to say; step 1 ledgers the line.
* `UNTESTED` (MI250X/MI210, Radeon, NVIDIA, ...) -- print this warning, filled in, **at the
  top of the first output**, then continue:

  ```markdown
  > **⚠️ Untested GPU: <name> (<arch>).** mad-debug-nan is tested only on AMD Instinct
  > MI300X or newer (gfx942, gfx950). All other GPU types are untested, and the skill is
  > likely not to work on them: the capture, replay and kernel tools assume ROCm and the
  > MI300X-class instruction set. Results here may be wrong or missing.
  ```

  Ledger it (`$LG note step1 "GPU UNTESTED: <name> <arch>"`) once the ledger exists, and
  repeat it in the step-9 report.
* `NO_GPU` -- this host has none (a login node, or SLURM): the verdict comes from step 1's
  `<GPU_EXEC>` check, which runs `python3 -m nan_doctor gpu-check --torch` on the model's
  GPUs; an `UNTESTED` there prints the same warning before step 2 or 4 starts.

1. **Ask everything up front, one question at a time with a default** -- or read it from
   the intake file. Model and run command (then the PyTorch gate; derive the rest from it;
   then create `<ARTIFACT_DIR>` and `<LEDGER>`), step 2, run geometry, step-7/8 wish,
   groups to replay (default: largest only), budget.
2. **Measure** (always optional, ask): >= 8 runs; skipped = *unmeasured* (never a prior
   run's rate) -> skip 3.
3. **Raise** (optional, interactive only): skip if high (ledger with environment); else
   oversubscribe. The automated `mad-nan-hunt` workflow never runs it (it fights the GPU
   lock).
4. **Capture**: the model's run with the `nan_doctor` tool, armed after init; stop at the
   first capsule. World size > 1: every rank stops together and rank 0 writes
   `manifest.json` (origin rank, arrivals). Wrong finite values: `NAN_CAPTURE_BOUNDS` /
   `NAN_CAPTURE_VERIFY_EVERY`, with a known-good control (`step4-explode.md`).
5. **Read** the grouping (`madengine nan group`) and each 5b verdict
   (`madengine nan replay`); with `<GROUPS_TO_REPLAY>=all`, replay the other groups one
   after another (default `largest`: G1 only). World size > 1: `madengine nan group --job`;
   replay the origin rank's capsule (R1), climbing R2/R3/5N only as `step4-distributed.md`
   says. Clean while the model is not: same kernel? then the model, then a synthetic
   training-step stand-in, as co-tenant (`step5-pressure.md`).
5N. **Native fallback** (automatic): no capsule from 4, or G1 (the largest group) did not
   reproduce in 5 -> `madengine nan handoff` turns the capsule path's failures into
   constraints, then write a standalone script at the model's shapes and load, reduce it
   while it NaNs, verdict with an fp64 check; a hit goes to 6.
6. **Finish the reproducer** (`madengine nan package`) -- each reproducing group's
   directory and tarball (confirm the README's CHECK line), or the native one
   (`repro-native/`). This ends reproducer generation only; the NaN is not root-caused or
   fixed yet.
7. **Root-causing** (experimental, optional; the user gives `<RC_BUDGET>` on entry): a loop
   over the step-6 reproducer, each iteration resuming from the last one's notes; track (a)
   native LLM debugging always, track (b) a special recipe in parallel when one exists for
   `<NAN_KIND>` (today: non-deterministic -> the mutation sweep). Ends with `<ROOT_CAUSE>`
   or `NO_ROOT_CAUSE` when the budget runs out.
8. **Fixing** (experimental, optional; only with a valid `<ROOT_CAUSE>`; the user gives
   `<FIX_BUDGET>` on entry): a loop -- resume from the last iteration, propose a fix
   (native LLM), validate it on the reproducer; a model-level workaround (a component
   turned off) is a candidate too. Ends with `<FIX>` or `NO_FIX`. A proposal only.
9. **Restore and report** (`madengine nan report`): restore any wired source, hand over the
   artifacts, write the report with the timing table.

## Agent Notes
- Claude Code, Cursor and Codex all load this skill (Codex through `.agents/skills/`). Only
  Claude Code substitutes `$ARGUMENTS`; elsewhere it means the text the user gave after the
  skill name.
- Claude Code runs this skill in the `mad-nan-doctor` subagent. In Cursor, delegate to the
  `mad-nan-doctor` subagent (Cursor reads `.claude/agents/`); in Codex, spawn the
  `mad_nan_doctor` agent (`.codex/agents/`). Otherwise run it inline and read
  `.claude/agents/mad-nan-doctor.md` first for the conventions it assumes.
- For an unattended run of steps 1-6 and 9 (plan by default, `--execute` to run), use the
  `mad-nan-hunt` workflow.
