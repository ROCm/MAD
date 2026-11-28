---
name: debug-numeric
description: Numeric Doctor (mad-numeric-doctor). Root-cause a numeric failure -- a NaN, an Inf, a magnitude spike (loss / grad-norm explosion) or a wrong finite value -- produced by a PyTorch model on AMD Instinct GPUs, MI300X or newer -- MI300X, MI325X, MI350X/MI355X, MI455X -- (other GPUs are untested and get a warning; PyTorch only -- a JAX, TensorFlow or other non-PyTorch model is refused with a warning) -- bisect the component, capture the op that first produced the bad value as a capsule from the live model, replay it without the model, package a reproducer, then optionally root-cause the failure from that reproducer and fix it. Use for "my training run NaNs", "debug this NaN", "the loss explodes", "the model outputs garbage / wrong values", "is this a kernel or library bug", "root cause the NaN", "debug <script or example> with numeric doctor".
allowed-tools: Read, Edit, Write, Bash, Glob, Grep, Agent
---

# Debug a numeric failure

A numeric failure is a NaN, an Inf, a magnitude spike or a wrong finite value; the
steps below say "NaN" for all of them unless a step says otherwise (a magnitude or
wrong-value signal: `step4-explode.md`). A ladder, each rung narrowing the search and
able to end it: learn how the model runs, check that it runs and measure the failure,
capture a capsule from the live model, replay it without the model, package what
reproduces (step 6: the reproducer is finished -- the failure is **not** fixed yet),
and -- on by default, budgeted, most expensive -- root-cause the failure from that
reproducer (step 7) and fix it (step 8). Every investigation closes with step 9.

**Few words, one question.** The user reads as little and answers as little as
possible: by default the only question is step 1's question 1 (`<RUN_SCRIPT>`); every
other choice takes its default (`step1-ask.md`). **First output of every
investigation, right after step 0:** one line saying what the skill will do, then
question 1 -- no scope notice, no overview table:

```markdown
I'll first generate a standalone mini-reproducer for the numeric failure, then
root-cause and fix it, using defaults throughout (say *show the steps* or *ask me the
options* at any time). How do I run the model?
```

**This holds whenever the skill is loaded or invoked**, even with no failure described
and no run script given (e.g. "load the skill", `/debug-numeric` alone): loading it
starts an investigation, so run step 0 silently and **end that turn with those lines,
with no tool call after them** (text before a tool call can scroll off or collapse under
the tool output). Once per investigation (not again after a compaction or on resume). A
delegated or non-interactive run (the step-1 answers given up front, no user to wait
for): take the given answers and go on without ending the turn.

**The block below is shown only on request** -- the user asks for the steps, the
scope, what the skill does or what it supports ("show the steps", "what will you
do?"): print it verbatim, **the scope notice first, then the overview table**; never
the table without the notice, never a summary or paraphrase in its place. The notice
keeps its bullet form; never reflow it into a paragraph. A gate warning
(`step0-gates.md`) is not this block: it prints whenever its gate fires.

```markdown
> **⚠️ Scope of this version:**
>
> * **Mini-reproducer focus:** debug-numeric generates a standalone package that
>   triggers the NaN, Inf or wrong value without the model (steps 1-6).
> * **Experimental root-causing and fixing:** steps 7 (root cause) and 8 (fix) run
>   by default within a default time budget (1-4 h each); say so if you want to stop
>   at the reproducer or set your own budget.
> * **PyTorch only:** the model must be a PyTorch program. A JAX, TensorFlow or other
>   non-PyTorch model is detected from your run script and the skill stops there.
> * **AMD Instinct MI300X or newer only:** tested on MI300X/MI325X (gfx942),
>   MI350X/MI355X (gfx950) and MI455X (gfx1250) GPUs. On any other GPU you get a
>   warning, because the skill is likely not to work there.

| step | what it does | optional? |
|---|---|---|
| 1 | **Ask** which script runs the model; derive the NaN signal and GPU environment from it; create the artifact dir and ledger | required |
| 2 | **Run** the model to make sure that it runs in this environment and measure the NaN rate. On one node this happens inside the capture job (step 4), whose control runs measure it; a separate run first only on request, or for a multi-node model (one smoke run). No NaN line at all (the outputs may be silently wrong): **compare** the model's outputs across configurations -- graphs, compilation, fused ops, attention backend -- to find the one that is wrong | required (default) |
| 3 | If the NaN rate is too low in the given model run, we can try to **increase** the NaN rate first (e.g. by oversubscribing the GPUs) | optional (default: off); needs a separate step-2 run |
| 4 | **Capture** the chain of ops that produced the NaN as a capsule from the model run (a multi-node run: first find the fewest nodes that still produce the same NaN) | required |
| 5 | **Replay** the capsules without the model to confirm they reproduce | required |
| 5N | **Fallback**: write a **standalone repro** if no capsule exists, none replays, or the capture cannot reach the model | automatic, only if needed |
| 6 | **Package** the reproducer (tarball + README). Ends reproducer generation; the NaN is not fixed yet | required |
| 7 | **Root-cause** the NaN from the reproducer (experimental) | required (default), budgeted; you can turn it off |
| 8 | **Fix** it: the real fix to the faulty code (shipped even when it cannot be built here, labelled so), the narrowest **workaround** that touches only the affected calls, and the broad one (a component off) -- each validated on the reproducer and the model, with its cost, all packaged and reported side by side (experimental) | required (default), budgeted; needs a root cause from 7 |
| 9 | **Restore** sources, hand over the artifacts, **report** with a timing table, and the result card (`result.json`) | required |
```

**GPUs.** AMD Instinct: MI300X/MI325X (gfx942), MI350X/MI355X (gfx950) and MI455X (gfx1250).
The tools read the arch from the device (or the kernel's path / `_ISA` tag) and treat them alike;
every measured number in these files comes from gfx950 (MI355X: 288 GiB, 256 CUs, 8
XCDs) unless it says otherwise. On MI300X (192 GB, 304 CUs, 8 XCDs) re-derive memory
caps, tile/CU arithmetic and timings from this device, never from those numbers.

**This file is the map. On entering a step, read its file** (under `reference/`);
read a *conditional* file only when its trigger fires; never read a skipped step's
file.

| step | read | conditional: only when |
|---|---|---|
| 0 | -- (the step-0 command reads `step1-ask.md`; `env.md` follows in step 1) | `step0-gates.md`: a gate did not pass (`SKIP`/`ASK`/`UNTESTED`) |
| 1 | `step1-ask.md`, then `env.md` | `step1-contention.md`: the launcher is not the simple `gpu=trial%N` concurrent loop. `env-runtime.md` (any step): an image to pull/build/load, `RUNTIME=apptainer`, or an environmental failure |
| 2 | `step2-rate.md` | `step2-diff.md`: no `<NAN_SIGNAL>` at all -- the model prints nothing when it is wrong and the user suspects silently wrong outputs, or asks to compare configurations (differential mode makes the signal) |
| first model wave (2, 3, 4) | `waves.md` | |
| 3 | `step3-raise.md` | |
| 4 | `step4-run.md` | `step4-capture.md`: no capture launcher for this run script yet, or its source/arming/filter/cap changed. `step4-capture-memory.md`: untested setup, an OOM, snapshot-skip lines to judge. `step4-capture-speed.md`: setup dominates time to capsule. `step4-distributed.md`: ONE job over several ranks (torchrun, FSDP/TP/EP/PP, any node count). `step4-explode.md`: the signal is a magnitude (grad norm / loss spike) or a wrong finite value (no NaN at all), or finite spikes precede a rare NaN |
| 5 | `step5-read.md` | `step5-run.md`: running 5a/5b yourself (a failed capture-to-reproducer job, or extra groups beyond `next_groups.sh`). `step5-replay-env.md`: an `env:` banner not `matches capture`. `step5-replay-guardrail.md`: walk non-finite with no `DIVERGED`, or reproduces at concurrency 1. `contention-replay.md`: a single-process replay outside 5b. `step5-pressure.md`: 5b `NOT_SET_UP`, or `NOT_REPRODUCED` while the model reproduces, or a step-1 `TRIGGER` hint. `step4-distributed.md` (*Reading*, *Replay ladder*): world size > 1 |
| 5N | `step5n-native.md` | entered only from the exits it lists: no capsule (4), G1 (the largest group) did not reproduce (5), or a setup the capture cannot reach (1/4: `CAPTURE_UNREACHABLE`). `step5n-compiled.md`: N1 finds the NaN may be born inside torch.compile'd code (capsule `PROPAGATED` at the first eager op after a compiled module, or no capsule with most of the step compiled) |
| 6 | `step6-finish.md` | |
| 7 | `step7-rootcause.md` | `step7-recipe-sweep.md`: the reproducer's NaN is non-deterministic (recipe track (b)) |
| 8 | `step8-fix.md` | entered only with a valid `<ROOT_CAUSE>` from step 7 |
| 9 | `step9-report.md` | `step9-report-sweep.md`: step 7's recipe track ran |

`[E<n>]` tags cite the maintainers' evidence notes, which are not part of this
repository; the rule they back stands on its own.


## Rules at every step

1. **Paths.** `<REPO_ROOT>` (step 0) is substituted everywhere. Tools:
   `<REPO_ROOT>/scripts/*.py` (flat; on `PYTHONPATH` inside every runtime); the
   execution environment: `<REPO_ROOT>/scripts/nd` (below: `nd`), its layers in
   `scripts/env/`, its jobs in `scripts/jobs/` (`env.md`).
2. **Every GPU command runs through `nd` with `ND_ENV=<ENV>`** (steps 4-8): `nd run`
   for one command on this GPU node, `nd submit -- <job>` for a job. Blocks show the
   command as `nd run` takes it: single-quoted, paths written as `$ND_REPO` / `$ND_ART`
   (the repo and the artifact dir as the runtime mounts them), never host paths.
3. **Every file written goes under `<ARTIFACT_DIR>`** -- the runtime mounts the repo
   and the artifact dir and nothing else, and a bad path fails only after the GPU work:

   ```bash
   ARTIFACT_DIR=<REPO_ROOT>/artifacts/<RUN_TAG>   # fresh per investigation, e.g. 2026-10-05-llama-pretrain
   mkdir -p "$ARTIFACT_DIR"/capsules "$ARTIFACT_DIR"/asm &&
     chmod 777 "$ARTIFACT_DIR" "$ARTIFACT_DIR"/capsules "$ARTIFACT_DIR"/asm   # not -R: a rerun hits root-owned job files
   ```

4. **The ledger** (`<LEDGER>` = `<ARTIFACT_DIR>/ledger.md`, append-only) survives
   compaction; memory does not. **Append the moment a rung yields a number or
   verdict, before reporting it; re-read it before any choice that depends on an
   earlier rung** and for every number in the report. One line each, **<= ~200
   characters** -- detail goes in a file, the line gives its path:
   `<NAN_RATE>`/`<FULL_MODEL_RATE>` with script and environment (or the step-2 skip); image (from `<RUN_SCRIPT>`); `<CAPTURE_NODES>`; saved
   model-source path; `<CONTENTION>`; each 5b verdict **with `n/N`, concurrency,
   iterations, host**; each tarball path + `sha256sum`; each step-7/8 budget,
   iteration and outcome; each recipe-sweep row.
5. **Rates are environment-bound** -- (script, image, host and GPU arch, step count,
   containers per GPU, host threads per process, node count) -- and so are decisions cached from
   them (step-3 skip, step-4 length and geometry). **Environment changed
   or a re-measurement disagrees: re-read `<LEDGER>`, re-enter the lowest dependent
   rung** (usually 3). A low rate is a step-3 problem before a hardware one. Never
   carry a rate over from a previous run: unmeasured is unmeasured. **The step a NaN
   or spike appears on is non-deterministic** (MoE: always) unless proven for this
   case -- bit-identical per-step values from >= 2 runs on different nodes, same
   flags; environment-bound like a rate (`step4-explode.md`).
6. **Contention travels with every rate**: a contention-gated capsule replayed alone
   gives a meaningless clean `0/N` (`step1-ask.md`).
7. **Time steps with `scripts/ledger.py`** (it writes ledger lines and does
   the arithmetic):

   ```bash
   LG="python3 <REPO_ROOT>/scripts/ledger.py <LEDGER>"
   $LG start step4;  $LG note step4 "<text>";  $LG job step4 <jobid> "<what>"   # quote: ( ) break the shell
   $LG wait step4 start <question>;  $LG wait step4 end
   $LG end step4     # queue/compute (sacct; overlap once)/waiting-user/other
   $LG skip step2 "<why>"   # a step not run: no start/end; the table lists it as skipped
   $LG table         # step 9
   ```

   Record **every** batch job with `job` (`nd submit --step TAG` does it); relay the `TELL USER:` line `end` prints.
   Steps 5b-8 are per group: `start step6[G1]` (`[native]` for 5N's package); a bare `start step6` is
   refused. A re-entered step gets a new `start`.
8. **No skipping ahead, per group**: a group may go to 6-8 while others are in 5b;
   step 9 waits for every replayed group (`<GROUPS_TO_REPLAY>`) -- and, if G1
   did not reproduce or the capture could not reach the model, for step 5N's verdict. Step 7 only from a finished step-6
   reproducer -- whatever its verdict: a kernel / library fault and a model / numerics
   bug (undefined arithmetic, an overflow, a bad config or input) both go on to 7-8;
   step 8 only with a valid `<ROOT_CAUSE>`.
9. **Spend tokens on decisions, not output.** Every call re-reads the whole context
   (late in an investigation ~150-190k tokens), so cost = calls x context.
   * **Fewer calls.** Batch independent work into one command: the ledger lines a
     result needs, the next step's file and that step's first command go in the call
     that reads the result. Read each step's file once, whole, in one `cat` (not in
     `sed -n` slices, not with an offset); after a compaction re-read `<LEDGER>` first,
     then only the current step's file. Never check what a pending job's wait will
     report anyway. Keep one call's output under ~25k characters: a bigger one spills
     to a file you then read again. That bounds batching step files too: most are 5-20k,
     so batch two small ones at most. Four are over the limit alone (`step1-ask.md`,
     `step7-rootcause.md`, `step7-recipe-sweep.md`, `step8-fix.md`: 28-30k) -- read each of those whole
     with the Read tool (no offset / limit), not `cat`. `wc -c` the set when unsure.
   * **Smaller outputs.** Never `cat` a model log (~2,600 lines), job output, capsule
     dump or model source (map it with `find_capture_points.py`, step 4, and read the
     windows it names). Use the summarizers: a capsule -- `capsule_info.py <pt>` (op,
     dirty operands, dtypes/shapes/strides, small integer operands, the `rows` line -- zero /
     non-finite rows and the ids that fed them --, chain, model frames, the CPU fp64 0/0
     check; `capsule_operands.py` to load its operands in an experiment), never an ad-hoc torch.load (no torch on the login node: a
     one-minute `exec.sh --gpus 0` job, or `apptainer exec <SIF>` where it exists); a
     model wave -- `wave_summary.py <logs> --signal '<NAN_SIGNAL>'` (per-trial verdicts,
     per-arm rates, `!!` alerts); what the capture did, from any log or `logs.json.gz`
     -- `capture_digest.py`, never zgrep/unescape JSON by hand; exploding values --
     `grad_trajectory.py`; 5b -- the attempt's `summary.txt` / `LADDER_RESULT=`. Else
     `grep -m` / `tail -n` for the line the decision needs; widen only if ambiguous.
     Filter a runtime's per-kernel warnings off every ad-hoc command (`2>&1 | grep -vE
     '<pattern>'`). Quiet flags (`git --no-pager`, `-q`); quote the line that matters
     and give the path rather than echoing files or JSON.
   * **Wait on a job with the prompt cache warm.** After 5 idle minutes the cache
     expires and the next call re-writes the whole context (~150-190k tokens at
     1.25x, the price of ~10 warm checks). Every `nd` call needs `ND_ENV=<ENV>`
     (export it, or prefix each call where exports do not persist). One `nd status
     <name>` right after `nd submit` (jobs often start within seconds); then, with
     nothing else to do, **wait in the foreground: `nd wait <name> --max 240`** (Bash
     timeout >= 300000 ms), repeated -- exit 0: terminal, its last lines printed (only the
     tail: judge a short job -- `capsule_info.py`, a probe -- from its `LOG=` file, not the tail;
     2026-11-19 a full capsule summary read as "failed" from its last line);
     exit 3: still running (state and the log's last line; `LOG_IDLE=` = silent 10+ min: a quiet
     phase or a stalled node); exit 4: `HELD` (a launch failure
     never starts by itself) -- `nd stop`, resubmit at once with `--exclude <node>`. A package's `run.sh` /
     `validate.sh`, verified as a vendor runs it: `PACKAGE=<dir> [SCRIPT=validate.sh] nd submit ... --step <TAG> --
     package_run.sh` (empty environment, from the package's dir), then the same wait; one `sbatch`ed outside nd:
     `nd adopt <name> <id> --step <TAG>` first -- never a `squeue` / `sleep` loop. After a resume, the
     ledger's bare job ids work too (`nd status 144302`). **Never idle 5-40 min**
     (a `sleep`, a scheduled wakeup or a background wait that ends then: the cache
     miss with no gain). Only a job with > 40 min left (still `PENDING` after ~15
     min, a long sweep tier) gets ONE check near its expected end instead: a
     background `nd wait <name>` where `SINGLE_SESSION` is unset, else a scheduled
     `nd status`. **`SINGLE_SESSION=1` (one SSH session per login)**: only the foreground `--max <= 300`
     wait; never a background wait loop -- it holds the single SSH session and every
     other command fails (`flock: No locks available`). Useful work beats waiting
     (step 7's track (a) while (b) sweeps): its calls keep the cache warm. Typical
     durations: a reproducing capture-to-reproducer job 2-6 min; a reproducing 5b ~40 s (a deterministic one: ~1 min, the solo probe),
     a clean 5b attempt +5-20 min; an enumerate/`--list-mutations` job ~1 min; a sweep
     tier 4-10 min; fix validation ~10 min; a values check ~15 s. A status question
     from the user: `tail -n 5 <LEDGER>` (the capture-to-reproducer job writes its
     phase lines as it goes) plus one `nd status <name>`, nothing more. Then read only
     the job's last line (`CAPTURE_TO_REPRO_RESULT=`, `LADDER_RESULT=`) and the files
     it names.
10. **Report, then keep climbing.** When a rung yields a result, report it in a
    line or two and continue *in the same turn* to the next rung that needs no
    user decision (a reproducing 5b -> step 6 -> step 7 -> step 8 -> step 9 -- in an
    interactive session steps 7-8 run in a background fork and the turn ends on the step-6
    hand-over, `step6-finish.md` *Then* -- unless
    the user turned steps 7-8 off -- then step 6 -> step 9 once every replayed group
    has a verdict; no capsule, G1 clean, or a capture that cannot reach the model -> step 5N -> 6). Never end a turn on
    "next I'll ...". Stop only for a user decision (compute beyond the step-1 estimate
    or its threshold, an extension of a spent step-7/8 budget, an unresolvable CHECK,
    a model that does not run by itself -- `step2-rate.md`) or to wait on a queued
    job -- then as rule 9 says. Inside a step-7/8 loop, keep iterating until it ends.
    **One status line per milestone, also between rungs**: the image is building / ready,
    the capture-to-reproducer job is submitted / caught a capsule / replayed, a tool or
    setup problem found and its fix, a test job's result. Never more than ~5 tool calls, or
    a few minutes of setup, without a line the user can see -- a user who has to ask "where
    are we?" was left without one (2026-11-19/20: asked three times, during the image build,
    a tool fix and a multi-item change). One line, not a report: what just happened, what
    runs now.
11. **Talk to the user in their terms, not the skill's.** Call
    `scripts/jobs/capture_to_repro.sh` **"the capture-to-reproducer job"** and, the
    first time, say what it does: one 8-GPU job that runs the model until it
    captures the NaN, replays the capture without the model, and packages the
    reproducer.
    **Step 0 is internal: never mention it to the user** -- not by number, not as
    "step 0", "steps 0-6" or "the step-0 check". Its results (repo root, GPU and
    PyTorch gates) are reported only as what they found (e.g. the untested-GPU
    warning), and user-facing step ranges start at 1.
12. **Two announcements are never skipped, however few words the rest uses**: a model
    that does not run by itself (step 2, `step2-rate.md` *The model does not run*),
    and the hand-over from the finished reproducer to root-causing and fixing (step 6,
    `step6-finish.md` *Then*). Each is a short, clearly marked block of its own, not a
    clause inside a status line. **The hand-over ends the turn** (interactive session): text
    printed mid-turn folds under tool output and the user never sees it (2026-11-19; twice on
    2026-11-24, once repeated after step 7's first submit) -- so steps 7-8 go to a background
    fork (`step6-finish.md` *Then*) and the block is the turn's last text. Step 9 repeats it at
    the top of the report.
13. **`<ARTIFACT_DIR>/status.json`** is the investigation's state as data (reproducer, root
    cause, fix, current step and job), rewritten by `ledger.py` on every ledger line; the
    numeric-doctor panel and status line read it. Nothing to do but keep the milestone lines
    exact (`TARBALL`, `LADDER_RESULT=`, `HANDOVER announced`, `ROOT_CAUSE=` / `NO_ROOT_CAUSE`,
    `FIX=`); `python3 <REPO_ROOT>/scripts/ledger.py <LEDGER> status` rewrites it and prints its
    one line -- also the answer to a status question while a step-7/8 fork runs. The user can
    run it themselves (`! <command>`, in the hand-over block) at no token cost: a "where are
    we?" turn re-writes the whole idle context into the cache (2026-11-24/25: 3-5 per
    session, 0.1-0.2M tokens each).

## Placeholders (the step that sets each; its file defines it)

| name | what | step |
|---|---|---|
| `<REPO_ROOT>` | the checkout (inside a runtime: `$ND_REPO`) | 0 |
| `<ARTIFACT_DIR>`, `<LEDGER>` | rule 3, rule 4 | 1 |
| `<RUN_SCRIPT>`, `<NAN_SIGNAL>`, `<ENV>` | runs the model; how its output shows a NaN (or a threshold regex on a logged magnitude, or the model's own wrong-value line, `step4-explode.md`); `<ARTIFACT_DIR>/env.sh`, the profile `nd` reads as `ND_ENV` (`env.md`) | 1 |
| `<WORLD_SIZE>`, `<NODES>` | ranks / nodes of ONE job (1 for independent trials / a single node); > 1 -> `step4-distributed.md` (the node-count ladder for `<NODES>`) | 1 |
| `<PRIORS>` | `use` (default: earlier investigations of the script order the work -- a prior root cause is re-tested first) or `none` (fresh eyes; `NUMERIC_DOCTOR_NO_PRIORS=1`) | 1 |
| `<GROUPS_TO_REPLAY>` | `largest` (default: G1 only, the first line of `groups_priority.txt`; world size > 1: the origin group) or `all` (`next_groups.sh`) | 1 |
| `<NAN_RATE>` | e.g. "3 of 8 trials" **with its environment** (folded: the step-4 job's control arm); or *unmeasured* (step 2 skipped) | 2 (4) |
| `<MORE_NANS_RUN_SCRIPT>`, `<MORE_NANS_RATE>`; `<FULL_MODEL_RUN_SCRIPT>`, `<FULL_MODEL_RATE>` | step 3's rate-raising variant and its rate; the step-3 versions if it ran, else the originals (what 5N and step 8's workaround compare against) | 3 |
| `<CAPTURE_NODES>` | the fewest nodes that still produced the same bug (`CAPTURE_NODES=` ledger line; `<NODES>` without a ladder) -- every later model run uses it | 4 |
| `<CAPSULE_DIR>` | `<ARTIFACT_DIR>/capsules/`, empty before capture | 4 |
| `<CONTENTION>`, `<OUT_JSON_PATH>`, `<ORIGIN_RANKS>` | processes per GPU at capture, sources, env amplifiers; `<ARTIFACT_DIR>/groups.json`; the job manifest's origin rank(s) (world size > 1) | 5a |
| `<CAPSULE_FILE>`, `<CAPSULE_ITER_COUNT>`, `<CAPSULE_CONCURRENCY>` | a group's representative `.pt`; chain runs per copy (1000000); copies per GPU (capture's processes per GPU + 1; one less per OOM, down to 1) | 5b |
| `<RC_BUDGET>`, `<FIX_BUDGET>`, `<NAN_KIND>`, `<NAN_ORIGIN>` | the user's wall-clock budgets for the step-7 / step-8 loops; `deterministic` or `non-deterministic`, from the step-6 reproducer; `system` (a kernel / library fault) or `model` (overflow, undefined arithmetic: the model's values) -- both go through 7-8 | 7, 8 |
| `<REPORT_JSON>`, `<ASM_DIR>`, `<MUTATE_ONLY>` | `<ARTIFACT_DIR>/mutations.json`; `<ARTIFACT_DIR>/asm/`; the current sweep tier's `--mutate-only` flags | 7 (recipe) |
| `<ROOT_CAUSE>`, `<FIX>` | a validated root cause, or `NO_ROOT_CAUSE`; the fix artifact's path, or `NO_FIX` | 7, 8 |

## The ladder (the step files hold the rules)

**0. Resolve `<REPO_ROOT>` and run the gates -- ONE call, silently.** With
`<SKILL_DIR>` = this file's directory, and `<RUN_SCRIPT>` if the request names it:

```bash
S=<SKILL_DIR>; R=
for d in "$NUMERIC_DOCTOR_ROOT" "$S/../.." "$PWD" "$(git rev-parse --show-toplevel 2>/dev/null)"; do
  [ -n "$d" ] && [ -f "$d/scripts/replay_nan_op.py" ] && { R=$(cd -P "$d" && pwd -P); break; }
done
echo "SKILL_DIR=$(cd -P "$S" && pwd -P) REPO_ROOT=${R:-NONE} SKILL_COMMIT=$(git -C "$S" rev-parse --short HEAD 2>/dev/null || echo none)"
for t in nd compare_nan_capsules.py op_nan_capture.py kernel_mutations.py kernel_patch.py replay_nan_op.py; do
  [ -e "$R/scripts/$t" ] || echo "MISSING scripts/$t"; done
python3 "$R/scripts/gpu_check.py"                    # GPU gate: no torch needed
python3 "$R/scripts/framework_check.py" <RUN_SCRIPT>  # PyTorch gate: only if <RUN_SCRIPT> is known
```

In the same turn, read `<SKILL_DIR>/reference/step1-ask.md` whole with the Read tool (~30k: over a
`cat`'s budget, rule 9); env.md gets its own call, in step 1.

`REPO_ROOT` is the first match, verbatim. **`NONE` or a `MISSING` line: stop and ask**
-- never guess. No artifact dir or ledger yet: `<RUN_TAG>` names the model, so step 1
creates both once `<RUN_SCRIPT>` is known. The gates, each on its last line:

* **PyTorch gate** (`FRAMEWORK=<f> VERDICT=<v>`; static, login node) -- the first thing
  done with `<RUN_SCRIPT>`, before anything is created: here if the request names it,
  else right after step 1's question 1. Every rung hooks PyTorch's dispatcher, so a
  non-PyTorch model gets nothing. `RUN` (`pytorch`): continue; step 1 ledgers the line.
* **GPU gate** (`GPU=<name> ARCH=<arch> VERDICT=<v>`): `SUPPORTED` (gfx942, gfx950,
  gfx1250, newer Instinct MI<n>, n >= 300): nothing to say; step 1 ledgers the line.
  `NO_GPU` (exit 5, a login node): step 1's `<ENV>` check decides, on the model's GPUs,
  before any wave.
* **Anything else** -- `SKIP`, `ASK`, `UNTESTED` (here or in step 1's `<ENV>` check):
  read `step0-gates.md` before printing anything; it holds the warning to print
  and what to do.

1. **Ask** only for the run script (PyTorch gate after it); everything else takes its default
   unless the user asks to choose.
2. **Run and measure** (default on): the model must run by itself, else stop and tell the user.
   `<NODES>` = 1: folded into step 4's job (its control arm is the rate, `step2-rate.md` *Folded*);
   a separate wave only on request or for step 3. Skipped = *unmeasured*, never a prior run's rate.
   No signal at all (silently wrong outputs): differential mode first (`step2-diff.md`) -- its check mode
   is then `<NAN_SIGNAL>`.
3. **Raise** (default off; only on request): skip if high; else oversubscribe.
4. **Capture** with a control arm, as the capture-to-reproducer job (capture -> 5a -> 5b(G1)
   -> package); `<NODES>` > 1: the node-count ladder first.
5. **Read** the grouping and each 5b verdict (`<GROUPS_TO_REPLAY>`); clean while the model is
   not: `step5-pressure.md`.
5N. **Native fallback** (automatic): no capsule, G1 clean, or the capture cannot reach the model
   (`CAPTURE_UNREACHABLE`: skips 4-5) -> a standalone script; a hit goes to 6.
6. **Finish the reproducer** (tarball + README). Not root-caused or fixed yet.
   Announce the hand-over to root-causing and fixing.
7. **Root-causing** (default on, `<RC_BUDGET>` defaulted; a system bug and a model bug alike): track (a)
   always, (b) a recipe for `<NAN_KIND>` (system bugs).
8. **Fixing** (default on, `<FIX_BUDGET>` defaulted, needs `<ROOT_CAUSE>`): look for all three kinds -- the real
   fix (kernel / library / model / config), a targeted workaround (only the affected calls), a broad one --
   and ship every one that exists, labelled, recommended first.
9. **Close**: restore the source, hand over `<ARTIFACT_DIR>`, `$LG table`, report
   (reproducer, `<ROOT_CAUSE>`, `<FIX>`).
