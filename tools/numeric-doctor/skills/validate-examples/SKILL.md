---
name: validate-examples
description: Validate the debug-numeric skill (and with it the Numeric Doctor install) end to end -- run ../debug-numeric/SKILL.md unattended on every example under examples/training and examples/inference, one background agent per example, and judge each run's result.json against validation/expected/<example>.json with compare_results.py. Use for "validate numeric doctor", "validate mad-numeric-doctor", "validate the examples", "validate the skill", "is numeric doctor installed correctly", "run all examples", or a named subset ("validate softmoe_train_nan").
allowed-tools: Read, Edit, Write, Bash, Glob, Grep, Agent
---

# Validate debug-numeric against the examples

Each example under `examples/{training,inference}/<name>/` is a model with a known numeric bug; its answer --
verdict, root cause, fix -- is `validation/expected/<name>.json` (`validation/expected/README.md`). A
validation runs the whole debug-numeric ladder (steps 1-9) on every example, unattended, and compares.
`RESULT_COMPARE=SAME` per example is a pass. It is expensive: each example is a 17 min - 1.5 h agent run
with 8-GPU jobs (2026-11-20 round: 6 examples, 17-76 min each, all in parallel).

## 0. Resolve, check the install, scope -- before any agent

```bash
R=$(cd -P <this skill's dir>/../.. && pwd -P)    # <REPO_ROOT>: holds scripts/replay_nan_op.py and validation/expected/
cd "$R" && git --no-pager log --oneline -1 && git --no-pager status --short scripts skills examples validation
ls -d examples/training/*/ examples/inference/*/      # examples = dirs with a run.sh; never examples/tbd/
ls validation/expected/*.json
```

* **Install check first** (what "properly installed" means, cheap): the debug-numeric step-0 command
  (its SKILL.md *The ladder*, step 0) for `<REPO_ROOT>`, the tools and the GPU gate, then ONE GPU job
  through `nd` with any example's profile (`nd profile <site> --art <ART> IMAGE=<its run.sh image>`,
  `nd submit --name check --gpus 1 -- exec.sh --gpus 0 -- 'python3 -m gpu_check --torch'`). A failure here
  stops the validation: report it, launch nothing.
* **The example list**: every `examples/{training,inference}/<name>/run.sh`; the user may name a subset.
  Per example, `<RUN_SCRIPT>` and `<NAN_SIGNAL>` come from its card's `run_script` / `nan_signal` (the
  signal the expected result was validated with). An example **without a card** still runs; its outcome
  row says `no expected card` (not a pass). A card without an example dir: list it as `missing example`.
* **Uncommitted changes** under `scripts/ skills/ examples/`: the runs test the working tree, not a
  commit. Say so and ledger the `git status` lines; the outcome names the commit plus "dirty".
* **Ask the user once, one message** (the only stop; skip what they already said). **Ask nothing** when the
  user named the examples to run and the estimate is at most ~2 node-hours (one or two examples): take the
  defaults below, say them in the status line, and ledger them -- the request is the approval (2026-11-23:
  "try it on a single example" needed no cost question):
  1. the **cost**: N examples x ~1 node-hour each of 8-GPU jobs (more when a step-7 sweep runs), plus
     N agent runs' tokens -- OK?
  2. **priors**: `none` (default for a validation: fresh eyes, `NUMERIC_DOCTOR_NO_PRIORS=1`, measures
     discovery) or `use` (earlier artifacts order the work: a confirmation, faster). Not blind either way --
     see *Blindness*.
  3. **parallel** (default on Slurm: all at once; the queue serialises GPU jobs) or **sequential** (the
     default, and the only safe choice, on a `SCHED=local` site: nothing queues there, and parallel runs'
     8-GPU jobs share the host's GPUs, so each run's rates and contention change -- 2026-11-28, one host).

## 1. Set up the validation dir and ledger

```bash
X=$R/artifacts/$(date +%F)-validate-examples        # ls it first: taken -> -2, -3, ...
mkdir -p "$X" && chmod 777 "$X"
```

`$X/validate_examples_ledger.md` -- append-only progress, the file a new session resumes from. Header:
date, commit (+ dirty), cluster/site, GPU arch (from the install check), priors, parallel/sequential, the
resume rule below. Then one row per example:

```
| example | run_script | nan_signal | run artifact dir | agent id | status | started | compare | notes |
```

`status`: `PENDING` -> `RUNNING` -> `DONE` / `FAILED` (agent ended without step 9) / `NO_CARD`. Update the
row the moment it changes, before reporting it. **Tell the user the ledger path** once, right after it
exists: they can resume any time with "resume the validation from <path>".

## 2. Launch one background agent per example

With no background-agent tool (Codex CLI, or a Cursor session without one): sequential mode, inline --
follow the prompt below yourself for one example at a time, then judge it (step 4) before the next.

Agent tool, `general-purpose`, background, one per example (parallel: all in one message). `RUN_TAG` =
`<date>-<example>-val` (the debug-numeric artifact dir becomes `artifacts/<RUN_TAG>`; ls first, suffix if
taken). The prompt -- it is the agent's only context:

```
Unattended end-to-end validation of the debug-numeric skill. No user is available: never stop to ask.
Read <REPO_ROOT>/skills/debug-numeric/SKILL.md and follow it through step 9 (a delegated run: SKILL.md says
take the given answers and go on). Work from <REPO_ROOT>.
Step-1 answers: <RUN_SCRIPT> = <run_script>; <NAN_SIGNAL> = <nan_signal>; <RUN_TAG> = <RUN_TAG>;
<PRIORS> = <none|use> [none: export NUMERIC_DOCTOR_NO_PRIORS=1 in every shell].
Steps 7-8 (root cause, fix): YES, default budgets. Every other question: take the skill's default and
ledger "(default, unattended)". The compute budget is approved up to <N> node-hours for this example.
Site: <site profile, e.g. slurm-docker: Slurm + docker on compute nodes, login node has no docker>.
Do NOT read validation/ (the expected answers) or any other artifacts/ dir [PRIORS=none].
Do NOT edit anything under scripts/, skills/, examples/ or validation/: a broken tool or instruction is
recorded (command, output) in the run's ledger and your final message; work around it only as the skill
allows, and go on.
Finish step 9 including result.json (`result_card.py check` = RESULT_CARD_CHECK=OK). End with: the artifact
dir, each step's verdict, result.json's path, and every defect you hit in the skill, scripts or example
(file, what failed, what you did instead).
```

Ledger each agent id and `RUNNING`. Sequential mode: launch the next when one ends. Subagents die with the
session -- the resume rule covers it.

## 3. While they run

Background agents notify on completion; don't poll them. In between, a status question gets the ledger
rows plus `tail -n 3 artifacts/<RUN_TAG>/ledger.md` per running example -- nothing more. Never take over
or "help" a running example's jobs.

## 4. Judge each finished example (as it lands)

```bash
python3 scripts/result_card.py check artifacts/<RUN_TAG>          # must be RESULT_CARD_CHECK=OK first
python3 scripts/compare_results.py validation/expected/<example>.json artifacts/<RUN_TAG>
grep -c 'TODO(agent)' artifacts/<RUN_TAG>/report.md                # 0 = step 9 finished
```

* **SAME** -> pass. **REFINED** (validation/expected/README.md): the card narrower = the run found the right
  thing less minimally (pass with a note); the run narrower = a better answer than the card (pass; propose
  a card update). **DIFFERENT** -> not a pass until explained; read the compare's lines and the run's
  report, then classify:
  * `skill regression` -- the run missed or got wrong what the card holds (a defect to fix);
  * `card stale` -- the run is right and the card predates a skill change (2026-11-23: the capture began
    catching Inductor kernels directly, so `NATIVE_KERNEL_BUG`/`kernel_capsule` became
    `KERNEL_BUG`/`op_capsule`); propose the card edit, never apply it without the user;
  * `card malformed` -- the run's own card is wrong (a list field written as a string compared char by
    char); fix the run's result.json, re-check, re-compare, and note it;
  * `card ambiguous` -- run and card agree in substance but fill a field the card schema leaves open
    differently (2026-11-23: `root_cause.location.source` -- the model's call site vs the code that emits the
    faulty call, both FOUND at the same kernel, mechanism and fix). Propose a definition in
    `skills/debug-numeric/reference/step9-report.md` (*Result card*) and the card edit; apply neither unasked;
  * `environment` -- another GPU arch than the card's (cards are gfx950 unless noted), an image that did not
    load, a node problem: not a skill verdict; say what would make it comparable.
* **FAILED** (no step 9): from its ledger, the last step reached and why; relaunch once with a RESUME
  prompt (below) if it died of the session, not of the skill.
* Collect the agent's defect list verbatim into the outcome, per example.
* Ledger the row: status, compare (bug / root_cause / fix), wall time (`$LG table` total of the run).

## 5. Outcome

`$X/validate_examples_outcome.md`, written once every example has a final row:

1. **Bottom line**: n of N SAME (pass), REFINED, DIFFERENT (with class), FAILED, NO_CARD.
2. Setup: commit (+ dirty), site, GPU arch, priors, parallel, how blind (below).
3. Table: example | wall | verdict (step 5/5N) | root cause | fix | compare (bug/rc/fix) | class | run dir.
4. Per DIFFERENT / FAILED: what differed, the evidence, the class, the proposed action.
5. Defects reported by the runs, grouped by example, each with file and symptom; mark ones already fixed.
6. Checks: `git status --short scripts skills examples validation` unchanged by the runs; every run's
   report has 0 TODO; no run read `validation/` (grep each run's transcript summary / ledger for it).

Then tell the user the bottom line and the table in a few lines, point to the outcome file, and list proposed
card updates and defects to fix -- applying either is a separate request.

## Resume (a new session, or after agents died)

Read `$X/validate_examples_ledger.md`. Per row: `DONE` -> keep. `RUNNING` -> open `artifacts/<RUN_TAG>/`:
`report.md` with 0 TODO and `result.json` checking OK = it finished (judge it, step 4); else launch a new
agent with the step-2 prompt plus `RESUME: the investigation artifacts/<RUN_TAG> exists -- read its
ledger.md first (SKILL.md rule 4), continue from the last open step, never restart it`. `PENDING` -> launch.
Ledger `resumed <date>` in the header.

## Blindness (say it in the outcome)

`PRIORS=none` keeps earlier investigations out of the work, not the answer out of reach:
* the example dirs are hint-free (2026-11-25: symptom and workload only; the root cause, fix and controls stay
  out of the repo) -- keep them so: a cause named in an example README, docstring,
  comment or arm name makes a run confirm more than it discovers;
* `nd` mounts the whole repo into every container: `artifacts/` (earlier runs, fix paths) and
  `validation/` (the expected cards) are visible to an agent that searches the filesystem;
* the session's git context shows recent commit subjects -- keep example names out of them.
A truly blind run needs a stripped copy of the repo (no `artifacts/`, `validation/`, `examples/tbd/`, answer
READMEs) with the agent pointed at that copy's SKILL.md -- set up only on request.
