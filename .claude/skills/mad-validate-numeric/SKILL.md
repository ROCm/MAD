---
name: mad-validate-numeric
description: Validate mad-debug-numeric end to end on MAD against Numeric Doctor's examples — run the skill unattended on each example model (tag numeric_examples) through the mad-numeric-doctor agent, then compare each result with the example's expected card, read from a Numeric Doctor checkout. Use when the user asks to validate the numeric doctor, mad-debug-numeric or its madengine integration on MAD.
argument-hint: "[<example model> ...] [--answers DIR] [--priors none|use]"
disable-model-invocation: true
allowed-tools: Bash(madengine discover *) Bash(madengine nan *) Bash(amd-smi *) Bash(rocm-smi *) Bash(bash .claude/skills/mad-common/preflight.sh) Bash(bash .claude/skills/mad-common/mad-run.sh *) Read Write Glob Grep Agent
---

# Validate mad-debug-numeric against the examples

Each `numeric_examples` model is a Numeric Doctor example in MAD form: a workload with a
known numeric bug and no hint of it. Its answer -- verdict, root cause, fix -- is a card in
Numeric Doctor's `validation/expected/<example>.json`, outside MAD on purpose: **the agent
under test never sees it**. A validation runs the whole `mad-debug-numeric` ladder on each
example, unattended, and compares. `RESULT_COMPARE=SAME` per example is a pass. It is
expensive: each example is a 20-90 minute agent run with GPU jobs.

## 0. Resolve and scope (no GPU)

```bash
bash .claude/skills/mad-common/preflight.sh
madengine discover --tags numeric_examples --json
```

* **The examples**: every model the tag resolves to, or the ones named in `$ARGUMENTS`. Model
  `pyt_numeric_<example>` answers to card `<example>.json` (`pyt_numeric_softmoe_train_nan`
  -> `softmoe_train_nan.json`).
* **The answers dir** (`<ANSWERS>`): `--answers DIR`, else `$NUMERIC_DOCTOR_ANSWERS`, else ask
  the user for their checkout's `mad-numeric-doctor/validation/expected`. Read each card's
  `nan_signal` only. A model without a card still runs; its row says `no expected card`
  (not a pass). Never copy a card into MAD, into `nan_hunts/`, or into a prompt.
* **The GPU** (`amd-smi static --asic` or `rocm-smi --showproductname`): the cards come from
  Numeric Doctor's own runs, mostly on gfx950 (MI350X / MI355X). Ledger the arch; on another
  arch a `DIFFERENT` on a kernel name or a rate can be the arch, not the skill -- say so.
* **Ask once, one message**, unless the user named one or two examples (then take the
  defaults and say them): the cost (N examples x up to ~1.5 h of agent time and GPU jobs,
  one at a time -- `mad-run.sh` serialises GPU runs anyway), and `<PRIORS>` (default `none`:
  fresh eyes, it measures discovery; `use` reads earlier investigations of the model).

## 1. The validation dir and ledger

`X=nan_hunts/<date>-validate-numeric` (taken: `-2`, `-3`, ...). `$X/validate_ledger.md` is
append-only progress and the file a new session resumes from. Header: date, MAD and
madengine commits (`git -C <repo> log --oneline -1` of each, as the user's shell reports
them), GPU arch, priors. One row per example:

```
| example | nan_signal | artifact dir | status | started | compare | notes |
```

Resume: a row with `status=done` is not run again; `running` with no live agent restarts.

## 2. One example: intake, then the agent

Per example, its `<ARTIFACT_DIR>` = `$X/<example>/` (create it), and
`$X/<example>/intake.json`:

```json
{"schema": 1, "source": "mad", "env": "madengine",
 "model_tag": "<model>", "run": "madengine run --tags <model> --live-output",
 "nan_signal": {"regex": "<the card's nan_signal, escaped>"},
 "world_size": <n_gpus>, "artifact_dir": "$X/<example>",
 "preferences": {"measure_rate": true, "capture_data": "values", "rootcause": true,
                 "fix": false, "groups": "largest", "budget_minutes": 60,
                 "priors": "<none|use>"}}
```

`madengine nan intake $X/<example>/intake.json` must exit 0. Then start ONE
`mad-numeric-doctor` agent (`Agent`, `subagent_type: mad-numeric-doctor`), prompt: *"Run the
mad-debug-numeric skill for `<model>` with the intake `$X/<example>/intake.json`
(`reference/step1-intake.md`), unattended, through step 9. Fill `result.json` as
`step9-report.md` *Result card* says. End with the `report.md` path and the
`LADDER_RESULT=` / `ROOT_CAUSE=` lines."* Nothing else: no card, no expected verdict, no
hint from an earlier run. Ledger `status=running`. One example at a time.

## 3. Compare

When the agent ends, `$X/<example>/result.json` must exist (missing: `compare=NO_RESULT`).

```bash
madengine nan exec --image <the model's image> --mount <ANSWERS>:<ANSWERS>:ro -- \
    python3 -m nan_doctor_rt.compare_results <ANSWERS>/<example>.json $X/<example>
```

Ledger its last line (`RESULT_COMPARE=<overall>  (bug <..>, root_cause <..>, fix <..>)`) and
the notes it prints. Step 8 never runs unattended under MAD (`step1-intake.md`), so the fix
level is `DIFFERENT` by construction: **a pass is bug and root_cause `SAME` or `REFINED`**,
and the report says the fix was not run. `REFINED` names the narrower side; `DIFFERENT`: read the run's `report.md` and say in one
line where it went another way (verdict, root cause or fix). Rates, hosts and call sites
differ run to run and are never a failure.

## 4. Report

One table, one row per example: example, verdict found, compare result, run time, the
artifact dir. Then the totals (`SAME` k of N) and, for each non-`SAME` row, its one line.
Say what was not run and why (no card, another arch, the user's subset).
