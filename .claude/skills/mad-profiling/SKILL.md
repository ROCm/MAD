---
name: mad-profiling
description: Profile the collective communication of a madengine run and turn the artifacts into reports. Use when the user wants to see which collectives a workload issues, how large its messages are, how balanced its ranks are, or what a profiled run costs -- for a training engine (primus/Megatron-LM) or a disaggregated inference engine (sglang PD-disagg), and for engines not yet supported, which are added as one module under scripts/collprof/engines/. Covers what a run must set to be measurable at all (NCCL_DEBUG, torch profiler points, rocprofv3, and the flags that stop a framework from bypassing every profiler), how to build per-phase reports with scripts/collective_report.py, how to rebuild a whole campaign from a catalog with scripts/regen_reports.py, how to leave a finished run's logs compressed with scripts/compress_logs.py, and how to read the numbers without overstating them (references/interpretation.md).
compatibility: Requires python 3.10+ for the report tooling, standard library only except openpyxl, which adds profile.xlsx and is installed with `python3 -m pip install openpyxl` into the same interpreter that runs the scripts. Profiling a run additionally requires a cluster run configured through the mad-slurm-multinode skill, RCCL/NCCL with debug output, and optionally rocprofv3 from ROCm.
metadata:
  author: mkuznet1 (mikhail.kuznetsov@amd.com)
  version: "0.1"
---

# mad-profiling

Two halves, usable independently:

1. **Configure a run so it can be measured** — which env vars, manifest blocks and
   framework flags a profiled run needs, and which of them change the workload enough
   that throughput has to come from a second, unprofiled run. The additions are kept as
   overlays on a mad-slurm-multinode manifest in
   [assets/manifest-overlay/](assets/manifest-overlay/), one per engine, applied with
   `jq -s '.[0] * .[1]' base.json <engine>.overlay.json`.
2. **Turn the artifacts into reports** — `scripts/collective_report.py` for one job,
   `scripts/regen_reports.py` for a campaign, `scripts/compress_logs.py` to leave the
   run's logs on disk in the form the parsers read anyway.

> **Skill paths — read first.** When this skill activates you are given the absolute
> path to its directory (the folder containing this `SKILL.md`). All `scripts/`,
> `references/`, `assets/`, and `reports_template/` paths below are relative to THAT
> directory, not to the run directory. Set it once:
>
> ```bash
> export SKILL_DIR="<absolute path to this skill's directory>"
> ```

## When this fires

- "Profile the collectives of this run" / "which collectives does this workload issue".
- "Build a communication report for job `<id>`" / "rebuild all the reports".
- "How large are the messages", "are the ranks balanced", "what does profiling cost".
- "Add support for `<engine>`" to the report tooling.

Out of scope:
- Configuring and launching the run itself (repos, `mad.env`, manifests, SLURM) —
  that is the **mad-slurm-multinode** skill. This skill states what a manifest must
  set for a run to be measurable and hands the rest over.
- Kernel-level performance analysis and roofline work. The reports are about
  communication volume, message sizes and balance; the only durations they carry come
  from rocprofv3, and where those are absent the report says so.
- Deriving bandwidth. Nothing here divides a volume by a duration — the two never
  cover the same window ([references/interpretation.md](references/interpretation.md)).

## Responsibilities — who does what

- **mad-slurm-multinode** brings up the node and launches the run.
- **this skill** says what the run must set to be measurable, and owns everything
  downstream of a finished run: parsing artifacts, building reports, and stating the
  limits of what was measured.
- **the engine module** (`scripts/collprof/engines/<engine>.py`) owns every fact that
  is true of one engine only, including the sentences its reports print about scope.
- **`scripts/collprof/core/`** owns everything that is true regardless of engine and
  is not edited to add an engine.

The core/engine boundary is the design, not a detail. It exists because report prose
once branched on phase names, so any engine that called its phases `prefill` and
`decode` inherited another engine's claims — output that was wrong and looked right.
A test pins this
(`scripts/tests/test_report.py::test_a_new_engine_does_not_inherit_another_engines_scope_note`).

## Required inputs — ask before exploring

For **report building**, only two are needed, and a missing one is a question rather
than a search:

1. **Run directory** holding the per-node logs of a finished job.
2. **Output root** for the reports (conventionally `reports/<name>` in the run dir).

Everything else has a working default: the engine is detected from the log layout,
traces are found and matched to phases by the engine, and sanity bounds come from the
engine.

For **configuring a profiled run**, the concrete additions are
[assets/manifest-overlay/](assets/manifest-overlay/) — the `tools` block (rocprofv3
plus the power and VRAM profilers), the `NCCL_DEBUG` / `NCCL_DEBUG_SUBSYS` values in
both env blocks, the trace directories and mounts, and for Primus the experiment-YAML
profiler keys. What each one buys, what disappears without it, and what it costs is in
[references/measurement-setup.md](references/measurement-setup.md) and in that
directory's README. The additions need `<FILL_PROF_ROOT>` (a shared-FS directory for
this run's artifacts) and the engine's own values; the rest of the manifest stays the
mad-slurm-multinode skill's business.

## Workflow — reports from a finished run

### Step 1 — Pin the interpreter, check the artifacts are there

```bash
PY="${PY:-python3}"                             # every command below runs under this one
"$PY" -c "import openpyxl" 2>/dev/null || "$PY" -m pip install openpyxl
ls "$RUN_DIR"                                   # per-node logs, one per node or per role per node
"$PY" "$SKILL_DIR/scripts/collective_report.py" --list-engines
```

Only the standard library is needed to parse and to write `report.md` and the CSVs;
`openpyxl` adds `profile.xlsx`, so installing it up front is worth one command and
removes the failure that used to cost half a rebuild. Nothing else is required —
python 3.10+ and no other third-party package.

`$PY` is pinned because the scripts start with `#!/usr/bin/env python3` and therefore
inherit whatever interpreter is on `PATH`. That is how a campaign once ran under a
python without `openpyxl` and lost every workbook, so a conda or venv environment is
named here rather than assumed: `PY=~/miniforge3/envs/<env>/bin/python3`.

Four artifact classes can be present, and each answers a different question. Which
ones a run produced is worth establishing before parsing, because the report's shape
follows: RCCL debug logs (message sizes for the whole run), torch profiler traces
(sizes and process groups per collective, for a few steps), rocprofv3 stats
(durations, no sizes), and the benchmark or perf CSV (throughput). Details:
[references/measurement-setup.md](references/measurement-setup.md).

**Guard:**
- **If the logs hold no `NCCL INFO` collective lines** → either the run was not configured for it (fix `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS` including `COLL`, then rerun the job rather than parsing harder), or it was launched with `RCCL_LOG_DIR` and the records are in per-rank files somewhere else; point `--rccl-dir` at them.
- **If no engine recognises the directory** → the tool says what it looked for. Either pass `--engine`, or add an engine ([references/engines.md](references/engines.md)).

### Step 2 — One report per phase

```bash
"$PY" "$SKILL_DIR/scripts/collective_report.py" \
  --run-dir "$RUN_DIR" --out-dir reports/<name> \
  --parse-cache reports/.parse-cache/<job>.pkl
```

A phase is one comparable stretch of a run: a datatype for training, a role for
disaggregated serving. Each becomes `reports/<name>_<phase>/` holding `report.md`,
the CSVs behind every table, and `profile.xlsx`.

Worth knowing before the first run:

- **`--parse-cache` is not optional in practice.** A 2 GB decode log takes tens of
  minutes to read, mostly waiting on shared storage, and the report text usually goes
  through several passes. The cache is keyed on input identity and parser version, so
  changed bytes and changed logic are reparsed.
- **Traces are found automatically.** `--torch-trace PHASE=PATH` pins them explicitly
  when needed; `--no-auto-traces` skips them, which is the fast path when only the
  log-derived numbers are wanted.
- **Per-rank RCCL logs are read as well.** A run launched with `RCCL_LOG_DIR` wrote one
  file per process (`NCCL_DEBUG_FILE`) instead of interleaving eight ranks in a node's
  stdout. They are picked up under the run directory by default and with `--rccl-dir`
  when they live elsewhere, which is the training case. The header of every report says
  which stream its records came from.
- **Parse one job at a time.** Two parsers on the same shared filesystem slow each
  other down about fivefold.

**Guard:**
- **If a phase is skipped as having logged no collectives** → that phase started and produced nothing parseable, usually a missing env var or an early failure. Read its log before rerunning the tool.
- **If the report warns that records hit a sanity bound** → the bound may be too low for this run's scale. Raise `--max-msg-bytes` / `--max-nranks` and reparse ([references/data-quality.md](references/data-quality.md)); do not ignore it, the volume is missing those records.
- **If a discovered capture directory holds no traces** → it is named in the warning and in the report, and the rest of the phase is still built. An idle replica captures nothing, so check whether that process did any work before treating it as a coverage gap. A path pinned with `--torch-trace` stays strict and fails.
- **If the run warns that `openpyxl` is missing** → `report.md` and the CSVs hold every number, only `profile.xlsx` was skipped, and the warning names the interpreter that lacks the package. Install it into that same interpreter (`"$PY" -m pip install openpyxl`) and rerun; the parse cache makes the second pass seconds rather than minutes. Installing is the normal course of action, not a decision to bring to the user.

### Step 3 — Compress the logs

```bash
"$PY" "$SKILL_DIR/scripts/compress_logs.py" --run-dir "$RUN_DIR" [--rccl-dir <prof-root>/rccl]
```

The logs are the largest artifact a measured run leaves, and with `RCCL_LOG_DIR` nothing is
filtered out of them: about 8 GB for a 4-node serving job, half a gigabyte per datatype
phase of a training one. The text repeats, so gzip takes an inference run's 8 GB under
100 MB, and every parser here reads `.log.gz` as readily as `.log`. This is not a tidying
step to do eventually — a shared home directory filling up has truncated a file mid-write,
marked a finished job FAILED at teardown, and cost a parse cache.

The engine's globs decide what counts as a log, so exactly the set
`collective_report.py` reads is compressed and nothing else. Each file is verified by
digest before its plain copy is dropped, and the run is left untouched on any failure.

**Guard:**
- **If it reports a log sitting beside its own `.gz`** → both match the parser's globs, so that node's records would be counted twice. It compresses nothing and names the pair; keep whichever copy is complete and delete the other.
- **If the job is still running** → do not run this yet. Whatever a log receives after the copy is made would be lost.

### Step 4 — A campaign from a catalog

```bash
cp "$SKILL_DIR/assets/jobs.example.json" reports/jobs.json    # edit: one entry per job
"$PY" "$SKILL_DIR/scripts/regen_reports.py" --catalog reports/jobs.json --list
"$PY" "$SKILL_DIR/scripts/regen_reports.py" --catalog reports/jobs.json
```

The catalog is the only place job ids and artifact paths live, so adding a job is an
entry rather than an edited script, and rebuilding everything after a tooling change
is one command. Runs are sequential on purpose; one failing job does not stop the rest
and the failures are listed at the end.

### Step 5 — Cross-run comparison, if asked

A report describes one phase of one job. Comparing runs or models is a written
document, and the templates in [reports_template/](reports_template/) exist so that
the comparison states its own method: which job each number came from, what was
different about the profiled configuration, and what the method cannot see. Copy a
template and fill it; do not let a comparison table stand without its scope note.

**Guard:**
- **If two runs used different configurations** (attention backend, quantisation, profiling flags) → say so in the comparison. It is the first thing that explains a difference and the easiest to leave out.
- **If a per-run figure is a throughput** → it must come from a run without profiling, and the comparison says which job that was.

## Adding an engine

One module under `scripts/collprof/engines/` and one line in its registry; nothing in
`core/` changes and no existing engine is touched. The checklist, the meaning of every
field, and the tests to add are in [references/engines.md](references/engines.md).

**Guard:**
- **If adding an engine seems to require editing `core/`** → the spec is missing a field. Add the field to `core/spec.py` with a default that preserves current behaviour, rather than special-casing the engine in the core.

## Reading the numbers

[references/interpretation.md](references/interpretation.md) is the file to read
before quoting anything from a report. It covers what each channel can and cannot
support, why no bandwidth is derived anywhere, and the specific blind spots of a
disaggregated serving profile — including that the KV-cache transfer that defines the
topology does not appear in RCCL at all.

[references/data-quality.md](references/data-quality.md) covers the input side: why a
fraction of a percent of records arrive torn, how they are detected, and what to do
when a sanity bound rejects records that are real.

<!-- Authoring note: this skill is written declaratively (statements of how it
behaves) rather than imperatively, which reads more uniformly across models and
languages. New content follows the same convention. -->
