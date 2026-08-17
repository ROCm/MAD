# Adding an engine

An engine is a module under `scripts/collprof/engines/` holding one `EngineSpec`, plus
one line in `engines/__init__.py`. Nothing in `core/` changes, and no existing engine is
touched — that is the whole point of the split, and the reason it is worth keeping.

## The contract

`core/` reads an `EngineSpec` and never names an engine. Everything below is data:

| field | what it answers | example |
|---|---|---|
| `name`, `summary` | what the report header says it profiled | `sglang-disagg` |
| `logs` (`LogLayout`) | where the per-node logs are and what a phase is | `prefill_NODE*.log`, phase from the file name |
| `metrics` (`LogMetric`) | which scalars the log prints that the report should carry | `elapsed time per iteration (ms): 250.5` |
| `iteration_metric` | whether the engine has iterations to divide by | `iter_ms` for training, empty for serving |
| `traces` (`TraceLayout`) | how trace files are named and which phase each belongs to | `-TP-3.trace.json.gz`, role from the profile-point log |
| `limits` (`SanityLimits`) | the scale at which a record stops being plausible | 512 MiB per message, 64 ranks per communicator |
| `notes` (`ReportNotes`) | what a reader must be told to not misread the numbers | the scope of a measurement configuration |
| `fingerprints` | other artifacts that corroborate a detection | `perf_sglang-disagg-*.csv` |

`notes` is the field to take seriously. Every sentence in it is a claim about how one
engine was measured, and it is inserted into the report as written. A new engine that
leaves `notes` empty produces a report that claims nothing beyond what the core can
prove — which is correct, if terse. A new engine that copies another engine's `notes`
produces a plausible report full of false statements.

## Checklist

1. **Read one artifact of the engine by hand first.** A collective line, the phase
   boundary, one trace file name, and the throughput line. Everything below follows
   from those four.

2. **Create `engines/<engine>.py`** with a module docstring stating the layout in
   prose, then the regexes, then `SPEC = EngineSpec(...)`. Use `primus.py` (phases from
   markers inside one log per node) or `sglang_disagg.py` (phases from file names, one
   log per role per node) as the closer starting point.

3. **Decide where the phase comes from.**
   - `PHASE_FROM_MARKER`: one log covers several phases in sequence, so the log
     announces them. Supply `phase_marker` (group 1 is the name) and `marker_guard`, a
     literal the line always contains — logs reach gigabytes and every regex is guarded
     by a cheap substring check.
   - `PHASE_FROM_FILENAME`: each log belongs to one phase. Supply `phase_of_name`.

4. **Declare the metrics** rather than adding them to the parse loop. Each is a
   `guard` substring, a pattern whose group 1 is the value, and a `label` with a
   `{value}` placeholder for the report. Several metrics may match one line.

5. **Map traces to phases in code, not by hand.** Prefer whatever the engine itself
   recorded: the sglang implementation reads the `output_dir` each profile-point log
   asked its workers for, because it is exact and covers every node of the role.
   Timestamps look tempting and are unsound — trace directory names come from the
   container's clock and file mtimes from the shared filesystem's, and on this cluster
   the two sit ~460 s apart, enough to attribute a whole role's traces to the other
   role. When the mapping cannot be established, raise with what was found; a wrong
   mapping poisons a whole report silently, while a raise costs one `--torch-trace`.

6. **Set the sanity bounds to the engine's scale.** The defaults suit eight-GPU
   communicators and messages up to 512 MiB. An engine that legitimately builds wider
   communicators or moves larger messages says so here, or its real records are counted
   as damaged — the report warns about it either way, but the default should not be
   wrong for the engine.

7. **Write the notes.** Ask: what would a reader conclude from these numbers that is
   not true? For a measurement configuration that differs from the tuned one, for
   traffic that bypasses RCCL entirely, or for a trace window that is not an iteration,
   the answer belongs in `notes` — the fields are `rank_coverage`, `communicator`
   (`{nranks}` is substituted), `scope`, `damage_cause`, `unmarked_window`,
   `trace_vs_log`. Leaving one empty omits the sentence, which is the honest default:
   `damage_cause` explains why *this* engine's logs tear, so an engine that has not been
   investigated states the count and the breakdown and claims no cause.

8. **Register it** in `engines/__init__.py`.

9. **Add tests.** The existing ones parametrise over the registry, so an engine gets
   the contract checks for free (`test_engines.py`), but add:
   - a fixture in `tests/conftest.py` building a minimal run of the engine;
   - discovery and phase attribution (`test_rccl_log.py`);
   - trace-to-phase resolution, including the failure mode (`test_torch_trace.py`);
   - that the engine's report states its own scope and none of another engine's
     (`test_report.py`).

10. **Run the suite.** It needs no cluster and no artifacts, only pytest, which is
    installed rather than worked around:
    ```bash
    "$PY" -c "import pytest" 2>/dev/null || "$PY" -m pip install pytest
    "$PY" -m pytest scripts/tests -q
    ```

## What not to do

- **Do not branch on a phase name.** `prefill` is not an engine, and two engines can
  use the same phase names. Everything that varies by engine reaches the core through
  the spec.
- **Do not add an engine name to `core/`.** If something cannot be expressed as spec
  data, add a spec field with a behaviour-preserving default and document it here.
- **Do not lower a sanity bound to make a run look clean.** The bounds exist to catch
  spliced digits; a real record above them is a reason to raise the bound and reparse.
- **Do not infer a phase from ordering when the engine records it.** Sorting by
  timestamp is how one node of every role went missing from three reports.
