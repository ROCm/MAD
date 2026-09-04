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
| `rccl_logs` (`LogLayout`) | where `NCCL_DEBUG_FILE` put one log per process, if the engine's launcher can be asked for them | `rccl/prefill_NODE0.<host>.<pid>.log` |
| `metrics` (`LogMetric`) | which scalars the log prints that the report should carry | `elapsed time per iteration (ms): 250.5` |
| `iteration_metric` | whether the engine has iterations to divide by | `iter_ms` for training, empty for serving |
| `traces` (`TraceLayout`) | how trace files are named and which phase each belongs to | `-TP-3.trace.json.gz`, role from the profile-point log |
| `limits` (`SanityLimits`) | the scale at which a record stops being plausible | 512 MiB per message, 64 ranks per communicator |
| `run_config` (`RunConfigLayout`) | where the engine states the configuration it actually ran with | `server_args=ServerArgs(disable_cuda_graph=False, ...)` |
| `steps` (`StepTimingLayout`) | how a step time is recoverable from what the engine already prints | `#running-req: 16, ..., gen throughput (token/s): 234.5` |
| `a2a` (`A2AKernels`) | how the expert all-to-all is named in this engine's traces, when a backend carries it outside RCCL | `mori_ep_dispatch_kernel`, `deep_ep::combine` |
| `benchmark` (`BenchmarkLayout`) | where the harness left its numbers and what its CSV calls them | `perf_*.csv`, `isl1024_osl1024_con64`, `mean_itl_ms` |
| `counters` (`CounterLayout`) | where the RDMA adapter samples landed, and how this fabric's driver spells its operations | `rdma/decode_NODE2.csv`, `rx_write_req` against `rx_write_requests` |
| `notes` (`ReportNotes`) | what a reader must be told to not misread the numbers | the scope of a measurement configuration |

The last five are optional and default to reporting nothing: an engine that declares none of them
produces the report it produced before they existed. They are worth filling in for anything that
serves, because they answer questions the other channels cannot — `run_config` is what makes two
runs comparable or not, `steps` is the only duration channel a serving run has, `a2a` is the
only channel that names traffic a backend carries itself, and `benchmark` is where throughput
and latency come from at all, since a profiled run's own log carries neither.

`benchmark` is also the field the rule above was tested by. The comparison module began with one
harness's schema written into it — the key shape `isl…_osl…_con…`, the metric names `mean_itl_ms`
and `mean_ttft_ms`, the glob `perf_*.csv` — and a second serving engine would have had to edit
`core/` to be read at all. Naming them here is what keeps the end-to-end split arithmetic over
whatever the harness called its columns rather than over three names `core/` assumes.

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

   The node label works the same way: `NODE_FROM_PARENT` for a per-node directory,
   `NODE_FROM_STEM` with `node_of_name` when the name carries more than the node, as a
   per-rank file does.

3a. **Add `rccl_logs` if the engine can write one RCCL log per process.** It is a second
   `LogLayout` with `written_by=LOG_PER_RANK`, read in addition to `logs`, and it is worth
   having: it removes the torn records instead of detecting them, and for a launcher that
   filters which ranks reach stdout it is the only way to see all of them. The phase has to
   be readable from the file name, since these files hold nothing but RCCL — the markers and
   the metrics stay in the shared log, which is why both are read. What sets the variable is
   the engine's launch script; see `RCCL_LOG_DIR` in
   [measurement-setup.md](measurement-setup.md).

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

6a. **Say where the engine states its own configuration, and its step time.** Both are one
   `guard` plus one pattern, and both answer a question no other channel can.
   - `run_config`: the line where the engine prints its effective settings, defaults applied.
     Prefer that over a command line — a launcher passes a subset and the framework fills in the
     rest, and it is the filled-in value that ran.
   - `steps`: a line carrying enough to derive a step time. For a server that logs its running
     batch and its generation rate, the rate over the batch is the step frequency. The optional
     `graphed` group records whether a captured graph was replayed over those steps, which is worth
     supplying wherever the engine says: it is the configuration difference most likely to explain
     a per-step gap, and it belongs in band rather than inferred from startup.

6b. **Declare `a2a` if a backend carries collectives outside RCCL.** An expert-parallel MoE
   backend does: MoRI over IBGDA, DeepEP over rocSHMEM, neither reaching an RCCL log nor a
   `record_param_comms` event. Patterns are matched against trace event names, first match wins, so
   order them from specific to general. Getting them wrong is cheap and visible — the report lists
   the busiest unclassified device events precisely so the patterns can be corrected against a real
   trace.

7. **Write the notes.** Ask: what would a reader conclude from these numbers that is
   not true? For a measurement configuration that differs from the tuned one, for
   traffic that bypasses RCCL entirely, or for a trace window that is not an iteration,
   the answer belongs in `notes` — the fields are `rank_coverage`, `communicator`
   (`{nranks}` is substituted), `scope`, `damage_cause`, `unmarked_window`,
   `trace_vs_log`, `step_basis`, `graphs_off`, `a2a_outside_rccl`. Leaving one empty omits the
   sentence, which is the honest default: `damage_cause` explains why *this* engine's logs tear, so
   an engine that has not been investigated states the count and the breakdown and claims no cause.
   The same applies to the last three — the core reports that graph replay was off, and only the
   engine knows whether that is a deliberate measurement setting or a fault.

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
