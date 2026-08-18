# Making a run measurable

A profiled run is a different run. Every setting below buys visibility and most of them
cost performance, so the rule that follows from it is: **volumes come from the profiled
run, throughput comes from a second run without any of this.** A report that quotes both
from the same job is quoting a slowed-down configuration as if it were the tuned one.

This file is the reasoning; the settings themselves are ready to apply in
[../assets/manifest-overlay/](../assets/manifest-overlay/), as an overlay per engine on
a mad-slurm-multinode manifest plus the Primus experiment-YAML fragment. Read this to
decide what to include, apply that to include it.

## The four artifact classes

| artifact | needs | answers | does not answer |
|---|---|---|---|
| RCCL debug log | `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS` incl. `COLL` | every collective of the whole run: name, count, datatype, nranks, so message sizes | durations |
| torch profiler trace | a profile point in the framework (`--profile`, `/start_profile`) | size, dtype and process group per individual collective | anything outside the captured window; its durations do not survive a cross-check |
| rocprofv3 stats | `rocprofv3 --rccl-trace` wrapping the process | per-API and per-kernel durations | message sizes; per-phase attribution |
| benchmark / perf CSV | the run's own harness | throughput and latency | anything about individual collectives |

The report joins the first three and states which are present. Two of them missing is a
usable report; the RCCL log missing is not.

## What a framework hides from a profiler by default

This is the part that is easy to miss, because the run succeeds and the report comes out
nearly empty rather than failing:

- **A framework's own all-reduce kernel bypasses RCCL.** sglang's intra-node custom
  all-reduce never reaches `torch.distributed`, so it appears in no RCCL log and in no
  `record_param_comms` event. A profiled run that leaves it enabled produced a report
  showing 8 bytes per rank — the init barrier and nothing else. It is disabled for
  measurement with `--disable-custom-all-reduce`.
- **Graph capture makes collectives invisible and breaks the profiler.** A replayed HIP
  graph dispatches one packet, so the collectives inside it are not individually
  observable, and rocprofv3 aborts on the malformed AQL packet. Decode is measured with
  `--disable-cuda-graph`.
- **Both of those change performance.** That is the reason for the second, unprofiled
  run, and the reason the reports carry a scope note.

## Applying the flags where every node reads them

Two traps, both of which produced a run that looked profiled and was not:

- **Set profiling flags in one place per role.** A launcher that appends the flags at
  one call site and assembles the first node's command separately will profile three
  nodes out of four. Verify from the logs rather than the script: every node of a role
  must report the same configuration (e.g. `disable_custom_all_reduce=True`).
- **Only the container's env block reaches the workload.** In a madengine manifest,
  `context.docker_env_vars` reaches the container; `deployment_config.env_vars` stops at
  the SLURM launcher. A variable that gates a code path (`SGLANG_USE_AITER=0`) is
  useless in the wrong block, and the failure is a stack trace deep in a kernel dispatch,
  not a message about the variable. The mad-slurm-multinode skill's
  `scripts/validate_manifest.sh` checks for the asymmetry.

## Shut the workload down gracefully or lose the stats

`rocprofv3` writes its CSVs while the process exits. A bare `kill` returns immediately,
the launcher exits, the container is torn down, and the stats directory is left empty —
with no error anywhere. Send `SIGINT` first, wait for the process to unwind (allow
minutes when profiling), and escalate only if it does not. Even then, serving runs have
produced no rocprofv3 CSVs at all; treat their absence as expected and read
[interpretation.md](interpretation.md) before promising kernel timings.

## One RCCL log per rank, not one per node

`NCCL_DEBUG=INFO` sends every rank's records to the process's stderr, so the eight ranks of
a node write into one stream and a fraction of a percent of the records arrive spliced.
`NCCL_DEBUG_FILE` ends that: each process gets its own line-buffered file, `%h` expanding to
the host and `%p` to the pid. Set `RCCL_LOG_DIR` and MAD's launchers do the rest —
`auto` for serving (the files land in `/run_logs/<job>/rccl`, beside the server logs) and a
path under the profiling root for training, where the run directory belongs to madengine and
is not writable from the container.

Two things follow, one per engine:

- Serving stops producing torn records at all, so the connectivity table and the volumes
  come from every line rather than from the survivors.
- Training also gains the ranks it was missing: `--local-ranks-filter` decides who reaches
  stdout, and a 32-rank job was reported from 18 ranks. A file per process has no filter.

The directory has to be writable **from inside the container**, which on this cluster means
world-writable on the host: the container runs as a root that NFS squashes to `nobody`, and a
directory created by an ordinary `mkdir` is read-only to it. A `chmod -R 0777` on the profiling
root before launching is the whole fix, and it is worth not skipping — RCCL that cannot open its
debug file drops the output rather than falling back to stdout, so the run finishes with no RCCL
data at all. MAD's launchers check for it and say so in the log, leaving the records in stdout.
The same applies to `PRIMUS_WORKSPACE`, where a run that cannot write simply fails.

The datatype phases of a training run are distinguished by the file name (`BF16.node_1...`),
because the markers Megatron prints, and its throughput lines, stay in stdout. Both sets are
read, and the report says which stream its records came from. Point the analysis at the
files with `--rccl-dir` when they are not under the run directory:

```bash
collective_report.py --run-dir <run> --rccl-dir <prof-root>/rccl --out-dir reports/<name>
```

Budget the disk for it: nothing is filtered away any more, so a 4-node serving job leaves about
8 GB of RCCL text and a training job with two datatype phases about half a gigabyte per phase.
The lines repeat, so `gzip` takes an inference run's 8 GB down to under 100 MB in seconds, and the
parser reads `.log.gz` as readily as `.log` — compress them once the run is done.

## Log volume is a capacity question

`NCCL_DEBUG=INFO` writes a line per collective. A decode node produces about 2 GB per
run, and a four-node job with two roles fills 4 GB in minutes. On a shared home
directory this has hit 100% capacity, which truncated a file mid-write and marked a
finished job as failed at teardown.

- Plan the space before the run, and check it afterwards.
- Keep the logs gzipped; they compress about 24x and every parser here reads `.gz`
  directly.
- The parse cache means the expensive read happens once.

## Checklist before launching a profiled run

0. The engine's overlay merged into the manifest and every `<FILL_...>` filled, then
   mad-slurm-multinode's `scripts/validate_manifest.sh` run on the result — it checks
   the leftovers, the two env blocks agreeing, and `nodes == nnodes`.
1. `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS` including `COLL`, in the block that reaches
   the container, and `RCCL_LOG_DIR` so each rank logs to its own file.
2. The framework's profiler-bypassing paths disabled, for every node of every role.
3. A profile point that writes traces, and enough disk for them.
4. rocprofv3 only if kernel durations are actually needed; it is not what makes a
   profiled run slow, but it adds a failure mode.
5. Graceful shutdown wired up.
6. Space for the logs, and a plan to compress them.
7. A second run, same manifest minus the profiling, for the throughput numbers.
