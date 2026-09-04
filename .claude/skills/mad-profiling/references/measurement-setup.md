# Making a run measurable

A profiled run is a different run. Every setting below buys visibility and most of them
cost performance, so the rule that follows from it is: **volumes come from the profiled
run, throughput comes from a second run without any of this.** A report that quotes both
from the same job is quoting a slowed-down configuration as if it were the tuned one.

This file is the reasoning; the settings themselves are ready to apply in
[../assets/manifest-overlay/](../assets/manifest-overlay/), as an overlay per engine on
a mad-slurm-multinode manifest plus the Primus experiment-YAML fragment. Read this to
decide what to include, apply that to include it.

## The six artifact classes

| artifact | needs | answers | does not answer |
|---|---|---|---|
| RCCL debug log | `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS` incl. `COLL` | every collective of the whole run: name, count, datatype, nranks, so message sizes | durations |
| torch profiler trace | a profile point in the framework (`--profile`, `/start_profile`) | size, dtype and process group per individual collective; which named operations ran | anything outside the captured window; its durations do not survive a cross-check |
| rocprofv3 stats | `rocprofv3 --rccl-trace` wrapping the process | per-API and per-kernel durations | message sizes; per-phase attribution |
| benchmark / perf CSV | the run's own harness | throughput and latency | anything about individual collectives |
| the engine's own logging | nothing — it is there in every run | step time as a distribution per node, the configuration each node ran with | attribution of that time to anything |
| RDMA adapter counters | `RDMA_COUNTERS=1`; reading sysfs, so nothing is degraded | how many operations of each verb crossed the fabric, and therefore whether two arms doing the same work put different amounts of traffic on it | causality — a reply can itself be a write, so verbs cannot prove a protocol waited; also no per-rank or per-kernel attribution, and no way to separate the KV transfer from the exchange |

The report joins whichever of these a run produced and states which are present. **No single one
is required**, including the RCCL log: a phase is skipped only when every channel is empty, and a
run with step times, a configuration and adapter counters but no `COLL` records still produces a
report — which is the normal shape of a *tuned* run, since an unprofiled job leaves
`NCCL_DEBUG_SUBSYS` at `INIT,NET,GRAPH`. What disappears with the RCCL log is the volume sections,
and the report says so rather than pretending they were measured as zero.

The last two rows cost nothing and survive profiling — the engine computes its own step time and
the adapter counts its own operations, so neither needs an instrument attached to the workload.
That is what makes them comparable between a profiled run and an unprofiled one, which is
otherwise the hardest comparison here to make: it is how the cost of the settings below is
quantified rather than only warned about.

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
minutes when profiling), and escalate only if it does not.

## Why sglang-disagg yields no rocprofv3 CSVs: two causes fixed, one left

Serving runs here produce no rocprofv3 CSVs at all. That was recorded as an observation for a
while; it is not a flush problem and not a missing package, and the cause is worth stating so the
next attempt does not spend a run rediscovering it.

madengine's `tools` block wraps **the model run command** — `bash run.sh` — and rocprofv3 follows
that process tree. For a PD-disaggregated launcher the tree is the orchestration, not the work.
A four-node Kimi-K2 run (job 224068, exit 0, `--kernel-trace --rccl-trace --stats --output-format
csv`) opened **668 profiling sessions across 48 distinct commands**, every one of them a shell
helper. Counted across all four ranks, since a session on any of them is one rocprofv3 attach:

| profiled, by command | sessions | ranks |
|---|---:|---:|
| `date` | 594 | all |
| `sleep` | 540 | all |
| `tee` | 53 | all |
| `python3` helpers (not a server) | 20 | all |
| `cut`, `awk` | 32 | all |
| **total** | **668** | 580 on rank 0, 26–31 on each of the others |

Rank 0 accounts for 580 of them because it hosts the proxy and runs the readiness-polling loop, so
the count scales with bringup time rather than with the workload.

`python3 -m sglang.launch_server` is **not among them**. The servers are started as
`eval "$DECODE_CMD" 2>&1 | tee -a "$DECODE_LOG" &`, and rocprofv3 attached to the `tee` on the
right of that pipeline while the server on the left escaped. Nothing that touched a GPU was
profiled, so there were no kernel or RCCL statistics to write — the empty output directory is the
correct result of profiling `sleep`.

The readiness-polling loop is what produces most of those sessions, which is also why the count
scales with bringup time rather than with the workload.

**One cause was in the launcher, and it is fixed.** The servers were started as
`eval "$CMD" 2>&1 | tee -a "$LOG" &` with the pid taken from `$!` — and for a pipeline bash sets
`$!` to the **last** command in it, so `$!` was the `tee`. The careful SIGINT of the previous
section went to `tee`, which died at once; the wait loop saw its pid gone and reported a clean
stop; the server was left to hit a broken pipe and die without unwinding. Every "exited on
SIGINT" line in these logs was a false report. Measured three ways, with a child that prints when
its handler fires:

| launch shape | `$!` is | SIGINT reaches the server |
|---|---|---|
| `eval "$CMD" \| tee -a log &` | `tee` | no |
| `eval "$CMD" >> log &` | a bash subshell, which ignores it as an async job | no |
| `eval "exec $CMD" >> log &` | the server | **yes** |

The launcher now uses the third, under `set -m` so the job gets a process group of its own, and
`_shutdown_server` signals the group rather than the pid — a server forks scheduler children, and
they have to hear it too.

**The other cause is not fixed, and wrapping the server by hand does not fix it either.** Two
2-node validation jobs (239109, 239268) ran rocprofv3 around the server command instead of around
`run.sh`. The wrapping works — the profiler instruments the server and each of its children, all
logging their own PIDs — and after the group-signal fix every one of them logs the signal. And
sglang still does not unwind: five minutes of SIGINT, three of SIGTERM, then SIGKILL, with empty
output directories. Nothing was written either time.

So the remaining obstacle sits inside the profiled process, not in the plumbing, and a switch for
it is deliberately **not** shipped here: an option that cannot produce output is worse than none.
What is worth trying next, in order — whether a scoped capture around `/start_profile` avoids
needing a graceful exit at all, since a persistent server is a poor fit for a profiler that
finalises at process end; and what keeps sglang's scheduler processes alive under instrumentation.

Until one of those lands, a serving report stays volume-only and
[interpretation.md](interpretation.md) governs what trace durations can support — shares within
one capture, never absolute milliseconds.

Training runs were never affected: Primus runs the workload in the wrapped process itself.

## What crossed the fabric: adapter counters

The channel for the question the other four cannot reach. A trace names the kernels of an
all-to-all but says nothing about their network behaviour, and a kernel that writes once and
signals is indistinguishable there from one that waits for a reply. Across a fabric that
difference *is* the cost, which is why an intranode arm collapses a gap that a kernel-share table
cannot explain.

`rdma_counters.sh` samples `/sys/class/infiniband/*/ports/*/{counters,hw_counters}` into
`rdma/<role>_NODE<n>.csv` while the servers run; the launcher starts it with `RDMA_COUNTERS=1`,
which defaults on whenever `PROFILE_ENABLE=1`. The report sums each counter's steps from sample to sample, per
node, and groups the counters into writes, reads and atomics. Read the grouping
asymmetrically: reads and atomics **in quantity** are positive evidence of a protocol that waits,
because something asked for data or performed a remote update; their absence is not evidence of
one that does not, since a reply can itself be an RDMA write and a transport acknowledgement is
neither verb.

Three limits, all structural:

- **Per adapter and per node.** Never per rank, never per kernel.
- **Every user of the NIC is included**, the KV transfer among them, so an absolute count is a
  ceiling for the exchange rather than a measurement of it. Two arms serving the same requests
  differ by the backend, which is why this channel is reported as a comparison.
- **Cumulative.** A counter that decreased wrapped or was reset; those are dropped and named, and
  the column is then a floor.

Unlike every other channel here it costs nothing and degrades nothing — reading sysfs perturbs no
kernel — so it is the only one that can be collected on a **tuned** run rather than on a run that
had to be spoiled to be measurable. `RDMA_COUNTERS=1` without `PROFILE_ENABLE` is a supported
combination for exactly that reason.

**The container cannot see the counters by default, and says nothing about it.** A class entry
under `/sys/class/infiniband` is a symlink into `/sys/devices/...`, and docker hands a container
the class directory with its symlinks intact while the targets are absent: ten adapters listed,
zero counter files under any of them, every sample empty. With `-v /sys/devices:/sys/devices:ro`
in the docker run options the same probe returns 53 counters per port.

**Only `run_xPyD_models.slurm` carries that mount.** The overlay cannot: `additional_docker_run_options`
is a single string in the base manifest and a `jq` merge replaces it rather than appending, which
is why the overlay only names the mount in a comment. Applying the overlay therefore does **not**
make the counters visible — add the bind to that string by hand, as the README describes for the
`/run_logs` bind, which has the same problem for the same reason.

The sampler refuses to paper over it: a requested adapter with no counter directory is named, the
remedy is printed, and a partial set is refused rather than sampled, because a total over some of
a run's adapters is not a smaller truth but a wrong number that reads like a right one.

What a validated run looks like (job 239268, 1P+1D, Llama-3.1-8B, ~31 minutes):

| node | rx write req | rx bytes | tx bytes |
|---|---:|---:|---:|
| prefill_NODE0 | 0 | 817 kB | 1.158 GB |
| decode_NODE1 | 52,736 | 1.158 GB | 817 kB |

One direction, exactly balanced, and carried by **write requests with no reads and no atomics**.
That is consistent with the KV transfer's one-sided RDMA and it is not a proof of it: these are
verb counts, and a reply can itself be an RDMA write while a transport acknowledgement is neither
verb. What the run does establish is that the channel reports the right traffic on the right
nodes, in the right direction, at a volume that matches what was transferred — which is what makes
it worth reading on traffic whose shape is the question.

Read the asymmetry when interpreting it: reads and atomics in quantity are positive evidence of a
protocol that waits, because something asked for data or performed a remote update; their absence
is not evidence of one that does not. The comparison is where the counts earn their keep — two
arms serving the same requests whose operation counts differ are doing different amounts of
network work, whatever the verbs are called.

Comparing two arms takes one more thing than the tooling can check. The counters are
whole-window totals, so they only mean anything if the arms served the same requests — and the
perf CSV keeps one row per point and metric, never a request count, so a different
`BENCHMARK_ITR`, a retry, or extra profile-point traffic leaves every point key matching while
one arm did more work. `compare_runs.py` therefore requires `--counters-same-workload` before it
will render the table: a check that cannot be made is better demanded of the person who knows
than guessed at.

One caveat from the same run: the samples land on the shared filesystem, and that filesystem
returns **zeros instead of an error** when a read falls in a bad window. One sample of 138 came
back as a megabyte of NUL bytes, which stops `csv` outright. The parser skips such lines, counts
them and says so in the report; [data-quality.md](data-quality.md) documents the same failure mode
for trace reads.

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
The lines repeat, so gzip takes an inference run's 8 GB down to under 100 MB in seconds, and the
parser reads `.log.gz` as readily as `.log`. Once the job has finished:

```bash
compress_logs.py --run-dir <run> [--rccl-dir <prof-root>/rccl]
```

It compresses what the engine's globs call a log and nothing else, verifies each copy by digest
before dropping the plain one, and refuses to touch a log that already sits beside its own `.gz`.
Such a pair is what `gzip -k` or an interrupted gzip leaves behind; the engine's globs match both
halves, so discovery keeps the plain file, ignores the `.gz` and says which logs it found doubled.
Delete one of each pair to silence that warning.

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
4. rocprofv3 only if kernel durations are actually needed, and read the section above first: on
   this engine it is not wired, and wrapping the server by hand was measured to produce nothing.
4b. `RDMA_COUNTERS=1` if the question is what crossed the fabric. It is free, so the only reason
   to leave it off is that nobody will read it.
5. Graceful shutdown wired up.
6. Space for the logs, and a plan to compress them.
7. A second run, same manifest minus the profiling, for the throughput numbers. Its step-time
   section is also the reference the profiled run's is measured against, which is what turns "both
   of those cost performance" into a number.
8. When the run will be compared against another, `--compare-config` pointed at it, so the report
   states what else differed. For two collective backends, read
   [backend-comparison.md](backend-comparison.md) first.
