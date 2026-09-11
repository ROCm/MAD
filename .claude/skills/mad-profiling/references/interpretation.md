# Reading the reports without overstating them

Read this before quoting a number from a report. Each channel is strong at one thing and
silent about others, and most of the ways to be wrong here come from combining them.

## What each channel supports

| channel | window | supports | does not support |
|---|---|---|---|
| RCCL debug log | the whole run, including setup and weight loading | message sizes, call counts, per-rank and per-node volume, connectivity | any duration; anything a backend carries outside RCCL |
| torch profiler trace | the few steps a profile point covered | size, dtype and process group per collective; the collective mix; which named operations ran | rates; its own durations |
| rocprofv3 | a whole process, all phases, initialisation included | host API and device kernel durations | message sizes; per-phase attribution |
| engine step log | every logging interval of the run, profiled or not | step time as a distribution, per node, and whether graphs were replayed | attribution of that time to anything |
| reported configuration | the run's startup | what each node actually ran with, defaults applied | why it was set that way |
| RDMA adapter counters | from the first sample to the last, which the launcher brackets around the servers | how much crossed the fabric, and in what verbs: bytes, packets, writes, reads, atomics | per-rank or per-kernel attribution; which user of the NIC it belonged to; causality |

### What the counter channel is for, and what it cannot say

The only channel that reaches traffic RCCL never sees — an expert all-to-all on its own transport.
Three limits are structural:

* **Per adapter and per node.** Never per rank, never per kernel. A node's GPUs share what its
  NICs report, and nothing here divides that up.
* **Every user of the NIC is included**, the KV transfer among them. An absolute count is a
  **ceiling** for any one exchange, not a measurement of it.
* **Verbs, not causality.** Reads and atomics in quantity are positive evidence of a protocol
  that waits for replies; their absence proves nothing, because a reply can itself be a write and
  an acknowledgement is neither verb.

The load-bearing use is comparative: two arms serving the same requests, where the other traffic
is common-mode and the difference is the exchange. Even then the operation counts are comparable
only when both arms' transports increment the same counters -- one that posts through a path the
adapter's verb counters do not see reports a handful of writes for a terabyte, and the comparison
drops that row rather than showing it.

## Why no bandwidth is derived anywhere

Three independent reasons, each sufficient:

1. **The windows differ.** Volumes are per phase; rocprofv3 durations cover the whole
   process including initialisation. Dividing them mixes two different runs of time.
2. **RCCL fuses collectives into a few generic device kernels**, so device time cannot be
   attributed to an individual collective even in principle.
3. **Trace durations fail a physical check.** The same kernel averages tens of
   microseconds in the trace and milliseconds in rocprofv3, a factor of thousands, and the
   trace value puts multi-GiB collectives in tens of microseconds. Summing every kernel in
   a trace accounts for a few percent of its own `ProfilerStep` wall time, while rocprofv3
   kernel time fills the step. Sizes from the trace, durations from rocprofv3, never a
   ratio of the two.

## Sizes: per-rank shard or total message

The log-derived sections count the **per-rank shard**; the trace-derived sections count the
**total message across the group**, following nccl-tests. They differ by the group size,
and the report states the factor. Comparing a number from one section with a number from
the other without it is off by 8 on these runs.

## What a disaggregated serving profile cannot see

All of these are by construction, and the reports say so:

- **The KV-cache transfer.** Prefill hands the cache to decode over mooncake RDMA, never
  through RCCL. The traffic that defines the topology appears nowhere in the numbers; what
  is measured is the intra-node tensor-parallel exchange within each role.
- **The tuned configuration.** Collectives are only visible because the framework's own
  all-reduce and graph capture were disabled for measurement. Throughput from such a run
  is not the product's throughput — take it from a run without profiling.
- **Kernel timing, in practice.** rocprofv3 produced no stats for these servers even
  after the shutdown was made graceful and the processes were confirmed to exit on SIGINT
  within two seconds. Those reports are volume-only, and say so.
- **A steady-state window in the traces.** sglang's `/start_profile` emits no
  `ProfilerStep` markers, so a capture is an unmarked window holding roughly one forward
  pass. The mix and the sizes are sound; the counts are per capture, not per iteration.
- **The expert all-to-all, in RCCL terms.** A MoE backend carries its own transport — MoRI over
  IBGDA, DeepEP over rocSHMEM — so the exchange appears in no RCCL log and in no
  `record_param_comms` event. The trace names those operations and the report classifies them by
  name: a discovery aid, not a measurement. It reports what matched and stays silent rather than
  reporting zero when nothing did.

## Comparing runs and models

Normalise before comparing, and say what by:

- **Per rank, always.** Each node is an independent replica, so totals across a role
  depend on how many replicas were busy rather than on the workload.
- **Per step or per iteration** when comparing two runs of different length. Decode steps
  come from the server log (`grep -c "Decode batch"`); training iterations are already in
  the report.
- **Check the result against the model.** A number that reflects the model rather than the
  logging is the strongest evidence a parse is right: 170 collectives per decode step per
  rank for an 80-layer dense model that all-reduces twice per layer, 85 for a 36-layer
  mixture of experts. If a per-step count does not resemble the architecture, suspect the
  parse.
- **Report the busiest message size, not the largest.** The maximum is sensitive to
  surviving splices — one report's "largest" decode message was 172 MiB, a torn 176 KiB
  record carrying 0.004% of the volume. The size that carries the most volume is stable and
  is what the size mix table ranks.
- **State the configuration difference first.** Attention backend, quantisation and
  profiling flags explain more differences than anything in the communication numbers, and
  are the easiest thing to leave out of a comparison. `--compare-config <other run>` puts them in
  the report and marks the ones that move throughput on their own. A comparison of two collective
  backends has its own file: [backend-comparison.md](backend-comparison.md).
- **Prefer the step time to a throughput ratio.** At fixed concurrency a decode throughput is
  determined by the step time, so quoting both as evidence double-counts one measurement. The step
  time is also a distribution: a gap constant across intervals is a fixed per-step cost, one that
  grows with the batch is volume-limited.

## A checklist for a claim

Before writing a number into a document:

1. Which job, and was it profiled? (If it is a throughput, it must not have been.)
2. Per rank, and over how many active ranks?
3. Per step or per iteration, or over the whole run?
4. Shard or total message?
5. Does the report warn about discarded records or a sanity bound?
6. Does the count resemble the model's architecture?
