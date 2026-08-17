<!--
Template for the one report the tooling cannot generate: a comparison across runs,
models or configurations. It is written by hand because every claim in it depends on
which jobs are being compared and what was different about them.

How to use it:
  1. Copy to reports/<name>_comparison.md and fill every <FILL_...>.
  2. Take per-phase numbers from the generated reports, not from a log by hand.
  3. Delete a section only if it does not apply -- an empty section is a signal that
     something was not established, and that is worth seeing.
  4. Keep the last two sections. A comparison without its scope is the artifact most
     likely to be quoted out of context later.

Conventions that make a comparison hold up (references/interpretation.md):
  - every figure per rank, and say over how many active ranks;
  - throughput only from runs without profiling, and name the job;
  - normalise by step or iteration when run lengths differ;
  - busiest message size, not largest;
  - state configuration differences before explaining any gap.
-->

# <FILL_workload> on <FILL_cluster>: <FILL_variant_a> and <FILL_variant_b>

<FILL_one paragraph: what is the same across the runs -- topology, parallelism, request
or batch shape -- so that what differs is visible.> Everything below comes from these
runs:

| job | variant | what it is |
|---|---|---|
| <FILL_id> | <FILL_variant> | baseline, no profiling — the throughput numbers |
| <FILL_id> | <FILL_variant> | baseline, no profiling — the throughput numbers |
| <FILL_id> | <FILL_variant> | profiled<FILL_, with rocprofv3 attached> |
| <FILL_id> | <FILL_variant> | profiled |

<!-- Forced differences first: a reader who does not know them will attribute them to
the model. If there are none, say so in one sentence. -->
<FILL_paragraph: which configuration differences were forced rather than chosen --
attention backend, quantisation, a kernel unavailable on this architecture -- and why.>

## Throughput

From the runs without profiling:

| <FILL_axis, e.g. concurrency> | <FILL_variant_a> | <FILL_variant_b> |
|---:|---:|---:|
| <FILL> | <FILL> | <FILL> |

<FILL_paragraph: scaling, latency behaviour, and the one place the variants part.>

## What profiling costs

<FILL_axis and value>, against a baseline of <FILL> for <FILL_variant>:

| run | <FILL_metric> | <FILL_metric> |
|---|---:|---:|
| baseline (<FILL_id>) | <FILL> | <FILL> |
| profiled with rocprofv3 (<FILL_id>) | <FILL> | <FILL> |
| profiled without rocprofv3 (<FILL_id>) | <FILL> | <FILL> |

<FILL_paragraph: what the cost is actually attributable to. On these runs it was the
measurement configuration and the debug logging rather than the profiler tool.>

This is why throughput and communication volume come from different runs, and why no
number in the sections below should be read as a rate.

## Communication, per rank, over the whole run

<FILL_sentence: which job each variant's figures come from, and why that job. Prefer a
run where every replica was busy.> Each figure is per GPU, and each node is an
independent replica, so these are the numbers that carry across the group; totals across
a role are not.

| variant, phase | collectives | volume | how it splits | busiest message size |
|---|---:|---:|---|---:|
| <FILL> | <FILL> | <FILL> | <FILL> | <FILL> |

The last column is the single message size carrying the most volume, not the largest one
seen: surviving splices can sit above it in size while carrying a rounding error of the
traffic.

<!-- Normalisation: without it, a longer run looks like a heavier workload. -->
<FILL_paragraph: steps or iterations per run, and the resulting per-step figures.>

<FILL_paragraph: a sanity check that the counts reflect the architecture -- collectives
per step against layer count, for instance. If they do not, say so rather than
explaining it away.>

<FILL_two or three short subsections: where the bytes go in each phase, and how message
sizes differ. Name the regime each phase is in -- bandwidth-bound at large messages,
latency-bound at small ones -- because it decides what tuning would even help.>

## Load balancing between replicas

<FILL_paragraph: how requests were spread, per role. Distinguish spread within a node
(expected to be even) from spread across nodes (how the router behaved). Confirm an
imbalance on a second run before drawing a conclusion from it.>

## What this method cannot see, by construction

<!-- Keep this section. Copy the points that apply from references/interpretation.md and
add anything specific to these runs. -->
- **<FILL_e.g. the KV transfer>.** <FILL_why it is invisible and what is measured instead.>
- **The tuned configuration.** <FILL_which flags were needed to make collectives visible.>
- **Kernel timing.** <FILL_whether rocprofv3 produced anything; if not, say the reports are volume-only.>
- **<FILL_trace window>.** <FILL_whether the traces carry step markers.>
- **Whole records.** <FILL_share of records discarded and why; name the sanity bound if any were rejected by it.>

<FILL_optional operational note: anything that cost a job -- disk filling, a shutdown
that lost stats -- so the next person does not repeat it.>

## Where the detail lives

| report | run |
|---|---|
| `reports/<FILL>_prefill/`, `reports/<FILL>_decode/` | <FILL_id> |

Each directory holds `report.md`, the CSVs behind every table, and `profile.xlsx` with
the same content as sortable sheets plus the rank matrix as a heatmap.
`scripts/regen_reports.py --catalog reports/jobs.json` rebuilds all of them.
