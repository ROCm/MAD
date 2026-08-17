<!--
Template for the write-up of one profiling campaign: what was run, what had to be fixed
to get measurable artifacts, and what the numbers turned out to be.

This is the document that saves the next person the days the campaign cost. The failures
are the valuable part -- a run that produced empty rocprofv3 output, a flag that reached
three nodes out of four, a kernel missing for the architecture -- so keep them even when
they are resolved, and say how each was detected rather than only what it was.

Copy to reports/<name>_findings.md and fill every <FILL_...>.
-->

# <FILL_workload> on <FILL_cluster>: profiling campaign

**Goal.** <FILL_one or two sentences: what question the campaign was to answer.>

**Outcome.** <FILL_two or three sentences: what is now known, and what remains open.>

## Runs

| job | configuration | result |
|---|---|---|
| <FILL_id> | <FILL_what was different about it> | <FILL_succeeded / failed with what> |

## What had to change before anything could be measured

<!-- One subsection per obstacle. Each states the symptom as it appeared, the cause, the
fix, and how it was verified -- verification is what makes the entry reusable. -->

### <FILL_symptom as first observed>

- **Symptom.** <FILL_what the artifacts looked like, e.g. "8 bytes per rank in the report".>
- **Cause.** <FILL_the mechanism.>
- **Fix.** <FILL_the change, and where it belongs: manifest, launcher, framework flag.>
- **Verified by.** <FILL_the check, e.g. "all four node logs report disable_custom_all_reduce=True".>

## Numbers

<FILL_the figures that answer the goal, per rank and per step, each naming its job. Point
at the generated reports for the tables rather than copying them.>

## Limits of what was measured

<!-- Copy the applicable points from references/interpretation.md. -->
- <FILL>

## Environment notes

<FILL_anything about the cluster or filesystem that a repeat run needs to know: log
volume and where it went, compression, disk headroom, shared paths, parse times.>

## Open items

- <FILL_what is still unknown, and what would settle it.>
