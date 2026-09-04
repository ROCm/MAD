# Comparing two collective backends

A backend comparison asks a narrower question than a profile does — *is this transport faster than
that one* — and it is the easiest question here to answer wrongly, for two structural reasons:

1. **The traffic under test reaches no RCCL log.** An expert all-to-all backend carries its own
   transport: MoRI over IBGDA, DeepEP over rocSHMEM. Neither emits an RCCL record nor a
   `record_param_comms` event, so the channel the report is strongest in is silent about the one
   operation being compared, while the report still looks complete. What it measures is the
   tensor-parallel and DP-attention exchange, which is identical in both arms.
2. **Switching the backend tends to switch other things with it.** A backend whose low-latency
   kernels are unavailable falls back to a throughput mode, that mode is not graph-capturable, and
   graph capture then comes off — three changes from one decision, of which only the first was
   intended.

Both cost the same way: the difference gets attributed to the transport because the transport is
what was nominally changed.

## The case this file comes from

A Kimi-K2-Instruct FP8 comparison on 4P+4D EP32 reported MoRI 5–14% ahead of DeepEP end to end. The
arms also differed in three settings besides the backend:

| knob | MoRI arm | DeepEP arm |
|---|---|---|
| `moe_a2a_backend` | `mori` | `deepep`, `deepep_mode=normal` |
| decode graph capture | on | **off** (`--disable-cuda-graph`) |
| `mem_fraction_static` | 0.73 | **0.92** |
| `max_running_requests` | 8192 | not set |

Two independent signs that the transport was not the whole story:

- **The gap has a fixed-cost shape.** At EP32 / ISL 8192 the TPOT difference was 19.8, 19.6 and
  19.7 ms at concurrency 64, 128 and 256 — constant across a fourfold change in batch. A
  bandwidth-limited cost scales with tokens in flight; a per-step host cost does not.
- **A third of it survives where the transport cannot act.** At EP8 (1P+1D) each server occupies
  one node, so the expert all-to-all is entirely intranode and no internode dispatch exists to
  blame — and the gap was still ~6.6 ms.

The microbenchmark pointed the other way for the same pair (DeepEP low-latency at ~3085 µs against
MoRI's ~4000 µs), because it exercised the low-latency kernels the application run did not have.

## What the tooling now does about it

- `--compare-config <other run>` puts the settings that differ in the report and marks the ones
  that move throughput on their own. Four of the settings above are on that list, so this pair
  cannot come out looking comparable.
- The **step-time section** reads the engine's own accounting, per node, as a distribution rather
  than a mean — which is what separates a fixed per-step cost from occasional stragglers — and
  carries whether graph replay was on for those steps. That last field is the confound above,
  observed in band rather than inferred from a startup line.
- The **expert all-to-all section** classifies trace events by name into dispatch, combine, permute
  and transport. It is a discovery aid, not a measurement: when no name matches it says so and
  lists the busiest unclassified device events, which is what it takes to extend
  `A2A_PATTERNS` in the engine module. Its durations carry the same caveat as every other trace
  duration in [interpretation.md](interpretation.md) — the share of a category within one trace is
  comparable, an absolute figure is not.

## The arms a comparison needs

One knob per row. Rows A and B are the controls a backend comparison is usually missing.

| # | arm | what it isolates |
|---|---|---|
| A | the faster backend, **graph capture off**, everything else as tuned | the graph-free penalty, in ms of step time |
| B | the faster backend, tuned, with the other arm's `mem_fraction_static` | the KV-budget difference |
| C | the slower backend with the faster one's `mem_fraction_static` | the same, from the other side |
| D | the slower backend in its **low-latency mode with graphs**, if that path works | the only true apples-to-apples |
| D' | the **faster** backend forced into the slower one's mode, when D is impossible | the same isolation from the other side |
| E | both arms at the smallest topology that keeps the collective intranode | a floor: whatever gap survives here is not internode transport |
| F | both arms at long input, concurrency swept | the prefill-side component, which the decode step time does not carry |

**Arm D' is the one to reach for, because D usually cannot be built.** A backend whose
low-latency kernels do not work on a fabric cannot be moved onto them — that is the premise of the
whole problem. The backend that *does* have both modes can be moved onto the other's, and that
isolates the same quantity. Running D' alongside the uncontrolled comparison gives two one-factor
comparisons whose gaps must add up to the uncontrolled one, and that sum is the check on the
design: on the Kimi-K2 pair the mode step was -4.1 ms, the backend step at matched mode +14.7 ms,
and the uncontrolled measurement +10.6 ms, agreeing to 0.0 ms over three independent runs.

## What the arms measured on the Kimi-K2 pair

All at a matched kernel variant, so each row is one factor. Step time is the median over each
arm's own accounting, 66k to 196k intervals per arm.

| arm | topology / input | MoRI | DeepEP | gap |
|---|---|---:|---:|---:|
| D' | EP16 2P+2D, ISL 1024 | 226.0 ms | 240.7 ms | **+14.7 ms** |
| E | **EP8 1P+1D**, ISL 1024 | 229.8 ms | 232.4 ms | **+2.6 ms** |
| F | EP16 2P+2D, **ISL 8192** | 229.6 ms | 245.7 ms | **+16.1 ms** |

**Arm E is the one that changes the reading.** A metadata-dominated exchange invites the
conclusion that the cost is the protocol itself, and that conclusion is wrong. The protocol shape
is *unchanged* intranode. Counting the same kernels on both sides — the notify pair against the
ones that move tokens — DeepEP spends **96.6%** of the exchange on metadata intranode against
**90.9%** internode, so the shape is if anything more lopsided where the gap is smaller. Yet the
gap collapses from 14.7 ms to 2.6 ms.

So roughly 12 of the 14.7 ms is internode-specific: those metadata round-trips are nearly free
over XGMI and expensive over the fabric. A share within the exchange says where time goes *inside*
the exchange, and says nothing about how much of a step the exchange is worth — arm E is what
separates the two, and no amount of staring at kernel shares substitutes for it.

A kernel share also cannot say what a kernel put on the wire, and that is the next question after
arm E. Two implementations of the same exchange differ by nothing a trace can see: one writes its
metadata into the peer's memory and signals, the other waits for a reply, and only the second pays
a round trip -- free inside a node, dominant across a fabric, which is the shape arm E exposes
without explaining. The adapter counters are the channel that explains it: writes against reads
and atomics, per node, over a window. They cost nothing to collect and do not degrade the run, so
an arm set built after [measurement-setup.md](measurement-setup.md) should carry them.

Read arm E as a floor rather than a transport ablation: both backends switch to a different
implementation intranode (`EpDispatchIntraNodeKernel` against `EpDispatchInterNodeV1Kernel`,
`deep_ep::intranode::` against `deep_ep::internode::`), so this is each backend's intranode path
against its internode one, not the same code with the transport swapped. The `variant` column
reports that scope, so which of the two ran is read off the report rather than remembered — the
axis was added after arm E, when every intranode kernel came back unlabelled.

**Arm F separates the two mechanisms that a single throughput number conflates.** At ISL 8192 the
decode cost stays put while the prefill cost grows with load:

| concurrency | throughput gap | TTFT gap | ITL gap | share of the E2E gap from decode |
|---:|---:|---:|---:|---:|
| 64 | -8.6% | +26.8% | +7.3% | 79.6% |
| 128 | -8.4% | +48.5% | +7.4% | 72.4% |
| 256 | -12.3% | +51.4% | +6.6% | 60.2% |
| 512 | -13.5% | +53.8% | +7.9% | 57.9% |

ITL moves by less than a point and a half across an eightfold change in concurrency; TTFT doubles.
"The gap widens with load" is therefore a statement about prefill saturation, not about the decode
step, and quoting the throughput percentage alone hides which of the two moved.

Which knob moves the mode is engine-specific and is usually not an `--flag`. For MoRI it is
`SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD`, a token count: at or below it the
low-latency kernel runs, above it the throughput one. The launcher sets it to twice
`MORI_MAX_DISPATCH_TOKENS_DECODE` for decode, so the condition holds on every step and decode is
always low-latency; prefill never exports it and lands on normal. Setting it to `0` puts every step
above the threshold.

**An environment variable of this kind is invisible to the configuration channel**, which reads the
engine's `ServerArgs`. A comparison of two arms differing only in it reports *no setting differs*,
which is correct at the level it reads and misleading here. The kernel-variant column is what
carries the difference, and it is read from the trace rather than from a flag.

**Arm A needs a control of its own.** It is tempting to use the profiled run for it —
`PROFILE_ENABLE=1` does make the run graph-free — but that flag also adds
`--disable-custom-all-reduce`, which routes the intra-node TP exchange through RCCL instead of the
engine's own kernel. A profiled run therefore differs from the tuned one in **two** settings, and
the difference between them is the cost of both together, not the graph-free penalty this arm
exists to isolate. Build arm A with `--disable-cuda-graph` alone.

That the two are conflated is easy to miss precisely because one flag sets both, which is why it
is worth stating: a measurement configuration is not a control unless it varies one thing.

## Normalising

- **Per step, not per second.** A step-time difference in milliseconds is a quantity that can be
  explained; a percentage of throughput at fixed concurrency is the same measurement restated. At
  ISL 1024 in the run above, the whole throughput difference was recoverable from the step time
  alone, so quoting both as separate evidence double-counts one number.
- **Per rank**, as everywhere else in these reports.
- **Compare medians** when the step-time distribution has a tail, and say which it was.

## A claim about a backend

Before writing one, the questions from [interpretation.md](interpretation.md) plus three:

1. Did the arms differ in anything besides the backend? (`--compare-config` answers this.)
2. Was graph capture the same in both?
3. Was each backend in the same mode as the other for the phase being measured, and does the
   report say which mode that was?

If the answer to 3 is no, the honest claim is about the mode that was available, not about the
backend. "This backend's low-latency path is unavailable on this stack, which forces a graph-free
decode and costs X%" is a different statement from "this backend's all-to-all is X% slower", and
it points at different work.

Question 3 used to read "the mode it is *meant* to be in — low-latency kernels for decode,
throughput kernels for prefill". Arm D' contradicted that premise and it has been dropped: MoRI's
low-latency decode kernel measured **slower than its own throughput kernel** on an otherwise
identical run, 230.1 ms against 226.0 ms per step, so the launcher default of
always-low-latency-in-decode costs 4.1 ms a step. A mode named for latency is not thereby the
faster mode at a given batch size, and which one wins is a measurement rather than a default. For
a *comparison* what matters is only that both arms are in the same mode; which mode is faster is a
separate question, and arm D' answers both at once.

The kernel name is the evidence for which mode ran, not the flag that was passed:
`EpDispatchInterNodeV1KernelLowLatency_fp8_fnuz` against `EpDispatchInterNodeV1Kernel_fp8_fnuz`.
The comparison reports it as a `variant` column and says so unprompted when the two arms differ,
so a mode mismatch is caught by reading the report rather than by remembering to check.
