# Step 4-5, world size > 1: one job, many ranks (torchrun, FSDP/TP/EP/PP, any node count)

Read when the model runs ONE job over several ranks (torchrun, a distributed trainer
launched by madengine). Not for independent single-process runs that share GPUs
(co-tenants: those are separate jobs; `step1-contention.md`).

## Wiring

* The `nan_doctor` tool arms every rank with no edits (`step4-capture.md`). Source
  wiring is the same three lines, in the per-step method every rank runs;
  `cap.step()` at the top of each step on **every** rank, outside any other collective
  (it calls `sync_stop()`: all ranks stop together, none hangs in RCCL).
* **The framework owns the loop** (Megatron, torchtitan, a `Trainer` class): don't edit
  it. If the tool's arming does not reach its per-step path, patch that method from a
  wrapper script that then runs the framework's entry point
  (`torchrun ... wrapper.py <framework args>` in place of the framework's module).
  Build the capture at the first step (after model, optimizer, checkpoint load), never
  around `main()`.
* Capsule dir on a filesystem every node sees; ranks write `rank<N>_<uuid>.pt` side by
  side; rank 0 writes `manifest.json` there at the stop (on SLURM each node's
  `node_<N>/nan_doctor_output/` holds its ranks' capsules).

## Running it

The capture is ONE `madengine run` (multi-rank, as the model's own launcher does it)
with the `nan_doctor` tool, started through `mad-run.sh` (`env-madengine.md`). There is
no control arm in the same job; a control is a second run without the tool.

Seeing `manifest.json`, 5a groups with `--job`; `groups_priority.txt` puts the group holding
the **origin rank's** capsule first (tag `ORIGIN`, represented by that rank's capsule), tags
groups that are only allocator garbage `FALSE_POSITIVE?` and groups whose every member
arrived through a collective `PROPAGATED?`; 5b replays the origin capsule (R1).

## What the capture handles by itself (no knob)

| situation | behaviour |
|---|---|
| `DTensor` and other wrapper subclasses (FSDP2 params, TP activations, Float8) | deferred to the subclass: its local ops are checked as plain tensors |
| fake / functional / meta tensors (sharding propagation, tracing) | passed through unchecked: no data |
| higher-order operators (`flex_attention`, `cond`, ...) | run unchecked: their bodies are invisible, like a compiled region |
| FSDP2 padding from `torch.empty` (partly written all-gather buffers) | origin `uninitialized`: never a capsule on its own, not an escape when gathered; saved once as `.masked-<n>`; garbage written into a tensor that outlives the step IS captured |
| a NaN that reached this rank through a collective | capsule `dist_causality.arrived_via` = (group, seq, op); the rank's `first_arrival` |
| a **synchronous collective** -- a framework's own all-reduce op (vLLM's `vllm.all_reduce`: QuickReduce / custom all-reduce / RCCL behind one op) -- returning non-finite from finite inputs | captured AT the collective on every member rank (each capsule: that rank's pre-call input, `collective` = group, seq); the manifest's `collective_origin`: *Synchronous collectives*, below |

## Synchronous collectives (the collective made the NaN)

Unlike async c10d, `vllm.all_reduce` & co. return their result in stream order, so the capture checks
their output like any op's. The manifest verdict reads `COLLECTIVE: vllm.all_reduce.default #<seq> of
group tp:0 returned non-finite from finite inputs on rank(s) [0, 1]`; 5a tags the group `COLLECTIVE`.
No recapture: the member ranks' capsules ARE the R3 record. Replay the collective under `torchrun`
through the framework that ran it:

```bash
madengine nan exec --image <IMAGE> -- python3 -m nan_doctor_rt.collective_r3 check <capsule dir>   # no GPU work: COLLECTIVE_FAULT / INPUT_NONFINITE / CORRECT
madengine nan exec --image <IMAGE> --gpus <one per rank> -- torchrun --nproc_per_node=<ranks> -m nan_doctor_rt.collective_r3 replay <capsule dir> [--iters N]
```

`check` compares each rank's output with the fp64 reduction of every rank's input and adds
`NARROW_INTERMEDIATE? fp16` (fp8) when the inputs and the true result fit the dtype but exceed a
narrower float's range: a cast or packing inside the collective. `replay` (`--backend vllm`, chosen
by the op) initialises vLLM's distributed environment and TP group, restores the capsules' `VLLM_*`
env (so vLLM picks the same all-reduce for the same size) and calls the recorded op; last line
`LADDER_RESULT=REPRODUCED n/N`. The replay goes through `mad-run.sh`. A control run with one env
setting changed (`NAN_R3_KEEP_ENV=<VAR>` keeps the caller's value) is step 7's first experiment.

## The one knob: compiled entry points

`torch.compile` refuses to trace under any user dispatch mode. The run dies with
`non-infra torch dispatch mode present, this is not supported today in torch.compile`
(or `found no compiled frames`). Name the compiled entry point the model calls from
eager code -- the outermost frame in that traceback that is model code:

```bash
NAN_CAPTURE_SUSPEND=<pkg.module:Class.attr>[,...]
# e.g. a FlexAttention wrapper:
NAN_CAPTURE_SUSPEND=<pkg.attention_module>:FlexAttention.compiled_flex_attn
```

Those calls run with the capture off; the capsule's `suspended` field lists them. Say in
the report that their bodies were not checked. A whole compiled model cannot be
captured this way: run it eager (the user's call; it may change kernels).

## Parallelism bugs: replica check (no NaN needed)

A dropped or doubled reduction, or wrong placements, makes no op wrong -- replicas just
drift apart (loss diverges, grad norm off, sometimes NaN much later). Turn on the
replica check: `cap.capture.watch_replicas(model.named_parameters(), every=1)` in the
wiring (or `NAN_CAPTURE_REPLICAS_EVERY=1` where the wrapper reads it). At each step boundary it
compares an fp64 digest of every parameter across the ranks that must hold the same
values (a DTensor along each `Replicate` mesh dimension; a plain tensor across the job,
DDP). A mismatch stops all ranks; the manifest verdict reads `PARALLELISM: replicas of
<param> diverged across ranks [...] at step N` -> step 5N (multi-rank native). Cost:
one small all_gather per group per check.

## Reading the result (5a)

```bash
madengine nan group <CAPSULE_DIR> --manifest build_manifest.json --out-dir <ARTIFACT_DIR> --job [--recursive]
```

* `JOB MANIFEST ... verdict`: **origin rank(s)** -- whose NaN was made locally (inputs
  finite, not arrived) -- or, if none, the earliest arrival (group, seq): the origin is
  a member of that group the capture could not see into (a suspended/compiled region,
  or the collective itself).
* Per rank: `CAPTURED` / `nan-seen` / `clean`, the capsule's op, and where its first NaN
  arrived from. Ranks that only received the NaN are not the bug: replay the origin's.
* `--job` groups without geometry (EP/PP ranks run one site on different shapes).
* Ledger: origin ranks, arrivals, capsule of the origin rank.

## Replay ladder (cheapest first; each run judges the capsule's own trigger)

| rung | what runs | for | how |
|---|---|---|---|
| R1 solo | the origin rank's chain on one GPU (copies per GPU = concurrency) | local op bugs, deterministic or racy | the usual 5b (`madengine nan replay <origin capsule> ...`) -- a chain stops at collectives, whose outputs are recorded roots, so it needs no other rank |
| R2 concurrent | every rank's capsule at once, one per GPU, no collectives | faults that need all GPUs busy | `madengine nan exec --image <IMAGE> --gpus all -- torchrun --nproc_per_node=<W> -m nan_doctor_rt.replay_nan_op_dist '<dir>/rank{rank}_*.pt'` (per-rank mode; `--collective-noise MB`), through `mad-run.sh` |
| R3 collective | the suspect collective on its recorded inputs, every member rank | the manifest names no origin, and the earliest arrival is a collective (group, seq) -- or a `collective_origin` (synchronous collectives: no recapture, *Synchronous collectives* above) | re-run the capture with `NAN_CAPTURE_RECORD_COLLECTIVE=<group>:<seq>[:<step>]` (the manifest's earliest arrival): every member saves that collective's inputs and outputs. Then `madengine nan exec --image <IMAGE> -- python3 -m nan_doctor_rt.collective_r3 check <CAPSULE_DIR> --group G --seq S` (no GPU work): `COLLECTIVE_FAULT ranks [...]` (inputs finite, output wrong -> RCCL/transport), `INPUT_NONFINITE ranks [...]` (look upstream there) or `CORRECT`; and `torchrun --nproc_per_node=<group size> -m nan_doctor_rt.collective_r3 replay ...` under `madengine nan exec --gpus all` re-runs it on the recorded inputs to reproduce an intermittent fault without the model. Group and seq numbering only line up across runs when the collective structure is the same (same config) |
| 5N multi-rank | a minimal parallel program | parallelism bugs: every op correct, the combination wrong (a dropped reduction) -- no op check fails, only cross-rank invariants | `step5n-native.md` |

## Cost and time

Capture armed at step 1: first step a few seconds slower; a false positive at step 0-1
shows up within a minute. All ranks stop at the next step boundary after any capture; the
manifest costs one `all_gather_object`.
