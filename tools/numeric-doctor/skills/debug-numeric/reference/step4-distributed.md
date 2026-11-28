# Step 4-5, world size > 1: one job, many ranks (torchrun, FSDP/TP/EP/PP, any node count)

Read when `<RUN_SCRIPT>` launches ONE job over several ranks (torchrun, `srun` of a
distributed trainer). Not for independent single-process trials that share GPUs
(co-tenants: those are separate jobs; `step1-contention.md`). Validated 2026-11-09 on
torchtitan Qwen3 MoE, FSDP 4 x TP 2 x EP 4, 8 x MI355X.

## Wiring

* Same three lines as `step4-capture.md`, in the per-step method every rank runs;
  `cap.step()` at the top of each step on **every** rank, outside any other collective
  (it calls `sync_stop()`: all ranks stop together, none hangs in RCCL).
* **The framework owns the loop** (torchtitan, Megatron, a `Trainer` class): don't edit
  it -- patch its per-step method from a wrapper that then runs the framework's entry
  point. Template: `scripts/torchtitan_capture_train.py` (run it with
  `torchrun ... torchtitan_capture_train.py <torchtitan args>` in place of
  `-m torchtitan.train`). Build the capture at the first step (after model, optimizer,
  checkpoint load), never around `main()`. The wrapper already patches the step: the
  locator's `wired=no ... NEXT: wire_capture.py apply` line does not apply to it.
  **torchtitan default: `NAN_CAPTURE_SUSPEND=torchtitan.models.common.attention:FlexAttention.compiled_flex_attn`**
  in `CAPTURE_ENV` from the first run (its compiled FlexAttention cannot run under the
  capture; *The one knob* below).
* **Keep every rank's log**: `torchrun --log-dir <run dir>/ranklogs --redirects 3 --tee 3`.
  A launcher that tees rank 0 only (`LOG_RANK=0`) shows the failing rank as a bare
  `ChildFailedError` and SIGTERMs; the real error is in that rank's own log.
* Capsule dir on a filesystem every node sees; ranks write `rank<N>_<uuid>.pt` side by
  side; rank 0 writes `manifest.json` there at the stop.

## Running it: the capture-to-reproducer job, multi-rank mode

The wave runs ONE multi-rank job as one trial: a one-line spec with GPU field `all` (the
job owns the node; log `<WAVE_DIR>/all_all.cap.<job>.<host>.log` ending `TRIAL_RC=`).
The command is the run script's launcher with `torchtitan_capture_train.py` in place of
the framework's entry point -- a launcher script of your own in `$ND_ART/` is fine:

```
all|cap|<the launcher's env>|torchrun --nproc_per_node=8 ... $ND_REPO/scripts/torchtitan_capture_train.py <framework args>
```

```bash
SPEC=<ARTIFACT_DIR>/capture.spec NAN_SIGNAL='<NAN_SIGNAL>' CAPTURE_ENV='NAN_CAPTURE_DATA=1 NAN_CAPTURE_SUSPEND=...' \
  CONTROLS=none nd submit --name c2r --gpus 8 --exclusive --step step4 -- capture_to_repro.sh
```

`CONTROLS=none`: a control arm would need a second job. Validated 2026-11-11 on Slurm
with apptainer and with docker (docker needs the docker runtime's `nofile` limit, now default). An
existing multi-rank launcher that honours `CAPSULE_DIR` also works: `WAVE_SCRIPT=<it>
WAVE_LOGS='<its trial log glob>'` instead of `SPEC`.

Seeing `manifest.json`, 5a groups with `--job`; `groups_priority.txt` puts the group holding
the **origin rank's** capsule first (tag `ORIGIN`, represented by that rank's capsule), tags
groups that are only allocator garbage `FALSE_POSITIVE?` and groups whose every member
arrived through a collective `PROPAGATED?`; 5b replays the origin capsule (R1).
Validated 2026-11-09 (torchtitan MoE, 8 ranks): G1 = rank 4's `_grouped_mm` `ORIGIN`.

## The node-count ladder (`<NODES>` > 1): capture on the fewest nodes that show the bug

A run that needs `<NODES>` > 1 nodes is captured on `<NODES>`, then `<NODES>`/2, ... down to 1
node, and the rest of the investigation uses the fewest nodes that still produced **the same
bug** -- `<CAPTURE_NODES>`. All rungs run in ONE capture-to-reproducer job of `<NODES>` nodes
(a multi-node job can queue for hours: one queue wait, not one per rung); rung K uses the job's
first K nodes. Validated 2026-11-12 on Slurm + apptainer:
the qwen3 MoE example's `run.sh 4` (HSDP, dp_replicate = nodes) -- rungs 4 -> 2 -> 1 each captured
rank-local `_grouped_mm` at trainer.py:762 on the first try (4 and 2 nodes: same stack digest; 1
node: same model frames, different digest -- the frames fallback matters), `CAPTURE_NODES=1`,
5b REPRODUCED c2, kernel 14013; ~2.5 min for the whole ladder. **Same bug = same site, not
same kernel**: the 2-node capture of one run replayed in kernel 14024, the 4- and 1-node ones
in 14013 (same op, same model line; the GEMM picked for that rung's token counts differs) --
report the kernel of the `<CAPTURE_NODES>` reproducer, not of the `<NODES>` run. The other exits, same
setup (attempts forced clean by a `case $NAN_CAPTURE_DIR` guard in the spec): one run -- the
2-node rung clean on try 1, SAME on try 2; 1 node clean twice -> `CAPTURE_NODES=2 (1 node(s) did
not capture the same bug in 2 attempt(s))`, replayed from `nodes/n2/try2`; another --
`NODE_LADDER_MIN=2` on 2 nodes -> `CAPTURE_NODES=2 (the ladder reached its floor of 2)`, `REPLAY=0`
-> `WAVE_ONLY`. Re-test after the srun `--wait=60` fix
(a second run): the same verdicts, every rung `cap=1/1 died=0`, each node
ending `TRIAL_RC=1` (the capture's stop) -- none killed.

* **Same bug** = the rung's site capsule (the origin rank's, else the first group that is not
  `FALSE_POSITIVE?`) has rung `<NODES>`'s ATen op AND its stack digest or its last 4 model
  frames (`scripts/same_site.py`; dtypes and shapes are noted, not required). A capsule
  elsewhere counts as this rung NOT reproducing -- ledger it anyway: it is a second bug.
* **Only the data-parallel degree shrinks** (DP replicate, else DP shard / DDP). TP, EP, PP,
  CP degrees, the per-rank micro-batch and the sequence length stay, so each rank runs the same
  shapes; the global batch shrinks with the nodes -- say so in the report. The ladder stops at
  the fewest nodes whose DP degree is still >= 1 with the other degrees unchanged -- work it out
  from the run script and pass it as `NODE_LADDER_MIN` (default 1; e.g. TP 8 x PP 2 = 16 ranks
  per replica: 2 nodes). Odd K halves down (3 -> 1).
* **The spec**: ONE line, GPU field `all`, its command written with `$ND_NNODES`,
  `$ND_NODE_RANK`, `$ND_MASTER_ADDR` (`wave.sh` gives each node's `nd run` these):
  `torchrun --nnodes=$ND_NNODES --node_rank=$ND_NODE_RANK --master_addr=$ND_MASTER_ADDR
  --master_port=29500 --nproc_per_node=8 ...` and the DP degree from `$ND_NNODES` (torchtitan:
  `--parallelism.data_parallel_replicate_degree $ND_NNODES`; a global-batch setting divided
  accordingly, or set the local batch instead). A dump/output dir per run: include
  `$ND_JOB_ID-n$ND_NNODES`. Example: the test's `capture.spec`.
* **Attempts per rung** (`NODE_LADDER_TRIES`), from `<FULL_MODEL_RATE>` p measured at
  `<NODES>` nodes (runs that NaN / runs; step 2's `rate.sh`, `step2-rate.md`): `ceil(ln 0.1 / ln(1 - p))` -- the attempts after which
  a rung as likely to NaN as `<NODES>` nodes would have shown it with 90% probability (p = 1: 1;
  0.5: 4; 0.3: 7). Unmeasured (step 2 skipped): 2, and say that a clean rung may be bad luck.
  More than 4 attempts: show the cost and ask before submitting (CLAUDE.md). Every rung,
  rung `<NODES>` included, gets that many attempts before it counts as clean.
* **Walltime** `<T>`: (attempts x rungs) x one run's length (+ ~2 min start-up each) + 30 min
  for 5a, the replay and packaging; rungs = log2(`<NODES>`) + 1. A passing rung ends at its
  first capsule, so the usual job is far shorter.

```bash
SPEC=<ARTIFACT_DIR>/capture.spec NAN_SIGNAL='<NAN_SIGNAL>' CAPTURE_ENV='...' CONTROLS=none \
  NODE_LADDER_TRIES=<n> NODE_LADDER_MIN=<m> nd submit --name c2r --gpus 8 --nodes <NODES> --exclusive --time <T> \
  --step step4 -- capture_to_repro.sh
```

The job ledgers one `node-count ladder n=<K> try=<t>: capsules=.. -> REFERENCE | SITE=SAME (..)
| SITE=DIFFERENT .. | NO_CAPSULE` line per attempt, then `CAPTURE_NODES=<K> (of <NODES>; why it
stopped) -> nodes/n<K>/try<t>`. That rung's capsules (`nodes/n<K>/try<t>/capsules` = the new
`<CAPSULE_DIR>`) and groups (copied to `<ARTIFACT_DIR>/groups*`) go on to 5a-6 as usual; the
replay is single-node anyway (R1). `NODE_LADDER=0`: one wave on all `<NODES>` nodes, no ladder
(the user's call, e.g. the node count is part of the bug report).

**Every later model run uses `<CAPTURE_NODES>`** -- a recapture (`NAN_CAPTURE_RECORD_COLLECTIVE`
for R3, real values), R2's `torchrun` (world size = `<CAPTURE_NODES>` x 8), the 5N multi-rank
stand-in's starting size, and step 8's model A/B (`model_ab.sh --nodes`, `step8-fix.md`). Rates
are bound to it (SKILL.md rule 5): a rate measured on `<NODES>` nodes is not one on
`<CAPTURE_NODES>`.

## What the capture handles by itself (no knob)

| situation | behaviour |
|---|---|
| `DTensor` and other wrapper subclasses (FSDP2 params, TP activations, Float8) | deferred to the subclass: its local ops are checked as plain tensors |
| fake / functional / meta tensors (sharding propagation, tracing) | passed through unchecked: no data |
| higher-order operators (`flex_attention`, `cond`, ...) | run unchecked: their bodies are invisible, like a compiled region |
| FSDP2 padding from `torch.empty` (partly written all-gather buffers) | origin `uninitialized`: never a capsule on its own, not an escape when gathered; saved once as `.masked-<n>`; garbage written into a tensor that outlives the step IS captured |
| a NaN that reached this rank through a collective | capsule `dist_causality.arrived_via` = (group, seq, op); the rank's `first_arrival` |
| a NaN that reached a **pipeline stage** through the pipeline's send / recv (outside the dispatcher: no arrival is recorded) | the manifest tags every rank `pp<i>/tp<j>` (`NAN_CAPTURE_RANK_TAG`, set by `scripts/vllm_capture`); of the ranks that hold the NaN with no tracked arrival, only the EARLIEST stage is named origin (2026-11-24: PP 2, the NaN made in stage 0 -- both stages had been named) |
| a **synchronous collective** -- a framework's own all-reduce op (vLLM's `vllm.all_reduce`: QuickReduce / custom all-reduce / RCCL behind one op) -- returning non-finite from finite inputs | captured AT the collective on every member rank (each capsule: that rank's pre-call input, `collective` = group, seq); the manifest's `collective_origin`: *Synchronous collectives*, below |

## Synchronous collectives (the collective made the NaN)

Unlike async c10d, `vllm.all_reduce` & co. return their result in stream order, so the capture checks
their output like any op's. The manifest verdict reads `COLLECTIVE: vllm.all_reduce.default #<seq> of
group tp:0 returned non-finite from finite inputs on rank(s) [0, 1]`; 5a tags the group `COLLECTIVE`;
`capsule_info.py` prints a `collective` line. No recapture: the member ranks' capsules ARE the R3
record. The capture-to-reproducer job packages all of them (`repro-G<i>/capsule/`) and replays the
collective under `torchrun` through the framework that ran it:

```bash
python3 scripts/collective_r3.py check <capsule dir>          # no GPU: COLLECTIVE_FAULT / INPUT_NONFINITE / CORRECT
torchrun --nproc_per_node=<ranks> scripts/collective_r3.py replay <capsule dir> [--iters N]   # model's image
```

`check` compares each rank's output with the fp64 reduction of every rank's input and adds
`NARROW_INTERMEDIATE? fp16` (fp8) when the inputs and the true result fit the dtype but exceed a
narrower float's range: a cast or packing inside the collective. `replay` (`--backend vllm`, chosen
by the op) initialises vLLM's distributed environment and TP group, restores the capsules' `VLLM_*`
env (so vLLM picks the same all-reduce for the same size) and calls the recorded op; last line
`LADDER_RESULT=REPRODUCED n/N`. A control run with one env setting changed
(`NAN_R3_KEEP_ENV=<VAR>` keeps the caller's value) is step 7's first experiment.

## The one knob: compiled entry points

`torch.compile` refuses to trace under any user dispatch mode. The run dies with
`non-infra torch dispatch mode present, this is not supported today in torch.compile`
(or `found no compiled frames`). Name the compiled entry point the model calls from
eager code -- the outermost frame in that traceback that is model code:

```bash
NAN_CAPTURE_SUSPEND=<pkg.module:Class.attr>[,...]
# torchtitan FlexAttention:
NAN_CAPTURE_SUSPEND=torchtitan.models.common.attention:FlexAttention.compiled_flex_attn
```

Those calls run with the capture off; the capsule's `suspended` field lists them. Say in
the report that their bodies were not checked. A whole compiled model cannot be
captured this way: run it eager (the user's call; it may change kernels).

## Parallelism bugs: replica check (no NaN needed)

A dropped or doubled reduction, or wrong placements, makes no op wrong -- replicas just
drift apart (loss diverges, grad norm off, sometimes NaN much later). Turn on the
replica check: `cap.capture.watch_replicas(model.named_parameters(), every=1)` in the
wiring (torchtitan wrapper: `NAN_CAPTURE_REPLICAS_EVERY=1`). At each step boundary it
compares an fp64 digest of every parameter across the ranks that must hold the same
values (a DTensor along each `Replicate` mesh dimension; a plain tensor across the job,
DDP). A mismatch stops all ranks; the manifest verdict reads `PARALLELISM: replicas of
<param> diverged across ranks [...] at step N` -> step 5N (multi-rank native). Cost:
one small all_gather per group per check.

## Reading the result (5a)

```bash
python3 <REPO_ROOT>/scripts/compare_nan_capsules.py <CAPSULE_DIR> --job [--recursive]
```

* `JOB MANIFEST ... verdict`: **origin rank(s)** -- whose NaN was made locally (inputs
  finite, not arrived) -- or, if none, the earliest arrival (group, seq): the origin is
  a member of that group the capture could not see into (a suspended/compiled region,
  or the collective itself).
* Per rank: `CAPTURED` / `nan-seen` / `clean`, the capsule's op, its tag (`pp<i>/tp<j>`), and where its first
  NaN arrived from. Ranks that only received the NaN are not the bug: replay the origin's.
* `--job` groups without geometry (EP/PP ranks run one site on different shapes).
* Ledger: origin ranks, arrivals, capsule of the origin rank.

## Replay ladder (cheapest first; each run judges the capsule's own trigger)

| rung | what runs | for | how |
|---|---|---|---|
| R1 solo | the origin rank's chain on one GPU (copies per GPU = concurrency) | local op bugs, deterministic or racy | the usual 5b (`replay_ladder.sh`, `CAPSULE=<origin capsule>`) -- a chain stops at collectives, whose outputs are recorded roots, so it needs no other rank |
| R2 concurrent | every rank's capsule at once, one per GPU, no collectives | faults that need all GPUs busy | `torchrun --nproc_per_node=<W> scripts/replay_nan_op_dist.py '<dir>/rank{rank}_*.pt'` (per-rank mode; `--collective-noise MB`) |
| R3 collective | the suspect collective on its recorded inputs, every member rank | the manifest names no origin, and the earliest arrival is a collective (group, seq) -- or a `collective_origin` (synchronous collectives: no recapture, *Synchronous collectives* above) | re-run the capture with `NAN_CAPTURE_RECORD_COLLECTIVE=<group>:<seq>[:<step>]` (the manifest's earliest arrival): every member saves that collective's inputs and outputs. Then `python3 scripts/collective_r3.py check <CAPSULE_DIR> --group G --seq S` (no GPU): `COLLECTIVE_FAULT ranks [...]` (inputs finite, output wrong -> RCCL/transport), `INPUT_NONFINITE ranks [...]` (look upstream there) or `CORRECT`; and `torchrun --nproc_per_node=<group size> collective_r3.py replay ...` re-runs it on the recorded inputs to reproduce an intermittent fault without the model. Group and seq numbering only line up across runs when the collective structure is the same (same config) |
| 5N multi-rank | a minimal parallel program | parallelism bugs: every op correct, the combination wrong (a dropped reduction) -- no op check fails, only cross-rank invariants | `step5n-native.md` |

## Cost and time (2026-11-09, torchtitan debug MoE, 8 ranks)

Capture armed at step 1: first step ~3 s slower; a false positive at step 0-1 shows up
within a minute. All 8 ranks stop at the next step boundary after any capture; the
manifest costs one `all_gather_object`.
