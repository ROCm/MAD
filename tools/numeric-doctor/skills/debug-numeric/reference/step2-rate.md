# Step 2: run the model, measure the rate (on by default)

**Runs by default, without asking** (step-1 question 2): it is the check that the model
runs by itself in this environment before a capture is spent on it, and it measures the
rate. Skip it only when the user asked to; skipped: `$LG skip step2 "<why>"` and `$LG
skip step3 "no rate to raise"` (no `start`: a skipped step has no visit). Don't cite or
recommend from earlier runs' rates (script comments, old ledgers).

* run: `<RUN_SCRIPT>`'s own command, as the user gave it -- no capture wrapper -- to
  confirm it runs and NaNs and measure the rate (below). If step 1 had to add a NaN
  check to a working copy, run that copy; on a failure, run the original once too, so
  the failure is pinned on the right one;
* `<NODES>` > 1: by default one smoke run, not a rate (*A model that runs on more than one
  node*, below);
* skip (on request): `<NAN_RATE>` *unmeasured* -- **never** a rate carried over from a
  previous run or supplied by the user in its place. Skip step 3 too; step 4's controls
  become the first rate -- don't cut step 4 below the model's check window
  (`step4-capture.md` 4.3).

## Folded into step 4 (one node: the default)

`<NODES>` = 1 and no step 3 wanted: **no separate wave.** A step-2 wave and the capture wave run the
same model on the same node type; the capture wave's control arm measures what step 2 would, and the
capture-to-reproducer job's preflight (`gpu_check --torch`, the image, the repo) is the "does it run
here" check. On 2026-11-19 a separate step-2 wave was 4.6 of the 18 minutes to a reproducer.

* Ledger `$LG skip step2 "folded into step 4: the control arm measures the rate"` (and `$LG skip step3
  "folded"`) -- the table shows it as folded, not missing.
* The step-4 spec keeps **at least 2 control lines** (more when the GPUs allow: the rate's
  denominator), and the job runs with `WAIT_CONTROLS=1`: the early stop waits for the controls to end
  (NaN, or the run's length) instead of killing them at the first capsule. Their cost is what
  step 2 would have spent anyway.
* `<NAN_RATE>` = the job's `ctl rate n/m` (`wave_summary.<tag>.txt`) with its environment, plus the
  capture arm's NaN'd trials as supporting evidence (hooked: "at least"). Below 8 control trials it
  is **provisional** -- say so; it still settles "does it run" and "does it NaN here". Two to six
  controls cannot tell a 30% rate from 0 (2026-11-20 qwen_tf32: controls 0/6, the same model's
  stock arm in step 8 3/12): never read a clean control arm as "rare" or "gone" -- the capture arm
  NaN'd. **Step 8's model A/B stock arm is a second measurement** in the same environment: the
  report quotes both, and the larger denominator as the rate (`step9-report.md`).
* **World size > 1 (one job over all GPUs): no control arm** -- the capture job runs with
  `CONTROLS=none` (`step4-distributed.md`: a control would be a second job), so a folded step 2
  gives **no rate**: ledger `$LG skip step2 "folded, world size > 1: no control arm -- NAN_RATE
  unmeasured"`, take the capture's own trial (NaN'd: yes/no) as "does it NaN here", and the rate,
  if any, from step 8's stock arm. A rate needed before step 4 (step 3, a budget decision): run
  `rate.sh` (`env.md`) as an explicit step 2 instead of folding it.
* **All controls `ERROR` / `DIED`** (and not the external-kill signature): *The model does not run*,
  below, exactly as for a step-2 wave -- the capture's result is void.
* Not folded: the user asks for a rate first, or step 3 (raising a rate needs one measured first), or
  `<NODES>` > 1 (the smoke run below), or no capture job will run (`CAPTURE_UNREACHABLE`,
  `step5n-native.md`: the rate at the original setup is 5N's evidence and its rate to beat).
  A model whose one run takes several GPUs of one node (vLLM / SGLang TP 2-4, a `torchrun` on 2-4
  GPUs): a `wave.sh` spec line per run with a GPU **list** as its GPU field (`0,1|ctl||<cmd>`,
  `2,3|ctl||<cmd>`, ...: one trial over those GPUs, `HIP_VISIBLE_DEVICES` set to the list), so 8
  GPUs give 8/TP runs per wave; the size-1 check rides in the same wave on its own GPUs.

## The model does not run

**Every** scored run failed -- `wave_summary.py` scores all trials `ERROR` / `DIED`, or
`RATE_RESULT` has no scored run (`0/0 err=<n>`) -- and it is not the environment
Numeric-doctor broke: not the external-kill signature (`waves.md`), not a failed `<ENV>`
check (`TORCH_GPU=NO`), not a missing image. Some runs failing and some running is not
this (an odd GPU or contention: *Partition by GPU*, below). Then **stop the ladder**:
no step 3, no capture. Ledger `MODEL_DOES_NOT_RUN <host> <the error line>` and end the
turn with this block, filled in, and nothing else:

```markdown
> **⚠️ The model does not run here -- even without Numeric-doctor's capture.**
> `<RUN_SCRIPT>` failed in every run (<n> of <n>) before producing a result:
>
>     <the 3-5 lines of the error, from the log>
>
> Command: `<the command step 2 ran>` on `<host>` (image `<image>`). This is not
> the NaN, and Numeric-doctor cannot debug a model that does not run. **Please first make
> sure `<RUN_SCRIPT>` works by itself, without Numeric-doctor**, then tell me to continue.
> If it does run for you, tell me how you ran it (host, image, environment): then the
> problem is in how I launched it, and I'll fix that.
```

The user says it works by itself: compare their way of running it with `<ENV>`
(`env.md`, `env-runtime.md`), fix the difference, and run step 2 again (a new `start`).

## Measure a rate, not "it NaN'd"

* **>= 8 trials, concurrently across the node's GPUs (one wave)**; one run
  (m = 1) is not a rate -- if the user insists, mark it provisional.
* **k <= 1: run a second wave and pool** (1/4 and 1/24 are the same single event).
* **Record the environment**: host, image digest, step count, containers per GPU,
  host threads per process (`OMP_NUM_THREADS`/`torch.get_num_threads()`; unset =
  all cores *per process*) vs cores and process count, `rocm-smi` Perf level and
  SCLK (`perf_determinism` != `auto`).
* **Host-thread oversubscription**: N processes x all cores. Tell: CPU-bound
  phases (data gen, init, optimizer setup) much slower in the full wave than alone.
  It also changes how fast each process feeds its GPU.
* **Partition by GPU before quoting a denominator**. Devices disagree: re-run
  pinned to the good ones for the denominator; probe odd ones alone (no partner) to
  separate bad GPU from bad-under-contention.
* **Drop trials that lost their co-tenant**: the tell is a late step-rate jump,
  not the verdict (`wave_summary.py` flags `ORPHAN?`); note the drop.
* Fanning out: check the fixed-container-name trap (`step3-raise.md`) first. Run
  with step 1's declared containers per GPU.

**The wave, on one node (`<NODES>` = 1): `wave.sh` with a control-only spec** -- the step-4
spec format (`step4-run.md`), every line arm `ctl` (no capture), one line per trial with its
GPU, the env the run script sets per trial and the model command (`$ND_REPO` / `$ND_ART`
paths). Its lines become step 4's control lines unchanged. Not an ad-hoc `exec.sh` loop: its
logs miss the `<gpu>_<i>.<arm>.` names `wave_summary.py` attributes trials by (2026-11-19: 8
logs, "trial/arm unknown").

```bash
# <ARTIFACT_DIR>/rate.spec:   0|ctl||python3 $ND_REPO/<model command> --seed 0   (... one per trial)
SPEC=<ARTIFACT_DIR>/rate.spec NAN_SIGNAL='<NAN_SIGNAL>' WAVE_DIR=<ARTIFACT_DIR>/wave_rate \
  nd submit --name rate --gpus 8 --exclusive --time <T> --step step2 -- wave.sh
```

`wave.sh` ends with `wave_summary.py`'s verdicts (also in `<ARTIFACT_DIR>/wave_summary.<tag>.txt`):
`<NAN_RATE>` is its `ctl rate n/m`. `wave.sh` runs trials only: the `<ENV>` check (step 1,
`gpu_check --torch`) is its own 1-minute job, submitted first. A run script that starts its own
containers: the spec runs the command *inside* them (step 1, *A run script that starts its own
containers*).

## A model that runs on more than one node (`<NODES>` > 1)

One trial is then a whole `<NODES>`-node run, so a wave of trials sharing one node does not
exist: `scripts/jobs/rate.sh` runs `TRIALS` uncaptured runs of the model, each ONE job
over `NODES=<NODES>` nodes, and scores them all with `wave_summary.py`.

**By default a smoke run: `TRIALS=1`, one group (`--nodes <NODES>`).** It answers only
"does it run": `RATE_RESULT=1/1` = it runs and NaN'd (a provisional rate, not one to size
the ladder from); `0/1` = it runs, no NaN in this run; `0/0 err=1` or `died=1` = *The model
does not run*, above. Either way `<NAN_RATE>` stays *unmeasured* and step 3 is skipped (`$LG
skip step3 "smoke run only"`). **A full rate (`TRIALS` >= 8, below) only when the user asks
for it.**

```bash
CMD='<the model once: torchrun --nnodes=$ND_NNODES --node_rank=$ND_NODE_RANK
     --master_addr=$ND_MASTER_ADDR --master_port=$MASTER_PORT ...>' NAN_SIGNAL='<NAN_SIGNAL>' \
  NODES=<NODES> TRIALS=8 nd submit --name rate --gpus 8 --nodes <NODES x G> --exclusive \
  --time <T> --step step2 -- rate.sh
```

* `CMD` is `<RUN_SCRIPT>`'s per-node command with the original entry point (no capture
  wrapper), written like a capture spec's `all` line (`step4-distributed.md`); each run gets
  its own `RUN_OUT` dir for dumps and checkpoints.
* **Size the allocation, G groups of `<NODES>`**: the groups run trials side by side, so
  `TRIALS` runs take ceil(`TRIALS` / G) rounds. G = 1 is cheapest to queue for, slowest to
  finish; `<T>` = rounds x one run's length + ~3 min start-up each. State the node-hours
  (`<NODES>` x G x `<T>`) in step 1's budget.
* Result: the job's `RATE_RESULT=<nan>/<scored> nodes_per_run=.. copies=.. runs=..` line
  (ledgered by the job; ERROR / DIED runs are excluded and counted), and
  `<OUT>/summary.txt` for per-run verdicts and NaN steps. It is `<NAN_RATE>` at `<NODES>`
  nodes -- what the node-count ladder's attempts per rung (`NODE_LADDER_TRIES`) are sized from.
* Validated 2026-11-12 on a Slurm + apptainer cluster: qwen3 MoE
  `run.sh` per-node command, `NODES=2 TRIALS=4` on 4 nodes = 2 groups side by side, 2 rounds,
  5 min; `RATE_RESULT=4/4` on the magnitude signal (`grad_norm` >= 100 at step 2 in every run --
  at 2 nodes this model explodes but did not reach NaN in 100 steps), err=0 died=0.
