# Step 4, run: the capture wave (always read)

`step4-capture.md` first only if there is no capture spec yet for this
`<RUN_SCRIPT>` (or its source, arming, filter or cap changed). A `capture.spec` next to
`<RUN_SCRIPT>` (or an older `*capture*.sh` launcher) that has run is the validated setup --
its header's `Validated <date>` line, or a ledger that names it (step 1's earlier-investigations
grep) with `CAPTURE_TO_REPRO_RESULT=REPRODUCED` or a passing wave line. It: state it, don't re-derive it.
`<PRIORS>`=none (step1-ask.md Q7) keeps the spec beside `<RUN_SCRIPT>` -- the user's setup --
and drops the earlier-investigation half: knobs from the spec header's submit line, not an
earlier artifact dir's `capture.submit`.
**Its knobs come verbatim from a file, never from ledger notes**: the earlier artifact dir's
`capture.submit` (the capture-to-reproducer job writes it: `SPEC`, `NAN_SIGNAL`, `CAPTURE_ENV`,
`CONTROLS`, ... as it ran them, plus its `CAPTURE_TO_REPRO_RESULT` and `SPEC_SHA256`, the sha of its spec's trial
lines: the job refuses another spec, and a `WAIT_ARMS` arm no spec line has) -- copy it with ITS spec, the one
beside it in that artifact dir (not the example's: 2026-11-23 the two differed by the C0/C1 lines), and
`set -a; ART=<ARTIFACT_DIR>; . <ARTIFACT_DIR>/capture.submit; set +a` before the submit -- else the
spec header's submit line. A ledger line abbreviates: rebuilt from one, `NAN_CAPTURE_SUSPEND=
FlexAttention.compiled_flex_attn` lost its module and every rank died at its first step
(2026-11-13). Neither file has the knobs: re-derive them (`step4-capture.md`), don't guess.
If its capture lines run a wired COPY at `$ND_ART/model/<file>` (a torchtitan spec running
`scripts/torchtitan_capture_train.py` does not), a fresh `<ARTIFACT_DIR>` does not have it yet.
Make it first:

```bash
mkdir -p <ARTIFACT_DIR>/model && cp <model.py> <ARTIFACT_DIR>/model/ &&
python3 <REPO_ROOT>/scripts/wire_capture.py apply <ARTIFACT_DIR>/model/<model.py basename> --backup-dir <ARTIFACT_DIR>
```

Controls run the original from `$ND_REPO`/the user's tree; step 9's `wire_capture.py restore`
then targets the copy. Every command goes through `nd` with `ND_ENV=<ENV>` (`env.md`).

**The capture spec** -- the wave, one trial per line (`scripts/jobs/wave.sh` header):

```
# <gpu>|<arm>|<env VAR=VALUE ...>|<command>   arm cap = capture armed, ctl = control
0|ctl|NUM_STEPS=100 MASTER_PORT=29500|python3 $ND_REPO/examples/<x>/train.py
2|cap|NUM_STEPS=100 MASTER_PORT=29502|python3 $ND_ART/model/train_capture.py
```

**Capture the user's configuration only.** Every capture and control line runs the arm the
user named (or `<RUN_SCRIPT>`'s default) -- not a variant of it (another backend, a debug
switch, an arm meant as a control), however cheap the spare GPUs make it. A variant's
capsules form groups of their own, at other ops, and muddy `groups_priority.txt` and the
report (2026-11-19: two `SGLANG_FORCE_FUSED_OP_BACKEND=torch` lines added a second group at a
downstream GEMM no replay could resolve). Spare GPUs take more capture lines, or the 5N
compiled track's C0/C1 (*Compiled NaN predicted*, below). Variants are step-7 experiments.

Translate `<RUN_SCRIPT>`'s trial loop into it: one line per trial with that trial's GPU,
the env the script sets per trial (step counts, a `MASTER_PORT` per trial when trials
share a GPU), the model command with `$ND_REPO` / `$ND_ART` paths; capture lines run the
wired copy of the source (`step4-capture.md`), controls the original. Runtime flags,
image and hand-set env are not in the spec -- they are `<ENV>`. `wave.sh` gives capture
lines `NAN_CAPTURE_DIR`, `NAN_CAPTURE_CONTAINERS_PER_GPU` (trials on that GPU, so
Declare-contention is automatic for a spec), `NAN_PROGRESS_EVERY`; `CAPTURE_ENV` adds
the rest (`NAN_CAPTURE_DATA=1`, `..._ARM_AFTER_STEPS=1`, `..._SUSPEND=...`); `LOCATE=1`
gives the first capture line the loop locator. Save it as `<ARTIFACT_DIR>/capture.spec`;
templates: `examples/*/*/capture.spec`.

## 4.4 Run with a control arm

* Confirm `NAN_CAPTURE_CONTAINERS_PER_GPU` is set in the script about to run (a
  step-3 file may lack it; it cannot be added to a capsule later).
* **Timing-sensitive fault: no-capture control arm in the same wave**, interleaved
  (e.g. 12 capture + 4 control); the hook is not neutral on a race:
  * capture 0/12 vs control 4/4 -> hook suppresses: cut overhead (filter, cheaper
    check -- extra GPU work perturbs) or raise the rate (step 3);
  * capture ~6/12 vs control 4/4 -> hook taxes; fine.
  * Prefer controls paired with controls (a slow capture partner de-stresses).
    **Pair controls by GPU in the spec**: the lines sharing GPU g run together, so put
    the controls' lines on the same GPUs (e.g. both trials of GPU 0 and of GPU 1) --
    controls spread one per GPU sit beside capture trials. (An old launcher's
    `CONTROL_TRIALS`: with `gpu = trial % NGPU` the pairs are `0 NGPU 1 NGPU+1`.) Ledger it.
  * `<NAN_RATE>` *unmeasured*: controls that exited before the stop give "at least
    k of m"; ledger it with this wave's environment, or keep the rate *unmeasured*.
* **Stop every trial, controls included, at the first settled capsule** -- one
  capsule is enough for step 5, and every later trial only delays the replay on the
  same GPUs. Ledger the stopped trials as `STOPPED`, never as clean. Run on (or
  `WAIT_CONTROLS=1`) only when the user asks for a control rate or a second capsule.
* **A wave on its own** (no replay -- the 5N C0/C1 wave, step 8's single-process A/B):
  `SPEC=<file> NAN_SIGNAL='<NAN_SIGNAL>' nd submit --name wave --gpus 8 --exclusive -- wave.sh`;
  same early stop; stop it early with `nd stop wave` -- never a pattern `pkill` (another
  session's trials share the node, and auto mode refuses it); `nd stop` terminates the
  job's process group and every descendant group (torchrun's `setsid` ones).
  A sequential-attempt launcher (`attempt-NN/run.log`, `attempts.tsv`) is read by
  `wave_summary.py <wave dir>` directly; `capture_digest.py` on a log with no
  `[op-nan]` lines follows the run dir the log names (harnesses that log to their
  own facts files).
* Read the wave with `wave_summary.py --capsules <CAPSULE_DIR>` (SKILL.md rule 9).

**Run 4 -> 5a -> 5b(G1) -> step-6 verification as one job** (default, every runtime):

```bash
$LG start step4
SPEC=<ARTIFACT_DIR>/capture.spec NAN_SIGNAL='<NAN_SIGNAL>' CAPTURE_ENV='<NAN_CAPTURE_* knobs>' \
  nd submit --name c2r --gpus 8 --exclusive --time <T> --step step4 -- capture_to_repro.sh
```

**A failure known to hit every run, on a site that allocates GPUs per job** (`SLURM_GPUS_FLAG`
set in `<ENV>`): the user said every run fails at the same step, or a smoke / rate run was n/n with
n >= 2. Then 4 lines are enough -- 2 `cap` + 2 `ctl` (the folded rate keeps its 2 controls) -- on
`nd submit --name c2r --gpus 4` without `--exclusive`; the ladder and replay start at concurrency 1
anyway. Anything less certain, or a site where nodes come whole (`SLURM_GPUS_FLAG=` empty: the 8
GPUs are held either way, and a deterministic wave stops early on its own), keeps the full wave.
Ledger which (`wave: 4-line, deterministic per <evidence>`). Untested on a per-GPU site as of
2026-11-25 (the test site allocates whole nodes).

**Step 2 folded** (`<NODES>` = 1, the default: `step2-rate.md` *Folded*): add `WAIT_CONTROLS=1` and
keep >= 2 control lines -- the control arm is the step-2 rate. World size > 1: `CONTROLS=none`, so no
rate (`step2-rate.md` *Folded*, last item). The job's `rates` ledger line holds it.

**Compiled NaN predicted** -- the preflight's `VLLM` line says *default mode: torch.compile + HIP
graphs*, or `LAZY` shows a `torch.compile` over most of the step: the capture will likely end
`PROPAGATED` at the first eager op (`step5n-compiled.md`). Put the compiled track's **C0 and C1
lines in this same wave** instead of two of the plain controls -- they cost the same GPUs, and C1
then names the kernel, and its kernel capsule is saved, before the capture job ends (2026-11-19: a
separate C0/C1 wave after it was ~4 of 18 minutes):

```
# 4 capture + 2 control + C0 + C1 (8 GPUs); WAIT_CONTROLS=1 WAIT_ARMS='ctl c0 c1'
6|c0|<the model's compile-off switch, e.g. nothing here and --enforce-eager 1 in the command>|<model command, original>
7|c1|TORCHINDUCTOR_NAN_ASSERTS=1 TORCH_LOGS=output_code TORCH_LOGS_OUT=$ND_ART/output_code.c1.log PYTHONPATH=$ND_REPO/scripts/inductor_capsule INDUCTOR_CAPSULE_DIR=$ND_ART/capsules_inductor|<C1 command: vLLM's variant copy with HIP graphs off, step5n-compiled.md *vLLM*>
```

`WAIT_ARMS='ctl c0 c1'` makes the early stop wait for them (C1 ends at its assert, C0 at the NaN).

**vLLM with graphs (its default): capture lines run graphs off, plus a graph-check line.** The
capture watches compiled kernels (`step5n-compiled.md`), but a replayed HIP graph launches them
without Python. So the capture lines turn graph replay off and keep `torch.compile` -- no model edit
-- and one graphs-on line checks every graph replay for a NaN only graphs make (*Graph-only NaN*,
`step5n-compiled.md`); 3 capture + 2 control + `g` + C0 + C1,
`WAIT_ARMS='ctl g c0 c1'`:

```
0|cap|PYTHONPATH=$ND_REPO/scripts/vllm_graphs ND_VLLM_CUDAGRAPH_MODE=NONE|<model command, wired copy>
5|g|PYTHONPATH=$ND_REPO/scripts/vllm_graphs ND_GRAPH_CHECK=1 ND_GRAPH_CHECK_DIR=$ND_ART/capsules_graph|<model command, original>
```

The controls stay as the user runs the model (graphs on): they are the rate. 2026-11-22 (a served Qwen 0.5B):
graphs-off capture lines `REPRODUCED G1` at a fused Inductor kernel (C1's kernel) where graphs-on capture
lines had ended `NO_REPLAYABLE_GROUP`. **Capture lines clean
while the controls NaN** = the NaN needs graphs: read the `g` line's `GRAPH_CHECK` (below) and go to
5N's graph-only route, not to `step5-pressure.md`. TP > 1 (`scripts/vllm_capture` on the capture
lines): put both on `PYTHONPATH`, `vllm_graphs` first. SGLang: `scripts/sglang_graphs` -- capture lines
`ND_SGLANG_DISABLE_CUDA_GRAPH=1` (graphs off, no edit), one `g` line `ND_GRAPH_CHECK=1` (`GRAPH_ONLY` /
`NOT_GRAPH_ONLY`); `step1-ask.md` *SGLang*.
**Not when one trial takes several GPUs** (vLLM TP > 1: a GPU-list field, `0,1|...`): 8 GPUs are
only 8/TP trial slots, and the two controls the folded rate needs come first. There, vLLM's own
custom ops -- its all-reduce, attention, GEMMs -- still run as ops in compiled mode and are captured,
so a NaN made by one of them needs no C0/C1; run the compiled track as its own wave only if the
capture does end `PROPAGATED` at a compiled boundary. Slots, TP 2: 2 capture + 2 control; TP 4:
1 + 1; TP 8: one trial, no control (the rate from step 8's stock arm, `step2-rate.md`).
`PYTHONPATH` in a spec line is prepended to nd's own. Read them in 5N (`step5n-compiled.md`:
*already in the capture wave*), don't re-run them.

**Node**: no `--node` / `--prefer-node` needed -- `nd submit` pins an idle node that already holds the
image (docker / podman: `IMAGE_NODE ... pinned`), and falls back to any node if it is taken before
the job starts (`PINNED ... resubmitted unpinned`). A fresh node pulls the image first (~2 min for
20 GB).

**`<NODES>` > 1** (a multi-node run): this job is submitted with `--nodes <NODES>` and first runs
the node-count ladder -- read `step4-distributed.md` (*The node-count ladder*) before writing the
spec.

`<T>`: the wave's run length (the spec's steps x the step time from step 1/2, or the run
script's own walltime) + 30 min for 5a, the ladder and packaging; 02:30:00 when unknown -- the
early stop ends the job long before (a reproducing qwen-tf32 job: ~2 min).

**With a TRIGGER** (step 1 ledgered a load note): add the co-tenant to this first
submit, so G1's 5b and package run beside the load instead of ending `NOT_SET_UP`
solo -- `CAPSULE_COTENANT='python3 -m transformer_hog --mode full
--procs <capture's processes per GPU - 1> <the model's sizes>'` -- the sizes from the model
source's config as `transformer_hog.py`'s flags `--hidden --inter --heads --kv-heads --head-dim
--layers --vocab --seq --batch --qkv-bias --tf32 --dtype` (`--head-dim` = hidden / heads when the
config has none; seq and batch from the run script, not the config:
`find_capture_points.py` or a grep of the config, not the hog's defaults) (a decoder
transformer; `step5-pressure.md` 5d for the sizes and other models) and
`CAPSULE_MODE=op`; a shapes-only capsule also needs `CAPSULE_EXTRA_ARGS=--synthesize`.
**A co-tenant replaces the copies**: with `CAPSULE_COTENANT` set the ladder runs ONE copy per
GPU beside the load (`concurrency 1` in its result is right, not a bug) -- the load is the
contention. Ledger that the solo case was not run. The 2026-11-06 qwen run (TRIGGER ledgered,
solo 5b first): solo `NOT_SET_UP`, then a 5d sweep and a repackage before the
co-tenant reproducer -- ~25 min that this saves.
Preflight, wave with **early stop** (>= 1 settled capsule; controls are stopped too),
5a, package the top non-false-positive group, 5b from the extracted copy (early stop
at 10 hits), kernel id, README + tar + sha256 -- ledger notes and timing written by
the job, **including `end step4`** and its own `step5a` / `step5b[G]` / `step6[G]` visits, and
`<ARTIFACT_DIR>/capture.submit` (its knobs verbatim, for the next reuse), and
`<ARTIFACT_DIR>/capsule_info.<G>.txt` -- `capsule_info.py` of the top group, replayable or not (read it
instead of submitting `capsule_info.py` yourself):
don't end them again (a second `end` counts the step twice). Your reading of its result is
`$LG start step5` AFTER the job ended; before any further job of your own, `$LG end step5`
and start that job's step (`step5b[G]`, ...) -- `$LG table` warns when visits overlap. A later observation about step 4 (the control arm, a
log you read afterwards) is noted under the step you are in (`$LG note step5 'step4 ctl: ...'`):
a note for a closed step is written without a warning (it has no time to lose), but it sits
out of order in the ledger -- prefer the open step. It also writes `<ARTIFACT_DIR>/next_groups.sh`: one GROUP-only submit line
per further replayable group (same script, `GROUP=G<i>`, skips wave and 5a).
Wait for it once (SKILL.md rule 9), read its last line `CAPTURE_TO_REPRO_RESULT=...` and the
files it names, then go to `step5-read.md`. Knobs (header of `scripts/jobs/capture_to_repro.sh`
and `wave.sh`): `SPEC` (or an old launcher as `WAVE_SCRIPT` + `WAVE_LOGS`) and `NAN_SIGNAL` are
required; the image is `<ENV>`'s. The early stop waits for
nothing else by default; with `WAIT_CONTROLS=1` it also waits for the control logs it
can see (no control arm: `CONTROLS=none`, else it runs to the end). `EARLY_STOP=0` for an exact capture-arm rate (else "at least k of m"; killed trials
show `STOPPED`); `REPLAY=0` = wave only; an old launcher not shaped like the examples
(`<gpu>_<trial>.<cap|ctl>.<jobid>.<host>.log` with `TRIAL_RC=`, honouring
`CAPSULE_DIR`): set `WAVE_LOGS` / `CONTROL_LOGS` or `EARLY_STOP_CMD`. A rare race (a
per-launch rate far below 1 / the capsule's op index, E40): `CAPSULE_FUTILITY=0`, or the
ladder ends `NOT_SET_UP` long before its first hit; the package's `run.sh` keeps the value.

## 4.5 Judge the result

* **From `<CAPSULE_DIR>`, not exit codes or the log.** At world size 1 the capture
  raises in place (a `__torch_dispatch__` traceback is success -- don't "fix" it);
  an `except Exception` loop swallows it, exits 0, and may print the model's own
  `SUCCESS` (the message contains "nan"). World size > 1: `sync_stop()` tears all
  ranks down the step after.
* **Why a capsule (or none):** `capture_digest.py <run log or run dir>` (SKILL.md
  rule 9) -- the captured op, dirty operands, chain, `settle` and the census in
  ~40 lines; a clean run shows only the MASKED sites.
* **"No NaN" does not mean the trial ran**: the same handler swallows a step-0 OOM.
  Check `Error at step` first. No capsule + "no NaN" = no NaN, hook suppression, or
  died at once -- only the log separates them.
* **Capsule save errors** (`wave_summary.py` alert `capsule save error`, a
  `cannot pickle` traceback): the trial caught a NaN but `torch.save` failed. Since
  2026-11-11 the capsule is saved anyway: offending leaves (a module argument, a tensor
  carrying a module attribute, a bad dict key) and, as a last resort, the deepest objects
  `torch.save` itself refuses are replaced by markers and listed in the capsule's
  `unpicklable` field; the run log prints `replaced N leaf(s)` / `still not picklable ...
  replaced N object(s)` with each path. A replay of an op (or chain node) whose argument
  was replaced can fail: say so, and report the path -- it is a capture bug to fix. A
  `torch.memory_format` argument is no longer one (recorded as a tag). A capsule from an
  older build was lost (only a `.pt.tmp` left, the trial `TRIAL_RC=1`): ledger the lost site
  (op, file:line from its `pending:` line) as `not captured`, re-capture on the current
  build. Lost this way: 5 of 6 capsules of the 2026-11-06 qwen run, 6 of 6 of the
  2026-11-11 qwen-tf32 run (a `clone(memory_format=...)` in the chain).
* **Reproducer found:** tell the user where `<CAPSULE_DIR>` is and to save it.
* **No capsule:** `step5-read.md` *Empty `<CAPSULE_DIR>`* -- after its fixes, the
  fallback is step 5N (`step5n-native.md`), not the end of the ladder.
