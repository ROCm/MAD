# Step 1: ask for the run script (required), take the defaults, declare contention

Never skipped; runs nothing but the `<ENV>` check.

| ask for | notes |
|---|---|
| `<RUN_SCRIPT>` | the script that runs the model |
| `<NAN_SIGNAL>` | how its output says a NaN happened (`examples/training/softmoe_train_nan/run.sh` prints `Loss is not finite`; the user's may differ). A **magnitude** counts too -- `grad_norm: +([1-9][0-9]{2,}\|nan\|inf)` for exploding gradients: `step4-explode.md`. So does a **wrong finite value** with no NaN at all -- the model's own parity or determinism check (`RESULT ... wrong=yes`): take that line, `step4-explode.md`. **No signal at all** -- the model prints nothing when wrong, the user suspects silently wrong outputs or asks to validate a configuration: differential mode makes one (`step2-diff.md`) |
| `<WORLD_SIZE>` | **derive, don't ask**: ranks of ONE job (`torchrun --nproc_per_node` x nodes, `NGPU`, the parallelism degrees). > 1 -> `step4-distributed.md` (wiring through the framework's step, manifest, origin rank). Independent trials sharing GPUs are co-tenants, world size 1 each |
| `<NODES>` | **derive, don't ask**: nodes of ONE job (`sbatch -N`/`--nodes`, `torchrun --nnodes`, a node-count argument; the DP-replicate degree it sets). > 1 -> the node-count ladder (`step4-distributed.md`): also derive `NODE_LADDER_MIN` (the fewest nodes the other parallelism degrees fit in) and state both |
| `<ENV>` | **derive, don't ask**: `<ARTIFACT_DIR>/env.sh`, written by `nd profile <site> --art <ARTIFACT_DIR> IMAGE=...` from the site profile plus what `<RUN_SCRIPT>` says (runtime, image or SIF or venv, its container flags and hand-set env) -- `env.md`. Every GPU command of steps 4-8 uses it as `ND_ENV` |
| image/build | **don't ask**: the one `<RUN_SCRIPT>` uses (its default, or what the user's env override sets) |

**Verify `<ENV>` once:** `nd run --gpus all -- 'python3 -m gpu_check --torch'` (on a
GPU node; from a login node `nd submit --name check --gpus 1 -- exec.sh -- '...'`) prints torch's
view of device 0 (name, arch, CUs, memory) and a last line `GPU=... ARCH=... VERDICT=...`
(`TORCH_GPU=NO` = `<ENV>` is broken -- image, runtime flags: fix it first).
**Ledger the device line** (gfx942 = MI300X, gfx950 = MI350X/MI355X), CUs and memory:
rates, memory caps and reproducers are bound to it (SKILL.md *GPUs*). `VERDICT=UNTESTED`
here, not yet warned in step 0: print the untested-GPU warning (`step0-gates.md`) now,
before any wave, and continue.

**Injected vs stock: don't ask.** It matters only for reading step 7 (on an
injected image the root cause is the planted NaN, confirming the tooling, not a
bug), and the 5b `kernel.*.txt` or a `--list-mutations` enumeration tells:
`nans_to_nops` matches a site = injected, `PatchError` = stock. Neither: report it
as unknown.

**Ask only question 1; take the default for 2-7** -- users found the questions and
the reading too much (a run where the user answered `default` five times in a row cost
five round trips for nothing), and a question later in the ladder stalls for as long
as the user is away. Questions 2-7 below are *settings*: decide each from its default,
ledger it as `(default)`, and don't print them. After question 1, tell the user one
line: what was derived (signal, image, nodes, data mode) and "using defaults -- say
*ask me the options* to change them". Still ask only what has no safe default: a
`<NAN_SIGNAL>` the script does not show, the PyTorch gate's `ASK`, the compute check
(question 6, above its threshold). Never ask what you can read from `<RUN_SCRIPT>` --
state the derived value and let the user correct it.

**Ask-me mode** -- the user asks to choose ("ask me the options", "let me choose",
"ask me the questions"), before or during step 1: show questions 2-7 as one compact
block with their computed defaults (one line each), "reply `default` to accept all, or
name the ones to change" -- then ask only the ones the user names, one at a time, and
ledger `ask-me mode`. It also restores the step-7 budget question and the sweep-plan
stop. In order:

1. **`<RUN_SCRIPT>`** -- no default; asked as "How do I run the model?", so the answer may
   be a script, a directory or a command line. **A command line** (`python train.py
   --x`, `torchrun ...`, `sbatch run.sh`): the script it runs is `<RUN_SCRIPT>`, and its
   arguments and env are the run as the user gave it -- write them into
   `<ARTIFACT_DIR>/run.sh` once the dir exists, and ledger the original command
   verbatim. **A directory, not a script:** pick the plain run
   script -- the one that runs the model as-is, not a `*capture*` /
   `*timing*` / `*reproducer*` variant (those are derived launchers; a
   `*capture*` one is step 4's validated setup, note it) -- state the pick in one line
   and go on; ask only if two plain ones remain. **Then the PyTorch gate** (SKILL.md
   step 0, `framework_check.py`) before anything else: `SKIP` = print its warning
   (`step0-gates.md`) and stop -- no artifact dir, no ledger; `ASK` = its one question first. `RUN`: go on,
   and ledger its last line once the ledger exists. **Now create `<ARTIFACT_DIR>`**
   (SKILL.md rule 3; `<RUN_TAG>` = today's date + the model/run name from the
   script) -- **`ls` it first**: the tag is only date + name, so an earlier session's
   investigation of the same script today already owns it. Its ledger has jobs or
   capsules: suffix the tag with the next free number (`-2`, then `-3`, ...). Only step-1 lines: tell the user and resume it
   (ledger a `resumed` note). **`<PRIORS>` = none: do not open it** (another investigation's
   ledger is exactly what none excludes) -- take the next free suffix whatever it holds. Then start the ledger: `$LG start step1`, `$LG note step1` with `<REPO_ROOT>`
   as step 0 printed it (the physical path) and, if `pwd` shows another spelling (a symlink:
   `/home/...` -> `/lustre/...`), that one too -- tools print either and `<ARTIFACT_DIR>`. **Look for earlier investigations
   of the same script, any date** (skip with `<PRIORS>` = none, question 7), in the same call: `grep -l '<RUN_SCRIPT basename>'
   <REPO_ROOT>/artifacts/*/ledger.md`. A hit with a capture launcher, a passing wave
   or a tarball is a validated setup and maybe a finished reproducer: ledger it, and
   before step 4 offer the user a fresh capture with that launcher (it confirms the
   old result still holds) or reuse of the old reproducer -- **default: fresh capture**.
   Several hits: the one that counts is the newest with a verified tarball, else the
   newest with a passing wave -- decide by grep, not by reading: `grep -l -E
   'verified-from-extracted-copy|NATIVE_TOTAL=[1-9]' <dirs>/ledger.md` = a verified tarball;
   `CAPTURE_TO_REPRO_RESULT=(NO_CAPSULE|PREFLIGHT_FAILED)` = the wave did not pass; a ledger
   with both (a failed wave, then a native or later verified tarball) counts as verified.
   "Newest" = the latest timestamp in the ledger (`tail -1`), not the directory name or mtime;
   its launcher: `<dir>/capture.submit` (the knobs verbatim, `step4-run.md`), else `grep -m3 -E
   'capture.spec|SPEC=|WAVE_SCRIPT=' <dir>/ledger.md`. Ledger the
   others as one line (`earlier: <dirs>`). Never carry its rates over (rule 5).
   **Default mode: don't ask** -- take the fresh capture with that launcher and say so in
   the one-line summary ("reusing the capture setup from `<dir>`"); offer the old
   reproducer only in ask-me mode or if the user asks for it. **A reused
   setup brings its `<NAN_SIGNAL>`** (its `capture.submit`, or the spec header's submit line):
   take it, don't re-derive it -- a README-derived regex differed from the validated one
   (2026-11-13: grad_norm `{3,}` vs `{2,}`). Then read the script -- once, comments and blank
   lines stripped (`grep -vE '^\s*(#|$)' <RUN_SCRIPT>`); its README, Dockerfile and the model
   source only by `grep` for what is still missing (the preflight and the trigger grep below
   cover most of it) -- and derive `<NAN_SIGNAL>` (when nothing is reused),
   `<ENV>` (`nd profile`), the image, the trial geometry and **the data source** (public or
   test assets -- e.g. torchtitan's `c4_test` --, synthetic (random tokens / random init: as
   public), or the user's own data: it sets the
   capsule data mode, question 3); report them in one line, don't ask. Ask for
   `<NAN_SIGNAL>` only if the script doesn't show it (preflight `SIGNAL ?  the run
   script greps no NaN line` = its greps only format output; take one of the `the
   source raises/logs:` candidates the preflight prints) -- and not even then when the user
   describes no visible failure at all (silently wrong outputs): `step2-diff.md` makes the signal. Then, in ONE
   call, `python3 <REPO_ROOT>/scripts/capture_preflight.py <RUN_SCRIPT>
   --source <model.py>` (with `ND_ENV` set: the IMAGE lines depend on the runtime): `IMAGE ...
   MISSING` (apptainer) = the SIF is not there, `IMAGE ... BUILDS its image` = build it once --
   either way get it now (`env-runtime.md` *Images*; a ~35 min pull or a build is the user's call), before
   any job -- **don't wait for an answer**: tell the user in one line it is being pulled
   or built and how long it takes, then start it (ask-me mode: ask first);
   docker/podman `nothing to do now` = nd pulls or loads it per node; `SIGNAL ... ? not in the source as a regex` = the source prints it from data (a `RESULT
   ... nan_step=` line): accept it, the step-4 wave confirms it (no model log exists yet when
   step 2 is skipped); `SIGNAL 'RESULT .*wrong=yes' ... verdict field` = the script reads a
   result line's verdict field: a valid wrong-value signal (`step4-explode.md`) -- but when the
   next line says `the source also raises/logs: '<msg>' ... prefer it`, the NaN is printed mid-run
   too: take that line as `<NAN_SIGNAL>`; each candidate's `[arm ...]` tag says which `run.sh` arm
   it tests, and a grep of an arm that does not run `--source` (an op-level repro arm) is listed as
   `skipped`, not used (2026-11-24); a build file beside the script that preflight marks `ignored` is not
   this script's image -- its SIF/IMAGE line settles it; `SIGNAL ... NOT RAISED ANYWHERE` means the script cannot
   see the NaN it greps for (raise it now, with the user -- not after a wave). The
   default fix, when the model hides NaNs (`nan_to_num` on the loss): one line after
   `backward()` in a working copy, `after_backward(model, loss, parts)` from
   `scripts/grad_trajectory.py`. It raises `NonFiniteDetected("non-finite
   GRADIENT at step N after backward: <modules>")` (add `NonFiniteDetected` to
   `<NAN_SIGNAL>`), and with `NUMERIC_DOCTOR_GRADLOG` set it also logs the per-step gradient
   trajectory (step 8). Back the source up first, and restore it in step 9;
   `MASK` on a loss, `HOOK`, `ENV`, `LAZY` and `LOOP` settle step 4's wiring
   questions without further greps (`LOOP` is a static guess: the step-4 smoke
   test's locator confirms it); `VLLM` lines: *vLLM*, below; `SGLANG` lines: *SGLang*, below. A `SIGNAL ... MATCHES CLEAN
   OUTPUT` pattern is not a NaN signal (it matches the field's clean value, `nan_request=-`):
   take the source's not-finite line it lists instead.
   **Read the trigger notes too, in the same call**: `grep -niE
   'pressure|HBM|co-?tenan|share|race|stress|trials per GPU|control'` over the
   `<RUN_SCRIPT>` directory's `*.md` plus any non-`.md` `README*` (`README.md` matches both
   globs: name each file once) and the model source's docstring. A
   load trigger ("needs HBM pressure", "1 trial per GPU stays clean") means replay
   copies alone will likely not reproduce: ledger `TRIGGER: <quote, file:line>` and
   give the **first** capture-to-reproducer job its co-tenant (`step4-run.md`
   *With a TRIGGER*) instead of waiting for a solo 5b to come back `NOT_SET_UP` (a
   trainer's own note of the load it needs, read late, has cost ~2 h of clean
   replays; read on time but acted on after the solo 5b, a sweep and a
   repackage).
2. **Step 2** -- default **run** (`step2-rate.md`): it confirms the model runs by
   itself in this environment, and measures the rate. `<NODES>` = 1: **folded into the
   step-4 capture-to-reproducer job** -- its control arm is the run and the rate, and the
   preflight already checked the setup (`step2-rate.md` *Folded*); a separate step-2 wave
   first only when the user asks for one, or wants step 3 (a rate to raise). Skip it only when
   the user asks to (skipped means *unmeasured* and step 3 skipped, never a carried-over
   rate). `<NODES>` > 1: by default one
   **smoke run** (`rate.sh TRIALS=1`, one run of `<NODES>` nodes) -- a full rate (>= 8
   runs of `<NODES>` nodes: it sizes the node-count ladder's attempts per rung,
   `NODE_LADDER_TRIES`) only when the user asks for it; without it each rung gets 2, and
   a clean smaller rung may be bad luck. In ask-me mode word it as: "Run the model once to
   make sure that it runs in this environment. You CAN skip this step and go directly to
   the NaN capturing step (step 4)." -- with the smoke-run / full-rate node-hours for a
   multi-node model; don't quote a trial count or earlier runs' rates.
3. **Capture-wave geometry** (trials, trials per GPU, which are controls) --
   default: the launcher's own geometry, controls per `step4-run.md`; the wave
   stops (controls too) at the first settled capsule -- say so in the question.
   **Same question, the capsule data mode**: real values (`NAN_CAPTURE_DATA=1`)
   by default when the data is public/test assets, shapes only otherwise
   (`step4-capture.md` 4.1 item 1) -- state which and why, so the user can
   override it now rather than after a clean synthetic replay.
4. **Root-causing (step 7) and fixing (step 8) wanted?** (yes / ask me after 6 / no)
   -- default: **yes**. Both are loops over the step-6 reproducer with a wall-clock
   budget; by default step 7 takes the default budgets for both without asking and the
   step-6 hand-over announces them (`step7-rootcause.md` *Entry*); fixing needs a root
   cause. The user turns them off by saying so at any point ("stop at the reproducer",
   "no root-causing"): ledger it and go from step 6 to step 9. In ask-me mode, say that
   step 6 ends with a reproducer, not a fix, and that steps 7-8 are experimental in this
   version.
5. **`<GROUPS_TO_REPLAY>`: if the capture yields several capsule groups, replay only
   the largest, or all?** (largest / all) -- default: largest, i.e. the replayable
   group with the most capsules (`G1` in `groups_priority.txt`; the capture-to-reproducer job replays
   it anyway). `all` adds one 5b+6 job per further group (`next_groups.sh`).
   **The first time "largest" reaches the user** (this question, or its line in the
   ask-me block, or the one-line summary), define it: the group holding the most capsules -- size is the
   capsule count, not tensor size or memory. World size > 1 (`--job` grouping): G1 is
   the **origin** group (the rank and op where the NaN was born, `manifest.json`), even
   when a propagated group holds more capsules; `groups_priority.txt` is the order.
6. **Allocation budget for the whole run** -- the one setting that may still stop for
   the user (CLAUDE.md confirmation). Compute the estimate; **at most 8 node-hours and
   `<NODES>` = 1: don't ask** -- ledger it as the budget and say it in the one-line
   summary. **Above 8 node-hours, or `<NODES>` > 1: one line, "this will use about N
   node-hours (<what>), OK?"**, nothing more. The estimate counts only the jobs the
   settings imply (the step-2 wave or smoke run unless skipped; the capture-to-reproducer job sized
   from the geometry and the script's walltime; with `all`, 1 node x 2 h per extra
   group; plus a reserve for the step-5N fallback -- default ~6 one-node 30-min
   candidate jobs and one locating wave -- spent only if no capsule reproduces;
   name it when you ask). `<NODES>` > 1: the capture-to-reproducer job is `<NODES>` nodes for the
   node-count ladder's walltime (`step4-distributed.md`), in node-hours.
   Within it, submit waves, the capture-to-reproducer job and per-group 5b jobs without
   re-confirming (this is the CLAUDE.md confirmation); ask again only beyond it. The
   step-7/8 default budgets and a recipe sweep's plan are checked against the same
   8-node-hour threshold where they start (`step7-rootcause.md` *Entry*,
   `step7-recipe-sweep.md`).
7. **`<PRIORS>`: use earlier investigations of this script?** (use / none) -- default
   **use**; **none** when the user asks for it ("fresh eyes", "no priors", "from scratch")
   or `NUMERIC_DOCTOR_NO_PRIORS=1` is set (a validation run of the skill itself). `use`: the
   greps below (an earlier capture launcher, step 5N's and step 7's *prior work*) order the
   work -- a prior root cause is re-tested first. `none`: skip all three greps and read
   nothing under other artifact dirs; build the 5N candidates and the step-7 hypotheses from
   this run's evidence only. What `none` does NOT exclude: what ships beside `<RUN_SCRIPT>` --
   its README (the trigger grep below reads it in both modes: a user's notes are evidence)
   and a `capture.spec` there, which is the user's validated setup (`step4-run.md`: reuse it;
   its header holds setup, not results) -- only earlier investigations' artifact dirs
   (their `capture.submit`, ledgers, reports) -- and never `validation/expected/`, the
   examples' answers, kept for grading runs (`compare_results.py`). No spec beside the script: build it
   (`step4-capture.md`). A validation run that must be blind runs on a copy of the script
   without its notes; with an example's README read, say in the report that the run was
   not blind (2026-11-20: four examples' READMEs or docstrings name their cause). Why it exists: with
   priors every step 7 of 2026-11-20 found its root cause in iteration 1 by re-testing an
   earlier `ROOT_CAUSE.md` -- right for a user, but a validation run then measures
   confirmation, not discovery, and inherits stale setup (a co-tenant nobody re-checked, a
   moved example path). Ledger `PRIORS=use|none`; the report says which (`step9-report.md`).

**Verify `<ENV>` without a separate job when the next job runs it anyway**: the
capture-to-reproducer job's preflight runs the same `gpu_check --torch` before the wave's setup
is spent -- whether or not step 0's GPU gate said `NO_GPU`. Run the standalone check
(`gpu_check.py --torch`, its own ~1 min job, right after the step-1 questions) only if the
first GPU job is something else: a step-2 wave, or a capture that cannot be submitted yet
(an image still to build, a spec still to write) -- so an untested-GPU warning comes before
work is spent. Submitted beside the capture-to-reproducer job it only waits in the queue
(2026-11-13: still pending after the capture job had finished, then `launch failed requeued
held`, which `nd stop` cannot cancel).

**Ledger:** each answer (and whether it was the default) -- `RUN_SCRIPT=<path>` and `NAN_SIGNAL='<regex>'`, quoted
(`result_card.py` reads them from these notes) -- the derived inputs and image, the `nd profile` line, the `<ENV>` check,
containers per GPU and how derived.

## A run script that starts its own containers

The run script `docker run`s / `podman run`s / `apptainer exec`s the model itself (every
example's `run.sh`; `run.sh <arm>` picks one configuration). Neither step 2 nor a capture spec can
run it as given -- `nd` would start a container inside a container. **Unwrap it, don't ask:**

* the **image** goes to the profile: the one it runs, or -- when it builds a `Dockerfile` that only
  sets `ENV` and copies files (preflight: `no build needed`) -- that Dockerfile's base;
* docker / podman **runtime flags nd already sets**: `--device=/dev/kfd /dev/dri`, `--group-add`,
  `--user`, `--security-opt seccomp=unconfined`, `--ipc=host`, `--network=host` and `--shm-size`
  (`DOCKER_SHM`, default 64g: a run script's smaller value is covered) -- drop them; any other
  flag goes to `DOCKER_ARGS` / `PODMAN_ARGS`;
* its **container flags** (`apptainer exec --cleanenv`, ...) to `APPTAINER_FLAGS`, and every
  `-e K=V` / `--env K=V` / Dockerfile `ENV` to `EXTRA_ENV`, except what `nd` sets itself
  (`USER`, `LOGNAME`, `HOME`, `HIP_VISIBLE_DEVICES`, `TORCHINDUCTOR_CACHE_DIR`);
* the **command** it runs inside the container, with the chosen arm's arguments (its arms table
  row for `run.sh <arm>`), is the model command of step 2's spec and step 4's lines, written with
  `$ND_REPO` paths.

The image's own `ENTRYPOINT` is handled by `nd`: a program entrypoint (vLLM's `["vllm", "serve"]`)
is cleared (`DOCKER_ENTRYPOINT=auto`), a shell wrapper kept. Ledger the unwrapped command beside
the user's (e.g. `run.sh` -> `python3 $ND_REPO/examples/inference/qwen_vllm_serve_nan/qwen_vllm_serve.py --seed 0 ...`).

## vLLM

The capture preflight's `VLLM` lines, read before step 2:

* **In-process engine, or nothing is captured.** vLLM V1 runs the engine core -- and the model --
  in a subprocess unless `VLLM_ENABLE_V1_MULTIPROCESSING=0`; a capture wired into the serving
  loop then sees no model op and the wave reads `NAN_NO_CAPSULE`. Not set in the source or run
  script: add it to `EXTRA_ENV` (`nd profile ... EXTRA_ENV="VLLM_ENABLE_V1_MULTIPROCESSING=0"`).
* **Tensor parallel > 1 on one node** puts the model in one worker process per rank: the capture
  goes INTO the workers. Capture lines get `PYTHONPATH=$ND_REPO/scripts/vllm_capture` in their env
  field and run the **original** script (no wired copy, nothing to restore in step 9); the GPU field
  is the run's GPU list -- one trial over those GPUs (`wave.sh`):

  ```
  0,1|cap|PYTHONPATH=$ND_REPO/scripts/vllm_capture|<the run command, --tp 2>
  2,3|cap|PYTHONPATH=$ND_REPO/scripts/vllm_capture|<the same>
  4,5|ctl||<the same>
  6,7|ctl||<the same>
  ```

  (2 capture + 2 control: the folded rate's minimum; no C0/C1 lines -- `step4-run.md` *Compiled NaN
  predicted*, the multi-GPU exception.)

  `CAPTURE_ENV='NAN_CAPTURE_DATA=1 NAN_CAPTURE_ARM_AFTER_STEPS=1'` (values: a collective is judged on
  them). The logs show `[vllm-capture] capture built in the worker ... tp 2` once per rank, and `capture
  suspended in N vLLM warm-up / graph-capture method(s)`: vLLM's dummy forwards and HIP-graph capture can run after
  the capture is armed, and their padded rows are no model output -- unsuspended, a warm-up all-reduce was captured at
  "request 0" instead of the model's overflow at request 9 (2026-11-23). N = 0 after a vLLM upgrade: re-check the
  method names in `scripts/vllm_capture/sitecustomize.py` (`_WARMUPS`) before trusting a request-0 capsule. A NaN
  made by the all-reduce itself is captured at `vllm.all_reduce` on every rank (a COLLECTIVE group:
  `step4-distributed.md` *Synchronous collectives*) -- in the default compiled mode too, where
  vLLM's custom ops still run as ops. Its rate (step 2) folds into this wave's `ctl` lines.
  **Pipeline parallel > 1 and several nodes** (vLLM's own `--nnodes` / `--node-rank` / master address; Ray
  clusters are not covered): the same shim. Every worker of every stage gets every `execute_model`, so the
  stop stays in lockstep; the job manifest tags each rank `pp<i>/tp<j>`, and a later stage's NaN arrives
  through the pipeline's send / recv, which the capture does not see -- of the ranks that hold the NaN, the
  EARLIEST stage is the origin (the manifest's verdict says so). Several nodes: `all` lines, one job over
  `<NODES>` (`step4-distributed.md`), the driver given `ND_NNODES` / `ND_NODE_RANK` / `ND_MASTER_ADDR` and
  `MASTER_PORT` for vLLM's `nnodes` / `node_rank` / `master_addr` / `master_port`. Only the head node can
  run the engine: the offline `vllm.LLM` cannot be a follower ("collective_rpc should not be called on follower
  node") -- the head runs `vllm serve` through `serve_run.sh` (below), every other node `vllm serve --headless
  --node-rank <ND_NODE_RANK>` (`tests/coverage/vllm_multinode.sh` is the shape). `nd` gives multi-node trials
  the routable socket interface (`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`) and address (`VLLM_HOST_IP`,
  `ND_NODE_IP`): in a container RCCL sees no RDMA device and its TCP fallback picked a backend NIC that does not
  route between nodes (a hang in `ncclSocketPollConnect`). Every node must SEE exactly its share of the GPUs
  (TP x PP / nnodes): `nd`'s `all` trials see all 8, and vLLM 0.33 counts a rank as on the reader's node by
  `rank // device_count` -- fewer ranks per node than visible GPUs and the head's engine stops at `assert
  remote_message_queue is not None`. Export `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` = `0..<share-1>` in
  the driver on every node (the test launcher does). Validated: TP 1 x PP 2 and TP 2 x PP 2 on one node
  (2026-11-24); TP 2 over two nodes serving, and TP 2 x PP 2 over two nodes captured with the origin on stage 1
  (ranks 2,3 = the second node), `REPRODUCED` (2026-11-25).
* **`vllm serve`** (an OpenAI server): run the server itself, with the user's failing requests --
  `bash $ND_REPO/scripts/serve_run.sh --server 'vllm serve <model> --port {port} ...' --requests
  <requests.jsonl>` as the spec line's command (it starts the server, waits for `/health`, sends the
  requests through `scripts/serve_client.py`, stops the server; `{port}` = 18000 + 10 x the trial's first
  GPU). The capture shims act in the server's worker processes as for an offline driver. `<NAN_SIGNAL>` =
  `Logprobs are not finite` (the client's line -- also when vLLM answers HTTP 400 because it cannot
  serialise a NaN logprob). The requests come from the user (a JSONL of prompts or token ids); never
  invent traffic. Ledger the request file.
* **Compiled by default** (`enforce_eager=False`: `torch.compile` + HIP graphs): a NaN born in
  compiled code reaches the capture only `PROPAGATED` at the next eager op (usually
  `vllm.unified_attention_with_output`) -- expected, and the route to `step5n-compiled.md` (its
  *vLLM* section: C0's switch is `enforce_eager=True`, C1 needs HIP graphs off). Don't force
  eager mode for the capture: the user's NaN is the compiled run's.
* The capture's locator also reports vLLM's start-up forwards (`warmup.py`,
  `cudagraph_utils.py capture()`, a `NONE ... profile_run` line): engine initialization, not loops
  to wire; its `OK:` / `NEXT:` verdict is what counts.

## SGLang

The capture preflight's `SGLANG` lines, read before step 2:

* **The model runs in the scheduler subprocess, always** (`sglang.Engine` spawns it; there is no
  in-process switch). **Don't wire the request loop** -- the capture goes into the scheduler:
  capture lines of the spec get `PYTHONPATH=$ND_REPO/scripts/sglang_capture` in their env field and
  run the **original** script (no wired copy, nothing to restore in step 9). Its `sitecustomize`
  patches `Scheduler.run_batch`: the capture is built at the first batch (after model load and HIP
  graph capture) and `cap.step()` starts every batch; the logs show `[sglang-capture] patched ...`
  and `capture built in the scheduler`. Use `NAN_CAPTURE_ARM_AFTER_STEPS=1`. No locator run: there
  is no user-side loop to confirm.
* **Seen and not seen:** the prefill runs eagerly (SGLang's default on ROCm), so the capture sees
  it, aiter / `sgl_kernel` custom ops as one op each (`aiter.rmsnorm2d_fwd`); decode steps are HIP
  graph replays and are not seen. A NaN born only in decode reaches the capture `PROPAGATED` at the
  next prefill, or not at all: rerun with `disable_cuda_graph=True` in the engine arguments.
* **Tensor / pipeline / data parallel > 1** (`tp_size`, `pp_size`, `dp_size`): one capture per scheduler
  rank; a replica's ranks stop together over its default group (they share every batch); data-parallel
  replicas (not DP attention) each capture into `<CAPSULE_DIR>/.../dp<i>`. Validated 2026-11-24: TP 2 and
  DP 2, `REPRODUCED`. Under docker, several multi-GPU trials on one node need `--pid=host`, which `nd` adds
  by itself (`DOCKER_PID_HOST`): with per-container PIDs and the shared IPC namespace, 3 of 4 TP-2 trials
  died in `ncclCommInitRank` ("unhandled cuda error").
* **Decode steps are graph replays** (above): capture lines get `PYTHONPATH=$ND_REPO/scripts/sglang_graphs`
  (first) and `ND_SGLANG_DISABLE_CUDA_GRAPH=1` -- graphs off without an edit, for `sglang.Engine` and
  `launch_server` alike; `ND_GRAPH_CHECK=1` on one line tells a graph-only NaN (`GRAPH_ONLY` /
  `NOT_GRAPH_ONLY`, as vLLM's `scripts/vllm_graphs`).
* **`sglang.launch_server`** (an HTTP server): as `vllm serve` above, `serve_run.sh --api sglang --server
  'python3 -m sglang.launch_server --model-path <model> --port {port} ...'` (SGLang's native `/generate`).
* **ROCm images run as root** (`DOCKER_AS_USER=0` in the profile): aiter writes into root-owned
  directories (its JIT dir, `/tmp/aiter_configs`); as the host uid every process -- the replays too
  -- dies at `import aiter` (`shutil.Error ... Permission denied ... flydsl_cache`). The package's
  `run.sh` keeps the setting. A root container runs with `umask 000`, so what it writes stays
  deletable; a group's re-run (`GROUP=G<i>`) moves its earlier `repro-`/`verify-` dirs and tarball
  to `old/*.before-<tag>` itself.
* Template: `examples/inference/qwen_sglang_serve_nan/capture.spec`.

## Contention: the one rule (every later replay)

A capsule stores *values*; a race is carried by *timing* (how many processes issue
at the device at once), which no `torch.save` keeps. A contention-gated capsule
replayed alone never fails -- a clean `0/N` that reads like evidence. **Every
replay rate carries the contention it was measured under**; a solo `0/N` on a
contention-gated fault is an experiment not set up, not a negative.

## Declare-contention (step 1; again whenever trial geometry changes, e.g. step 3)

N = concurrent containers on the busiest **device**, injected into the working copy
(never `<RUN_SCRIPT>`) as `NAN_CAPTURE_CONTAINERS_PER_GPU` -- the one fact the
capture cannot see. Simple shape (`for trial ...; do ... &`, `gpu=$((trial % NGPU))`):
N = `ceil(trials / NGPU)`; ledger N and how derived. **Anything else** (no obvious
pinning, uneven, several containers per trial, sequential): `step1-contention.md`.
