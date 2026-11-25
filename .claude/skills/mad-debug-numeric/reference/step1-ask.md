# Step 1: ask for the model (required), take the defaults, declare contention

Never skipped; runs nothing but the `madengine nan exec` check (`<GPU_EXEC>` below = `madengine nan exec --image <IMAGE> --`).

| ask for | notes |
|---|---|
| `<MODEL_TAG>` | the madengine model/tag that runs the model (`madengine run --tags <tag>`), plus its `build_manifest.json` |
| `<NAN_SIGNAL>` | how its output says a NaN happened (a grep for a log line the model prints). A **magnitude** counts too -- `grad_norm: +([1-9][0-9]{2,}\|nan\|inf)` for exploding gradients: `step4-explode.md`. So does a **wrong finite value** with no NaN at all -- the model's own parity or determinism check (`RESULT ... wrong=yes`): take that line, `step4-explode.md`. **No signal at all** -- the model prints nothing when wrong, the user suspects silently wrong outputs or asks to validate a configuration: differential mode makes one (`step2-diff.md`) |
| `<WORLD_SIZE>` | **derive, don't ask**: ranks of ONE job (`torchrun --nproc_per_node` x nodes, `NGPU`, the parallelism degrees). > 1 -> `step4-distributed.md` (wiring through the framework's step, manifest, origin rank). Independent trials sharing GPUs are co-tenants, world size 1 each |
| `<IMAGE>` | **don't ask**: the `docker_image` of the model in `build_manifest.json` |
| `<GPU_EXEC>` | `madengine nan exec --image <IMAGE> [--gpus ...] [--mount ...] [--env K=V] --`; runs any `python3 -m nan_doctor_rt.<module>` in the model's image. Steps 4-7 need it |

**Verify `<GPU_EXEC>` once:** `<GPU_EXEC> python3 -m nan_doctor gpu-check --torch`
prints torch's view of device 0 (name, arch, CUs,
memory) and a last line `GPU=... ARCH=... VERDICT=...` (`TORCH_GPU=NO` = the image or
GPU access is broken: fix it first). **Ledger the device line** (gfx942 = MI300X,
gfx950 = MI350X/MI355X), CUs and memory: rates, memory caps and reproducers are bound
to it (SKILL.md *GPUs*). `VERDICT=UNTESTED` here, not yet warned in step 0: print the
SKILL.md *GPU gate* warning now, before any wave, and continue.

**Injected vs stock: don't ask.** It matters only for reading step 7 (on an
injected image the root cause is the planted NaN, confirming the tooling, not a
bug), and the 5b `kernel.*.txt` or a `kernel_mutations --list-mutations` enumeration
tells: `nans_to_nops` matches a site = injected, `PatchError` = stock. Neither: report
it as unknown.

**Ask only question 1; take the default for 2-7** -- a wall of questions overwhelms
(a user answering `default` five times in a row costs five round trips for nothing), and
a question later in the ladder stalls for as long as the user is away. Questions 2-7
below are *settings*: decide each from its default, ledger it as `(default)`, and don't
print them. After question 1, tell the user one line: what was derived (signal, image,
world size, data mode) and "using defaults -- say *ask me the options* to change them".
Still ask only what has no safe default: a `<NAN_SIGNAL>` the model does not show, the
PyTorch gate's `ASK`, the budget check (question 6, above its threshold). Never ask what
you can read from the model's run configuration -- state the derived value and let the
user correct it.

**Ask-me mode** -- the user asks to choose ("ask me the options", "let me choose",
"ask me the questions"), before or during step 1: show questions 2-7 as one compact
block with their computed defaults (one line each), "reply `default` to accept all, or
name the ones to change" -- then ask only the ones the user names, one at a time, and
ledger `ask-me mode`. It also restores the step-7 budget question and the sweep-plan
stop. In order:

1. **`<MODEL_TAG>`** -- no default; asked as "How do I run the model?", so the answer
   may be a tag, a directory or a command line. **A command line** (`madengine run
   --tags <tag> --additional-context '...'`): the tag and its context are the run as the
   user gave it -- ledger the command verbatim once the ledger exists, and use its context
   for every run. **If the user names a directory, not a tag:**
   pick the plain model entry -- the one that runs the model as-is, not a capture,
   timing or reproducer variant -- state the pick in one line and go on; ask only if
   two plain ones remain. **Then the PyTorch gate** (SKILL.md step 0,
   `madengine nan framework-check`) before anything else: `SKIP` = print its warning
   and stop -- no artifact dir, no ledger; `ASK` = its one question first. `RUN`: go
   on, and ledger its last line once the ledger exists. **Now create `<ARTIFACT_DIR>`**
   (SKILL.md rule 3; `<RUN_TAG>` = today's date + the model name) -- **`ls` it
   first**: the tag is only date + name, so an earlier session's investigation of the
   same model today already owns it. Its ledger has jobs or capsules: suffix the tag
   (`-2`). Only step-1 lines: tell the user and resume it (ledger a `resumed` note).
   **`<PRIORS>` = none: do not open it** (another investigation's ledger is exactly what
   none excludes) -- take the next free suffix whatever it holds.
   Then start the ledger: `$LG start step1` (`$LG` = `madengine nan ledger <LEDGER>`),
   `$LG note step1` with `<MAD_ROOT>` and `<ARTIFACT_DIR>`. **Look for earlier
   investigations of the same model, any date** (skip with `<PRIORS>` = none, question 7),
   in the same call: `grep -l '<MODEL_TAG>'
   <ARTIFACT_DIR>/../*/ledger.md`. A hit with a passing capture run or a tarball is a
   validated setup and maybe a finished reproducer: ledger it. **Default mode: don't
   ask** -- take a fresh capture with that setup (it confirms the old result still holds)
   and say so in the one-line summary ("reusing the capture setup from `<dir>`"); offer the
   old reproducer only in ask-me mode or if the user asks for it. Never carry its rates
   over (rule 5). **A reused setup brings its
   `<NAN_SIGNAL>`** (its `madengine run` command and ledger): take it, don't
   re-derive it -- a README-derived regex differed from the validated one. Then read
   the model's run script -- once, comments and blank lines stripped (`grep -vE
   '^\s*(#|$)' <run script>`); its README, Dockerfile and the model source only by
   `grep` for what is still missing -- and derive `<NAN_SIGNAL>` (when nothing is
   reused), the image, the trial geometry and **the data
   source** (public or test assets, or the user's own data: it sets the capsule data
   mode, question 3); report them in one line, don't ask. Ask for `<NAN_SIGNAL>` only
   if the script doesn't show it (preflight `SIGNAL ?  the run script greps no NaN
   line` = its greps only format output; take one of the `the source raises/logs:`
   candidates the preflight prints) -- and not even then when the user describes no
   visible failure at all (silently wrong outputs): `step2-diff.md` makes the signal.
   Then, in ONE call, `madengine nan preflight
   <run script> --source <model.py>`: `IMAGE ... MISSING` = the image is not built or
   pulled -- get it now, before any run -- **don't wait for an answer**: tell the user in
   one line it is being pulled or built and how long it takes, then start it (ask-me mode:
   ask first); `SIGNAL 'RESULT .*wrong=yes' ... verdict field` = the script reads a result
   line's verdict field: a valid wrong-value signal (`step4-explode.md`) -- but when the
   next line says `the source also raises/logs: '<msg>' ... prefer it`, the NaN is printed
   mid-run too: take that line as `<NAN_SIGNAL>`; a `SIGNAL ... MATCHES CLEAN OUTPUT`
   pattern is not a NaN signal (it matches the field's clean value, `nan_request=-`): take
   the source's not-finite line it lists instead; `SIGNAL ...
   NOT RAISED ANYWHERE` means the script cannot see the NaN it greps for (raise it now,
   with the user -- not after a wave). The default fix, when the model hides NaNs
   (`nan_to_num` on the loss): one line after `backward()` in a working copy,
   `after_backward(model, loss, parts)` from `nan_doctor_rt.grad_trajectory`. It raises
   `NonFiniteDetected("non-finite GRADIENT at step N after backward: <modules>")` (add
   `NonFiniteDetected` to `<NAN_SIGNAL>`), and with `NAN_DOCTOR_GRADLOG` set it also
   logs the per-step gradient trajectory (step 8). Back the source up first, and
   restore it in step 9; `MASK` on a loss, `HOOK`, `ENV`, `LAZY` and `LOOP` settle
   step 4's wiring questions without further greps (`LOOP` is a static guess: the
   step-4 smoke test's locator confirms it). A vLLM model: *vLLM*, below; SGLang:
   *SGLang*, below.
   **Read the trigger notes too, in the same call**: `grep -niE
   'pressure|HBM|co-?tenan|share|race|stress|trials per GPU|control'` over the model
   directory's `README*`/`*.md` and the model source's docstring. A load trigger
   ("needs HBM pressure", "1 trial per GPU stays clean") means replay copies alone
   will likely not reproduce: ledger `TRIGGER: <quote, file:line>` and give the
   **first** replay its co-tenant (`step4-run.md` *With a TRIGGER*) instead of waiting
   for a solo 5b to come back `NOT_SET_UP`.
2. **Step 2** -- default skip (madengine has no capture job with a control arm to fold
   it into: every step-2 run is a separate model run). Running it also measures the rate
   (`step2-rate.md`). Always optional; skipped means *unmeasured*
   (and step 3 skipped), never a carried-over rate (`step2-rate.md`). In ask-me mode word
   it as: "Run the model once to make sure that it runs in this environment. You CAN skip
   this step and go directly to the NaN capturing step (step 4)." -- don't quote a run
   count (the wave's geometry depends on the model) or earlier runs' rates.
3. **Capture-wave geometry** (number of runs, which are controls) -- default: the
   model's own geometry, controls per `step4-run.md`. A wave is N sequential
   `mad-run.sh` runs of `madengine run`; it stops at the first settled capsule -- say
   so in the question. **Same question, the capsule data mode**: real values
   (`NAN_CAPTURE_DATA=1`) by default when the data is public/test assets, shapes only
   otherwise (`step4-capture.md` 4.1 item 1) -- state which and why, so the user can
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
   group with the most capsules (`G1` in `groups_priority.txt`). `all` adds one 5b+6
   pass per further group. **The first time "largest" reaches the user** (this
   question, or its line in the ask-me block, or the one-line summary), define it: the
   group holding the most capsules -- size is the capsule count, not tensor size or memory.
6. **Time budget for the whole run** -- the one setting that may still stop for the
   user. Compute the estimate; **at most 8 hours of the host's GPUs (the GPU lock holds
   them all): don't ask** -- ledger it as the budget and say it in the one-line summary.
   **Above: one line, "this will use the GPUs for about N hours (<what>), OK?"**, nothing
   more. The estimate counts only the runs the settings imply (waves only if step 2 runs;
   the capture run sized from the geometry and the model's runtime; with `all`, ~2 h per
   extra group; plus a reserve for the step-5N fallback -- default ~6 candidate runs of
   ~30 min and one locating wave -- spent only if no capsule reproduces; name it when you
   ask). Within it, start waves, the capture run and per-group 5b replays without
   re-confirming; ask again only beyond it. The step-7/8 default budgets and a recipe
   sweep's plan are checked against the same 8-hour threshold where they start
   (`step7-rootcause.md` *Entry*, `step7-recipe-sweep.md`).
7. **`<PRIORS>`: use earlier investigations of this model?** (use / none) -- default
   **use**; **none** when the user asks for it ("fresh eyes", "no priors", "from scratch")
   or `NUMERIC_DOCTOR_NO_PRIORS=1` is set (a validation run of the skill itself). `use`: the
   greps below (an earlier capture setup, step 5N's and step 7's *prior work*) order the
   work -- a prior root cause is re-tested first. `none`: skip all three greps and read
   nothing under other artifact dirs; build the 5N candidates and the step-7 hypotheses from
   this run's evidence only. What `none` does NOT exclude: what ships with the model -- its
   README (the trigger grep below reads it in both modes: a user's notes are evidence) --
   only earlier investigations' artifact dirs (their ledgers, reports). A validation run
   that must be blind runs on a copy of the model without its notes; with a README that
   names the cause read, say in the report that the run was not blind. Why it exists: with
   priors a step 7 finds its root cause in iteration 1 by re-testing an earlier
   `ROOT_CAUSE.md` -- right for a user, but a validation run then measures confirmation,
   not discovery, and inherits stale setup. Ledger `PRIORS=use|none`; the report says
   which (`step9-report.md`).

**Verify `<GPU_EXEC>` without a separate run when the next run exercises it anyway.**
Run the standalone `gpu-check` only if the first GPU job is something else -- **or if
step 0's GPU gate said `NO_GPU`**: then the GPU type is still unknown, so run it as its
own short command right after the step-1 questions, before any wave, so an
untested-GPU warning comes before work is spent.

**Ledger:** each answer (and whether it was the default), the derived inputs and image,
the `<GPU_EXEC>` check, containers per GPU and how derived.

## vLLM

Read before step 2 when the model serves through vLLM (`<RT>` below =
`/myworkspace/scripts/common/tools/nan_doctor/nan_doctor_rt`, where madengine puts the
runtime in the container during a run):

* **In-process engine, or nothing is captured.** vLLM V1 runs the engine core -- and the model --
  in a subprocess unless `VLLM_ENABLE_V1_MULTIPROCESSING=0`; a capture in the serving loop then
  sees no model op. Not set by the model: add it to the capture run's `docker_env_vars`. At
  TP 1 the capture run gets `"PYTHONPATH": "<RT>/vllm_capture"` too, as below: an inference
  loop has no optimizer step for the tool's hook.
* **Tensor parallel > 1 on one node** puts the model in one worker process per rank: the capture
  goes INTO the workers. The capture run gets `"PYTHONPATH": "<RT>/vllm_capture"` in
  `docker_env_vars` and runs the model unchanged (no wired copy, nothing to restore in step 9),
  with `NAN_CAPTURE_DATA=1 NAN_CAPTURE_ARM_AFTER_STEPS=1` (values: a collective is judged on
  them). The log shows `[vllm-capture] capture built in the worker ... tp 2` once per rank, and
  `capture suspended in N vLLM warm-up / graph-capture method(s)`: vLLM's dummy forwards and
  HIP-graph capture can run after the capture is armed, and their padded rows are no model
  output -- unsuspended, a warm-up all-reduce was captured at "request 0" instead of the model's
  overflow at request 9. N = 0 after a vLLM upgrade: re-check the method names in
  `nan_doctor_rt/vllm_capture/sitecustomize.py` (`_WARMUPS`) before trusting a request-0 capsule.
  A NaN made by the all-reduce itself is captured at `vllm.all_reduce` on every rank (a
  COLLECTIVE group, `step5-read.md`) -- in the default compiled mode too, where vLLM's custom ops
  still run as ops. **Pipeline parallel > 1 and several nodes** (vLLM's own `--nnodes` /
  `--node-rank`; Ray clusters are not covered): the same shim. Every worker of every stage gets
  every `execute_model`, so the stop stays in lockstep; the job manifest tags each rank
  `pp<i>/tp<j>`. A later stage's NaN arrives through the pipeline's send / recv, which the capture
  does not see -- of the ranks that hold the NaN, the EARLIEST stage is the origin (the manifest's
  verdict says so). Several nodes: only the head node runs the engine (an offline `vllm.LLM`
  cannot be a follower), so the head runs `vllm serve` through `serve_run.sh` (below) and every
  other node `vllm serve --headless --node-rank <rank>`; every node must SEE exactly its share of
  the GPUs (TP x PP / nnodes: vLLM 0.33 counts a rank as on the reader's node by
  `rank // device_count`) -- export `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` = `0..<share-1>`
  on every node. Validated upstream on one node (PP 2, TP 2 x PP 2) and over two nodes
  (TP 2 x PP 2); not yet run through madengine's SLURM launch.
* **`vllm serve`** (an OpenAI server): run the server itself, with the user's failing requests,
  as one trial in the model's image: `<GPU_EXEC> 'PYTHONPATH=$NAN_DOCTOR_RT/vllm_capture:$PYTHONPATH
  bash $NAN_DOCTOR_RT/bin/serve_run.sh --server "vllm serve <model> --port {port} ..." --requests
  <requests.jsonl>'` (plus the capture's `NAN_*` variables as `--env`). It starts the server, waits
  for `/health`, sends the requests through `serve_client.py` and stops the server; `{port}` =
  18000 + 10 x the trial's first GPU. The capture shims act in the server's worker processes as for
  an offline driver. `<NAN_SIGNAL>` = `Logprobs are not finite` (the client's line -- also when
  vLLM answers HTTP 400 because it cannot serialise a NaN logprob). The requests come from the
  user (a JSONL of prompts or token ids); never invent traffic. Ledger the request file.
* **Compiled by default** (`enforce_eager=False`: `torch.compile` + HIP graphs): a NaN born in
  compiled code reaches the capture only `PROPAGATED` at the next eager op (usually
  `vllm.unified_attention_with_output`) -- expected, and the route to `step5n-compiled.md` (its
  *vLLM* section). Don't force eager mode for the capture: the user's NaN is the compiled run's.

## SGLang

* **The model runs in the scheduler subprocess, always** (`sglang.Engine` spawns it; there is no
  in-process switch). The capture goes into the scheduler: the capture run gets
  `"PYTHONPATH": "<RT>/sglang_capture"` in `docker_env_vars` and runs the model unchanged. Its
  `sitecustomize` patches `Scheduler.run_batch`: the capture is built at the first batch (after
  model load and HIP graph capture) and `cap.step()` starts every batch; the log shows
  `[sglang-capture] patched ...` and `capture built in the scheduler`. Use
  `NAN_CAPTURE_ARM_AFTER_STEPS=1`.
* **Seen and not seen:** the prefill runs eagerly (SGLang's default on ROCm), so the capture sees
  it, aiter / `sgl_kernel` custom ops as one op each (`aiter.rmsnorm2d_fwd`); decode steps are HIP
  graph replays and are not seen. A NaN born only in decode reaches the capture `PROPAGATED` at the
  next prefill, or not at all: put `<RT>/sglang_graphs` first on `PYTHONPATH` with
  `ND_SGLANG_DISABLE_CUDA_GRAPH=1` -- graphs off without an edit, for `sglang.Engine` and
  `launch_server` alike; `ND_GRAPH_CHECK=1` on one run tells a graph-only NaN (`GRAPH_ONLY` /
  `NOT_GRAPH_ONLY`, as vLLM's `vllm_graphs`).
* **Tensor / pipeline / data parallel > 1** (`tp_size`, `pp_size`, `dp_size`): one capture per
  scheduler rank; a replica's ranks stop together over its default group (they share every
  batch); data-parallel replicas (not DP attention) each capture into `<CAPSULE_DIR>/.../dp<i>`.
  Validated upstream: TP 2 and DP 2, `REPRODUCED`. `madengine nan exec` adds `--pid=host` to
  multi-GPU trials: with per-container PIDs and the shared IPC namespace, concurrent TP-2 trials
  died in `ncclCommInitRank` ("unhandled cuda error").
* **`sglang.launch_server`** (an HTTP server): as `vllm serve` above, with `--api sglang` and
  `--server "python3 -m sglang.launch_server --model-path <model> --port {port} ..."` (SGLang's
  native `/generate`).
* **ROCm images run as root**: aiter writes into root-owned directories (its JIT dir); a replay
  as the host uid (`madengine nan exec --as-user`) dies at `import aiter` -- don't pass it.

## Contention: the one rule (every later replay)

A capsule stores *values*; a race is carried by *timing* (how many processes issue
at the device at once), which no `torch.save` keeps. A contention-gated capsule
replayed alone never fails -- a clean `0/N` that reads like evidence. **Every replay
rate carries the contention it was measured under**; a solo `0/N` on a
contention-gated fault is an experiment not set up, not a negative.

## Declare-contention (step 1; again whenever the geometry changes, e.g. step 3)

N = concurrent containers on the busiest **device**, injected into the capture run's
environment (never the model's own files) as `NAN_CAPTURE_CONTAINERS_PER_GPU` -- the
one fact the capture cannot see. `madengine run` runs one job per invocation, so
sequential runs give N = 1; ledger N and how derived. Anything else (several
containers on one GPU, several containers per run, uneven): `step1-contention.md`.
