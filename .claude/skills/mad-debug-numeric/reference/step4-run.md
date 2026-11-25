# Step 4, run: the capture wave (always read)

`step4-capture.md` first only if there is no validated capture setup yet for this
model (or its source, arming, filter or cap changed). A `madengine run` command with
`"tools": [{"name": "nan_doctor"}]` whose ledger shows a passing run is the validated
setup: state it, don't re-derive it. **Its knobs come verbatim from that command**
(its `--additional-context` and environment as it ran), never rebuilt from a ledger
note: a note abbreviates, and a `NAN_CAPTURE_SUSPEND` that lost its module killed
every rank at its first step. Neither has the knobs: re-derive them
(`step4-capture.md`), don't guess. `<PRIORS>`=none (`step1-ask.md` question 7) drops the
earlier-investigation half: no earlier ledger's command is reused.

**Capture the user's configuration only.** Every capture and control run uses the model
and context the user named -- not a variant of it (another backend, a debug switch, a
context meant as a control). A variant's capsules form groups of their own, at other ops,
and muddy `groups_priority.txt` and the report (two runs with a fused-op backend switched
to `torch` once added a second group at a downstream GEMM no replay could resolve).
Variants are step-7 experiments.

## 4.4 Run with a control arm

A capture = the model's `madengine run --tags <tag> ...` with `"tools": [{"name":
"nan_doctor"}]` in `--additional-context`; capsules land in
`nan_doctor_output/capsules/` (on SLURM, `node_<N>/nan_doctor_output/`). Start each
run detached: `bash .claude/skills/mad-common/mad-run.sh start <run-id> -- madengine
run ...`, then `mad-run.sh wait <run-id>` until `MAD_RUN_STATUS=succeeded|failed`. A
**wave is N sequential runs**; controls are runs without the tool (`env-madengine.md`,
`waves.md`).

* Confirm `NAN_CAPTURE_CONTAINERS_PER_GPU` is set in the capture run's environment (it
  cannot be added to a capsule later).
* **Timing-sensitive fault: no-capture control runs in the same wave**, interleaved
  (e.g. 12 capture + 4 control); the hook is not neutral on a race:
  * capture 0/12 vs control 4/4 -> hook suppresses: cut overhead (filter, cheaper
    check -- extra GPU work perturbs) or raise the rate (step 3, interactive only);
  * capture ~6/12 vs control 4/4 -> hook taxes; fine.
  * Runs are sequential, so a control has no capture partner on its GPU; if the fault
    needs co-tenants, give both arms the same declared load (`step1-contention.md`).
  * `<NAN_RATE>` *unmeasured*: controls that ended before the stop give "at least
    k of m"; ledger it with this wave's environment, or keep the rate *unmeasured*.
* **Stop at the first settled capsule** -- one capsule is enough for step 5, and every
  later run only delays the replay on the same GPUs. After each run, check
  `<CAPSULE_DIR>`; start no further run once a capsule exists. Ledger the runs not
  started as `STOPPED`, never as clean. Run on only when the user asks for a control
  rate or a second capsule.
* Never stop a run with a pattern `pkill`: that can hit another session's runs. Stop
  through `mad-run.sh`, then confirm the GPU is empty.
* Read the wave with `madengine nan waves --capsules <CAPSULE_DIR>` (SKILL.md rule 9).

**vLLM with graphs (its default): capture runs with graph replay off, plus a graph-check
run.** The capture watches compiled kernels (`step5n-compiled.md`), but a replayed HIP graph
launches them without Python. So the capture run turns graph replay off and keeps
`torch.compile` -- no model edit -- and one graphs-on run checks every graph replay for a
NaN only graphs make (*Graph-only NaN*, `step5n-compiled.md`). In `docker_env_vars` (`<RT>`:
`step4-capture.md`): capture run `"PYTHONPATH": "<RT>/vllm_graphs", "ND_VLLM_CUDAGRAPH_MODE":
"NONE"` with the tool; graph-check run (no tool) `"PYTHONPATH": "<RT>/vllm_graphs",
"ND_GRAPH_CHECK": "1", "ND_GRAPH_CHECK_DIR": "/myworkspace/<ARTIFACT_DIR>/capsules_graph"`.
The controls stay as the user runs the model (graphs on): they are the rate. **Capture runs
clean while the controls NaN** = the NaN needs graphs: read the graph-check run's
`GRAPH_CHECK` line and go to 5N's graph-only route, not to `step5-pressure.md`. The capture
run puts both on `PYTHONPATH` (`<RT>/vllm_graphs:<RT>/vllm_capture`, `vllm_graphs` first) at
any TP: `vllm_graphs` alone arms no capture in an inference loop. SGLang:
`<RT>/sglang_graphs` (first) -- capture runs `ND_SGLANG_DISABLE_CUDA_GRAPH=1` (graphs off, no
edit), one run `ND_GRAPH_CHECK=1` (`GRAPH_ONLY` / `NOT_GRAPH_ONLY`); `step1-ask.md` *SGLang*.

After the capture run: `madengine nan group` (5a, `step5-read.md`), then 5b
`madengine nan replay` (`step5-run.md`), then step 6 `madengine nan package`
(`step6-build.md`) -- each long command started through `mad-run.sh start`. Ledger each
with `madengine nan ledger <LEDGER> job` / `wait`. Read the last line of each command
and the files it names, then go to `step5-read.md`.

**With a TRIGGER** (step 1 ledgered a load note): give the first 5b the co-tenant, so
it runs beside the load instead of ending `NOT_SET_UP` solo --
`madengine nan replay ... --mode op --cotenant 'python3 -m nan_doctor_rt.transformer_hog
--mode full --procs <capture's processes per GPU - 1> <the model's sizes>'` (a decoder
transformer; `step5-pressure.md` 5d for the sizes and other models); a shapes-only
capsule also needs `--extra-args "--synthesize"`. Ledger that the solo case was not
run.

## 4.5 Judge the result

* **From `<CAPSULE_DIR>`, not exit codes or the log.** At world size 1 the capture
  raises in place (a `__torch_dispatch__` traceback is success -- don't "fix" it);
  an `except Exception` loop swallows it, exits 0, and may print the model's own
  `SUCCESS` (the message contains "nan"). World size > 1: `sync_stop()` tears all
  ranks down the step after.
* **Why a capsule (or none):** `madengine nan digest <run log or run dir>` (SKILL.md
  rule 9) -- the captured op, dirty operands, chain, `settle` and the census in
  ~40 lines; a clean run shows only the MASKED sites. It follows the run dir the log
  names when the log has no `[op-nan]` lines.
* **"No NaN" does not mean the run worked**: the same handler swallows a step-0 OOM.
  Check `Error at step` first. No capsule + "no NaN" = no NaN, hook suppression, or
  died at once -- only the log separates them.
* **Capsule save errors** (`waves` alert `capsule save error`, a `cannot pickle`
  traceback): the run caught a NaN but `torch.save` failed. Current builds save anyway
  with the offending argument leaves replaced and list them in the capsule's
  `unpicklable` field (a replay of an op whose argument was replaced can fail: say
  so); an older build lost the capsule -- ledger the lost site (op, file:line from its
  `pending:` line) as `not captured`.
* **Reproducer found:** tell the user where `<CAPSULE_DIR>` is and to save it.
* **No capsule:** `step5-read.md` *Empty `<CAPSULE_DIR>`* -- after its fixes, the
  fallback is step 5N (`step5n-native.md`), not the end of the ladder.
