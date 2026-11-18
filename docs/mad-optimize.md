# mad-optimize

`mad-optimize` is a Claude Code workflow (`.claude/workflows/mad-optimize.js`). It profiles one MAD model, runs GEAK `kernel_workflow` on that model's kernel source, and keeps the patch only when two clean MAD runs beat the unpatched spread. It does not change `mad-tune-search`. That workflow still stops when a tracked file changes.

Plan mode prints the clone command and the GEAK handoff. It does not clone, copy, or start a GPU job. `--execute` does.

```bash
mad-optimize --tag pyt_fam_fastwam_geak \
  --smoke-tag pyt_fam_fastwam_geak2_smoke \
  --budget 1 \
  -c "{'docker_mounts': {'/myworkspace/fam_cache': '/var/tmp/fam_cache'}, 'docker_env_vars': {'FAM_CACHE': '/myworkspace/fam_cache'}}"
```

Add `--execute` to run it. `--budget` defaults to 1 and is printed, not spent, in plan mode. `--repeats` defaults to 2 and is clamped to 2..5.

## What execute does

1. Resolve the tag with `madengine discover`. The tag must be exactly one model.
2. Setup. `GEAK_HOME` defaults to `$HOME/GEAK`, outside the MAD repo. A checkout inside the repo would show up in `git status`. If `kernel_workflow/kernel_workflow.js` is already at `fa86fd1e9bec56dadf27d6575a0c8f8ab7875c57`, Setup leaves it. If the directory is missing, Setup clones `https://github.com/AMD-AGI/GEAK.git` and checks out that commit. A checkout that exists at another commit is an error. Setup does not delete it. Setup does not `pip install` and does not install or replace Claude Code. `claude --version` must already be at least 2.1.177. `e2e_workflow` is not used; it optimizes SGLang and vLLM serving.
3. Source. GEAK edits `.py`, `.triton`, or `.hip` files. It does not decompile a wheel or a `.so`. `--kernel-path` uses that host directory and does not copy. Otherwise the model image is copied with `docker cp` to `$HOME/.mad-optimize/<model>/src`. For a `pyt_fam` GEAK model the container path is `/opt/faster-action-models/src/fastwam-geak`. If the copy has no editable source, the workflow stops with "this image has no kernel source; pass --kernel-path" and does not start GEAK.
4. Two clean baseline runs of the image tree. They do not mount the copied source.
5. A warm-up run fills the code-object cache. A rocprofv3 kernel trace on that cache, ranked by `kernel-tools.py rank`, names the top kernel; `rocm_trace_lite` would miss kernels replayed from HIP graphs. Then one rocprofv3 counter run of that kernel (built by `.claude/skills/mad-common/kernel-tools.py context`, image tree, no source mount) gives its kernel card, and `kernel-tools.py asm` summarizes its code object. The GEAK task names the kernel and, when the card has one, its bound and what that bound asks for: MFMA utilization for compute, fewer bytes for bandwidth, occupancy and stalls for latency. A counter run is stalled when its status is `running` and `rocm-smi --showuse` reads 0% at every check for 10 minutes. The workflow stops it with `mad-run.sh stop`, does not read its partial counters, and retries once with `kernel-tools.py context --serialize` (`AMD_SERIALIZE_KERNEL=3`) and a fresh run id. A card from the retry is marked `serialized` and its line says to read it as ratios, since serialization shifts the kernel's duration. If the retry stalls too, or the run fails, the kernel card is recorded as missing and the task falls back to the dispatch line. Plan mode prints the retry commands as `kernelRetryCommands`. Neither profiled `performance:` value is the score.
6. `kernel_workflow`, with a real args object: `kernel_path`, `workflow_dir`, `budget`, `gpu_ids` `"0"`, `task`, `apply_to_original` `"true"`. An empty `kernel_workflow/exp/` directory is an error (GEAK issue 300) and the patched score does not run. GEAK's geomean is recorded and does not decide.
7. Two clean runs with the host tree mounted over the container path (`docker_mounts` maps the container path to the host path). `gate()` is the same rule as `mad-tune-search`: both sides need at least two samples, and the latency improvement must beat `max(baseline spread, candidate spread, 1%)`. Lower milliseconds win.
8. When the gate passes and `--smoke-tag` is set, one run of that tag with the same mount. A success rate below a recorded unpatched smoke rejects the patch. When the gate fails, the smoke tag is not run.

`ANTHROPIC_API_KEY` stays in the shell profile. The workflow never writes it into the repo, an argument, or a transcript.

## Checks before execute

```bash
claude --version          # >= 2.1.177; the workflow does not upgrade it
rocminfo | grep gfx950
command -v rocprofv3 || command -v rocprof-compute || command -v rocprof
test -f "$HOME/GEAK/kernel_workflow/kernel_workflow.js" \
  && git -C "$HOME/GEAK" rev-parse HEAD
```

The last command fails until the first `--execute`, which clones the pin when `$HOME/GEAK` is absent. Confirm the clone landed on the pin and that `kernel_workflow.js` is there. Do not check out `main`.

`kernel_workflow.js` is itself a Workflow script, and workflows do not nest, so the GEAK phase cannot call it from inside `mad-optimize`. It starts it in its own headless `claude -p` session through `.claude/skills/mad-common/geak-run.sh` (`start`/`wait`/`status`/`stop`, same shape as `mad-run.sh`), the same way GEAK's own `interface/run_e2e.py` does. The helper passes a fixed command line, takes the `kernel_workflow` args from a JSON file, and sets `eval_dir` so the result location is known up front. A run counts as succeeded only when the session exits 0 **and** `<eval_dir>/final_patch.diff` exists, because `claude -p` can return before the workflow finishes. The session runs with `--permission-mode auto`; review that before running on a shared host.

GEAK serializes GPUs with `kernel_workflow/scripts/gpu_lock.sh`. MAD serializes them with `/tmp/mad-gpu.lock`. `geak-run.sh` takes the MAD lock for the whole GEAK session, so GEAK and `madengine` runs queue against each other. `mad-optimize` still runs one phase at a time.

`node --test tools/tests/*.test.mjs` covers plan mode, a missing `kernel_workflow.js`, an image with no editable source, an empty exp dir, the GEAK task built from a kernel card, and the gate. It does not clone GEAK and does not use a GPU.
