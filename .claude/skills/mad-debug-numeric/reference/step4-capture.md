# Step 4, prepare: arm the capture

Read only when there is **no capture setup for this model yet**, or its source, arming
point, op filter, snapshot cap or capture code changed; otherwise `step4-run.md` alone.
Short commands run directly; `madengine nan exec ...` GPU commands go through
`mad-run.sh` (`env-madengine.md`). Conditional: `step4-capture-memory.md` (untested
setup, an OOM, snapshot-skip lines to judge), `step4-capture-speed.md` (setup dominates
the time to a capsule).

## 4.1 Choose the wiring

**Default: the `nan_doctor` madengine tool.** The run is the model's normal
`madengine run --tags <tag> ...` with `"tools": [{"name": "nan_doctor"}]` in
`--additional-context`. A `.pth` hook arms the capture in every rank and worker: no
source edits, nothing to restore in step 9. Capsules land in
`nan_doctor_output/capsules/` (on SLURM `node_<N>/nan_doctor_output/`); group with
`--recursive`. The `NAN_CAPTURE_*` knobs below reach the run as environment variables
(`env-madengine.md`).

**Inference servers the optimizer hook cannot see.** Put the matching directory
on `PYTHONPATH` for that run, in `docker_env_vars`, as the absolute in-container path
`<RT>/<dir>` with `<RT>` = `/myworkspace/scripts/common/tools/nan_doctor/nan_doctor_rt`
(madengine populates `scripts/common/tools/` in the workspace during a run).
Nothing arms without `NAN_CAPTURE_DIR`.

| Workload | `PYTHONPATH` entry | What it patches |
|---|---|---|
| vLLM: tensor parallel > 1, or TP 1 in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) | `<RT>/vllm_capture` | `Worker.execute_model` in each worker (`step1-ask.md` *vLLM*) |
| vLLM CUDA graphs | `<RT>/vllm_graphs` | graph-only NaN check; graph replays are invisible to the op hook |
| SGLang | `<RT>/sglang_capture` | the SGLang scheduler (`step1-ask.md` *SGLang*) |
| `torch.compile` / Inductor | `<RT>/inductor_capsule` | the kernel Inductor's NaN asserts caught, replayed later with `python3 -m nan_doctor_rt.replay_inductor_kernel` (the capture itself watches Inductor kernel launches without it: `NAN_CAPTURE_INDUCTOR=0` turns that off) |

**Fallback: source wiring**, only for loops the hook cannot see: a loop with no
optimizer step, or a framework that owns the loop. Map the source, don't read it whole
(a model source is often 1,000+ lines):

```bash
madengine nan capture-points <model.py> <launcher>
```

It prints, with line numbers: the last import, the ranked work loops (with a code
window), `LAZY` builders, swallowing `except`s, the model's own NaN checks and their
step guard (the control floor, 4.3), `DIST`, and the fan-out 5a needs. Module-level loops
rank like function loops, and a `with torch.no_grad():` / `inference_mode()` block (or
decorator) counts for the outermost loop in it -- a script-style decode loop is found
without `--loop-line`. Read only those
windows. `no LOOP candidate` = read the entry point yourself. **The ranks are static, one
file at a time**: the top loop can be one that never runs (the trainer is reached through
a registry or dispatch) or one that runs but does not train (a memory probe). Wire it;
the smoke test (4.2) confirms or corrects it.

**Copy the source aside before editing** (or write the capture variant as a new file):
step 9 restores it. **Ledger: the saved path.** For a plain loop, one call does backup +
the three lines below + compile check:
`madengine nan wire apply <model.py> --backup-dir <ARTIFACT_DIR>` (loop from
`capture-points`, or `--loop-line N`); `status` / `restore` (step 9, sha256-checked).
`madengine nan preflight` (step 1) already answered HOOK/ENV/LAZY.

### The three lines, all defaults from the environment

```python
from nan_doctor_rt.op_nan_capture import capture_from_env
cap = capture_from_env()      # after model/optimizer construction, before the loop
    cap.step()                # top of EVERY iteration of the work loop (each worker's, if any)
cap.close()                   # after the loop
```

`nan_doctor_rt` must be importable in the model's process: `madengine nan wire apply`
writes the lines that make it so; use them verbatim. `NAN_CAPTURE_DIR` is an **in-container** path, **empty** and **writable in the
container**; unset = a no-op. The same defaults and knobs apply to the tool and to
source wiring:

1. `NAN_CAPTURE_DIR`: workers or ranks write `rank<N>_<uuid>.pt` side by side.
   **Synthetic data is the default: leave `NAN_CAPTURE_DATA` unset.** The capsule
   records each floating-point tensor's shape plus statistics (finite
   min/max/mean/std, zero/NaN/Inf counts; no values, no digests; `data: shapes` in
   `capsule_info`). **Integer tensors keep their real values** (`cu_seqlens`,
   gather/scatter indices, offsets): they are small and carry structure, not the
   user's data, and random ints are not a valid layout -- the op would reject them
   (`cu_seqlens_q must start with 0`) instead of replaying. The capsule stays small.
   It replays only with `--synthesize`, which draws the float operands from the
   statistics. Pass it on **every** replay: `madengine nan replay --extra-args
   "--synthesize"`, hand-run walks, the step-6 `run.sh`.
   * **What synthetic can and cannot show.** It reproduces a fault that does not
     need the actual float values: a race, or a kernel that NaNs on any input of
     that layout. Every verdict holds for the synthetic operands only, with no
     `DIVERGED` check. A clean synthetic replay proves nothing. The report says
     every replay ran on synthetic float operands. GEMM-family ops are still
     checked against a higher-precision recompute on the same synthetic operands
     (`WRONG VALUE` in the walk, counted by the loop): a kernel that leaves output
     unwritten reads leftover memory, which in a fresh replay process is mostly
     finite -- judged by finiteness alone it reads clean.
   * **Clean synthetic 5b while the model reproduces**, or an `ERROR` from the
     op's own argument checks: **ask the user before recapturing with
     `NAN_CAPTURE_DATA=1`**, which stores all real values (MBs to GBs per capsule,
     containing their data; a buffer the chain holds several times -- views, a
     pre-call snapshot -- is stored once, and an allocator's unwritten output not at
     all: replay allocates it). Then replay without `--synthesize`.
   * **`NAN_CAPTURE_DATA=1` from the start** when the user asks for real values,
     or when step 1 found the data **public / synthetic test assets** (a public test
     split, a random-token benchmark, a repo fixture): nothing to protect, and
     real values are what make operand checks possible (a `DIVERGED` row, the fp64
     walk on recorded operands, `--written-operands`). Step 1 states the choice in
     its one-line summary; the user can override it there, not after a clean replay.
2. Op filter (`NAN_CAPTURE_OP_CLASSES`): unset. Tightening once every capsule so far
   landed in one class (e.g. `gemm`) is fine (cuts overhead).
3. **`chain_depth=4`** (`NAN_CAPTURE_CHAIN_DEPTH`): the op's data-dependency ancestry --
   the only way to tell a kernel bug from a kernel faithfully computing garbage from a
   finite-but-wrong operand. Costs memory; old producers age out as `values_evicted`.
4. **Snapshots on, capped at 4 GiB** (`NAN_CAPTURE_SNAPSHOT`,
   `NAN_CAPTURE_SNAPSHOT_MAX_BYTES`): pre-call operands of in-place ops (optimizer
   updates); larger ops skip it (`args_are_pre_call=False`). Sizing:
   `step4-capture-memory.md`.
5. **World size > 1:** `cap.step()` calls `sync_stop()` every step on every rank at
   the top of the iteration -- so put it **outside any other collective**; otherwise
   only the NaN rank unwinds and the rest hang in RCCL. DTensor/FSDP2/TP, compiled
   entry points (`NAN_CAPTURE_SUSPEND`), the manifest: `step4-distributed.md`.
6. **Arm after initialisation, never around `main()`**: `aten.empty*` memory is often
   NaN bit patterns; a capture live during construction fires on it with
   `inputs_finite=True`. The allocator family (`aten.empty*`, `new_empty*`) is
   always excluded, but views of it cannot be (`detach`, ...) -- placement is the
   defence. **`LAZY` hit** (torch.compile, TrainPipeline*, CUDA graphs) or setup
   messages after the start-of-loop banner: `NAN_CAPTURE_ARM_AFTER_STEPS=1`
   (costs step 0; if the NaN is *at* step 0, don't -- add `NAN_CAPTURE_EXCLUDE_OPS`).
7. `NAN_PROGRESS_EVERY=N` prints `[op-nan] Progress step N` lines (budget 4.3).
8. **Settle mode, on by default** (`NAN_CAPTURE_SETTLE=1`): a NaN is written only if
   it outlives its step -- written into or read from a tensor older than the step,
   handed to a collective, or read to the host -- so torch's own compute-then-mask
   NaNs (`sinc_backward`'s 0/0 at x == 0, masked by a `where`) no longer stop the
   run. A masked NaN prints `MASKED(<origin>)` and lands in the census printed at
   `close()` and stored in the capsule (`masked_sites`). `nan_origin` comes from an
   fp64 re-run of a pointwise op: `math` (undefined arithmetic; counted only),
   `overflow` / `unexplained` / `propagated` / `no_reference` (masked ones saved
   once per site as `<capsule>.masked-<n>` -- masked is not benign: the model's own
   `nan_to_num` masks a real kernel NaN too). **A loop that rebuilds its tensors
   every step and stops on its own NaN check** (`if t.grad.isnan().any(): return`)
   never lets the NaN outlive its step; the candidate is written at process exit
   instead (`settle: unsettled_at_exit`), also from mp.Process workers and from a
   worker its parent `terminate()`s (the capture writes before SIGTERM kills it).
   `SIGKILL`, or a model with its own SIGTERM handler, gets no such write: there,
   or if the first run shows `pending:` lines and no capsule, set
   `NAN_CAPTURE_SETTLE=0`.
   The world-size-1 traceback is raised
   where survival was seen (often the optimizer), not at the op that made the NaN;
   the capsule is still that op's, and its `settle` record names both.
   **The capsule is the surviving NaN's root**: every non-finite output downstream of a
   pending candidate inherits it, and a survival writes that root -- a `math` one too (an
   eps-0 norm's `0 * rsqrt(0)`, a model / numerics NaN), never the downstream op that only
   carried it (a `math` birth op once lost to the next GEMM: `aten.rsqrt` of a PyTorch
   RMSNorm was captured as `aiter.gemm_a16w16`, a capsule no replay could resolve).
   Fused norms and gated activations of aiter / sgl_kernel / vLLM `_C` get an fp64 origin
   too (`nan_doctor_rt.math_refs`), so their capsules say `math` / `overflow` /
   `unexplained` instead of `no_reference`.
   **Do not reach for `NAN_CAPTURE_EXCLUDE_OPS` against a recurring benign NaN**:
   it blinds the capture to every op of that name. `NAN_CAPTURE_SETTLE=0` restores
   capture-at-first-non-finite (e.g. to compare with an old run).

**Framework-owned loop** (`capture-points` shows `HOOK`, no `LOOP`): the tool already
arms every rank; if the framework's per-step path never reaches it, pass the same
capture as a callback -- `L.Trainer(..., callbacks=[lightning_capture_callback()])`
(Lightning) or `Trainer(..., callbacks=[hf_capture_callback()])` (HF); both from
`nan_doctor_rt.op_nan_capture`, same env knobs; `step()` runs at the start of each train
batch / optimizer step.

Need something the helper does not expose: `capture_from_env(**overrides)` passes
keyword arguments to `OpNanCapture` (see its docstring).

**Hook cost.** Per watched op: one max-|value| reduction per output/written arg;
pure views skipped (`check_views=True` to include). Under co-tenancy the cost is
**extra kernels per step**; a many-kernel check slows steps severalfold and can
suppress the race. Capture slow, or 0 capsules while controls NaN: check the build has
the single-reduction check first.

**The model's kernel, for later (hipBLASLt GEMMs only):** give ONE capture run
`HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE=<in-container path under the artifact mount>/model_hipblaslt.log`
through the same route as `NAN_*`. It records the solution the model dispatched for
every GEMM shape, so a clean 5b can be checked with `madengine nan kernel-match` at once
(`step5-pressure.md`) instead of with an extra model run. One run only: the log costs
host time per call.

## 4.2 Smoke test -- only when the setup is new

Checks (also applied to a capture run's first steps):

* **0 capsules in the first ~15 steps.** An early capsule is almost always an
  alloc/view false positive: `inputs_finite=True`, 1-op chain on
  `empty`/`new_empty`/`detach`/`alias`. **Not when the op is real compute
  and the fault is a race:** the hook's extra kernels shift timing and can make it
  fire *sooner* -- a GEMM NaN can appear at steps 0-13 under capture while the
  uncaptured model NaNs much later. A GEMM/attention capsule with a multi-op chain and
  a finite fp64 reference is a finding, early or not; judge by the op and chain, not
  the step.
* **Read the `MASKED` lines / census** (settle mode): a `math` site firing every
  step is expected and costs nothing after its first settle; any `.masked-<n>`
  file (`unexplained` above all) is worth a look before the run is trusted -- it
  is a NaN the model hid.
* **No OOM** (`Error at step N: CUDA out of memory`, `pre-call snapshot ... failed`);
  snapshot-skip lines name only expected ops (`step4-capture-memory.md`).
* **Plumbing:** capsule dir writable; `[op-nan] capsule data:` matches the
  mode chosen in 4.1 (`shapes only` by default, `values` with
  `NAN_CAPTURE_DATA=1`); the armed line shows the expected process counts; env knobs
  reached the container. `madengine nan digest` summarizes the log.
* **The loop that ran** (source wiring): if the digest shows `CAPTURE NEVER
  CONSTRUCTED`, or another loop has more steps than the wired one,
  `madengine nan wire restore` the wrong file and re-apply with `--loop-line N`
  on the right loop. `HOOK` -- a framework owns the loop: its callback (4.1). The
  locator (`"PYTHONPATH": "<RT>/locate"` in the smoke run's `docker_env_vars`) hooks
  `Optimizer.step` and `backward` -- and, for inference, top-level forwards with grad
  disabled (each forward of a `generate()`) -- and names the loop around them. No
  locator lines: either `PYTHONPATH` did not reach the process (fix the route, not the
  wiring), or the model has nothing the locator hooks (plain tensor code: no
  `nn.Module` forward, no `backward`, no optimizer) -- then judge the wiring from the
  capture's own lines (`armed`, `Progress step`, a capsule).

Separate smoke test (~15 steps) **only** for: unknown memory headroom (a big model,
snapshots near the cap), or a changed arming point, filter or capture code you have
reason to doubt. A first capture of a model that fits easily needs none: the capture
run's first steps are the smoke test, and a separate one only repeats them.
**Otherwise the capture run is the smoke test** (a broken capture ends the run in its
first steps anyway): if it shows the false-positive signature or an OOM, fix and re-run.
Ledger: which smoke test or run validated which configuration.

## 4.3 Run length

* Cut to a few dozen steps -- the capture detects at the op, not at the model's late
  check; raise only if nothing is captured.
* **A NaN step already measured** (a step-2 wave, or an earlier run of the same environment in
  `<LEDGER>`): length = about 1.5 x the latest NaN step seen, never below the floor below.
  Unmeasured: the model's own length.
* **Floor: the control run's first check window.** Below it every control reads "no
  NaN" by construction. Find the interval in the source (a model that checks every
  1000 steps needs at least 1000).
* Budget wall time from the capture's own step rate; log progress every few steps.
