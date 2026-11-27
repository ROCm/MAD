# Step 4, prepare: wire the capture into the model and write the capture spec

Read only when there is **no capture spec for this `<RUN_SCRIPT>` yet**, or its
source, arming point, op filter, snapshot cap or capture code changed; otherwise
`step4-run.md` alone. Every command goes through `nd` (`env.md`). Conditional: `step4-capture-memory.md`
(untested setup, an OOM, snapshot-skip lines to judge), `step4-capture-speed.md`
(setup dominates the time to a capsule).

## 4.1 Prepare the model source

* **Map it, don't read it whole** (a model source is often 1,000+ lines):

  ```bash
  python3 <REPO_ROOT>/scripts/find_capture_points.py <model.py> <launcher>
  ```

  prints, with line numbers: the last import, the ranked work loops (with a code
  window), `LAZY` builders, swallowing `except`s, the model's own NaN checks and
  their step guard (the control floor, 4.3), `DIST`, and the fan-out 5a-load needs.
  Module-level loops rank like function loops, and a `with torch.no_grad():` /
  `inference_mode()` block (or decorator) counts for the outermost loop in it -- a
  script-style decode loop is found without `--loop-line` (2026-11-24).
  Read only those windows. `no LOOP candidate` = read the entry point yourself.
  **The ranks are static, one file at a time**: the top loop can be one that never
  runs (the trainer is reached through a registry or dispatch) or one that runs but
  does not train (a memory probe). Wire it; the smoke test's locator (4.2) confirms
  or corrects it.
* **Wire a copy, not the original** (`<ARTIFACT_DIR>/model/<file>`, the form the capture specs
  use; `step4-run.md`), or copy the source aside before editing in place: the control arm
  needs the original; step 9 restores it.
  **Ledger: the saved path.** For a plain loop, one call does backup + the three
  lines below + compile check (loop from `find_capture_points.py`, or `--loop-line`):
  `python3 <REPO_ROOT>/scripts/wire_capture.py apply <model.py>
  --backup-dir <ARTIFACT_DIR>`; `status` / `restore` (step 9, sha256-checked).
  Its preflight (`capture_preflight.py`, step 1) already answered HOOK/ENV/LAZY.

### Wire the capture -- three lines, all defaults from the environment

```python
sys.path.insert(0, os.environ["NUMERIC_DOCTOR_ROOT"] + "/scripts")   # after the last import
from op_nan_capture import capture_from_env
cap = capture_from_env()      # after model/optimizer construction, before the loop
    cap.step()                # top of EVERY iteration of the work loop (each worker's, if any)
cap.close()                   # after the loop
```

`NUMERIC_DOCTOR_ROOT` (every `nd run`) and `NAN_CAPTURE_DIR` (capture lines of the spec,
`wave.sh`) are set for you, as the runtime sees them. `NAN_CAPTURE_DIR` unset = a no-op, so the same source can serve as
the control arm. What the defaults encode, and the knobs (env vars) that change them:

1. `NAN_CAPTURE_DIR` = `<CAPSULE_DIR>/<trial>`: **empty**, **writable in the container**
   (`chmod 777`), under `<REPO_ROOT>`; group with `--recursive`. Workers or ranks
   write `rank<N>_<uuid>.pt` side by side.
   **Synthetic data is the default: leave `NAN_CAPTURE_DATA` unset.** The capsule
   records each floating-point tensor's shape plus statistics (finite
   min/max/mean/std, zero/NaN/Inf counts; no values, no digests; `data: shapes` in
   `capsule_info.py`). **Integer tensors keep their real values** (`cu_seqlens`,
   gather/scatter indices, offsets): they are small and carry structure, not the
   user's data, and random ints are not a valid layout -- the op would reject them
   (`cu_seqlens_q must start with 0`) instead of replaying. The capsule stays small.
   It replays only with `--synthesize`, which draws the float operands from the
   statistics. Pass it on **every** replay: 5b `CAPSULE_EXTRA_ARGS=--synthesize`,
   the hand-run walks, the step-6 `run.sh`.
   * **What synthetic can and cannot show.** It reproduces a fault that does not
     need the actual float values: a race, or a kernel that NaNs on any input of
     that layout. Every verdict holds for the synthetic operands only, with no
     `DIVERGED` check. A clean synthetic replay proves nothing. The report says
     every replay ran on synthetic float operands. GEMM-family ops are still
     checked against a higher-precision recompute on the same synthetic operands
     (`WRONG VALUE` in the walk, counted by the loop): a kernel that leaves output
     unwritten reads leftover memory, which in a fresh replay process is mostly
     finite -- judged by finiteness alone it reads clean. Before 2026-11-09 that
     check was off under `--synthesize`, and a hipBLASLt GEMM wrong on every call
     replayed 0/186000.
   * **Clean synthetic 5b while the model reproduces**, or an `ERROR` from the
     op's own argument checks: **ask the user before recapturing with
     `NAN_CAPTURE_DATA=1`**, which stores all real values (MBs to GBs per capsule,
     containing their data; a buffer the chain holds several times -- views, a
     pre-call snapshot -- is stored once, and an allocator's unwritten output not at
     all: replay allocates it). Then replay without `--synthesize`.
   * **`NAN_CAPTURE_DATA=1` from the start** when the user asks for real values,
     or when step 1 found the data **public / synthetic test assets** (torchtitan's
     `c4_test`, a random-token benchmark, a repo fixture): nothing to protect, and
     real values are what make operand checks possible (a `DIVERGED` row, the fp64
     walk on recorded operands, `--written-operands`). Step 1 states the choice in
     its one-line summary; the user can override it there, not after a clean replay
     (2026-11-09: the synthetic-first default cost one capture job and a round
     trip on public test data).
2. Op filter (`NAN_CAPTURE_OP_CLASSES`): unset. Tightening once every capsule so far
   landed in one class (e.g. `gemm`) is fine (cuts overhead).
3. **`chain_depth=4`** (`NAN_CAPTURE_CHAIN_DEPTH`): the op's data-dependency ancestry --
   the only way to tell a kernel bug from a kernel faithfully computing garbage from a
   finite-but-wrong operand. Costs memory; old producers age out as
   `values_evicted`.
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
   or if the first wave shows `pending:` lines and no capsule, set
   `NAN_CAPTURE_SETTLE=0`.
   The world-size-1 traceback is raised
   where survival was seen (often the optimizer), not at the op that made the NaN;
   the capsule is still that op's, and its `settle` record names both.
   **The capsule is the surviving NaN's root**: every non-finite output downstream of a
   pending candidate inherits it, and a survival writes that root -- a `math` one too (an
   eps-0 norm's `0 * rsqrt(0)`, a model / numerics NaN), never the downstream op that only
   carried it (before 2026-11-20 a `math` birth op lost to the next GEMM: `aten.rsqrt` of a
   PyTorch RMSNorm was captured as `aiter.gemm_a16w16`, a capsule no replay could resolve).
   Fused norms and gated activations of aiter / sgl_kernel / vLLM `_C` get an fp64 origin
   too (`scripts/math_refs.py`), so their capsules say `math` / `overflow` / `unexplained`
   instead of `no_reference`.
   **Do not reach for `NAN_CAPTURE_EXCLUDE_OPS` against a recurring benign NaN**:
   it blinds the capture to every op of that name. `NAN_CAPTURE_SETTLE=0` restores
   capture-at-first-non-finite (e.g. to compare with an old run).

**Framework-owned loop** (scan shows `HOOK`, no `LOOP`): pass the same capture as a
callback -- `L.Trainer(..., callbacks=[lightning_capture_callback()])` (Lightning) or
`Trainer(..., callbacks=[hf_capture_callback()])` (HF); both from `op_nan_capture`,
same env knobs; `step()` runs at the start of each train batch / optimizer step.

Need something the helper does not expose: `capture_from_env(**overrides)` passes
keyword arguments to `OpNanCapture` (see its docstring).

**Hook cost.** Per watched op: one max-|value| reduction per output/written arg;
pure views skipped (`check_views=True` to include). Under co-tenancy the cost is
**extra kernels per step**; a many-kernel check slows steps
severalfold and can suppress the race. Capture slow, or 0 capsules while
controls NaN: check the build has the single-reduction check first.

**The model's kernel, for later (hipBLASLt GEMMs only):** give ONE capture trial (one container) the
hipBLASLt call log -- `HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE=$ND_ART/model_hipblaslt.<trial>.log`
in that one spec line's env field. It
records the solution the model dispatched for every GEMM shape, so a clean 5b can be
checked with `kernel_match.py` at once (`step5-pressure.md`) instead of with an
extra model run. One trial only: the log costs host time per call.

## 4.2 Smoke test -- only when the setup is new

Checks (also applied to a wave's first steps):

* **0 capsules in the first ~15 steps.** An early capsule is almost always an
  alloc/view false positive: `inputs_finite=True`, 1-op chain on
  `empty`/`new_empty`/`detach`/`alias`. **Not when the op is real compute
  and the fault is a race:** the hook's extra kernels shift timing and can make it
  fire *sooner* -- the 2026-11-06 qwen capture caught a GEMM NaN at steps 0-13 in
  every capture container while the uncaptured model NaN'd at ~16-281. A
  GEMM/attention capsule with a multi-op chain and a finite fp64 reference is a
  finding, early or not; judge by the op and chain, not the step.
* **Read the `MASKED` lines / census** (settle mode): a `math` site firing every
  step is expected and costs nothing after its first settle; any `.masked-<n>`
  file (`unexplained` above all) is worth a look before the wave is trusted -- it
  is a NaN the model hid.
* **No OOM** (`Error at step N: CUDA out of memory`, `pre-call snapshot ... failed`);
  snapshot-skip lines name only expected ops (`step4-capture-memory.md`).
* **Plumbing:** capsule dir writable; `[op-nan] capsule data:` matches the
  mode chosen in 4.1 (`shapes only` by default, `values` with
  `NAN_CAPTURE_DATA=1`); `contention:` banner / armed line shows the
  expected process and container counts; env knobs reached the container.
* **The loop that ran** -- run the smoke test with the locator: add
  `PYTHONPATH=$ND_REPO/scripts/locate` (`nd run --env`, or `LOCATE=1` on a spec wave) to
  the env that reaches the trainer (the same route as `NAN_*`). It hooks
  `Optimizer.step` and `backward` -- and, for inference, top-level forwards with grad
  disabled (each forward of a `generate()`) -- and names the loop around them; `capture_digest.py`
  shows its `[op-nan-locate]` lines:
  * `OK: the wired loop ...` -- done;
  * `NEXT: wire_capture.py apply <file> --loop-line N` (beside `CAPTURE NEVER
    CONSTRUCTED`, or a summary where another loop has more steps than the wired
    one) -- `wire_capture.py restore` the wrong file, apply the named one, re-run
    the smoke test;
  * `HOOK` -- a framework owns the loop: its callback (4.1);
  * no locator lines -- either `PYTHONPATH` did not reach the process (or it ran with
    `python -I`): fix the route, not the wiring; or the model has nothing the locator hooks
    (plain tensor code: no `nn.Module` forward, no `backward`, no optimizer -- 2026-11-20/21, an
    inference script of bare functions): then judge the wiring from the capture's own lines
    (`armed`, `Progress step`, a capsule), not from the locator.

  Smoke test only, not the wave: the hooks cost a stack walk per step and graph-break
  a compiled optimizer step.

Separate smoke test (~15 steps, planned trials-per-GPU on the fewest GPUs) **only**
for: unknown memory headroom (a big model, snapshots near the cap), or a changed arming
point, filter or capture code you have reason to doubt. A first capture of a model that
fits easily needs none: the capture-to-reproducer job's preflight and first steps are the
smoke test, and a separate one only repeated them (2026-11-21: one extra minute, the same
capsule). **Otherwise the wave is the smoke test**
(a broken capture ends every capture trial in its first steps anyway): exclude
trials with the false-positive signature or an OOM; if most fail the same way, fix
and re-run. Ledger: which smoke test or wave validated which configuration.

## 4.3 Run length

* Cut to a few dozen steps -- the capture detects at the op, not at the model's late
  check; raise only if nothing is captured. A validated launcher or template spec keeps
  its own step count (e.g. 300): the wave's early stop ends every trial at the first
  settled capsule, so the length only bounds a run that catches nothing.
* **A NaN step already measured** (a step-2 wave, or an earlier wave of the same environment in
  `<LEDGER>`): length = about 1.5 x the latest NaN step seen, never below the floor below. With step 2
  folded (`WAIT_CONTROLS=1`) the job waits for every control, so a control that never NaNs runs the
  spec's full length: 200 requests where the NaN came at 49 (2026-11-19) would have been 75. Unmeasured:
  the run script's own length.
* **Floor: the control arm's first check window.** Below it every control reads "no
  NaN" by construction. Find the interval in the source (the 1st NaN checks at 998,
  1998, ... so `NUM_STEPS=1000` was the floor).
* Budget wall time from the capture's own step rate; log progress every
  few steps.
