# Step 0's gates: not PyTorch, or not a tested GPU

Read only when a gate did not pass: `framework_check.py` said `SKIP` or `ASK`, or
`gpu_check.py` said `UNTESTED` (in step 0, or in step 1's `<ENV>` check). Step 0 is
internal: report what a gate found, never "step 0".

**PyTorch gate -- the first thing done with `<RUN_SCRIPT>`, before anything is
created.** Every rung hooks PyTorch (the capture is a `__torch_dispatch__` mode, a
capsule is saved ATen ops, the replay re-dispatches them); a non-PyTorch model yields
no capsule and no replay, only empty waves. As soon as `<RUN_SCRIPT>` is known --
already in step 0 if the user's request names it (then gate before printing
anything), else right after step 1's question 1 -- run, on the login node:

```bash
python3 <REPO_ROOT>/scripts/framework_check.py <RUN_SCRIPT> [--source <model.py>]
```

Static (no torch, no GPU); last line `FRAMEWORK=<f> VERDICT=<v>`:

* `RUN` (`pytorch`) -- continue; step 1 ledgers the line.
* `SKIP` (`jax`, `tensorflow`, ...) -- **do not apply the skill**: no artifact dir,
  ledger, wave or job. End the turn with this warning, filled in, and nothing else
  of the skill:

  ```markdown
  > **⚠️ debug-numeric skipped: `<RUN_SCRIPT>` is not a PyTorch program** -- it runs
  > <framework> (<the evidence line, e.g. `entry: jax (jax, flax.linen, optax)`>).
  > debug-numeric works only on PyTorch models: it captures the failing op through
  > PyTorch's dispatcher (`__torch_dispatch__`), saves it as PyTorch tensors and
  > replays it through PyTorch, so on <framework> it would capture nothing and could
  > build no reproducer. Nothing was created or run.
  ```

  If the user insists, repeat why it cannot work; never wire the capture into a
  non-PyTorch model.
* `ASK` (`mixed(...)` -- e.g. torch only loads the data -- or `unknown`, e.g. the
  model runs inside an image the script cannot read) -- one question: "Is the model's
  compute (forward/backward) written in PyTorch?" with the evidence lines. Yes ->
  continue and ledger the answer; no -> the `SKIP` warning above.

**GPU gate -- warn on anything below MI300X.** The tooling assumes ROCm and the
GFX9-family ISA of AMD Instinct MI300X and newer; other GPUs are untested. Step 0
runs, on this host, in the same call as the `<REPO_ROOT>` search:

```bash
python3 <REPO_ROOT>/scripts/gpu_check.py     # no torch needed
```

Last line `GPU=<name> ARCH=<arch> VERDICT=<v>` (no GPU here: `GPU=none VERDICT=NO_GPU`, exit 5 --
a login node; the check is repeated in a job in step 1):

* `SUPPORTED` (gfx942 MI300X/MI300A/MI325X, gfx950 MI350X/MI355X, gfx1250 MI455X, newer
  Instinct MI<n>, n >= 300) -- nothing to say; step 1 ledgers the line.
* `UNTESTED` (MI250X/MI210, Radeon, NVIDIA, ...) -- print this warning, filled in,
  **at the top of the first output** (above the first lines and question 1), then continue:

  ```markdown
  > **⚠️ Untested GPU: <name> (<arch>).** debug-numeric is tested only on AMD Instinct
  > MI300X or newer (gfx942, gfx950, gfx1250). All other GPU types are untested, and the skill
  > is likely not to work on them: the capture, replay and kernel tools assume ROCm
  > and the MI300X-class instruction set. Results here may be wrong or missing.
  ```

  Ledger it (`$LG note step1 "GPU UNTESTED: <name> <arch>"`) once the ledger exists,
  and repeat it in the step-9 report.
* `NO_GPU` -- this host has none (a login node): the verdict comes from step 1's
  `<ENV>` check, which runs `gpu_check.py --torch` on the model's GPUs before
  any wave; an `UNTESTED` there prints the same warning before step 2 or 4 starts.
