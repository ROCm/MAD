# Step 2: measure the rate (optional)

Always optional, default skip (step-1 question 2): it runs when the user asks for it
(or ask-me mode chose it), ledger the answer; skipped: `$LG skip step2 "<why>"` and `$LG skip step3 "no rate to raise"` (no
`start`: a skipped step has no visit). Don't cite or recommend from earlier runs' rates (script comments, old
ledgers). No `<NAN_SIGNAL>` at all (silently wrong outputs): differential mode first
(`step2-diff.md`) -- its check mode is then `<NAN_SIGNAL>`.

* measure: run the model (`madengine run` via `mad-run.sh`, no `nan_doctor` tool) to
  confirm it runs and NaNs and measure the rate (below). If step 1 had to add a NaN
  check to a working copy, run that copy; on a failure, run the original once too, so
  the failure is pinned on the right one;
* skip: `<NAN_RATE>` *unmeasured* -- **never** a rate carried over from a previous
  run or supplied by the user in its place. Skip step 3 too; step 4's controls
  become the first rate -- don't cut step 4 below the model's check window
  (`step4-capture.md` 4.3).

## The model does not run

**Every** scored run failed -- `madengine nan waves` scores all runs `ERROR` / `DIED` --
and it is not the environment the skill broke: not the external-kill signature
(`waves.md`), not a failed `<GPU_EXEC>` check (`TORCH_GPU=NO`), not a missing image. Some
runs failing and some running is not this (an odd GPU or contention: *Partition by GPU*,
below). Then **stop the ladder**: no step 3, no capture. Ledger `MODEL_DOES_NOT_RUN <host>
<the error line>` and end the turn with this block, filled in, and nothing else:

```markdown
> **⚠️ The model does not run here -- even without the NaN capture.**
> `<MODEL_TAG>` failed in every run (<n> of <n>) before producing a result:
>
>     <the 3-5 lines of the error, from the log>
>
> Command: `<the madengine run command step 2 ran>` on `<host>` (image `<image>`). This
> is not the NaN, and the skill cannot debug a model that does not run. **Please first
> make sure `madengine run --tags <MODEL_TAG>` works by itself, without the `nan_doctor`
> tool**, then tell me to continue. If it does run for you, tell me how you ran it (host,
> image, additional context): then the problem is in how I launched it, and I'll fix that.
```

The user says it works by itself: compare their way of running it with the run's
`--additional-context` (`env-madengine.md`), fix the difference, and run step 2 again (a
new `start`). The same block when the step-4 clean runs all fail this way.

## Measure a rate, not "it NaN'd"

* **>= 8 runs (one wave: sequential `mad-run.sh` runs, `waves.md`)**; one run
  (m = 1) is not a rate -- if the user insists, mark it provisional.
* **k <= 1: run a second wave and pool** (1/4 and 1/24 are the same single event).
* **Record the environment**: host, image digest, step count, containers per GPU,
  host threads per process (`OMP_NUM_THREADS`/`torch.get_num_threads()`; unset =
  all cores *per process*) vs cores and process count, `rocm-smi` Perf level and
  SCLK (`perf_determinism` != `auto`).
* **Host-thread oversubscription**: N processes x all cores. Tell: CPU-bound
  phases (data gen, init, optimizer setup) much slower with load than alone. It also
  changes how fast each process feeds its GPU.
* **Partition by GPU before quoting a denominator.** Devices disagree: re-run
  pinned to the good ones for the denominator; probe odd ones alone (no partner) to
  separate bad GPU from bad-under-contention.
* **Drop runs that lost their co-tenant**: the tell is a late step-rate jump, not the
  verdict (`madengine nan waves` flags `ORPHAN?`); note the drop.
* Run with step 1's declared containers per GPU.
