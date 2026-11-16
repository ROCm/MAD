# Step 2: measure the rate (optional)

Always optional: **ask** whether to measure, skip only with agreement, ledger the
answer. Don't cite or recommend from earlier runs' rates (script comments, old
ledgers) when asking.

* measure: run the model (`madengine run` via `mad-run.sh`) to confirm it NaNs and
  measure the rate (below);
* skip: `<NAN_RATE>` *unmeasured* -- **never** a rate carried over from a previous
  run or supplied by the user in its place. Skip step 3 too; step 4's controls
  become the first rate -- don't cut step 4 below the model's check window
  (`step4-capture.md` 4.3).

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
