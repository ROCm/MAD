# Step 3: raise the NaN rate (optional, off by default, interactive only)

Interactive only: it oversubscribes GPUs, so it must not run under MAD's GPU lock.
The automated workflow skips this step.

**Off by default**: runs only when the user asks for it. Otherwise `$LG skip step3 "off by
default (<NAN_RATE>)"` and go to step 4; a low `<NAN_RATE>` (step 2 saw no NaN or under ~1 in
8): say in one line that the capture may need longer or more runs and that they can ask to
raise the rate first -- don't wait for an answer. In ask-me mode (`step1-ask.md`) a low rate
gets the question instead: "raise the NaN rate first?" (default: no).

* **Skip when `<NAN_RATE>` is already high** (extra load risks an all-`TIMEOUT`
  sweep). `$LG skip step3 "<rate, environment>"` -- it expires on a new host,
  image, geometry or a lower re-measurement.
* **A low rate is a step-3 problem before a hardware one**: try the levers on this
  node before calling it unsuitable.
* Otherwise (asked for), run Increase-NaN-rate.
* `<FULL_MODEL_RUN_SCRIPT>` / `<FULL_MODEL_RATE>` = the step-3 versions if it ran,
  else the originals.
* Lever was concurrent load (`hogger`, fanned runs)? Ledger it; take that load off
  before step 7 or pin `--mutate-timeout` (the recipe sweep's own `--contend`
  co-tenants stay).

## Increase-NaN-rate

1. Levers, strongest first:
   * **Oversubscribe -- more containers than GPUs** (16 on 8, not 8 on 8).
   * **More steps** -- buys check windows, not likelier windows; if NaNs land in
     window 1, use concurrency.
   * **Concurrent load** (`madengine nan exec --image <IMAGE> -- python3 -m
     nan_doctor_rt.hogger ...`).
   * **Right-size host threads**: `OMP_NUM_THREADS` ~ cores / concurrent processes,
     computed from the trial count. Can raise a race's rate sharply; direction
     unpredictable -- A/B within one wave; ledger the value.
   * **Flip `HSA_DISABLE_CACHE`** (0 <-> 1) -- cheap, often inert.
   * NaN-injected image -- only as a known-answer validation.

   One lever at a time where possible (apply the first two to both arms, A/B a third).
2. **Fixed container names break fan-out**: a launcher that sets one container name
   makes n-1 concurrent runs fail "name is already in use" (`ERROR`, or not-NaN) and
   its trap reaps one name. Use a per-trial name (`"${CONTAINER}-${trial}"`) and loop
   the trap; same for anything shared per trial.
3. Write `<MORE_NANS_RUN_SCRIPT>` as a new file; **re-run Declare-contention** on it.
4. Run; `<MORE_NANS_RATE>`. Not higher than `<NAN_RATE>`: say so, keep the original.
