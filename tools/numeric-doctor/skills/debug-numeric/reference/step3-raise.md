# Step 3: raise the NaN rate (optional, off by default)

**Off by default**: runs only when the user asks for it. Otherwise `$LG skip step3 "off by
default (<NAN_RATE>)"` and go to step 4; a low `<NAN_RATE>` (step 2 saw no NaN or under ~1 in
8): say in one line that the capture may need longer or more trials and that they can ask to
raise the rate first -- don't wait for an answer. In ask-me mode (`step1-ask.md`) a low rate
gets the question instead: "raise the NaN rate first?" (default: no).

* **Skip when `<NAN_RATE>` is already high** (extra
  load risks an all-`TIMEOUT` sweep). `$LG skip step3 "<rate, environment>"` --
  it expires on a new host, image, geometry or a lower re-measurement.
* **A low rate is a step-3 problem before a hardware one**: try the levers on
  this node before calling it unsuitable.
* Otherwise (asked for), run Increase-NaN-rate.
* `<FULL_MODEL_RUN_SCRIPT>` / `<FULL_MODEL_RATE>` = the step-3 versions if it ran,
  else the originals.
* Lever was concurrent load (`hogger.py`, fanned trials)? Ledger it; take that load
  off before step 7 or pin `--mutate-timeout` (the recipe sweep's own `--contend` co-tenants
  stay).

## Increase-NaN-rate

1. Levers, strongest first:
   * **Oversubscribe -- more containers than GPUs** (16 on 8, not 8 on 8;
     the stressed and unstressed launchers it was measured with differ only in trials
     1 -> 16 and `NUM_STEPS` 1000 -> 3000).
   * **More steps** -- buys check windows, not likelier windows; if NaNs land in
     window 1, use concurrency.
   * **Concurrent load** (`scripts/hogger.py`).
   * **Right-size host threads**: `OMP_NUM_THREADS` ~ cores / concurrent processes,
     computed from the launcher's trial count. Raised one race's rate sharply;
     direction unpredictable -- A/B within one wave; ledger the value.
   * **Flip `HSA_DISABLE_CACHE`** (0 <-> 1) -- cheap, was inert.
   * NaN-injected image -- only as a known-answer validation.
   * **A multi-node run (`<NODES>` > 1): copies of the same job on the same nodes** --
     `rate.sh` with `COPIES=k` runs k copies of the `<NODES>`-node job at once on one group
     of nodes (each its own `MASTER_PORT`, `RUN_OUT`, log): k processes per GPU, the
     multi-node form of oversubscribing. Every copy is a scored run, so `RATE_RESULT` is the
     rate at k per GPU -- compare it with the `COPIES=1` rate; memory must hold k copies
     (start at 2). If it is higher, the capture runs that way too: k copies of the spec's
     `all` line on the same nodes, and Declare-contention = k (`NAN_CAPTURE_CONTAINERS_PER_GPU`;
     `wave.sh` sets it from the line count).
     **Copies run without apptainer `--pid`** (`rate.sh` and `wave.sh` drop it themselves,
     `lib.sh nd_drop_pid_ns`): with a PID namespace per container the copies' ranks share
     PIDs and RCCL's intra-node P2P setup dies -- `hipIpcGetMemHandle failed : invalid
     argument`, HIP FATAL, in seconds (a probe; the same two copies without `--pid`
     trained and NaN'd). The capture's own process scan still records `gpu_procs=1`
     per rank without `--pid`, so the declared k is what carries the contention
     to the replay -- never declare 1 for copies. Validated 2026-11-12 on Slurm + apptainer, qwen3 MoE
     `run.sh` command, 2 nodes: `rate.sh COPIES=2` 4/4 runs, none died, ~1.5 s/step vs ~1.2
     solo; capture with 2 `all` lines: both copies captured (`cap=2/2`, 4
     capsules in `t0`, `t1`), capsules record `containers_per_gpu=2`.
     Docker/podman copies are untested (each container also has its own PID namespace).

   One lever at a time where possible (apply the first two to both arms, A/B a third).
2. **Fixed container names break fan-out**: a launcher that sets `CONTAINER="numeric-doctor"` once and
   fans out: n-1 trials fail "name is already in use" (`ERROR`, or not-NaN) and the trap reaps
   one name. Use `"${CONTAINER}-${trial}"` and loop the trap; same for anything
   shared per trial.
3. Write `<MORE_NANS_RUN_SCRIPT>` as a new file; **re-run Declare-contention** on it.
4. Run; `<MORE_NANS_RATE>`. Not higher than `<NAN_RATE>`: say so, keep the original.
