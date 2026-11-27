# Step 1: declare contention, the long form

Read when the launcher is not the simple shape in `step1-ask.md`.

Produces N = concurrent containers on the busiest **device**, injected as
`NAN_CAPTURE_CONTAINERS_PER_GPU` -- the one fact the capture cannot see itself.
Derive it from the script in front of you, never from an example.

1. Read `<RUN_SCRIPT>`:

   | find | typical shape |
   |---|---|
   | trial count, and **concurrent?** | `for trial in $(seq 0 "$last_trial_num")`, `&` on the launch, `wait` |
   | device pinning | `-e HIP_VISIBLE_DEVICES=$gpu`, `--gpus device=$gpu`, `gpu=$((trial % 8))` |
   | containers per trial | launches inside the loop body |

   No `&` = sequential = N is 1, nothing to declare. `gpu=$((trial % NUM_GPUS))` ->
   N = `ceil(trials / NUM_GPUS)` (16 over 8 -> 2), `NUM_GPUS` from the script.
2. Uneven (10 over 8): take the max, note the unevenness in `<LEDGER>`; 5a-load can
   correct from the capsule's `visible_devices`.
   **Trials inside one container** (one container per GPU running `--trials N`,
   `mp.spawn`, `torch.multiprocessing`): N is **1** -- the capture counts its sibling
   processes itself under `--pid` (`gpu_procs=N`). Capture with whole containers per
   arm (capture GPUs vs control GPUs), one log per container; template:
   `examples/training/qwen_train_nan/capture.spec`.
3. Unsure: ask the user ("how many copies of this container end up on one GPU?") or
   count a live wave (`env.md`: `docker ps`, `rocm-smi --showpids`). From `rocm-smi`, trust the per-device *count*,
   not the device *index*.
4. Inject into the working copy, **never into `<RUN_SCRIPT>`**, beside the existing
   per-trial variable (`HIP_VISIBLE_DEVICES`): `-e NAN_CAPTURE_CONTAINERS_PER_GPU=2`,
   or `export` before a wrapper that forwards env. Prefer computing it from the loop
   bounds.
5. Verify it landed (a spec wave sets it per GPU itself); the durable proof is the
   replay's `contention:` banner `N process(es) in this container x M container(s)
   per GPU`.
6. Ledger: N and how derived.
