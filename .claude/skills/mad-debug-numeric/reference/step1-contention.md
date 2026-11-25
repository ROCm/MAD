# Step 1: declare contention, the long form

Read when the capture run is not the plain case in `step1-ask.md` (one container per
GPU, runs started one after another).

Produces N = concurrent containers on the busiest **device**, injected as
`NAN_CAPTURE_CONTAINERS_PER_GPU` -- the one fact the capture cannot see itself.
Derive it from the model's run script in front of you, never from an example.

1. Read the model's run script (and the `madengine run` command that starts it):

   | find | typical shape |
   |---|---|
   | run count, and **concurrent?** | several `madengine run` invocations started at once, or a script that launches containers with `&` and `wait` |
   | device pinning | `HIP_VISIBLE_DEVICES=$gpu`, `--gpus device=$gpu`, `gpu=$((trial % 8))` |
   | containers per run | launches inside the loop body |

   Sequential runs = N is 1, nothing to declare. `gpu=$((trial % NUM_GPUS))` ->
   N = `ceil(trials / NUM_GPUS)` (16 over 8 -> 2), `NUM_GPUS` from the script.
2. Uneven (10 over 8): take the max, note the unevenness in `<LEDGER>`; 5a can
   correct from the capsule's `visible_devices`.
   **Trials inside one container** (one container per GPU running `--trials N`,
   `mp.spawn`, `torch.multiprocessing`): N is **1** -- the capture counts its sibling
   processes itself under `--pid` (`gpu_procs=N`). Capture with whole containers per
   arm (capture GPUs vs control GPUs), one log per container.
3. Unsure: ask the user ("how many copies of this container end up on one GPU?") or
   count a live wave. From `rocm-smi`, trust the per-device *count*, not the device
   *index*.
4. Inject it into the capture run's environment, **never into the model's own run
   script**, beside the existing per-trial variable (`HIP_VISIBLE_DEVICES`); see
   `env-madengine.md` for how the run's environment is set. Prefer computing it from
   the loop bounds.
5. Verify it landed in the container; the durable proof is the replay's `contention:`
   banner `N process(es) in this container x M container(s) per GPU`.
6. Ledger: N and how derived.
