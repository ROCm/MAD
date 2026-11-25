# Environment: madengine (MAD)

One environment: the model is a MAD model that `madengine run` launches (local Docker, or SLURM where madengine itself launches the workload). Every command runs from the MAD repo root (`<MAD_ROOT>`), where `models.json` lives and `perf.csv` is written. `<ARTIFACT_DIR>` is `nan_hunts/<hunt>/` there (gitignored; the intake names it): never write into `scripts/` or any tracked file -- MAD's workflows stop on a dirty tree.

| work | how | why |
|---|---|---|
| **the model** (step 2 runs, step 4 capture) | `madengine run`, started detached through `mad-run.sh` | madengine owns the image, data, credentials and SLURM launch; the helper holds MAD's host GPU lock |
| **everything that needs torch but not the model** (5a, 5b, 5N, 6, 7, 8) | `madengine nan exec|group|replay|package`, long ones through `mad-run.sh` | they replay capsules in the image madengine built: its `docker_image` in `build_manifest.json` |
| **bookkeeping** (`ledger`, `intake`, `summary`, `report`, `verdict`, `readme`, `arms`, `diff`, gates) | `madengine nan <command>` directly | standard library only, no GPU |

`<IMAGE>` = the model's `docker_image` in `build_manifest.json` (written by `madengine build` or the first `madengine run`; `madengine nan` also takes `--manifest build_manifest.json [--model NAME]` instead of `--image`). `<GPU_EXEC>` = `madengine nan exec --image <IMAGE> [--gpus 0|all|0,1] [--mount HOST[:CONT]] [--env K=V] --`: it mounts the working directory at its own path, so every `<ARTIFACT_DIR>` path is the same on the host and in the container; the `nan_doctor_rt` package is on `PYTHONPATH`. Default `--gpus 0`: a replay is reproducible only if it lands on the same device.

## Running the model

Never in the foreground (a tool call ends after 10 minutes). One run id per run:

```bash
bash .claude/skills/mad-common/mad-run.sh start <run-id> -- madengine run --tags <tag> --live-output --additional-context '<CTX>'
bash .claude/skills/mad-common/mad-run.sh wait <run-id>     # repeat until succeeded|failed
```

`<CTX>` is the intake's `ADDITIONAL_CONTEXT` (`docker_env_vars`, `slurm`, ...) plus, per step:

* **clean run (step 2, controls):** unchanged -- no `tools`. A NaN = the run log (`.mad-runs/<run-id>/log`) matches `<NAN_SIGNAL>`; `failed` alone is not a NaN.
* **capture (step 4):** add `"tools": [{"name": "nan_doctor"}]`, and capture knobs in `docker_env_vars` (merged with the intake's), e.g. `{"NAN_CAPTURE_DATA": "1", "NAN_CAPTURE_ARM_AFTER_STEPS": "1"}`. The tool sets `NAN_CAPTURE_DIR`, `NAN_CAPTURE_AUTO=optimizer` and `NAN_CAPTURE_CONTAINERS_PER_GPU=1` itself; **do not wire the source** -- a global optimizer-step hook arms the capture in every rank (`nan_doctor_rt/autocapture.py`: it cannot see the first iteration, GradScaler-skipped steps, or loops with no optimizer; then `step4-capture.md`'s source wiring). Every other `NAN_CAPTURE_*` knob of `step4-capture.md` applies unchanged.

* **a `nan_doctor_rt` directory on `PYTHONPATH`** (vLLM TP / graphs, SGLang, Inductor capsule, `diff_tap`, `locate`, `inductor_override`): the absolute in-container path in `docker_env_vars`, `"PYTHONPATH": "/myworkspace/scripts/common/tools/nan_doctor/nan_doctor_rt/<dir>"` -- madengine puts its runtime under the workspace's `scripts/common/tools/` during a run, and `/myworkspace` is `<MAD_ROOT>`.
* **differential arms (step 2, no signal):** one run per arm, no `tools`; `madengine nan arms arms ... --format context --out-dir <DIR>` prints each arm's context (`step2-diff.md`).

A run under the tool is **not** a performance measurement: never report its `performance:` line as a number.

## Where the capsules land

| target | path on the host |
|---|---|
| local | `<MAD_ROOT>/nan_doctor_output/` |
| SLURM | `<output dir>/<model>/<job tag>/node_<N>/nan_doctor_output/` (one per node, copied also when the node failed) |

Each `nan_doctor_output/` holds `capsules/` (`rank<N>_*.pt`, plus `manifest.json` for world size > 1) and `summary.txt` (`capsules=`, `manifest=`, `host=`). The `[op-nan]` lines are in the run log (`.mad-runs/<run-id>/log` locally; `node_<N>/stdout.out` on SLURM). **Copy** (not move) every `capsules/` into `<CAPSULE_DIR>/<run-id>/` the moment the run ends and ledger the count per node: the next madengine run with the tool moves a non-empty `capsules/` aside to `capsules.prev-<time>/` and its `summary.txt` is replaced, so the run a capsule came from is only known from the copy. `madengine nan group <CAPSULE_DIR> --recursive` groups across run subdirectories.

## Geometry and contention

madengine runs ONE job per invocation: there are no trials, no controls and no early-stop wave. A "wave" is N sequential `mad-run.sh` runs (`waves.md`); controls are runs without the tool. Containers per GPU = 1 (the GPU lock serialises runs; the tool declares `NAN_CAPTURE_CONTAINERS_PER_GPU=1`): ledger `CONTENTION=1 (madengine, one job per host)`. Step 3 (raising the rate by oversubscribing) fights the lock: do it only interactively, with the user's agreement, outside `mad-run.sh`, and say so.

## Replays and packages

```bash
bash .claude/skills/mad-common/mad-run.sh start <run-id> -- madengine nan group <CAPSULE_DIR> --manifest build_manifest.json --out-dir <ARTIFACT_DIR> --recursive
bash .claude/skills/mad-common/mad-run.sh start <run-id> -- madengine nan replay <CAPSULE_FILE> --manifest build_manifest.json --out-dir <ARTIFACT_DIR>/replay/<GROUP>
bash .claude/skills/mad-common/mad-run.sh start <run-id> -- madengine nan package --group <GROUP> --artifact-dir <ARTIFACT_DIR> --image <IMAGE> --capsule <CAPSULE_FILE> --attempt <attempt dir> --groups <OUT_JSON_PATH> --pack --ledger <LEDGER> --step 'step6[<GROUP>]'
```

then `mad-run.sh wait <run-id>`. A shipped package's own script -- the extracted reproducer's `run.sh`, a fix's `validate.sh` -- is verified as a vendor runs it: `mad-run.sh start <run-id> -- bash <ARTIFACT_DIR>/verify-<G>/repro-<G>/run.sh` (or `.../verify-fix/fix/validate.sh`) runs it from its own directory in an empty environment (HOME, USER, PATH and the locale only), so what is verified is what ships. Any other payload: `mad-run.sh start <run-id> -- madengine nan exec --image <IMAGE> --gpus <g> -- python3 -m nan_doctor_rt.<module> ...`. Never run `docker` or `replay_nan_op` bare: the replay would share GPUs with a madengine run, and its rate would be measured under contention nobody declared. The image must be on the host (`docker images <IMAGE>`); `--pull` fetches it from the registry the build used.

## SLURM

The submission host usually has no GPU and no docker. `madengine run` (clean and capture) goes through madengine's SLURM deployment as usual and each node writes its own `nan_doctor_output/`. Run `group`, `replay` and `package` on ONE GPU node that has docker and the image (e.g. `srun -N1 --gpus-per-node=8 madengine nan replay ...` from the MAD root on the shared filesystem); that is site-specific, so ask the user how GPU nodes are reached, and ledger it. Models with their own multi-node launchers (`slurm_multi`) do not get the tool on their ranks: capture is not available for them.

## Job hygiene

* `mkdir -p` an output directory before a command writes into it; a missing directory loses the output silently.
* Never name a helper script after a standard-library module (`inspect.py`, `json.py`, `types.py`): the container puts the script's directory first on `sys.path`, so `import torch` dies in a circular import.
* The container runs as another uid: `chmod -R 777 <ARTIFACT_DIR>` (`madengine nan exec --as-user` runs as the host user with `HOME=/tmp` instead).
* All-`ERROR` copies: `docker events --since <t> --filter event=die` shows `signal=9` / `exitCode=137` for an external kill. Containers per device in a live replay: `docker ps -q | xargs -r -n1 docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^HIP_VISIBLE_DEVICES=' | sort | uniq -c`.
* Never extract a reproducer tarball under `/tmp` to verify it: use a sibling directory of `<ARTIFACT_DIR>`.
