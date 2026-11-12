# mad-nan-hunt

`mad-nan-hunt` is a Claude Code workflow (`.claude/workflows/mad-nan-hunt.js`). It runs [nan-doctor](#nan-doctor) on one MAD PyTorch model: it checks that clean runs NaN, captures the op that first produced the NaN with madengine's `nan_doctor` tool, replays that op without the model, and packages a standalone reproducer. `/mad-debug-nan` is the interactive version of the same method, with every nan-doctor step available (including the standalone fallback and fix proposals).

Plan mode prints the clone command, the intake and every run. It does not clone or start a GPU job. `--execute` does.

```bash
mad-nan-hunt --tag dummy_nan                      # plan
mad-nan-hunt --tag dummy_nan --execute            # run
mad-nan-hunt --tag <model> --repeats 3 --capture-data values --groups all \
  --rootcause --budget-minutes 60 -c "{'docker_env_vars': {'HF_TOKEN': '...'}}" --execute
```

| option | default | meaning |
|---|---|---|
| `--nan-doctor-home DIR` | `$HOME/nan-doctor` | nan-doctor checkout, outside the MAD repo |
| `--repeats N` | 1 (max 5) | clean runs before capturing |
| `--skip-detect` | off | capture without the clean runs |
| `--nan-signal ERE` | `\b(loss\|grad_norm)\b.*\b(nan\|inf)\b` | what a NaN looks like in the run log, case-insensitive |
| `--capture-data shapes\|values` | `shapes` | `values` records operand values (exact replays; larger capsules; never for private data) |
| `--groups largest\|all` | `largest` | capsule groups to replay |
| `--rootcause`, `--budget-minutes M` | off, 60 | nan-doctor step 7 within M minutes. Fixes are never automated |
| `-c '<ctx>'` | none | madengine additional context for every run; must not carry `tools`. A `slurm` key runs on SLURM; K8s is not supported yet |

## What execute does

1. **Resolve.** `madengine discover`; the tag must be exactly one model.
2. **Setup.** nan-doctor is pinned in the workflow (`NAN_DOCTOR_REPO`, `NAN_DOCTOR_SHA`). A missing checkout is cloned at that commit; a checkout at another commit is an error and is left alone. The workflow writes `intake.json` (nan-doctor's answers to "how does this model run") into `<nan-doctor-home>/artifacts/mad-<date>-<model>/` and validates it with nan-doctor's `intake.py`. That artifact directory sits in the nan-doctor checkout because the replay container mounts only the checkout and `$HOME`.
3. **Detect.** `--repeats` clean runs through `mad-run.sh`. A run NaN'd when `grep -ciE <signal>` over its log is above zero; the workflow, not an agent, decides. No NaN in any run stops the hunt.
4. **Capture.** One run with `"tools": [{"name": "nan_doctor"}]`. madengine's tool arms nan-doctor's capture in every rank without editing the model, and the run stops when a NaN outlives its step. The capsules are copied into the artifact directory. No capsule: one more attempt, then the hunt stops and points at `/mad-debug-nan`.
5. **Replay.** nan-doctor step 5 groups the capsules and replays the chosen groups under load in the image madengine built, through `nan-doctor-run.sh`. A group reproduced when its replay has hits; zero hits is "not reproduced" whatever verdict name an agent reports, and the hunt stops there.
6. **Package.** nan-doctor step 6 builds the reproducer tarball for the first reproducing group and checks it.
7. **Root-cause** (only with `--rootcause`). nan-doctor step 7, within the budget.
8. **Verify.** A separate agent re-reads the replay logs and must agree with every group's hits and verdict. A disagreement ends the hunt as `reproducer-unverified`.
9. **Report.** nan-doctor step 9 writes `report.md`; the report, ledger, intake and tarball are copied to `nan_hunts/mad-<date>-<model>/` (gitignored).

Final status: `planned`, `no-nan`, `no-capsule`, `not-reproduced`, `package-failed`, `reproducer-unverified`, or `reproducer`.

A run under the `nan_doctor` tool is not a benchmark: its `performance:` number is never a measurement. Nothing in MAD is edited, and a proposed fix is never applied.

## GPU jobs and the lock

Model runs go through `mad-run.sh` (madengine only). Replays go through `.claude/skills/mad-common/nan-doctor-run.sh`, which starts only nan-doctor's two docker launchers (`replay`: one command in the image; `node5b`: step 5b on every GPU) and accepts only their environment keys. Both helpers hold `/tmp/mad-gpu.lock`, so model runs, replays and GEAK queue instead of sharing GPUs.

## Validating the path

`dummy_nan` (`scripts/dummy`, tags `dummies`, `nan_debug`) is a one-GPU training loop that plants a NaN with `sqrt()` of a negative activation at `--plant_step` (default 20). `mad-nan-hunt --tag dummy_nan --execute` on an MI300X-class host should end with status `reproducer`.

## nan-doctor

nan-doctor finds where a NaN in a PyTorch workload on AMD Instinct GPUs is first produced and packages it as a standalone reproducer. Its `debug-nan` skill is the method every phase follows; madengine carries only its capture runtime (`scripts/common/tools/nan_doctor/`).
