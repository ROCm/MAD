# mad-nan-hunt

`mad-nan-hunt` is a Claude Code workflow (`.claude/workflows/mad-nan-hunt.js`). It hunts a NaN in one MAD PyTorch model: it checks that clean runs NaN, captures the op that first produced the NaN with madengine's `nan_doctor` tool, replays that op without the model, and packages a standalone reproducer. `/mad-debug-nan` is the interactive version of the same method, with every step available (including the standalone fallback and fix proposals). The deterministic parts are madengine's `madengine nan` commands (`docs/nan-doctor.md` in madengine); this workflow and the skill are the agent layer on top.

Plan mode prints the intake and every command. It does not create files or start a GPU job. `--execute` does.

```bash
mad-nan-hunt --tag dummy_nan                      # plan
mad-nan-hunt --tag dummy_nan --execute            # run
mad-nan-hunt --tag <model> --repeats 3 --capture-data values --groups all \
  --rootcause --budget-minutes 60 -c "{'docker_env_vars': {'HF_TOKEN': '...'}}" --execute
```

| option | default | meaning |
|---|---|---|
| `--repeats N` | 1 (max 5) | clean runs before capturing |
| `--skip-detect` | off | capture without the clean runs |
| `--nan-signal ERE` | `\b(loss\|grad_norm)\b.*\b(nan\|inf)\b` | what a NaN looks like in the run log, case-insensitive |
| `--capture-data shapes\|values` | `shapes` | `values` records operand values (exact replays; larger capsules; never for private data) |
| `--groups largest\|all` | `largest` | capsule groups to replay |
| `--rootcause`, `--budget-minutes M` | off, 60 | step 7 within M minutes. Fixes are never automated |
| `-c '<ctx>'` | none | madengine additional context for every run; must not carry `tools`. A `slurm` key runs on SLURM; K8s is not supported yet |

Requires madengine with the `nan` command group and the `nan_doctor` tool.

## What execute does

1. **Resolve.** `madengine discover`; the tag must be exactly one model.
2. **Setup.** Creates `nan_hunts/mad-<date>-<model>/` (gitignored), writes `intake.json` (the answers to "how does this model run") and validates it with `madengine nan intake`, and starts the ledger.
3. **Detect.** `--repeats` clean runs through `mad-run.sh`. A run NaN'd when `grep -ciE <signal>` over its log is above zero; the workflow, not an agent, decides. No NaN in any run stops the hunt.
4. **Capture.** One run with `"tools": [{"name": "nan_doctor"}]`. The tool arms the capture in every rank without editing the model, and the run stops when a NaN outlives its step. The capsules are copied into the artifact directory. No capsule: one more attempt, then the hunt stops and points at `/mad-debug-nan`.
5. **Replay.** `madengine nan group` then `madengine nan replay` (step 5) replay the chosen groups under load in the image madengine built. A group reproduced when its replay has hits; zero hits is "not reproduced" whatever verdict name an agent reports, and the hunt stops there.
6. **Package.** `madengine nan package` builds the reproducer for the first reproducing group and the run verifies it from an extracted copy.
7. **Root-cause** (only with `--rootcause`). Step 7, within the budget.
8. **Verify.** A separate agent re-reads the replay logs and must agree with every group's hits and verdict. A disagreement ends the hunt as `reproducer-unverified`.
9. **Report.** `madengine nan report` writes `report.md` into the artifact directory.

Final status: `planned`, `no-nan`, `no-capsule`, `not-reproduced`, `package-failed`, `reproducer-unverified`, or `reproducer`.

A run under the `nan_doctor` tool is not a benchmark: its `performance:` number is never a measurement. Nothing tracked in MAD is edited, and a proposed fix is never applied.

## GPU jobs and the lock

Model runs and the GPU-using `madengine nan exec|group|replay|package` commands all start through `.claude/skills/mad-common/mad-run.sh`, which holds `/tmp/mad-gpu.lock`: model runs, replays and GEAK queue instead of sharing GPUs. `madengine nan` itself takes no lock.

## Validating the path

`dummy_nan` (`scripts/dummy`, tags `dummies`, `nan_debug`) is a one-GPU training loop that plants a NaN with `sqrt()` of a negative activation at `--plant_step` (default 20). `mad-nan-hunt --tag dummy_nan --execute` on an MI300X-class host should end with status `reproducer`.
