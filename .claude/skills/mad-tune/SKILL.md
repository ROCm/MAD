---
name: mad-tune
description: Tune a MAD model for better performance with a measure-change-measure loop. Use when the user wants to optimize, tune, or speed up an existing MAD model.
argument-hint: "<tag-or-model> [target: throughput|latency] [lever hints]"
disable-model-invocation: true
context: fork
agent: mad-tuner
allowed-tools: Bash(madengine *) Bash(rocm-smi *) Bash(amd-smi *) Bash(bash .claude/skills/mad-common/preflight.sh) Read Edit Grep Glob
---

Tune `$ARGUMENTS`.

## Pre-flight
```!
bash .claude/skills/mad-common/preflight.sh
```

## Task
1. Establish the baseline: read the model's `scripts/.../run.sh` and any config it
   references, plus its current `perf.csv` row if present. Record baseline perf + unit.
2. Propose tuning levers, changing ONE at a time so deltas are attributable:
   - env vars: `MAD_MODEL_BATCH_SIZE`, `PYTORCH_TUNABLEOP_ENABLED`, `NCCL_*`/`RCCL_*`,
     attention/backend flags;
   - args: tensor-parallel size, precision (fp16/bf16/fp8), sequence length,
     gpu-memory-utilization, max-num-seqs.
3. Re-measure each change. Keep improvements (`status == SUCCESS`); revert regressions.

If no AMD GPUs are present (`rocm-smi`/`amd-smi` absent), do NOT execute — produce a
ranked list of candidate changes with rationale and the exact `madengine run` command
to test each, then stop.

Never alter the `performance: <value> <unit>` output contract. Report: baseline, each
change tried with its measured effect, and the final recommended config with
before/after numbers.

For a deeper profiling-informed search across many candidates with adversarial
verification, use the `mad-tune-search` workflow instead.
