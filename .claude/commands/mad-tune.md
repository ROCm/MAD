---
description: Tune a MAD model for better performance with a measure-change-measure loop
argument-hint: <tag-or-model> [target: throughput|latency] [lever hints]
---

Tune `$ARGUMENTS`.

Use the `mad-tuner` subagent. It should:
1. Establish the baseline (current `run.sh`/config + `perf.csv` row).
2. Propose tuning levers (env vars like `MAD_MODEL_BATCH_SIZE`,
   `PYTORCH_TUNABLEOP_ENABLED`, `NCCL_*`/`RCCL_*`; or args like tensor-parallel
   size, precision, gpu-memory-utilization), changing ONE at a time.
3. Re-measure each change; keep improvements, revert regressions.

If no GPUs are present, produce a ranked list of candidate changes with rationale
and the exact `madengine run` command to test each, then stop. Report baseline,
changes tried with measured effect, and the final recommended config with
before/after numbers.
