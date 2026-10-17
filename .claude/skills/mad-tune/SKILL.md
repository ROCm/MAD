---
name: mad-tune
description: Tune a MAD training or inference model for better performance with a measure-change-measure loop. Use when the user wants to optimize, tune, or speed up an existing MAD model.
argument-hint: "<tag-or-model> [target: throughput|latency] [operator|dispatch|kernel]"
disable-model-invocation: true
context: fork
agent: mad-tuner
allowed-tools: Bash(madengine *) Bash(rocm-smi *) Bash(amd-smi *) Bash(bash *) Read Edit Grep Glob
---

Tune `$ARGUMENTS`.

The model tag is the first whitespace-delimited token of `$ARGUMENTS`. A later
token may be `throughput` or `latency`, and `operator`, `dispatch`, or `kernel`.
Use the tag token in every `madengine` command.

## Pre-flight
```!
bash ${CLAUDE_SKILL_DIR}/../mad-common/preflight.sh
```

## Task
1. `madengine discover --tags <tag>`. Classify the model as training
   (primus, pretrain, finetune, megatron, torchtitan, maxtext, torchtune) or
   inference (vllm, sglang, inference, serving). Record the clean baseline from
   `scripts/.../run.sh`, the config it references, and the current `perf.csv`
   row. Default target is `throughput`. Use `latency` when the user asks or the
   metric is milliseconds.
2. Profile once at the requested depth before changing anything. Default depth
   is `dispatch` (`rocm_trace_lite`), which works for training and inference.
   `operator` is TraceLens via `torch_profiler_dynolog` + `tracelens`
   (inference sets `TORCH_PROFILE_ITERATIONS=0`). `kernel` is one of
   `rocprofv3_compute`, `rocprofv3_memory`, `rocprofv3_communication`.
   Never combine `rocm_trace_lite` and `rocprofv3_*` in one run. A profiled
   perf number is not the baseline.
3. Propose levers for that family, changing ONE at a time so deltas are
   attributable:
   - training: batch, precision, TP/PP/EP, `RCCL_*`/`NCCL_*`, recompute,
     `PYTORCH_TUNABLEOP_ENABLED`;
   - inference: `gpu-memory-utilization`, `max-num-seqs`,
     `max-num-batched-tokens`, tensor-parallel size, precision, HIP graphs,
     attention backend;
   - kernel evidence, either family: `PYTORCH_TUNABLEOP_ENABLED`, hipBLASLt,
     the kernel flag the counters name.
4. Re-measure each change with a clean `madengine run` (`tools` removed). Keep
   improvements (`status == SUCCESS`) in the direction of the target; revert
   regressions.

If no AMD GPUs are present (`rocm-smi`/`amd-smi` absent), do NOT execute — produce a
ranked list of candidate changes with rationale and the exact `madengine run` command
to test each, then stop.

Never alter the `performance: <value> <unit>` output contract. Report: family,
baseline, the profile evidence, each change tried with its measured effect, and
the final recommended config with before/after numbers.

Primus projection is not part of this loop. For a multi-depth search, use the
`mad-tune-search` workflow. Pass `--expect` on that workflow only when you want
projected tokens/s for a Primus training model.
