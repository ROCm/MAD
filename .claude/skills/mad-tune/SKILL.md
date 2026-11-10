---
name: mad-tune
description: Tune a MAD training or inference model for better performance with a measure-change-measure loop. Use when the user wants to optimize, tune, or speed up an existing MAD model.
argument-hint: "<tag-or-model> [target: throughput|latency] [operator|dispatch|kernel|assembly]"
disable-model-invocation: true
context: fork
agent: mad-tuner
allowed-tools: Bash(madengine discover *) Bash(madengine report *) Bash(rocm-smi *) Bash(amd-smi *) Bash(bash .claude/skills/mad-common/preflight.sh) Bash(bash .claude/skills/mad-common/mad-run.sh *) Bash(python3 .claude/skills/mad-common/kernel-tools.py *) Read Edit Grep Glob
---

Tune `$ARGUMENTS`.

The model tag is the first whitespace-delimited token of `$ARGUMENTS`. A later
token may be `throughput` or `latency`, and `operator`, `dispatch`, `kernel`,
or `assembly`.
Use the tag token in every `madengine` command.

## Pre-flight
Claude Code runs this before the skill starts. Other agents: run it first from
the repo root and stop if it fails.
```!
bash .claude/skills/mad-common/preflight.sh
```

## Task
1. `madengine discover --tags <tag> --json`. It must resolve to exactly one
   model; otherwise list the matches and ask which one. Classify the model as
   training (primus, pretrain, finetune, megatron, torchtitan, maxtext,
   torchtune) or inference (vllm, sglang, inference, serving). Read its
   registry entry (`scripts/<dir>/models.json`, or the directory's
   `get_models_json.py`), its run script, and the config it references.
   Measure a fresh clean baseline (an old `perf.csv` row may come from a
   different commit or image). Default target is `throughput`. Use `latency`
   when the user asks or the metric is milliseconds.
2. Profile once at the requested depth before changing anything. Default depth
   is `dispatch` (`rocm_trace_lite`), which works for training and inference.
   `operator` is TraceLens via `torch_profiler_dynolog` + `tracelens`
   (inference sets `TORCH_PROFILE_ITERATIONS=0`). `kernel` and `assembly`
   follow `.claude/skills/mad-common/kernel-analysis.md`: dispatch first for
   the top kernel, then one counter run of that kernel through
   `kernel-tools.py context`, then `kernel-tools.py card` and `asm`. The card's
   `bound` (compute, bandwidth, latency or unknown) is the evidence. If a
   counter run stalls (status `running`, GPU use 0% at every check for 10
   minutes), stop it with `mad-run.sh stop`, skip `card` on its output, and
   rebuild the context with `--serialize` for one retry; if that stalls too,
   say the bound is unknown (kernel-analysis.md). Never
   combine `rocm_trace_lite` and `rocprofv3_*` in one run. A profiled perf
   number is not the baseline.
3. Propose levers for that family, changing ONE at a time so deltas are
   attributable:
   - training: batch, precision, TP/PP/EP, `RCCL_*`/`NCCL_*`, recompute,
     `PYTORCH_TUNABLEOP_ENABLED`;
   - inference: `gpu-memory-utilization`, `max-num-seqs`,
     `max-num-batched-tokens`, tensor-parallel size, precision, HIP graphs,
     attention backend;
   - kernel evidence, either family: take levers from the "Levers by bound"
     table in `kernel-analysis.md` that match the card's `bound`. A latency
     kernel does not get a bandwidth lever. Library kernels (hipBLASLt,
     rocBLAS, CK) take `PYTORCH_TUNABLEOP_ENABLED` or hipBLASLt tuning. When
     `bound` is `unknown`, say so and fall back to the family levers.
4. Re-measure each change with a clean `madengine run` (`tools` removed). Run
   the baseline and each change at least twice and compare medians; a delta
   inside the run-to-run spread is noise. Keep improvements
   (`status == SUCCESS`) in the direction of the target; revert regressions.

Start every `madengine run` with
`bash .claude/skills/mad-common/mad-run.sh start <run-id> -- <command>` and
repeat `mad-run.sh wait <run-id>` until it reports `succeeded` or `failed`.
Never run it in the foreground: a tool call ends after 10 minutes.

If no AMD GPUs are present (`rocm-smi`/`amd-smi` absent), do NOT execute — produce a
ranked list of candidate changes with rationale and the exact `madengine run` command
to test each, then stop.

Never alter the `performance: <value> <unit>` output contract. Report: family,
baseline, the profile evidence (the kernel card line when you went to the
kernel depth), each change tried with its measured effect, and
the final recommended config with before/after numbers.

Primus projection is not part of this loop. For a multi-depth search, use the
`mad-tune-search` workflow. Pass `--expect` on that workflow only when you want
projected tokens/s for a Primus training model.

## Agent Notes
- Claude Code, Cursor and Codex all load this skill (Codex through
  `.agents/skills/`). Only Claude Code substitutes `$ARGUMENTS`; elsewhere it
  means the text the user gave after the skill name.
- Claude Code runs this skill in the `mad-tuner` subagent. In Cursor, delegate
  to the `mad-tuner` subagent (Cursor reads `.claude/agents/`); in Codex, spawn the
  `mad_tuner` agent (`.codex/agents/`). Otherwise run it inline and
  read `.claude/agents/mad-tuner.md` first for the conventions it assumes.
