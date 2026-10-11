---
name: primus-projection
description: Project memory fit and tokens/s/GPU for a MAD Primus training model only. The argument must be a Primus training tag whose config lives under scripts/Primus. Do not use for vLLM, SGLang, diffusion, or any non-Primus model.
disable-model-invocation: true
allowed-tools: Bash(madengine discover *) Bash(bash scripts/Primus/primus-cli *) Bash(bash scripts/Primus/runner/primus-cli *) Bash(python3 -m primus.agents.tuning_agent *) Read Grep Glob
---

This is the Expect step for `mad-tune-search`. It supplies expected numbers for a
Primus training config. It does not run a MAD benchmark and it does not write
`perf.csv`.

## When to run
The tag is the first whitespace-delimited token of `$ARGUMENTS`. It must be a
Primus training model: `madengine discover --tags <tag>` shows `--config_path`
under `scripts/Primus/examples/**/*.yaml`, and that YAML's
`modules.pre_trainer.framework` or `modules.post_trainer.framework` is
`megatron`, `torchtitan`, `maxtext` (alias `jax`), or `torchrec_dlrm`.

If the tag is anything else, including vLLM, SGLang, diffusion, serving, or a
non-Primus trainer, stop. Say that `/primus-projection` applies only to Primus
training models, and do not call `primus-cli`. Do not emit a projection record.

`mad-tune-search` calls this same check when the user passed `--expect`. On a
non-Primus tag that workflow records `source: skipped` and continues the
profiler search. The slash command itself is not a way to probe other models.

## Steps
1. Resolve `<tag>` to the config path above. If `scripts/Primus` is missing,
   return `source: skipped` and name that as the reason.
2. When `scripts/Primus/skills/primus-projection/SKILL.md` exists, read it and
   use its command shapes. Do not invent flags it does not list.
3. From `scripts/Primus`, run memory projection, then performance projection
   with `--profiling-mode simulate`, using the resolved config. Prefer
   `primus-cli` at the repo root; fall back to `runner/primus-cli` on older
   trees.
4. Run `python3 -m primus.agents.tuning_agent` only when the user asked for the
   tuning agent. `--seed-only` is the form that returns a ranked legal grid
   without a second LLM. Skip the agent otherwise.
5. Return the record below. Leave `perf.csv` unchanged.

## Record
- `source`: `projection`, `planned` (commands printed, not run), or `skipped`
- `fits`: whether projected memory fits the GPU
- `memoryPerGpuGb`
- `expectedTokensPerSecPerGpu`
- `expectedTflops` (null when the tool does not report it)
- `recommended`: `{tp, pp, ep, cp, mbs, gbs, schedule, recompute}` or null
- `reason`: one line, including the skip reason when `source` is `skipped`
