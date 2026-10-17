---
name: primus-projection
description: Project memory fit and tokens/s/GPU for a MAD Primus training model only. The argument must be a Primus training tag whose config lives under scripts/Primus. Do not use for vLLM, SGLang, diffusion, or any non-Primus model.
disable-model-invocation: true
allowed-tools: Bash(madengine discover *) Bash(bash scripts/Primus/primus-cli *) Bash(bash scripts/Primus/runner/primus-cli *) Bash(python3 -m primus.agents.tuning_agent *) Bash(docker images *) Bash(docker run --rm *) Bash(which torchrun) Bash(rocm-smi *) Read Grep Glob
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
3. Run from `scripts/Primus`. Prefer `bash primus-cli` at the repo root; use
   `runner/primus-cli` when only that file exists. If `which torchrun` fails on
   the host, run the same command in one local image from
   `docker images 'rocm/primus*'`:
   ```bash
   cd scripts/Primus && docker run --rm --device=/dev/kfd --device=/dev/dri \
     --group-add video --ipc=host --network=host --security-opt seccomp=unconfined \
     -v "$PWD":/workspace/Primus -w /workspace/Primus IMAGE \
     bash -c 'bash primus-cli direct --script primus/cli/main.py -- ARGS'
   ```
   If neither the host nor a local Primus image can run it, return
   `source: planned` and print the command. Do not install packages.
4. Run memory projection, then performance projection with
   `--profiling-mode simulate --gpu-arch <mi300x|mi325x|mi355x> --target-nodes 1`.
   Take `gpu-arch` from the config directory (`MI355X` → `mi355x`). If simulate
   fails, retry once with `--profiling-mode benchmark`. Benchmark uses the
   GPUs: check `rocm-smi --showuse` first, and if they are busy return
   `source: planned` instead of starting it.
5. Run `python3 -m primus.agents.tuning_agent` only when the user asked for the
   tuning agent. `--seed-only` is the form that returns a ranked legal grid
   without a second LLM. Skip the agent otherwise.
6. Return the record below. Leave `perf.csv` unchanged.

## Record
- `source`: `projection`, `planned` (commands printed, not run), or `skipped`
- `fits`: whether projected memory fits the GPU
- `memoryPerGpuGb`
- `expectedTokensPerSecPerGpu`
- `expectedTflops` (null when the tool does not report it)
- `recommended`: `{tp, pp, ep, cp, mbs, gbs, schedule, recompute}` or null
- `reason`: one line: the profiling mode used (simulate or benchmark) and where it ran (host or image), plus the skip reason when `source` is `skipped`
