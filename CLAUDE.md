# CLAUDE.md

@AGENTS.md

## Claude Code specifics

Everything above is shared with Cursor and Codex. This section covers what only
Claude Code does.

- **Skills fork to subagents.** `/mad-benchmark` and `/mad-profile` run in
  `mad-benchmark-runner`, `/mad-tune` in `mad-tuner`, and `/mad-report` in
  `mad-perf-analyst` (`.claude/agents/`), via `context: fork`. `/mad-report`
  sets `background: false`, so its answer arrives in the same turn.
- **Pre-flight is injected.** The GPU skills run
  `bash .claude/skills/mad-common/preflight.sh` in a `` ```! `` block before
  the skill body loads; a failing pre-flight aborts the skill with its message.
- **Workflows** (`.claude/workflows/`) handle larger fan-out jobs. They **plan
  by default**: they resolve the model(s) and print every command without
  touching GPUs. Pass `--execute` to run.
  - `mad-benchmark-sweep --tags a,b [--ngpus 1,8] [-c '<ctx>'] [--execute]` — a
    matrix across tags. Planned cells run in parallel, executed cells one at a
    time.
  - `mad-tune-search --tag <one-model> [--target latency] [--repeats 3] [--execute]`
    — descends operator → dispatch → kernel → assembly profiling, drops kernel
    levers that do not match the kernel card's bound, then measures each
    candidate cleanly `--repeats` times (default 3). Medians, spread and the
    noise gate are computed in code; an adversarial verifier must also agree.
    A counter run that stalls (GPU idle for 10 minutes while `running`) is
    stopped and retried once with `--serialize`; the kernel rung then reports
    `serialized`, or `error` when the retry stalls too.
    After each executed candidate the workflow checks `git status` against a
    snapshot and stops if an edit leaked. `--expect` adds `/primus-projection`
    numbers for a Primus training model; it stays off otherwise.
  - `mad-optimize --tag <one-model> [--budget 1] [--repeats 2] [--smoke-tag <tag>] [--execute]`
    — profiles one model down to a kernel card, runs GEAK `kernel_workflow`
    as its own phase with a task that names the card's bound, and keeps the
    patched tree only when the same latency gate passes. It retries a stalled
    counter run the same way. See `docs/mad-optimize.md`.
  - The tests in `tools/tests/` run all three against a mock agent:
    `node --test tools/tests/*.test.mjs`. `kernel-tools.py` has its own
    GPU-free test on recorded rocprofv3 and Triton output:
    `python3 -m unittest discover -s tools/tests -p 'test_*.py'`.
- **Hook.** `.claude/settings.json` registers a PostToolUse hook
  (`.claude/hooks/validate-models-json.sh`) that validates any
  `scripts/**/models.json` Claude edits and reports errors back immediately.
- **Permissions.** `.claude/settings.json` pre-approves `madengine discover`,
  `run` and `build` (so GPU runs start without a prompt), `madengine report
  tracelens*`, read-only git commands, `rocm-smi`/`amd-smi`, the pre-flight,
  `mad-run.sh`, `kernel-tools.py`, the model validator, and the Primus CLI
  commands. It does not
  pre-approve `docker run` or `madengine database`; `/primus-projection` grants
  `docker run --rm` only while it runs.
