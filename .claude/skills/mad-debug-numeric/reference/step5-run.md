# Step 5, by hand: run 5a / 5b (extra groups, or a capture you ran yourself)

5a and 5b are `madengine nan group` and `madengine nan replay`. Both are long GPU
commands: start them detached with
`bash .claude/skills/mad-common/mad-run.sh start <run-id> -- madengine nan ...`, then
`mad-run.sh wait <run-id>` until `MAD_RUN_STATUS=succeeded|failed` (`env-madengine.md`).
Read results with `step5-read.md`.

## 5a. Group

```bash
madengine nan group <CAPSULE_DIR> --manifest build_manifest.json --out-dir <ARTIFACT_DIR> [--recursive] [--job]
```

`--recursive` for capsules in subdirectories (e.g. `node_<N>/nan_doctor_output/`).
Writes `groups.json`, `groups.txt`, `groups_priority.txt` (lines
`G<n> <count> <capsule> <op>[ FALSE_POSITIVE?|ORIGIN|PROPAGATED?]`). Show the user the
printed grouping. Empty: `step5-read.md`.

## 5b. Replay each group -- one command per group

For each group, `<CAPSULE>` = its capsule in `groups_priority.txt`. `madengine nan replay`
runs `<CAPSULE_CONCURRENCY>` copies on every GPU, each `replay_nan_op --chain
--chain-fallback --chain-iters <N> --restore-env`: the deterministic `[C1]` walk (fp64
reference per op), then the `[C2]` loop counting non-finite runs. Copies on a GPU are
each other's co-tenants -- no `--contend`.

```bash
madengine nan replay <CAPSULE> --manifest build_manifest.json \
  --out-dir <ARTIFACT_DIR>/replay/<GROUP> [--mode chain|op] [--iters N] [--concurrency C] \
  [--cotenant CMD] [--extra-args "..."] [--stop-hits N] [--futility N] [--seconds S] \
  [--ngpus N] [--env K=V]
```

The last output line is `LADDER_RESULT=...`; each attempt directory
`<out-dir>/c<N>.<pid>.<host>/` has `summary.txt` (`ATTEMPT_RESULT=`).
`madengine nan summary <attempt_dir>` re-prints it.

**Ladder** (inside one run, 1M runs per copy each): **start at the capture's
processes per GPU + 1** (the capsule's `gpu_procs x containers_per_gpu`, read by
`replay_nan_op --capture-procs`; 3 at capture -> 4 copies per GPU; none
recorded -> 2), so the replay is at least as loaded as the model was. `gpu_procs`
is the larger of the scans at arm and at capsule write (`gpu_procs_at_arm`): by the
write a sibling may have exited. Older capsules lack the arm scan and can read one
low -- if the banner is below the capture's processes per GPU, pass
`--concurrency <capture's + 1>`. **OOM: one
copy fewer, down to 1.** **Stop at the first attempt that runs to completion**:
reproduced -- its concurrency is the group's setting for steps 6-7 -- clean =
"not reproduced at c<C> x 1M", or `NOT_SET_UP` (below; no climbing, no stepping down
on a clean run). Knobs (flags; the former env names `CAPSULE_ITER_COUNT`,
`CAPSULE_CONCURRENCY`, `CAPSULE_MODE`, `CAPSULE_COTENANT`, `CAPSULE_EXTRA_ARGS`,
`CAPSULE_STOP_HITS`, `CAPSULE_FUTILITY`, `CAPSULE_SECONDS`, `NGPUS` still work as
defaults): `--iters`, `--concurrency` (overrides the start; OOM still steps down),
`--mode` (`chain` | `op`: the failing op alone), `--cotenant` (one replay per GPU beside
a load command, `step5-pressure.md`), `--extra-args`, `--stop-hits` (10), `--futility`
(100), `--seconds` (off), `--ngpus`.
**Overlap barrier:** a GPU's copies wait for each other before their first run, so
they are co-tenants from run 1 (`--start-barrier`), not from whenever each started.
**Early stop** (`--stop-hits`, default 10): a reproducing attempt stops every
copy together at that many hits across all GPUs.
**Futility stop** (`--futility`, default 100): the model hit the NaN within
the capsule's op call index `K` (the filtered ops counted up to it; `CAPTURE_OP_INDEX`,
printed at start); 0 hits in `100 x K` runs across all GPUs ends the attempt as
`NOT_SET_UP` -- the replay is not recreating the model's trigger (wrong kernel, missing
load), which no more runs fix: `step5-pressure.md`. `--futility 0` for a fault known to
be rarer than its model rate. Otherwise a clean attempt runs all `--iters`, so
`NOT_REPRODUCED` keeps its strength. Budget for a clean attempt ~5-20 min (20k runs: ~6 s
on a 2-op chain, ~22 s on 7-op); longer chains take longer. **Do not cut `--iters` to
save time** -- the stops above do that where the evidence allows; the capture-step rate
does not predict the replay rate, and short copies don't overlap. A 5d screening sweep
may cut it.
**Kernel id** runs on every completed ladder (`kernel.*.txt`), so a clean or
`NOT_SET_UP` ladder can be checked against the model's kernel (`madengine nan
kernel-match`).

**Several groups.** Priority = `count` (ties: `groups.json` order); the names are the
`G<n>` of `groups_priority.txt`. `<GROUPS_TO_REPLAY>=largest` (default): replay G1 only,
stop here. `all`: one `mad-run.sh` replay per group, **one after another** (they share
the host GPU lock), each with its own `--out-dir <ARTIFACT_DIR>/replay/G<n>`; ask before
exceeding the step-1 budget. Package each group (step 6) from its own reproducing
attempt.

* Ledger each replay (`madengine nan ledger <LEDGER> job step5b[G<n>] <run-id> count=<count> <capsule>`).
* A group already `PROPAGATED` from its capture line needs no ladder: run its `[C1]`
  walk on one GPU with `madengine nan exec --image <IMAGE> -- python3 -m
  nan_doctor_rt.replay_nan_op ...`.
