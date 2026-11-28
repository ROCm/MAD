# Step 5, by hand: run 5a / 5b (a failed capture-to-reproducer job, or a group outside it)

Both go through `nd` with `ND_ENV=<ENV>` (`env.md`). Read results with `step5-read.md`.

## 5a. Group

```bash
nd run --gpus 0 -- 'python3 -m compare_nan_capsules $ND_ART/capsules --recursive --json $ND_ART/groups.json'
```

Show the user the printed grouping. Empty: `step5-read.md`.

## 5b. Replay each group -- one command per group

For each `groups[]` entry, `representative` = `<CAPSULE_FILE>`. The launcher runs
`<CAPSULE_CONCURRENCY>` copies on every GPU, each `replay_nan_op.py --chain
--chain-fallback --chain-iters <CAPSULE_ITER_COUNT> --restore-env`: the
deterministic `[C1]` walk (fp64 reference per op), then the `[C2]` loop counting
non-finite runs. Copies on a GPU are each other's co-tenants -- no `--contend`.

```bash
# any runtime / scheduler (host paths; a runtime path from groups.json is mapped back)
$LG start 'step5b[<GROUP>]'      # a new visit, even if a capture-to-reproducer job closed one
CAPSULE=<CAPSULE_FILE> OUT_DIR=<ARTIFACT_DIR>/replay/<GROUP> \
  nd submit --name ladder_<GROUP> --gpus 8 --exclusive --step 'step5b[<GROUP>]' -- replay_ladder.sh
# already on the GPU node: ND_ENV=<ENV> bash <REPO_ROOT>/scripts/jobs/replay_ladder.sh
```

**Ladder** (inside one job, 1M runs per copy each; with `CAPSULE_COTENANT` set: ONE copy per GPU
beside the load, no ladder -- the load replaces the co-tenant copies): **start at the capture's
processes per GPU + 1** (the capsule's `gpu_procs x containers_per_gpu`, read by
`replay_nan_op.py --capture-procs`; 3 at capture -> 4 copies per GPU; none
recorded -> 2), so the replay is at least as loaded as the model was. `gpu_procs`
is the larger of the scans at arm and at capsule write (`gpu_procs_at_arm`): by the
write a sibling may have exited. Older capsules lack the arm scan and can read one
low -- if the banner is below the launcher's processes per GPU, set
`CAPSULE_CONCURRENCY=<launcher's + 1>`. **OOM: one
copy fewer, down to 1.** **Stop at the first attempt that runs to completion**:
reproduced -- its concurrency is the group's setting for steps 6-7 -- clean =
"not reproduced at c<C> x 1M", or `NOT_SET_UP` (below; no climbing, no stepping down
on a clean run). Env knobs: `CAPSULE_ITER_COUNT`, `CAPSULE_CONCURRENCY` (overrides
the start; OOM still steps down), `CAPSULE_MODE` (`chain` | `op`: the failing op
alone), `CAPSULE_COTENANT` (one replay per GPU beside a load command,
`step5-pressure.md`), `CAPSULE_EXTRA_ARGS`, `CAPSULE_STOP_HITS` (10),
`CAPSULE_FUTILITY` (100), `CAPSULE_SECONDS` (off).
**Overlap barrier:** a GPU's copies wait for each other before their first run, so
they are co-tenants from run 1 (`--start-barrier`), not from whenever each started.
**Early stop** (`CAPSULE_STOP_HITS`, default 10): a reproducing attempt stops every
copy together at that many node-wide hits.
**Futility stop** (`CAPSULE_FUTILITY`, default 100): the model hit the NaN within
the capsule's op call index `K` (the filtered ops counted up to it; `CAPTURE_OP_INDEX`,
printed by the launcher);
0 hits in `100 x K` node-wide runs ends the attempt as `NOT_SET_UP` -- the replay is
not recreating the model's trigger (wrong kernel, missing load), which no more runs
fix: `step5-pressure.md`. K = 38k gave ~4M runs (~20 min) instead of the hours
the copies would have run clean. `CAPSULE_FUTILITY=0` for a fault known to be
rarer than its model rate. Otherwise a clean attempt runs all
`<CAPSULE_ITER_COUNT>`, so `NOT_REPRODUCED` keeps its strength. Budget for a clean
attempt ~5-20 min (20k runs: ~6 s on a 2-op chain, ~22 s on 7-op); a job asks 2 h
-- raise `-t` for longer chains. **Do not cut `<CAPSULE_ITER_COUNT>` to save time**
-- the stops above do that where the evidence allows; the capture-step rate does
not predict the replay rate (same model, same capture rates: 7.5e-7 to 7.6e-5 per
run), and short copies don't overlap. A 5d screening sweep may cut it.
**Kernel id** runs on every completed ladder (`kernel.*.txt`), so a clean or
`NOT_SET_UP` ladder can be checked against the model's kernel (`kernel_match.py`).

**Several groups.** Priority = `count` (ties: JSON order); name `G1, G2, ...`.
`<GROUPS_TO_REPLAY>=largest` (default): replay G1 only, stop here. `all`: one
node per group, in parallel, within the step-1 budget (beyond it, ask; fewer nodes:
fill in priority order). Never two groups on one node. One `OUT_DIR` each. **After a
capture-to-reproducer job: just run `<ARTIFACT_DIR>/next_groups.sh` lines** (GROUP-only
mode packages, replays from the extracted copy -- step 6's verification -- and writes the
README); the loop below is for groups it cannot reach.

```bash
python3 -c 'import json, sys
gs = json.load(open(sys.argv[1]))["groups"]
for i, g in enumerate(sorted(gs, key=lambda g: -g["count"]), 1):
    print(i, g["count"], g["representative"])' <OUT_JSON_PATH> |
while read -r i n cap; do
  CAPSULE=$cap OUT_DIR=<ARTIFACT_DIR>/replay/G$i \
    nd submit --name ladder_G$i --gpus 8 --exclusive --step "step5b[G$i]" -- replay_ladder.sh
done
```

* `--step` ledgers each submission; add a note with `count=<n> <capsule>`.
* A group already `PROPAGATED` from its capture line needs no node: run its `[C1]`
  walk on one GPU (`nd run --gpus 0 -- 'python3 -m replay_nan_op <pt> --chain --chain-iters 1'`).
