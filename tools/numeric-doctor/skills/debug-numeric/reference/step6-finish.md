# Step 6: finish the reproducer tarball (not optional)

**"Finish" means reproducer generation is finished -- not that the NaN is fixed.**
Step 6 ends with a packaged reproducer; the NaN is still unexplained. Root-causing
(step 7) and fixing (step 8) are separate, budgeted steps -- on by default -- that start
from this tarball; the hand-over (*Then*) says so.

The one artifact a vendor can act on: capsule, scripts, one command -- no model,
dataset or checkout. **It is bound to the GPU arch it was captured on** (hipBLASLt
solutions and aiter/CK code objects are built per arch): the README names it, and a
`0/N` on another arch (MI300X vs MI355X) says nothing about the bug. **For every group that reproduced** (`KERNEL_BUG`,
`KERNEL_BUG_UNVERIFIED`), as soon as its 5b result lands -- and for every step-5N
hit: `NATIVE_KERNEL_BUG` / `NATIVE_UNVERIFIED`, and `NATIVE_MATH` / `NATIVE_FP32_OVERFLOW`
as a model-numerics reproducer (*Native reproducer*, below). G1 did
not reproduce: step 5N first (`step5n-native.md`); after a miss there, ledger that, go to step 9.

**After the capture-to-reproducer job** (any runtime; or `make_repro_readme.py --pack` on a hand-built one),
`repro-<G>/` has been built, verified by 5b from an extracted copy **through the package's own
`run.sh` in an empty environment** (no profile: exactly a vendor's `bash run.sh` on that node --
its baked runtime / SIF / flags are what was verified), given a `README.md` and packed; the
ledger has the tarball's path and `sha256`. **Op-only** (the result line says
`op-only(chain package kept: repro-<G>-chain)`): a chain capsule is replayed a second time with
the chain dropped (`capsule_slim.py`, `CAPSULE_MODE=op`, from its own extracted copy under
`verify-<G>-op/`); it reproduced, so `repro-<G>/` is that op-only package -- a fraction of the
size -- and the verified chain package is `repro-<G>-chain/` (its walk: `verify-<G>-chain/`), not
tarred: step 7 and a vendor who asks for the chain take it from there. The README's one command includes the site's
`sbatch` line (account, partition, GPU request -- from `run.sh`'s `# this site:`). The job closed its
`step6[<G>]` visit: open your own for what is left (`$LG start 'step6[<G>]'`, `end` it after the
repack), or the repack's ledger line warns `no open visit` and the work goes untimed. Left to you:

1. **Confirm the README's `<!-- CHECK -->` line** against the `[C1]` walk and the
   verdict table (`step5-read.md`); the script proposes, it does not decide. Fix the
   sentence and drop the marker. If it flags a deterministic walk or `FP32_OVERFLOW?`,
   resolve that first (`step5-replay-guardrail.md`) -- do not ship.
2. Add what only you know, briefly: where the capsule came from (model, run), groups
   folded into this one (same kernel, on what basis), a held-back loader path that
   `run.sh` must export (`env:` banner `held back:` -- e.g. `export
   LD_LIBRARY_PATH=/opt/conda/lib;` was once mandatory), and for a Tensile
   `KERNEL_BUG` the `--bench` one-liner -- already in the README's `## Kernel` section
   when the replay ladder identified the kernel (`kernel.*.txt`); add it by hand only
   if that section is missing.
   **`SYNTHETIC-INCONSISTENT` in the `operands:` banner**: the README must say the
   tarball shows the NaN but cannot validate a fix, and name the consistent-operand
   check that can (`step5-read.md`).
3. Edited anything? Re-pack without rewriting the README (seconds with `pigz`; a real-values
   package without it takes minutes -- give the call a timeout of >= 600000 ms either way):
   `python3 <REPO_ROOT>/scripts/make_repro_readme.py --repro <ARTIFACT_DIR>/repro-<G> --repack --ledger <LEDGER> --step 'step6[<G>]'`
   (tars, prints and ledgers the new sha256; the report takes the latest line). A README-only
   edit needs no re-run: say in the report that the job verified the earlier pack (its sha) and
   only `README.md` changed since; anything else changed (`run.sh`, the capsule, a script):
   verify the new tarball again (item 4 of *Native reproducer*, the same way).
4. **On other nodes** (always; submit it right after the job, it runs while you do 1-3): the package's
   `run.sh` on 2 OTHER nodes at once, each from its own copy, the way a vendor runs it --
   `PACKAGE=<ARTIFACT_DIR>/repro-<G> nd submit --name nodes --gpus 8 --nodes 2 --exclusive --exclude <the 5b
   host> --step 'step6[<G>]' -- verify_nodes.sh`, then `nd wait nodes --max 240`; last line
   `NODES_REPRODUCED=<k>/2 hosts=<host>:<verdict>,...`. Ledger it. `2/2`: say "reproduces on 3 nodes" in the README.
   `NODE_DEPENDENT` (k=1) or `REPRODUCES_NOWHERE_ELSE` (k=0): the node is a variable of the bug -- the README
   says on which hosts it reproduced and on which not, `<NAN_KIND>` cannot be `deterministic` without it, and
   step 7 reads it (a hardware or firmware difference, a tuning file, a node-specific library build). `ERROR`
   on a node: the package, not the bug -- fix `run.sh` and verify again. The 2026-11-23 qwen3 MoE capsule
   reproduced on the 5b node only in one step-8 op replay, and nothing in the package said so.
5. Tell the user path, size and the one command -- as *the reproducer is done*, not
   *the NaN is solved*. A no-chain capsule
   (`--chain-fallback`): the README must say the operands were never validated.
6. **A co-tenant reproducer** (built with `CAPSULE_COTENANT`, `step5-pressure.md`):
   the README's contention paragraph names the load. Add which loads ran clean (the 5d
   table: e.g. forward-only, one co-tenant process, no co-tenant) -- they bound the trigger
   for the vendor. **On the TRIGGER path** (co-tenant from the first submit) neither the solo
   replay nor 5d ran: say exactly that -- "the replay without the load was not run" -- never
   "does not reproduce alone", and no table; and
   that the co-tenant itself runs the same kernel family if it does (a TF32
   training step calls the F32X kernels too).

## Native reproducer (from step 5N)

No capsule, no capture-to-reproducer job, no `make_repro_readme.py` README -- you build it. Tag
`step6[native]`.

```
repro-native/
  run.sh  README.md  repro.py      # the minimal candidate, renamed
  native_run.sh                    # <REPO_ROOT>/scripts/native_run.sh
  failure_signature.py             # <REPO_ROOT>/scripts/failure_signature.py (the hit's SIGNATURE line)
  scripts/nd  scripts/env/         # the runtime layer
  cotenant/<file>                  # only if the reproducing load needs one
```

1. **Ship exactly what reproduced**: the minimal candidate as `repro.py`,
   `native_run.sh`, `cotenant/<file>` if the load needs one, and the runtime layer
   (`cp -r <REPO_ROOT>/scripts/nd <REPO_ROOT>/scripts/env repro-native/scripts/`), so it
   runs on docker, podman, apptainer or a venv. A multi-rank candidate (a `torchrun` program) ships the
   same way: `RANKS=<world size>` makes `native_run.sh` run each copy as one `torchrun` per GPU group --
   no runner of your own. `run.sh`:

   ```bash
   #!/bin/bash
   #SBATCH --job-name=nan-repro-native
   #SBATCH --nodes=1
   #SBATCH --exclusive
   #SBATCH --time=00:30:00
   # bash run.sh (this GPU node) | sbatch [<site args: -A <acct> -p <part> --gpus-per-node=8>] run.sh
   #   (Slurm, from this dir).  RUNTIME=, IMAGE=, SIF=, APPTAINER_FLAGS=, NGPUS=, COPIES=, RANKS= override.
   if [ -n "${SLURM_JOB_ID:-}" ]; then HERE=$SLURM_SUBMIT_DIR; else HERE=$(dirname -- "$0"); fi
   HERE=$(cd -- "$HERE" && pwd -P) || exit 2
   export ND_REPO_HOST=$HERE ND_ART_HOST=$HERE IMAGE=${IMAGE:-<image>}
   # what reproduced (from `nd info` of <ENV>): runtime and its flags; drop lines that do not apply
   export RUNTIME=${RUNTIME:-<runtime>} APPTAINER_FLAGS=${APPTAINER_FLAGS-'<flags>'}
   [ -n "${SIF:-}" ] || { [ -f "<SIF>" ] && export SIF="<SIF>"; }   # this site's SIF; else built/pulled from IMAGE
   exec "$HERE/scripts/nd" run --gpus all --env "COPIES=${COPIES:-<c>}" --env "NGPUS=${NGPUS:-8}" \
     --env "RANKS=${RANKS:-<k: 1, or the multi-rank candidate's world size>}" \
     [--env "COTENANT=<cmd with $ND_REPO/cotenant/...>"] -- 'bash $ND_REPO/native_run.sh $ND_REPO/repro.py'
   ```
   Without a profile (`ND_ENV` unset) `run.sh` gets nothing from `<ENV>`: what is not baked in
   above falls back to `RUNTIME=auto` and, under apptainer, a ~35 min `apptainer pull` of `IMAGE`
   -- impossible for a private image. A site that rejects jobs without a GPU request needs
   the site args on the `sbatch` line; put that line in the README.
2. **README**, short: what it reproduces (op or region, shapes, dtypes, precision
   flags) and the verdict with the fp64 check -- for a finite signal (`fp64_ref=WRONG_VALUE`
   hits) the first sentence says it reproduces **wrong values, not a NaN**, and names the
   reference and the tolerance a hit exceeds; for `NATIVE_MATH` /
   `NATIVE_FP32_OVERFLOW` the first sentence says it is **not a kernel bug**: the
   arithmetic (`0 * rsqrt(0) = 0 * inf` in an eps-less norm on all-zero rows) or the
   overflow, and the clean control that removes it (e.g. the eps); an `* Image:` line with the image in
   backticks (the report reads it); the one command; the rate **with the load** (copies x GPUs,
   co-tenant) and every clean load and removal that killed it (N5 ledger rows -- they
   bound the trigger); why the capsule path failed and how the native script differs
   from it (`NATIVE_TRIGGER`, `native/handoff.md`'s Ruled-out rows, the N6 difference); that a single solo run alone means nothing if the load matters.
3. Pack and ledger (writes `repro-native.tar.gz` + sha256):
   `python3 <REPO_ROOT>/scripts/make_repro_readme.py --repro <ARTIFACT_DIR>/repro-native --repack --ledger <LEDGER> --step 'step6[native]'`.
4. **Verify from an extracted copy** of that tarball: extract under `<ARTIFACT_DIR>/verify-native/`
   (never `/tmp`) and run *that* `run.sh` the way a vendor will -- from its directory, in an empty
   environment, not inside `nd run` (it starts its own runtime): `PACKAGE=<ARTIFACT_DIR>/verify-native/repro-native
   nd submit --name verify --gpus 8 --exclusive --step 'step6[native]' -- package_run.sh` (`scripts/jobs/package_run.sh`:
   `env -i` but for HOME/USER/PATH, no profile, no `SLURM_*`; last line `PACKAGE_RC=`), then `nd wait verify --max 240`
   -- never a `squeue` / `sleep` loop. (A plain `sbatch run.sh` + `nd adopt` is the same test where the session
   may submit outside nd; an unattended session's permissions refused it, 2026-11-23.) Pass = `NATIVE_TOTAL`
   `n/N > 0` near the ledger rate. A fix after a failed
   verify: change `repro-native/`, repack (3), verify again. Then on 2 other nodes, as item 4 of the capsule
   reproducer above (`verify_nodes.sh` reads `NATIVE_TOTAL`).
5. Tell the user path, size and the one command (reproducer done, NaN not yet
   root-caused -- or, for `NATIVE_MATH` / `NATIVE_FP32_OVERFLOW`, explained by the
   arithmetic, not fixed).

## Kernel capsule reproducer (from 5N's compiled track)

A kernel capsule that replays (`step5n-compiled.md` *Kernel capsule*: hits, inputs finite) is shipped
like a native reproducer -- it needs no model, and its inputs are the model's real values. Tag
`step6[kernel]`; when a hand-written candidate also hit, ship both (two packages).

```
repro-kernel/
  run.sh  README.md
  capsule/inductor_<pid>_<kernel>.pt     # the capsule
  source_graph.txt                       # Inductor's FX graph of the kernel's source nodes (C1's output code)
  scripts/replay_inductor_kernel.py      # imports torch only
  scripts/nd  scripts/env/  [image/]     # the runtime layer; a locally built image's build context
```

**The capture job builds it** on `NO_REPLAYABLE_GROUP` with a kernel capsule (`KERNEL_CAPSULE=` in its
result line): package, tar, extract, `run.sh` from the extracted copy in an empty environment (one GPU,
`ITERS=1000`), README, tarball, the `TARBALL` ledger line -- all under `step6[kernel]`. What is left:
the README's `CHECK` comment -- replace it with the reference verdict (the result line's `fp64_ref=`, the kernel's
fp64 twin; `unavailable`: eager / fp64 of the source nodes by hand, `step5n-compiled.md`: `NATIVE_MATH` or the
kernel), then repack (`make_repro_readme.py --repro
<ART>/repro-kernel --repack --ledger <LEDGER> --step 'step6[kernel]'`) and report as item 5 of *Native
reproducer*. Built by hand (a capsule saved outside the capture job): the same layout, `run.sh`'s
command `nd run --gpus 0 -- 'python3 $ND_REPO/scripts/replay_inductor_kernel.py $ND_REPO/capsule/<capsule>
--iters 1000'` (one GPU: the kernel replays alone; a contention-gated fault gets the native package's
`COPIES` and load instead); pack, verify from an extracted copy and report as items 3-5 of *Native
reproducer*. The kernel is recompiled from its Triton source by **the image's** torch / triton.

## Then

Every reproducing group packed: **announce the hand-over** -- always, as its own block
(SKILL.md rule 12), combined with item 5's path/size/command. Classify `<NAN_KIND>` first
(`step7-rootcause.md` *Entry* item 3) so the budgets quoted fit it. **Ledger it** in the same
call as `$LG end step6[<G>]` (`$LG note step6[<G>] "HANDOVER announced: <tarball> -> step 7
(RC <RC_BUDGET>, FIX <FIX_BUDGET>)"`; the panel's reproducer card keys on it, SKILL.md rule 13).
Step 7 must not have begun when the block appears: no `$LG start step7[<G>]`, `rootcause/` dir
or experiment before it.

**Interactive session: steps 7-8 run in a background fork, and the block ends the turn.**
An agent with no background-agent tool (Codex CLI, or a Cursor session without one) runs steps 7-8
inline instead, as under **No fork** below.
Text printed mid-turn folds under the next tool output and the user never sees it
(2026-11-19; 2026-11-24 twice, even repeated after step 7's first submit). So, in this order:

1. Spawn ONE fork per reproducing group -- `Agent` with `subagent_type: "fork"` (it inherits
   this conversation: the skill, `<LEDGER>`, `<ENV>`, the budgets) and `description: "debug-numeric
   steps 7-8 <G>"`, prompt: *"You are the step-7/8 fork for group <G> of <ARTIFACT_DIR>. Run step 7
   (`step7-rootcause.md`) and, with a valid root cause, step 8 (`step8-fix.md`) for <G> only, with
   RC <RC_BUDGET> / FIX <FIX_BUDGET>; ledger everything as the skill says (status.json follows).
   Do not run step 9 and do not talk to the user: end with three lines -- `ROOT_CAUSE=...`,
   `FIX=...`, and the `rootcause/` and `fix/` paths."* The fork runs the loops exactly as the main
   session would; its tool output stays out of this context.
2. Print the block (below) as the **last text of the turn -- no tool call after it** -- and end
   the turn.
3. While it runs, a status question gets `ledger.py <LEDGER> status` (SKILL.md rule 13), and *stop*
   gets `TaskStop` on the fork plus `nd stop` of its jobs (the ledger's open `job=` lines), then
   step 9. **`SINGLE_SESSION=1` (one SSH session per login)**: the main session issues no commands while the fork runs
   (one SSH session) -- answer status from `status.json` with `Read`, not a shell.
4. The fork's completion arrives as a notification: read its three lines (and `<LEDGER>`, never
   its transcript), relay root cause and fix in two lines, then step 9.

**No fork** -- a delegated or non-interactive run (the step-1 answers given up front: validate-examples,
`claude -p`), or no `Agent` tool: print the block and go on to step 7 in the same turn, as before.

```markdown
> **✅ Reproducer generated** -- `<tarball>` (<size>), run it with `<one command>`.
> It reproduces the NaN without the model; the NaN is **not** root-caused or fixed yet.
>
> **Next: root-causing, then fixing** (experimental) -- running now in the background, within
> <RC_BUDGET> for the root cause and <FIX_BUDGET> for the fix, on the reproducer only. Ask
> *where are we* at any time; say *stop at the reproducer* to end it, or give other budgets.
```

Step 9 repeats the block from the `HANDOVER announced` line (`step9-report.md` *Always report*).

Steps 7-8 turned off (step-1 Q4 `no`): the first two lines only, then step 9. Q4 "ask me
after 6" (ask-me mode): end the block with the question instead -- "root-cause and fix it
now?" with the budgets as defaults -- so a "yes" is a complete answer.
**Every verdict goes on to step 7**, a system bug and a model bug alike: a `KERNEL_BUG*`
reproducer, and a model / numerics one -- 5b `FP32_OVERFLOW?` (overflow, or undefined
arithmetic such as `0*inf`), 5N `NATIVE_MATH` / `NATIVE_FP32_OVERFLOW`. For a model / numerics
one the block's first line says so ("not a kernel bug: <the arithmetic>"); step 7 then finds
*why the model hands the op those values* (config, data, code) and step 8 fixes it in the
model (`step7-rootcause.md` *Model / numerics NaN*).
