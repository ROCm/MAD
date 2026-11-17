# Step 7, recipe track (b): non-deterministic NaN -> mutation sweep

> Experimental, interactive-only under MAD: the automated workflow skips step 7; mutated kernels are never applied to MAD or the image.

**You are here only from `step7-rootcause.md`, with `<NAN_KIND>` =
`non-deterministic`.** This runs **in parallel with** track (a) (native LLM), as
its own background run(s); each finished sweep is one recipe iteration -- write its
rows into the current `rootcause/iter-<k>.md` and let track (a) read them. A
deterministic NaN has no recipe: track (a) only.

The tooling is `replay_nan_op` (with `kernel_mutations` / `kernel_patch` behind it),
run as `madengine nan exec --image <IMAGE> [--gpus G] -- python3 -m
nan_doctor_rt.replay_nan_op ...`, started with `mad-run.sh start`. Below, `<EXEC>`
= `madengine nan exec --image <IMAGE> --` and `<REPLAY>` = `<EXEC> python3 -m
nan_doctor_rt.replay_nan_op`.

**No `identity`, no `nans_to_nops` in this recipe.** The step-6 reproducer already
showed the NaN reproduces through this replay path, and `nans_to_nops` only asks
about a NaN *injected* into the image. Never put either in `<MUTATE_ONLY>`, and
**always pass `--mutate-no-identity`** (else the tool adds `identity` itself).
Without the control, the asm read-back is the only proof a patch landed: keep
`--mutate-disasm` on (the tool refuses `none` with this flag) and read every CLEAN
row's `.diff` -- a CLEAN row flagged "disassembles IDENTICALLY" is void.

## The sweep replays exactly what 5b reproduced

The sweep's baseline is the 5b replay that reproduced, re-run under `--mutate`: same
capsule, same replay mode, same load, same flags. Nothing is picked by symptom.
Read the reproducing attempt from `<LEDGER>` (`LADDER_RESULT=REPRODUCED at
concurrency <C>`, its `summary=` path) and that attempt's `summary.txt`, then map
every setting:

| 5b reproducing attempt | sweep flag |
|---|---|
| the capsule it replayed (`<CAPSULE_FILE>`) | the same `.pt` |
| `--mode chain` (the default; copy lines say `chain runs`) | `--chain --chain-iters <N>` -- `[S3]` |
| `--mode op` (copy lines say `launches`) | `--iters <N>` -- `[S2]` (`[S1]` when C = 1) |
| reproducing concurrency `C` (`c<C>.` attempt dir) | `--contend <C-1>`; C = 1: none, on a quiet GPU (load causes all-`TIMEOUT` sweeps) |
| `--cotenant <cmd>` (a co-tenant reproducer, `step5-pressure.md`) | `--contend-with '<cmd>'` instead of `--contend` |
| `--extra-args` (e.g. `--synthesize`, `--written-operands`) | the same flags |
| the image and the `--env K=V` settings | the same `--image` and `--env`, on the sweep's `madengine nan exec` |
| `--restore-env` (5b always restores) | `--restore-env` |
| runs per copy | `<N>` = enough for >= 10 expected hits at 5b's rate (non-finite / runs actually done), capped at 5b's `--iters` |

* **Ledger the mapping** in one line before starting: the 5b attempt dir it came
  from and each flag. A setting you cannot read from the ledger or the attempt dir is
  a question for the user, not a default.
* **What cannot match:** the sweep is one process (+ its co-tenants) on one GPU; 5b
  ran the same per-GPU load on every GPU, all copies released together at a barrier.
  That per-GPU load is the sweep's geometry. Everything else above is copied, never
  substituted -- in particular never `[S2]` (the op alone) for a 5b that reproduced
  in chain mode, nor `--contend-auto` (the capture-time count, which may be
  under-counted or differ from what 5b needed).
* **Co-tenants must be running before the baseline counts.** `--contend-warmup` is
  gated on observed device memory, not time: it trips once the co-tenants have
  *allocated*, seconds before they *loop*. A baseline shorter than ~10x the
  co-tenants' setup (an op-mode loop at ~10k launches/s finishes 400k in under a
  minute) is measured mostly solo. Size `<N>` so the baseline runs for minutes, or
  match 5b's mode (chain loops are ~10x slower).
* **The baseline must reproduce before any row is read.** Baseline `NOT_REPRODUCED`
  (`skipping the mutation sweep`) or far below 5b's rate = the sweep is not 5b's
  setup: re-check the mapping line by line, then the warm-up above. Never
  `--mutate-force` past it, never report its rows.
* With `--mutate`, `--contend` co-tenants stay for the whole sweep (baseline and
  trials share the device) and keep running the **unpatched** kernel; only the
  measured process is patched. `--mutate-solo` drops them after the baseline. When
  the trigger needs a co-tenant running the same kernel, a `NAN` row does not rule
  the mutation out -- say so in the report.
* A real-model co-tenant (`--contend-with`): gate on its step line, ledger the
  geometry (`<REPORT_JSON>` won't know it).
* Contention-gated fault swept solo -> every row CLEAN: wrong setup, not a finding
  (with no `identity` row the sweep cannot say so -- check `--contend` first).
* Pin `--mutate-timeout` on a busy GPU. **Unrelated load** (hogger, parallel
  sweeps on the same device) comes off first; matched `--contend` co-tenants stay.

## What the sweep can reach

| source | reached by | flags |
|---|---|---|
| **Tensile** (hipBLASLt) | unbundle, patch, rebundle | `--patch-libdir`, `--patch-kernel` |
| **aiter** `.co` | bare HSA ELF under `<asm_root>/<arch>/`, patched in a symlink overlay | `--patch-asm-dir` |
| **CK** | HSA shim intercepts at load, names and dumps the kernel | -- |

* **Growth mutators** (`sbarrier_*`, +4 B per `s_barrier`) splice into padding, else
  rebuild the object -- possible only for single-kernel files (aiter `.co`, HSA dump).
  Tensile raises `PatchError`. **An `ERROR` row from a growth mutator is a tooling
  limit** (or a build predating rebuild support).
* **aiter trap:** with `AITER_ASM_DIR` unset aiter loads a copy embedded in
  `libtorch_hip.so` -- patching the file changes nothing, every row CLEAN. Loader
  trace: `[aiter] hipModuleLoad: /abs/path/...` = file-backed, sweepable;
  unprefixed `hipModuleLoad: hsa/...` = embedded. `ensure_aiter_asm_dir` sets it for
  baseline and trials; `trace_is_file_backed()` asserts it; embedded loads become
  `SKIPPED`. Check it in the enumeration.
* **aiter/CK op-to-kernel attribution is a heuristic** (family dir `fmha_v3_bwd/` vs
  `fmha_v3_fwd/`; `FmhaBwd*` vs `FmhaFwdKernel`); unreadable direction -> `SKIPPED`.
  Say so in the report.
* **`SKIPPED` rows stay in the table.** All `SKIPPED` = the recipe adds nothing;
  stop track (b), track (a) continues alone.

## Before sweeping

* **One capsule per distinct kernel**, not per group; report folds.
* **Reproduced at concurrency 1** (intermittently -- a fault that hits every run is
  deterministic and never reaches this file): read `step5-replay-guardrail.md`; the
  form still follows 5b's replay mode (chain by default -> `[S3]`).
* Identify the kernel: `--disasm <FILE>` (5b does not name it).
* **Enumerate** -- needs a GPU and **a reproducing baseline in 5b's form**, costs
  ~one 5b replay. A chain capsule that reproduced only under load lists nothing
  from a solo pass (its baseline is clean), so enumerate with the sweep's own
  mapping and a short loop (enough runs for a few hits at 5b's rate):

  ```bash
  <REPLAY> <CAPSULE_FILE> --chain --chain-iters <N> \
      --no-reference --contend <C-1> --restore-env <6B_EXTRA_ARGS> --list-mutations --mutate-no-identity
  ```

  The op-mode form (`<CAPSULE_FILE> --list-mutations`) is for a capsule 5b
  replayed as a single op. On synthetic operands its fp64 reference can be
  inconsistent (a backward's saved forward outputs): it says
  `synthetic_inconclusive` and lists anyway.

  GPU-free once you know the name: `--list-mutations --patch-kernel <KERNEL_NAME>`.
  Its mutator list is authoritative over the table below.

## Every applicable mutation, weakest first

One mutation = one trial of up to `--mutate-timeout` (an hour each when
contention-gated). The sweep covers **every mutator the enumeration lists** (less
`identity` and `nans_to_nops`), in tiers, weakest first, and prunes a stronger
mutation only when its weaker sibling has already cleared the NaN.

| mutator | asks | notes |
|---|---|---|
| `lgkmcnt0` | depends on relaxed LGKM waits? | GFX9 only (gfx90a, gfx942 MI300X, gfx950 MI355X) |
| `lgkmcnt0+sbarrier` | ... or cross-wave LDS ordering? | stronger than `lgkmcnt0` |
| `lgkmcnt0_each` | which relaxed wait(s)? timing or ordering? | one trial per site; only after a CLEAN `lgkmcnt0` (tier 3) |
| `sbarrier_ds_read_write` | readers before writers (WAR)? | unguarded LDS boundaries; 8 B/site |
| `sbarrier_ds_write_read` | writers before readers (RAW)? | 0 sites is a finding |
| `sbarrier_only_ds_read_write` | bare rendezvous enough? | 4 B/site, no drain -- a real fix's shape |
| `sbarrier_only_ds_write_read` | ... RAW direction | same |

**Localising a CLEAN `lgkmcnt0`** (tier 3, only after `lgkmcnt0` is a powered CLEAN):
`lgkmcnt0_each` proposes one trial per relaxed wait (`lgkmcnt0_each@0x<addr>`, the
same slot-width payload at that site only). It answers two questions the all-sites
row cannot: **which waits matter** (the minimal fix step 8 should ship), and **is it
timing alone** -- one drained site adds far less delay than six, so a single site
that clears the NaN at power, while others at the same loop depth do not, argues for
the ordering mechanism over a global slowdown (H "timing only"). All sites
individually `NAN` while all-at-once is CLEAN = the NaN needs several waits drained
(or it is timing): say which. Cost: one trial per site -- include it in the plan's
worst case.

**Run tier 3 in parallel, one sweep per GPU** (all GPUs of the machine, held for the
duration; this is interactive-only). Serially it is sites x trial time; split it is
minutes. Each GPU gets every Nth site by name (N = GPU count) -- `--mutate-only
lgkmcnt0_each@0x<addr>`, repeated (mutation names select single sites;
`NAN_MUTATE_EACH_ADDRS="0x.. 0x.."` does the same through the env) -- its own
baseline, its own co-tenant, and its **own** `--mutate-report`
(`mutations.t3.g<g>.json`: parallel sweeps cannot share one journal). Each GPU's
baseline must reproduce before its rows count. Iterations per site: enough for >= 30
expected hits at the tier-1 baseline rate.

**When `lgkmcnt0` is the only applicable mutator** (common on Tensile: the
`sbarrier_*` mutators find no unguarded boundary, and `lgkmcnt0+sbarrier` does not
fit the inter-kernel padding), there is no tier 2, and tier 3 is the experiment that
turns "a relaxed wait matters" into a root cause: put it in the confirmed plan up
front, run it as soon as tier 1 is a powered CLEAN.

### Strength is a partial order: three pairs

A mutation is *stronger* than another only when it patches the **same sites** and
enforces everything the other does plus more. That gives three pairs and nothing
across them (different sites, different hazards -- neither result implies the other):

| pair (same sites) | stronger | weaker | what the stronger adds |
|---|---|---|---|
| relaxed LGKM waits | `lgkmcnt0+sbarrier` | `lgkmcnt0` | an `s_barrier` after the drain |
| `ds_read` -> `ds_write` boundaries | `sbarrier_ds_read_write` | `sbarrier_only_ds_read_write` | an `lgkmcnt(0)` drain before the barrier |
| `ds_write` -> `ds_read` boundaries | `sbarrier_ds_write_read` | `sbarrier_only_ds_write_read` | an `lgkmcnt(0)` drain before the barrier |

Within a pair:

* **weaker `CLEAN` (powered: >= 3 expected hits) -> stronger is predetermined
  `CLEAN`**: it does everything the weaker one does at the same sites, and more.
  Don't run it; report it as *predetermined CLEAN (not run)*, never as measured or
  "skipped". The weaker one is the smaller change -- the shape of the fix step 8
  would ship.
* **weaker `NAN` -> run the stronger**: it asks whether what the stronger one adds
  (the barrier, or the drain) clears the NaN the weaker one could not.
* **Only a powered `CLEAN` prunes.** A weaker row that is `TIMEOUT`, `ERROR`,
  `SKIPPED` or a `PatchError` drop says nothing about the stronger one: run the
  stronger. A weaker `UNDERPOWERED` CLEAN: re-run it with more `<N>` first, then
  decide. (A `TIMEOUT` is mostly a barrier in divergent control flow hanging -- so a
  stronger barrier mutator may `TIMEOUT` too; report it as such.)
* Predetermination is about the mechanism: each mutation also shifts timing a
  little, so a race near threshold could in principle flip. That is why a pruned row
  is reported as predetermined, not as measured.

### Tiers

The tool runs the mutators you pass in **registration order** (`lgkmcnt0` before
`lgkmcnt0+sbarrier`, but drained `sbarrier_*` **before** bare `sbarrier_only_*`) and
never prunes, so one invocation cannot do this. Run tiers as separate sweeps with the
**same** `--mutate-report` (the journal key does not include the mutator set, so tier 2
appends to tier 1's report; rows already present are resumed, not re-run):

1. **Tier 1** -- the weaker member of every applicable pair (`lgkmcnt0`,
   `sbarrier_only_ds_read_write`, `sbarrier_only_ds_write_read`), plus the stronger
   member of any pair whose weaker member the enumeration dropped. `<MUTATE_ONLY>` =
   those names.
2. **Tier 2** -- the stronger member of every pair whose tier-1 row was **not** a
   powered `CLEAN`. Nothing left: no tier 2.

### Before tier 1: show the plan, confirm its cost

**Enumerate, show the plan, stop**:

* every listed mutator, its sites, its pair, its tier, and the `FAIL`/`PatchError`
  rows with reasons;
* per-trial cost at `<N>`, worst case (no pruning) and best case trial counts, and
  how much of `<RC_BUDGET>` each uses.

The user confirms the **cost**, not the set. If the worst case does not fit
`<RC_BUDGET>`, say so and ask which pairs to defer -- never drop a mutation silently.
Tier 2 needs no second confirmation if it is inside the confirmed worst case.

### Reading a pair

| weaker (tier 1) | stronger (tier 2) | means |
|---|---|---|
| CLEAN | *predetermined CLEAN* | the weaker change suffices: it is the minimal mechanism (and the fix's shape) |
| NAN | CLEAN | what the stronger adds is necessary: the barrier for `lgkmcnt0` (a wave reads an LDS tile a **sibling** hasn't finished -- cross-wave), the drain for `sbarrier_only_*` (rendezvous without retiring LDS ops is not enough) |
| NAN | NAN | this pair's sites do not hold the cause |

* A bare `sbarrier_only_*` `NAN`: read its census (description ends `N of M already
  drain lgkmcnt(0) first, K do not (<addrs>)`); at `0 of M` it rules out nothing on
  its own -- the drained sibling in tier 2 carries it.
* Direction (WAR vs RAW): compare the two `sbarrier_ds_*` pairs by the weakest
  member that cleared each. CLEAN in one direction only names it; CLEAN in both =
  rendezvous needed, direction unknown.

## Running

One command per tier, `<MUTATE_ONLY>` = that tier's mutators (one `--mutate-only` each)
-- **never omit `--mutate-only`** (that runs every mutator in one pass, in
registration order, with no pruning). Pick the form by **5b's replay mode** (the mapping table above), then fill
`<C-1>`, `<N>`, the co-tenant and the extra args from the same 5b attempt.
`<6B_EXTRA_ARGS>` = its `--extra-args`; with `--cotenant`, replace
`--contend <C-1>` with `--contend-with '<cmd>'`. Start each through `mad-run.sh start`.

```bash
# [S3] 5b ran --mode chain (the default) -- the usual sweep. <MUTATE_ONLY> runs
# per op.
<REPLAY> <CAPSULE_FILE> --chain --chain-iters <N> \
    --no-reference --contend <C-1> --restore-env <6B_EXTRA_ARGS> --mutate --mutate-timeout 3600 \
    --mutate-stop-on-finding --mutate-no-identity <MUTATE_ONLY> --mutate-report <REPORT_JSON> --mutate-asm-dir <ASM_DIR>

# [S2] 5b ran --mode op at concurrency C >= 2; ~an hour per trial --
# within the confirmed plan
<REPLAY> <CAPSULE_FILE> --iters <N> --mutate \
    --contend <C-1> --restore-env <6B_EXTRA_ARGS> --mutate-timeout 3600 \
    --mutate-no-identity <MUTATE_ONLY> --mutate-report <REPORT_JSON> --mutate-asm-dir <ASM_DIR>

# [S1] 5b ran --mode op at concurrency 1
<REPLAY> <CAPSULE_FILE> --iters <N> --mutate --restore-env \
    <6B_EXTRA_ARGS> --mutate-no-identity <MUTATE_ONLY> --mutate-report <REPORT_JSON> --mutate-asm-dir <ASM_DIR>
```

* `--restore-env` on the parent only; its `env:` banner must read `matches capture`
  (wrong env -> CLEAN control).
* `--mutate-only` repeats, one per mutator; `--mutate-no-identity` on every
  command. All rows CLEAN = suspect the setup (env banner, `--contend`) before the
  kernel.
* **Always `--mutate-report`**, and `<REPORT_JSON>`/`<ASM_DIR>` under `<ARTIFACT_DIR>`
  and visible to the container (`--mount`); written at the end, a bad path loses
  everything.
* **`--mutate-keep-fix <ARTIFACT_DIR>/fix/kept`** on every aiter / CK sweep: each
  CLEAN row's patched kernel plus an `env.sh` that loads it lands in
  `kept/<row>/`. The sweep's overlay is overwritten per trial and deleted at the
  end, and step 8 validates exactly these bytes. (hipBLASLt: `fix_overlay`.)
* **Resumable**: `--mutate-report` writes a `.jsonl` journal (one fsynced row per
  trial); re-run the identical command and only missing trials run (`(resumed from
  journal)`). Key = capsule, host, `payload_kind`, launch count, co-tenant geometry,
  `--restore-env` -- a mismatch renames it `.superseded`. Mutator set and timeout are
  not in the key; after raising the timeout use `--mutate-retry-incomplete`.
  **Say which rows were resumed** (`resumed_trials`); delete a line to re-run that
  trial; `--mutate-no-resume` re-measures all.
* `--mutate-asm-dir` writes `NAME.before.s`/`.after.s`/`.diff`. **Read the diff
  back**: identical = patch never landed; check changed-instruction count and
  addresses against the description. Growth mutators' diff size is address
  shift -- quote the description's site count.

**Chain sweeps:** `[S3]` runs the `[C1]` walk (`--no-reference` keeps it cheap) so
its culprit leads; seed order = `[C1]` culprit, then the loop's tally, then reverse
chain order; downstream first. `--mutate-stop-on-finding` stops at the first op
with a strict finding (some mutator CLEAN and not `UNDERPOWERED`; with
`--mutate-no-identity` no control is required); the rest are `SKIPPED`. **Don't narrow up front**
(`--mutate-chain-ops 1`); default 0 = whole chain.

## Power and budget

* CLEAN means something only if the trial could have caught the fault: zero hits
  with probability `exp(-p*n)`. Rows below 3 expected hits are `UNDERPOWERED`
  -- never a finding; raise the count and re-run. Size iters from 5b's rate.
* Auto `--mutate-timeout` = `2 x (setup + per_launch x iters)` from the baseline,
  re-derived from the first trial's real cost. Don't inflate `--iters` (hundreds,
  or pin the timeout); remove unrelated load; sweep serially.
* **`--mutate-force`** (5b clean): only `NAN` rows carry information. After `NOT_REPRODUCED` the default is not to sweep; if
  forced, say so in the report.
