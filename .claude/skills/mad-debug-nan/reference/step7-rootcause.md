# Step 7 (optional, budgeted): root-causing -- a loop over the step-6 reproducer

> Experimental, interactive-only under MAD: the automated workflow skips steps 7 and 8 and never fixes anything; findings and fixes are proposed, never applied to MAD or the image.

Step 6 left a packaged reproducer, not an explanation. Step 7 looks for **why**
the NaN happens, working only from that reproducer (tarball or `repro-native/`) --
never the model again. It ends with a valid `<ROOT_CAUSE>`, or `NO_ROOT_CAUSE` once
`<RC_BUDGET>` is spent. Tag `step7[<G>]`, one loop per distinct kernel/reproducer.
Tooling: `nan_doctor_rt` modules (`kernel_patch`, `kernel_mutations`, `gemm_shape_probe`, `attn_consistent_check`, `replay_nan_op`, ...) via `madengine nan exec --image <IMAGE> -- python3 -m nan_doctor_rt.<module> ...`; every GPU job is started detached with `mad-run.sh`.

## Entry

1. **Wanted?** Step-1 Q4. "ask me after 6": ask now, for steps 7 and 8 both. No ->
   ledger `step7 SKIP <why>` (and `step8 SKIP no root-causing`), go to step 9.
2. **Ask for `<RC_BUDGET>`** -- wall clock for the whole loop, both tracks (default:
   **1 h when the reproducer hits every run** -- deterministic: experiments are
   1-GPU jobs of seconds -- **4 h** when it is non-deterministic: sweeps and powered
   rates), and the GPU-hours it may spend. Classify `<NAN_KIND>` (item 3) first so
   the default you offer fits it. **Step 8 wanted too: ask for `<FIX_BUDGET>` in the
   same question** (its defaults, `step8-fix.md` *Entry*), so `default` settles both;
   a second budget question at step 8's entry stalls on a user who already said "do
   both". Ledger both.
3. **Classify `<NAN_KIND>`** from `<LEDGER>` (don't re-measure) and ledger it with
   the evidence:

   | `<NAN_KIND>` | evidence |
   |---|---|
   | `deterministic` | `[C1]` walk non-finite with no `DIVERGED`; or the reproducer hits every run (`n/N = N/N`), solo too (5N: "hits every run, also solo") |
   | `non-deterministic` | `0 < n < N` at the reproducing concurrency; contention-gated; a `DIVERGED` walk |

4. **Pick the tracks** -- (a) always; (b) only if a recipe exists for `<NAN_KIND>`:

   | `<NAN_KIND>` | (b) recipe |
   |---|---|
   | `non-deterministic` | the mutation sweep, `step7-recipe-sweep.md` (no `identity` / `nans_to_nops`) |
   | `deterministic` | none -- track (a) only |

   A new recipe is a row here plus its own `step7-recipe-*.md`.

5. **Prior work on the same kernel** -- if earlier investigations sit in the same
   artifact tree, one GPU-free command: `grep -lE '<solution index>|<kernel name
   prefix, ~60 chars>' <earlier ARTIFACT_DIRs>/rootcause/ROOT_CAUSE.md
   <earlier ARTIFACT_DIRs>/ledger.md` (both from `kernel.*.txt`). A hit is a
   **hypothesis, not a result**: read only its `ROOT_CAUSE.md` head (Where /
   Mechanism), enter it in `hypotheses.md` as `open (prior: <path>)`, and re-test it on
   *this* reproducer -- a different capsule, node or image can differ, so never carry
   its rates over. It does set the order: its causal experiment goes first. Ledger the
   hit (or `prior: none`).
6. `mkdir -p <ARTIFACT_DIR>/rootcause`; `$LG start step7[<G>]` -- **time the entry
   questions under the same tag** (`$LG wait step7[<G>] start <question>`), never a
   bare `step7` visit around the tagged one: its `other` would hold the whole group
   loop and the step-9 table would count it twice.

## Run (a) and (b) in parallel

The two tracks race to the same answer; neither waits on the other.

* **Start (b) first** -- its sweep is long and unattended: show the tiered plan
  (every applicable mutator, weakest first) and get its cost confirmed
  (`step7-recipe-sweep.md`), then start tier 1 through `mad-run.sh start` (`$LG job`)
  and schedule its one check (SKILL.md rule 9).
* **Run (a) while (b) runs**, on a different GPU. Never put load on a GPU (b) is
  sweeping: it skews (b)'s timings and contention (unrelated load comes off,
  `step7-recipe-sweep.md`).
* **Share findings at iteration boundaries**: a landed sweep row feeds (a)'s next
  hypothesis (a CLEAN `sbarrier_ds_write_read` = look for a RAW LDS race); an (a)
  finding can make the next recipe pair worth suggesting to the user.

## The loop body (one iteration = `rootcause/iter-<k>.md`)

**1. Gather.** Re-read `<LEDGER>`'s `step7` lines, `rootcause/hypotheses.md` and
the last `iter-<k-1>.md` (after a compaction: these first, nothing else), plus any
(b) rows that landed since (`<REPORT_JSON>` or its `.jsonl` journal). Open
`iter-<k>.md` with:

* **Status** -- elapsed vs `<RC_BUDGET>`, jobs in flight per track.
* **Learned last iteration** -- what worked, what didn't (an experiment that could
  not discriminate is a lesson, not a result).
* **Excluded** -- hypotheses ruled out, each with the experiment and its power.
* **This iteration** -- the open hypotheses to pursue, cheapest decisive first.

`rootcause/hypotheses.md` is the running table: `id | hypothesis | track | status
(open / excluded / supported / confirmed) | evidence path`. Never re-test an
excluded row without a stated reason.

**2. Track (a): native LLM debugging.** Your own reasoning and tools against the
reproducer. Per hypothesis, the cheapest experiment that can tell it apart from
the others. **Non-deterministic, first iteration -- three standard experiments**,
each one 1-GPU job beside (b)'s run, all with `--seconds` set below the job's
walltime (a starved loop then still reports a count):

* **Which op stops repeating** -- the 5b command with `--chain-repeat-check`
  (same capsule, `--contend <C-1>`, `--synthesize --restore-env`): same inputs
  every run, so a deterministic kernel repeats bit-for-bit. "First op to CHANGE in
  each non-finite run" names where the fault enters; when it is upstream of the
  op the finiteness tally blames, the blamed op is only computing from a finite
  but wrong operand. Read this before ordering hypotheses.
* **Does the load have to run the same kernel?** -- `--contend-with` a co-tenant
  that does not dispatch the suspect kernel (`transformer_hog --mode fwdbwd`
  uses explicit matmuls, no fused attention; `pressure_hog --mode copy` is HBM
  only). Reproduces = timing under load, not cross-process interference.
  For a GEMM (hipBLASLt) the cleanest "other kernel" is the same stand-in with the
  precision flipped -- `--tf32 0` (plain fp32 GEMMs) and `--dtype bf16` -- beside the
  `--tf32 1` positive control, one per GPU. Read the `STARVED` and `co-tenant said:`
  columns: a `pressure_hog` `triad`/`gemm` at `--procs 2` can starve the replay and
  `bust` can exit -- their zeros are not results.
* **Env amplifiers** -- each env var the run sets that touches the runtime
  (`HSA_DISABLE_CACHE`, `HSA_*`, `GPU_MAX_HW_QUEUES`, ...) flipped one at a time
  (`--env K=V`), both arms with the env set explicitly (no `--restore-env`, which
  would put it back). A var that takes the rate to 0 is a mitigation and bounds the
  trigger; it is not a fix.

**Deterministic, first iteration:** rebuild the operands *consistently* before
trusting any synthetic result. Synthetic derived operands overflow on their own.
Flash-attention varlen backward:
`python3 -m nan_doctor_rt.attn_consistent_check <capsule> [--layout nozero|short|long]`.
It keeps the capsule's real `cu_seqlens`, recomputes `out`/`lse` with the forward,
compares with an fp32 reference, and correlates the NaN rows with `lse < -88.72`
(the `exp(-lse)` overflow). For other ops, write the same thing: real integer
layout, statistics-drawn inputs, derived operands from the real producer, and a
reference.

**Deterministic GEMM** (`mm`/`addmm`/`bmm`/`_grouped_mm` wrong or NaN on every run) --
`nan_doctor_rt.gemm_shape_probe`, one cheap 1-GPU job per step, in this order:

1. `capsule <capsule>`: the culprit call on its recorded operands -- which expert
   groups are wrong (grouped GEMM: per-group row counts from `offs`), repeatable?,
   and one change at a time: contiguous copies, fresh storage, **random data at the
   recorded scale**, the bad group alone, **the bad group as a plain `torch.mm`**.
   Wrong on random data = shape-dependent, not value-dependent; wrong as plain mm =
   the library's GEMM, not the grouped path.
2. `sweep --k K --n N --dtype ...` at the bad group's K/N: every M on fresh data. Bad M
   in **bands** = kernel selection or tiling; it prints each band's tile count for
   the kernel's `MT<m>x<n>` (from `kernel.*.txt`: m tiles the output columns, n the
   rows); pass it as `--tile`, the default is one gfx950 kernel's. Look for what the
   bad counts share, against this device's `DEVICE ... CUs=` line (tile counts that
   are multiples of the XCD count and just below the CU count, with a stream-K
   kernel name, point at the stream-K grid). MI300X has 304 CUs, 8 XCDs; MI355X has
   256 CUs, 8 XCDs: the bands, if any, fall elsewhere per device.
3. The hipBLASLt problem and solution of the plain mm: `replay_nan_op <capsule>
   --bench` (a grouped capsule prints one plain-mm line per group size) or
   `HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE=...` on one call.
4. **Causal knobs**, each `sweep` in its own process (`--env VAR=VALUE`):
   `TENSILE_STREAMK_DYNAMIC_GRID=0`, `TENSILE_FIXED_WGMXCC=1`,
   `TENSILE_STREAMK_MAX_CUS=<CUs/2>`, `TENSILE_STREAMK_DATA_PARALLEL=1`,
   `TENSILE_SOLUTION_INDEX=<other>` (list them: `strings libhipblaslt.so.1 | grep
   -E '^TENSILE_'` inside the image). Bands that vanish under a knob and return without
   it, in the same job, are the causal evidence; re-check the knob on the capsule
   call (`capsule` mode under the env).
5. `sentinel --m <bad M> --k K --n N`: every wrong element still holding the
   sentinel = **tiles never written** (the model then reads leftover memory: NaN,
   Inf, or finite garbage = the grad-norm spikes); none = computed wrong.

rocBLAS is not an independent second implementation for bf16/fp16 here:
`torch.backends.cuda.preferred_blas_library("cublas")` fails identically (the same
Tensile kernels). Use an fp64/fp32 reference, not another library.

**Deterministic, value-dependent fault in a compiled (Triton / Inductor) kernel** --
5N already reduced it to one `tl.*` op over a value ladder (`step5n-compiled.md` C2).
One 1-GPU job of under a minute, four experiments:

1. **Who emits the sequence**: dump `h.asm["ttgir"]`, `["llir"]`, `["amdgcn"]` of
   the one-op kernel. The TTGIR op (e.g. `tt.precise_sqrt`) and the LLVM IR that
   replaces it tell Triton's own lowering (target intrinsics + arithmetic, e.g.
   `llvm.amdgcn.rsq.f32` + an fma chain) from LLVM's legalization (a generic
   `llvm.<op>.f32` call); also read the function's `"denormal-fp-math-f32"` and the
   ISA's `.amdhsa_float_denorm_mode_32` -- whether subnormals reach the instructions.
2. **Where it turns non-finite**: re-run the ISA's instruction sequence one
   `tl.inline_asm_elementwise` per instruction (`"=v,v,..."`, `is_pure=True,
   pack=1`) and store every intermediate for one failing input: the first `inf` /
   `NaN` names the instruction. Jit limits: no Python lists or `float("inf")` inside
   the kernel -- one `tl.store` per intermediate, compare against a finite bound.
3. **Causal**: the same instruction sequence with the mechanism removed (e.g. inputs
   pre-scaled by 2^32 and the result by 2^-16) in the same job as the unchanged one --
   correct vs non-finite on the same inputs.
4. **A correct twin in the same toolchain**: `libdevice.<op>` (ocml) and its ISA show
   what a correct expansion does differently (threshold compare, `v_ldexp_f32`).

The source file of the lowering is a reading aid, not required: the LLVM IR lines are
the *where*.

Then, per hypothesis, e.g.:

* read the kernel (`--disasm`, `<ASM_DIR>`, the aiter/CK source, the Tensile
  solution) and the op's code path in the library;
* vary one thing in the reproducer -- shape, stride/alignment, dtype, precision
  flag, `--solution-index`, workspace, concurrency, co-tenant -- and see whether the
  NaN follows;
* a second implementation (`step5-replay-guardrail.md` items 1-3), per-op fp64
  values, operand repair;
* minimal instrumentation or a patched build of the suspect code.

Each GPU run goes through `madengine nan exec` started by `mad-run.sh` (`$LG job`),
output under `rootcause/`. **Power as everywhere**: a "NaN gone" needs >= 3 expected
hits at the reproducer's rate; below that it is `UNDERPOWERED`, never an exclusion.

**3. Track (b): the recipe.** Fold rows that landed into `hypotheses.md` (a CLEAN
row with its `expected_hits` supports a mechanism). Its next sweep is tier 2: the
stronger sibling of every tier-1 row that was not a powered `CLEAN`
(`step7-recipe-sweep.md`); a powered `CLEAN` makes its stronger sibling
predetermined `CLEAN`, not run.

**4. Decide.** Ledger one line: `step7[<G>] iter <k>: <verdict>; <path>`.

* **Root cause found** -> write `rootcause/ROOT_CAUSE.md` and stop the loop (cancel
  (b)'s runs -- its journal resumes them if wanted -- and say so). Valid only if
  it has all of:
  * **where**: kernel + instruction addresses, or a source file:line / op;
  * **mechanism**: how a NaN comes out (e.g. a wave reads an LDS tile before its
    producer's write, a missing `s_waitcnt`, overflow in an fp16 accumulator);
  * **trigger**: why it fires at the reproducer's rate and load, and why it is
    `<NAN_KIND>`;
  * **causal evidence**: an intervention on the mechanism that makes the reproducer
    clean at sufficient power, and back to `NAN` when undone -- correlation or
    reading alone is a hypothesis, not a root cause;
  * consistent with every ledger fact (rates, clean loads, 5N removals).
* **Budget left** -> next iteration.
* **Budget spent** -> `<ROOT_CAUSE>` = `NO_ROOT_CAUSE`: write
  `rootcause/ROOT_CAUSE.md` anyway with the excluded and still-open hypotheses
  (the next investigator starts there), cancel in-flight runs or ask whether to let
  a nearly finished one land.

## Exit

`$LG end step7[<G>]`; ledger `<ROOT_CAUSE>` (valid + one-line summary, or
`NO_ROOT_CAUSE`) and the path. Tell the user in two lines. Valid and fixing wanted
(step-1 Q4) -> step 8; else step 9.
