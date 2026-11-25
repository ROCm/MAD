# Step 7 (on by default, budgeted): root-causing -- a loop over the step-6 reproducer

> Experimental under MAD: the automated workflow runs step 7 only with `--rootcause`, skips step 8 and never fixes anything; findings and fixes are proposed, never applied to MAD or the image.

Step 6 left a packaged reproducer, not an explanation. Step 7 looks for **why**
the NaN happens, working only from that reproducer (tarball or `repro-native/`) --
never a model run (a model / numerics NaN also reads the model's source and config,
statically: *Model / numerics NaN*, below). It ends with a valid `<ROOT_CAUSE>`, or `NO_ROOT_CAUSE` once
`<RC_BUDGET>` is spent. Tag `step7[<G>]`, one loop per distinct kernel/reproducer.
Tooling: `nan_doctor_rt` modules (`kernel_patch`, `kernel_mutations`, `gemm_shape_probe`, `attn_consistent_check`, `replay_nan_op`, ...) via `madengine nan exec --image <IMAGE> -- python3 -m nan_doctor_rt.<module> ...`; every GPU job is started detached with `mad-run.sh`.

## Entry

**A COLLECTIVE reproducer** (`step4-distributed.md` *Synchronous collectives*: `COLLECTIVE_FAULT`, the
member ranks' capsules replayed through the framework): the suspect is the collective's
implementation the framework picked for that size, not RCCL by default -- the replay's first line
names the communicator; the env that selects it is in the capsules (`VLLM_*`). A
`NARROW_INTERMEDIATE? <dtype>` hint is the first hypothesis: find the cast / packing to that dtype in
the implementation's source (the framework's installed package and its matching release's
sources) and test it with one env change at a time on the replay -- `bin/collective_arms.sh
<capsule dir> <arms file>` (`madengine nan exec --gpus <one per rank> -- bash
\$NAN_DOCTOR_RT/bin/collective_arms.sh ...`) runs each arm as its own torchrun and prints the
implementation each took (`path=`), keeping the code path (*An intervention must keep the code
path*); then *Why was it not prevented?*.

0. **The reproducer is finished first**: step 6's tarball verified from an extracted copy (the
   verify run's pass in `<LEDGER>`). While that verify run is queued or running, step 7 may
   **prepare** -- read priors, write `hypotheses.md`, plan experiments -- but starts no
   experiment and opens no `step7` visit: an experiment on an unverified package can be on the
   wrong artifact, and overlapping visits double-count in the time table.
1. **Wanted?** Step-1 Q4, default **yes** (an unattended run told to take defaults: yes).
   The user turned it off (Q4 `no`, or "stop at the reproducer" at any point, the step-6
   hand-over included) -> `$LG skip step7 "<why>"` (and `$LG skip step8 "no
   root-causing"`), go to step 9. "ask me after 6" (ask-me mode): the step-6 hand-over
   asked it.
2. **`<RC_BUDGET>`** -- **default mode: take the default, don't ask** (the step-6 hand-over
   already quoted it; a budget the user gave there wins). Ask only in ask-me mode, or when
   the defaults' GPU-hours for steps 7 and 8 together exceed the step-1 threshold (8 hours
   of the host's GPUs) -- one line. Wall clock for the whole loop, both tracks (default:
   **1 h when the reproducer hits every run** -- deterministic: experiments are
   1-GPU jobs of seconds -- **4 h** when it is non-deterministic: sweeps and powered
   rates), and the GPU-hours it may spend. Classify `<NAN_KIND>` (item 3) first so
   the default you offer fits it. **Step 8 wanted too: set (or ask) `<FIX_BUDGET>`
   together with it** (its defaults, so step 8's file need not be read yet: **1 h** for
   an env-setting workaround or a model / config fix on a deterministic reproducer,
   **4 h** for a kernel patch; plus GPU-hours), so `default` settles both; a second
   budget question at step 8's entry stalls on a user who already said "do both".
   Ledger both.
3. **Classify `<NAN_KIND>` and `<NAN_ORIGIN>`** from `<LEDGER>` (don't re-measure) and
   ledger both with the evidence:

   | `<NAN_KIND>` | evidence |
   |---|---|
   | `deterministic` | `[C1]` walk non-finite with no `DIVERGED`; or the reproducer hits every run (`n/N = N/N`), solo too (5N: "hits every run, also solo") |
   | `non-deterministic` | `0 < n < N` at the reproducing concurrency; contention-gated; a `DIVERGED` walk |

   And `<NAN_ORIGIN>`, from the step-5 / 5N verdict -- both go through steps 7-8:

   | `<NAN_ORIGIN>` | verdict |
   |---|---|
   | `system` | `KERNEL_BUG`, `KERNEL_BUG_UNVERIFIED`, `NATIVE_KERNEL_BUG`, `NATIVE_UNVERIFIED`: the op returns non-finite where the fp64 reference is finite (kernel, library, compiler, runtime, hardware) |
   | `model` | `FP32_OVERFLOW?` (overflow, or undefined arithmetic -- the walk's reason line), `NATIVE_MATH`, `NATIVE_FP32_OVERFLOW`: the op computes faithfully, the true result is not finite -- the model handed it those values |

4. **Pick the tracks** -- (a) always; (b) only if a recipe exists for `<NAN_KIND>` and
   `<NAN_ORIGIN>` is `system` (the mutation sweep hunts kernel faults; a `model` NaN is
   track (a) only, *Model / numerics NaN*):

   | `<NAN_KIND>` | (b) recipe |
   |---|---|
   | `non-deterministic` | the mutation sweep, `step7-recipe-sweep.md` (no `identity` / `nans_to_nops`) |
   | `deterministic` | none -- track (a) only |

   A new recipe is a row here plus its own `step7-recipe-*.md`.

5. **Prior work on the same kernel** (skip with `<PRIORS>` = none: ledger `prior: skipped
   (PRIORS=none)`) -- if earlier investigations sit in the same
   artifact tree, one GPU-free command: `grep -lE '<solution index>|<kernel name
   prefix, ~60 chars>' <earlier ARTIFACT_DIRs>/rootcause/ROOT_CAUSE.md
   <earlier ARTIFACT_DIRs>/ledger.md` (both from `kernel.*.txt`). A hit is a
   **hypothesis, not a result**: read only its `ROOT_CAUSE.md` head (Where /
   Mechanism), enter it in `hypotheses.md` as `open (prior: <path>)`, and re-test it on
   *this* reproducer -- a different capsule, node or image can differ, so never carry
   its rates over. It does set the order: its causal experiment goes first. Ledger the
   hit (`prior: <path>`, or `prior: none`); the report says "re-confirmed from <path>" for
   a root cause that came from it -- a confirmation, not a discovery.
6. `mkdir -p <ARTIFACT_DIR>/rootcause`; `$LG start step7[<G>]` -- **time the entry
   questions under the same tag** (`$LG wait step7[<G>] start <question>`), never a
   bare `step7` visit around the tagged one: its `other` would hold the whole group
   loop and the step-9 table would count it twice.

## Run (a) and (b) in parallel

The two tracks race to the same answer; neither waits on the other.

**Every experiment, both tracks (and step 8's validation): runs sized by
`madengine nan power`** from the reproducer's rate under the same load (`RUNS=`, 30
expected hits), not a round 1M; a CLEAN checked with `madengine nan power <rate>
--clean 0/<runs>`.

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

* **Which op stops repeating** -- only for a chain capsule (5b ran `--chain`; an
  op-mode 5b, or a co-tenant replay of one op, has no chain: skip it and ledger why).
  The 5b command with `--chain-repeat-check`
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

**Read of uninitialized memory** -- 5N's poison test hit (the rate follows the fill value) or
a capsule's kept `uninit` operand is read by the op. Name the buffer, then who should write it.
One 1-GPU job per step, on the reproducer's command:

1. **Which allocation**: `python3 -m nan_doctor_rt.alloc_poison --sites -- <cmd>` lists every
   `torch.empty*` call site (`ALLOC_SITE <file:line> count= <dtype(shape)>`); then one arm per
   candidate site, `--empty <bad value> --site <file:line>` (the rest left alone) -- the site
   whose poison alone flips the result is the buffer. Small integer buffers first (counts,
   indices, flags).
2. **Who reads it, who writes it**: grep the buffer's name in the library (Python wrapper and
   C++ / kernel launch code); a read guarded by a flag that does not check the arch, or a writer
   only on some path (one arch's kernel ABI, one dtype) is the defect. Prove "never written"
   directly: pre-fill it with a sentinel, run the writer alone, check it is unchanged.
3. **Causal**: the same reproducer with the buffer initialized to its meant value (or the read
   gated off) in the same job as the stock one -- clean vs NaN, every fill value.
4. **History**: `git log -S'<the allocation line>'` in the library -- an `empty` that replaced
   a `full`/`zeros` ("the kernel writes it") is the usual origin.

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

**Model / numerics NaN** (`<NAN_ORIGIN>` = `model`) -- the op is correct; the question is
why the model hands it operands whose true result is not finite. Usually deterministic,
1-2 iterations of 1-GPU jobs of seconds plus reading. In order:

1. **The arithmetic, on the recorded operands**: which elements go non-finite and the
   operand values that make them (a zero row into an eps-less norm, `|x|` above the
   dtype max into an `exp`, a zero denominator). Start from `capsule_info`'s `rows` line
   (the operand's all-zero / non-finite rows, the chain's token ids or indices at them, the
   eps: `input rows all zero: [46] <- embedding indices {46: 151819}; epsilon 0`); 5N's
   candidate for a native reproducer.
   **Experiments: `nan_doctor_rt.capsule_operands`, never a hand `torch.load`** -- the capsule
   stores tensors flat (`args[i]["base"]`, pool keys) and a hand script finds neither the
   input nor the embedding table: `c = load(<pt>, device="cuda")`; `c.arg("epsilon")`,
   `c.arg("input")`, `c.chain[i].args` (the embedding's table and ids), `c.func(...)` re-runs
   the op with one factor changed (item 3).
2. **Where those values come from**: walk the capsule's chain (`capsule_info`) back to
   its roots, then map each root to the model -- the module and frames the capsule names
   (`frames`), its source and config, read statically (`madengine nan capture-points`
   windows, `grep`): the config value (`eps`, a scale, a dtype), the parameter (an
   uninitialised or zeroed row), the input (an id outside the trained range), the code path
   (a missing mask or clamp).
3. **The conditions, each necessary?** Every factor gets its own causal experiment on the
   reproducer: change that one factor, keep the rest -- e.g. eps `0 -> 1e-6`, the zero row
   -> a trained-like row, the bad id -> an in-range one -- and the NaN must go, and come
   back when undone (a native candidate: its 5N removals already are these rows; a capsule:
   a script that rebuilds the op's inputs from the capsule with the one change). Two
   independent conditions that are each harmless alone are both the root cause: name both.
4. **The trigger in the model**: why the model reaches those values at the rate it NaNs
   (e.g. every 50th request carries an id >= the tokenizer size), from the run script,
   the data, the request pattern -- read, not re-run.

`ROOT_CAUSE.md`'s *where* is then the model source / config line(s), the *mechanism* the
arithmetic, the *causal evidence* the per-factor experiments of item 3.

Then, per hypothesis (a `system` NaN), e.g.:

* read the kernel (`--disasm`, `<ASM_DIR>`, the aiter/CK source, the Tensile
  solution) and the op's code path in the library;
* vary one thing in the reproducer -- shape, stride/alignment, dtype, precision
  flag, `--solution-index`, workspace, concurrency, co-tenant -- and see whether the
  NaN follows;
* a second implementation (`step5-replay-guardrail.md` items 1-3), per-op fp64
  values, operand repair;
* minimal instrumentation or a patched build of the suspect code;
* **the failure signature** -- how the output is wrong, not only that it is:
  `nan_doctor_rt.failure_signature` (`signature(out, ref, prev)`; the replays print its
  `SIGNATURE` line): NaN / Inf / saturated at a dtype max / zeros / garbage / the previous run's
  value / a small error, where (rows, aligned runs) and the reference's range there. Each mark
  points at a different mechanism; its `reading:` is a hypothesis to test, never a verdict;
* **a twin capsule's disagreement**: `nan_doctor_rt.twins.check(op, func, args, kwargs, pre)` on
  the capsule's operands (`capsule_operands.load`) returns both outputs -- which rows / tiles /
  splits differ, and by how much, is the first map of the fault (a stale split, a skipped tile,
  one head); a `native` twin is also the fix's value reference;
* **differential replay** (*Why was it not prevented?*, below): the same package on another
  axis of the environment -- arch, library version, build.

**An intervention must keep the code path.** Libraries dispatch on size, dtype, arch and env
(a size threshold, a fallback, a different kernel per arch): a change that makes the NaN go may
only have sent the call down another path, and says nothing about the one that failed. Log
which implementation ran in every arm (the library's own log / debug env, a print in a probe,
`LoadKernel` lines, the communicator a collective used), and when the change moves the call,
add an arm that forces the original path with only the suspect condition changed.

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
  * consistent with every ledger fact (rates, clean loads, 5N removals);
  * **why it was not prevented** -- *Why was it not prevented?*, below: the defence the code
    has for this condition (or "none"), and why it did not apply here.
* **Budget left** -> next iteration.
* **Budget spent** -> `<ROOT_CAUSE>` = `NO_ROOT_CAUSE`: write
  `rootcause/ROOT_CAUSE.md` anyway with the excluded and still-open hypotheses
  (the next investigator starts there), cancel in-flight runs or ask whether to let
  a nearly finished one land.

## Why was it not prevented? (before a root cause counts as found)

A causal mechanism ("X overflows", "Y reads before the write lands") is where the code went
wrong, not yet why: code usually means to handle the condition it failed on. Ask, in order,
and write the answers into `ROOT_CAUSE.md` (*Defence*):

1. **Intent.** Does the code already try to handle this condition? A guard is either **inline**
   at the site -- a bounds or range check, a clamp, a mask, a barrier / wait / fence, an assert, a
   fallback branch -- or **state set beforehand**, far from the failing line: a hardware mode
   (saturation, denormals, rounding, IEEE / clamp bits -- set once per kernel or per wave), an
   initialisation (a zero-fill, a sentinel), a library or framework flag, a validation of the
   inputs or config. So read the site AND everything that runs before it on the same data and
   thread: the kernel's prologue, the constructors / init calls of the objects it uses, the
   launch wrapper, the host function that configures the call. **Finding one guard does not end
   the search** -- one that runs after the damage, or covers another case, is a finding, not the
   answer: list every guard on the path and say for each whether it covers this condition.
   None -> *Defence: none* (the condition is unhandled -- a real finding). The source is the
   version that RAN: an installed wheel or image often ships no C++/HIP sources (vLLM, aiter,
   hipBLASLt) -- read them at the matching release tag of the public repo (`<pkg>.__version__`,
   the image's package list) and ledger the URL and tag. A web fetch to github.com may be
   refused; `curl -sL` often works: find the path at the tag with the tree API
   (`https://api.github.com/repos/<org>/<repo>/git/trees/<tag>?recursive=1` -- files move between
   releases), then `https://raw.githubusercontent.com/<org>/<repo>/<tag>/<path>` into
   `rootcause/src_<tag>/`. Neither reachable: say so and rest the Defence on the binary (item 3).
2. **Applicability.** There is one: why did it not apply to this run? A defence can be
   conditional on the arch (`#if defined(__gfx9xx__)`, a per-arch table), a build flag or macro
   (debug-only checks), a runtime switch (env, config, a mode the caller must set), a version
   (added in a later release than the one shipped), a dispatch condition (another path takes
   the guard), or an ordering (the guard runs after the damage). Find the condition and which
   side of it this run was on.
3. **Binary, not only source.** For compiled code, check that the branch the source implies is
   in the code object THIS GPU loads: `madengine nan exec --image <IMAGE> -- python3 -m
   nan_doctor_rt.gpu_objects <the library's .so / .co> --kernel <regex> --grep '<the guard's
   instruction or call>'` lists the code objects per arch and counts the pattern per kernel --
   `ARCH_DIFF` names arches whose code differs for the same kernels. Start from the guard
   signatures, whatever the source made you expect -- the source can be compiled out, the
   binary cannot lie:

   * a hardware mode set (saturation / overflow, denormals, rounding, IEEE, clamp):
     `s_setreg|s_denorm_mode|s_round_mode`
   * a clamp / saturate on an instruction: ` clamp` (the VOP modifier), `v_med3|v_min_f|v_max_f`
   * a wait / barrier / fence: `s_waitcnt|s_barrier|buffer_wbl2|buffer_inv`
   * a bounds or range check: `v_cmp_|s_cmp_` near the site, `s_cbranch`
   * a call to a checking / helper function: `s_swappc|s_getpc`, or the callee's name

   A pattern present in one arch's code and absent in another's is `ARCH_DIFF` -- then read the
   source for the conditional (`#if`) that explains it.
4. **Differential replay** (when 2-3 leave it open, or to confirm): run the SAME step-6 package
   with ONE axis of the environment changed -- another Instinct arch you can reach
   (`madengine nan gpu-check` names the arch), the library's previous or next release (another
   image), a build or config variant the library ships. Behaviour that differs localises the
   bug to what differs: a different output on another arch is code compiled differently per
   arch; a NaN in one release only is a regression between them. How: every step-6 package's
   `run.sh` takes `IMAGE=<another image>` (another release) and runs on any GPU host; a native
   package the same. Ledger which axes were reachable and which were not -- an axis not tried
   is not an exclusion; when no other arch is reachable, item 3's binary check (`ARCH_DIFF`) is
   the cross-arch evidence.

A defence that exists but did not apply is the root cause's real *where* (the condition, the
missing branch); the mechanism stays as evidence. Budget: this pass is part of step 7's loop,
one iteration or two -- reading and a few seconds of `gpu_objects`; a differential replay is
one run per axis.

## Exit

`$LG end step7[<G>]`; ledger `<ROOT_CAUSE>` (valid + one-line summary, or
`NO_ROOT_CAUSE`) and the path. Tell the user in two lines. Valid and fixing not turned
off (step-1 Q4) -> step 8; else step 9.
