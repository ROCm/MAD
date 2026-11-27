# Step 5N, compiled track: a NaN born inside torch.compile'd code

Read from `step5n-native.md` N1 when the NaN may be born inside a compiled region and the
capture did not see it. **Since 2026-11-21 the capture watches Inductor's kernels**
(`NAN_CAPTURE_INDUCTOR`, on by default; the capture log says `Inductor (torch.compile) kernels:
watched`): each launch of a generated kernel -- inside a `NAN_CAPTURE_SUSPEND` region too, where
the dispatcher ops stay unseen -- is an op `inductor.<kernel>.s<hash>`, and a NaN born there comes
back as an ordinary group that 5b replays (`step5-read.md`). That is not this track. This track is
for what the capture still cannot see: kernels replayed from a **CUDA / HIP graph** (vLLM's default
`cudagraph_mode`; for vLLM run the capture lines graphs off instead, `step4-run.md` *vLLM with graphs*,
and a NaN that needs graphs goes to *Graph-only NaN* below), the **C++ wrapper** (`cpp_wrapper`, AOTInductor), a capture run with
`NAN_CAPTURE_INDUCTOR=0`, or older tools. There the first capsule is the first *eager* op that reads
the NaN -- `PROPAGATED`, chain stopped at `outside_window` -- and names the consumer, never the
producer (2026-11-19 and earlier: every compiled region). This track replaces N2 (locate) and
the start of N3 (the first candidate) with Inductor's own instruments. N3-N6 (candidate contract,
load, reduction, verdict) still apply.

## Graph-only NaN (vLLM: the NaN needs graph replay)

The capture lines ran graphs off (`step4-run.md` *vLLM with graphs*) and stayed clean while the
graphs-on controls NaN'd; or a model-level A/B -- the run as is vs `PYTHONPATH=$ND_REPO/scripts/vllm_graphs
ND_VLLM_CUDAGRAPH_MODE=NONE` -- shows NaN only with graphs. Neither the capture nor C1 can see it:
both need graphs off. The `g` line (`ND_GRAPH_CHECK=1`) checked every graph replay; its log says

    [vllm-graphs] GRAPH_CHECK verdict=<GRAPH_ONLY|NOT_GRAPH_ONLY> piece=<i> mode=<PIECEWISE|FULL> batch=<descriptor> rows=<..> of <n> -> <file>

* `GRAPH_ONLY`: piece `<i>` replayed from its graph turned finite inputs into non-finite outputs, and
  the same piece run without the graph, on the same inputs in the same forward context, was finite.
  `batch` is the capture size it was padded to (`num_tokens`); `rows` the non-finite output rows --
  all of them past the real token count = padding rows only (harmless unless something reads them).
* `NOT_GRAPH_ONLY`: the piece itself makes the NaN, graph or not -- the capture lines should have caught
  it; check they really ran graphs off (`cudagraph_mode=NONE` in their logs).
* No line: no replay went non-finite from finite inputs -- the NaN is born outside the pieces (an eager
  op between them -- the capture sees those) or is propagated in.
* `runner=v2` (vLLM's V2 GPU runner -- newer models -- replays whole-step FULL graphs itself):
  the check is on the step's hidden states, the re-run without the graph uses the step's own capture closure, and
  `desc=...num_tokens=<n>, num_reqs=<r>` is the capture size; a non-finite row at index >= the real request count is a
  padding row. The capsule (`graph_replay_*.pt`) holds no piece: the model-level A/B is the reproducer. 2026-11-22:
  `GRAPH_ONLY runner=v2 num_tokens=4 rows=[3] of 4` -- a batch of 3 in the 4-token graph, its padding row non-finite.

**Reproducer**: `scripts/replay_graph_piece.py <file> --iters 20 [--compile] [--import <module>]` runs the
piece's saved FX code on the saved inputs directly and captured in a fresh CUDA graph:
`GRAPH_PIECE_REPLAY=GRAPH_ONLY eager=0/20 graph=20/20` reproduces it without the model -- package it as
a native reproducer (step 6 *Native reproducer*: `repro.py` = this call, the capsule beside it).
`NEITHER`: the trigger is outside the piece (the shared graph pool, vLLM's own capture order) -- the
model-level A/B above is the reproducer. A full graph (`mode=FULL`, attention inside) has no FX module
saved: the A/B again. 2026-11-22 (test injection: a custom op whose Python branch ran only when not
capturing): piece 4, `GRAPH_ONLY`, replay eager 0/10, graph 10/10, also `--compile`.

**Root cause (step 7)**: what a graph freezes at capture -- a Python branch or size evaluated once
(`torch.cuda.is_current_stream_capturing()`, a host scalar), a pointer to a host or freed buffer, a
size padded to the capture size and the rows it pads with, a stream or event outside the graph. Read
the piece's FX code (`replay_graph_piece.py --info`) for the custom ops it calls; their Python bodies
run only at capture.

## Compiled track or plain 5N?

| evidence (from N1) | use |
|---|---|
| capsule `PROPAGATED`, chain `outside_window`, frames at the first eager op after an `OptimizedModule` (or its dirty operand is a compiled module's output) | **compiled track** -- C0 first |
| `NO_CAPSULE` / `NO_REPLAYABLE_GROUP` while the model NaNs, and most of the step is compiled (whole-model `torch.compile`, `fullgraph`, a compiled loss or optimizer) | **compiled track** |
| C0: the compile-off model also NaNs | **plain 5N** -- model numerics or an eager kernel; compilation is a bystander (E50: `0 * rsqrt(0)`, NaN eager too). **Still read C1 from the same wave** before the first candidate: its kernel's source nodes and constants (an `eps` of `0.0`) make candidate 1 exact (2026-11-19 vLLM: C0 4/4 NaN, C1 named the fused embedding + `rms_norm` kernel, candidate 1 hit) |
| C1: the first assert is a *graph input* | **plain 5N** with the upstream eager producer as suspect |
| the faulting op is load- or timing-gated (needs a co-tenant or concurrency, 5c/5d reproduced) | **plain 5N** -- C1 adds a host sync after every kernel, which removes races; at most run C0. A NaN *step* that merely varies between trials (compiled kernels drift by 1e-5 and the trajectories part) is not this: every trial still NaNs, alone |
| the compiled region is one or two blocks and N1 already names a plausible suspect (a norm, an eps, a softmax, zero/padded rows) | **plain 5N bottom-up** -- a direct candidate costs one job (E51: 3 candidates, ~4 min). Fall back to this track if the candidate stays clean |

Both apply (a big compiled region *and* a cheap obvious suspect): one 5N candidate job and
C0 in parallel; whichever localises first wins. Ledger the choice: `5N track: compiled |
plain -- <which row>`.

**Already in the capture wave?** When step 4 carried the C0 and C1 lines (`step4-run.md` *Compiled
NaN predicted*), they ran: read arms `c0` / `c1` in the capture job's `wave_summary.<tag>.txt` and
logs, `output_code.c1.log` and `capsules_inductor/` -- don't submit the C0/C1 wave below again; go
to *Kernel capsule*. On `NO_REPLAYABLE_GROUP` the capture job has already replayed the newest kernel
capsule from an extracted `repro-kernel/` package (`KERNEL_CAPSULE=` in its result line,
`replay/kernel/run.<tag>.<host>.out`, the `step6[kernel]` ledger lines): read that, don't replay again.

## C0. Compile-off control (one wave, the discriminator)

Same geometry and step count as the step-4 wave, no capture, compilation off: the
model's own switch (`--compile-layers none`, `compile: false`), else
`TORCHDYNAMO_DISABLE=1` in the trainer's environment. Clean on every trial while the
compiled controls NaN'd at the ledgered step = born in compiled code (ledger `C0 clean
n/N`). Compile-off also NaNs: plain 5N. A different NaN step under compile-off is not
"clean": run it to the step-4 wave's length.

## C1. `nan_asserts` on the model (one wave, no capture)

Inductor checks, after every generated kernel, that its output buffers hold no NaN/Inf,
plus every graph input and output (`torch/_inductor/codegen/wrapper.py`,
`codegen/simd.py`, torch 2.10). Env on the trainer, via the same route as `NAN_*`:

```
TORCHINDUCTOR_NAN_ASSERTS=1  TORCH_LOGS=output_code  TORCH_LOGS_OUT=$ND_ART/wave_c0c1/output_code.<gpu>.log
PYTHONPATH=$ND_REPO/scripts/inductor_capsule  INDUCTOR_CAPSULE_DIR=$ND_ART/capsules_inductor
```

The last two save the failing kernel as a **kernel capsule** at the assert (*Kernel capsule*,
below): always set them.

Check `output_code.<gpu>.log` holds `assert not ... isnan()` lines (no
lines = the flag did not reach the trainer, or a cached graph was reused: give each
trial its own `TORCHINDUCTOR_CACHE_DIR`). Early stop at the first
`AssertionError`. Read the traceback's last frame -- a line in the generated wrapper
(`/tmp/inductor/.../*.py`) -- and the kernel call just above it in `output_code.<gpu>.log`:

| first assert | reading |
|---|---|
| a graph **input** (`# make sure graph inputs are not nan/inf`) | the NaN came from eager code upstream -- plain 5N, suspect upstream |
| a **kernel output** buffer | born in that fused kernel; its name and `# Topologically Sorted Source Nodes:` comment list the ATen ops fused into it (e.g. `triton_per_fused_add_div_mean_pow_sqrt_0`). Suspect = that kernel |
| a graph **output** only | born in the last kernel(s); same as above |

**Granularity is the fused kernel, not the op**: a whole RMSNorm is one kernel. **It
asserts on Inf too**: a legitimate `-inf` (attention mask, a max's initial value) stops
the run as well -- a kernel whose eager twin is also non-finite is a false
positive; strike it and re-run with that graph left eager, or read past it in a
debugger. A NaN made and masked *inside* one kernel never asserts (correct: it never
leaves). `TORCHINDUCTOR_RUNTIME_TRITON_NAN_ASSERTS=1` puts a `tl.device_assert` on every
value inside the kernels (per-op), but on ROCm 7.2 it **did not fire** on a real NaN
(device asserts compiled out) -- do not read its silence as clean.

Ledger: `C1 first assert: <input|kernel|output> <kernel name> <graph/module> step <n>`.

**Run C0 and C1 together, always** -- the wave costs the same, and C1 is what names the
kernel whichever way C0 comes out. C0 and C1 fit one node as one wave: one `wave.sh` spec (`step4-run.md` format) with C0
lines, arm `c0` (the compile-off flag), and C1 lines, arm `c1` (the env above); each trial
gets its own inductor cache (`nd run`). Both arms run the ORIGINAL source (no capture). Its
own wave and capsule dirs -- the defaults hold step 4's logs and capsules, and `CAPSULE_DIR`
must be empty:

```bash
ND_ENV=<ENV> SPEC=<ARTIFACT_DIR>/c0c1.spec NAN_SIGNAL='<NAN_SIGNAL>' WAVE_DIR=<ARTIFACT_DIR>/wave_c0c1 \
  CAPSULE_DIR=<ARTIFACT_DIR>/capsules_c0c1 <REPO_ROOT>/scripts/nd submit --name c0c1 --gpus 8 --exclusive -- wave.sh
python3 <REPO_ROOT>/scripts/wave_summary.py <ARTIFACT_DIR>/wave_c0c1 --signal '<NAN_SIGNAL>'
```

A C1 trial stops on Inductor's `AssertionError`, not the model's NaN line: `wave_summary.py` scores
it `NAN` with the note `inductor NaN assert` (the early stop's signal must still match the model's
line, or the C0 trials run to the end). A C1 trial scored `DIED` with an `AssertionError` in its log
is a summarizer miss, not a death -- read the log.

 The traceback's wrapper line maps into the log: the module's source
starts on the line after its `Output code:` header, so wrapper line N is log line
header + N.

## vLLM

vLLM compiles by default (`enforce_eager=False`) and replays the compiled pieces as HIP graphs
(`cudagraph_mode` `FULL_AND_PIECEWISE`, capture sizes 1..512 tokens); its attention ops run eager
between the pieces, so the first capsule is usually `vllm.unified_attention_with_output`
`PROPAGATED`, with a frame `__vllm_inlined_submods__<n>` (the compiled piece before it). The
capture preflight's `VLLM` lines say whether the engine is in-process and which switch is C0's.

* **C0**: `enforce_eager=True` (the script's switch, e.g. `--enforce-eager 1`); vLLM ignores
  `TORCHDYNAMO_DISABLE`.
* **C1 needs HIP graphs off, compilation kept**: `nan_asserts` read every buffer to the host, which
  is illegal during graph capture -- the engine dies at start-up. A variant copy of the serving
  script (`<ARTIFACT_DIR>/native/serve_c1.py`, never the original) passes
  `compilation_config={"cudagraph_mode": "NONE"}` to `LLM(...)`; the wave line runs it with the C1
  env. vLLM writes the graphs under its own cache (`~/.cache/vllm/torch_compile_cache`, `HOME=/tmp`
  in nd's containers: fresh per trial); the traceback's last frame is a file under
  `.../inductor_cache/`, found in `output_code.<gpu>.log` by its `Output code written to:` line.
* vLLM prints tracebacks with torch's `[rank0]: ` prefix; `wave_summary.py` strips it, so a C1
  trial stopped by Inductor's assert scores `NAN` (`inductor NaN assert`) with the model's signal
  alone (before 2026-11-19 the prefix hid it: 4 of 4 C1 trials scored `DIED`).

## Kernel capsule (C1's assert, saved)

`scripts/inductor_capsule/sitecustomize.py` (on C1's `PYTHONPATH`, with `INDUCTOR_CAPSULE_DIR`) catches
the first `nan_asserts` `AssertionError` where it is raised (Python 3.12 `sys.monitoring`) and saves the
kernel that wrote the asserted buffer: its Triton source, the launch line, every launch argument --
the inputs with their **real values**, outputs as shapes -- in / out / in_out roles, the model frames,
torch / triton / device. The trial log says `[op-nan-inductor] captured kernel <name> (assert on <buf>)
-> <file>; inputs finite`. Replay it without the model, in the model's image:

```bash
nd run --gpus 0 -- 'python3 $ND_REPO/scripts/replay_inductor_kernel.py $ND_ART/capsules_inductor/inductor_*.pt --iters 1000'
```

`INDUCTOR_REPLAY_RESULT=<hits>/<iters> ... nonfinite=<n> matches_model=<bool> fp64_ref=<status>` -- `fp64_ref`: the
checked output in the kernel's fp64 twin (`inductor_ref.py`; `finite`, `NON-FINITE` or `unavailable:<why>`); `matches_model`: every output
bit-identical to the model's launch (the capsule stores them); `--info` describes the capsule without a GPU.
Reading:

* **hits, inputs finite**: the kernel turns these real inputs into NaN on its own -- **this is the
  reproducer** (step 6 *Kernel capsule reproducer*); no hand-written candidate is needed to have
  one. 2026-11-19 (vLLM, embedding + `rms_norm` with eps 0): 100/100, the same 896 non-finite values
  as the model, real embedding table and token ids.
* The verdict: `fp64_ref=finite` -> the kernel or its lowering; `NON-FINITE` -> `NATIVE_MATH`. Only with
  `fp64_ref=unavailable:...`, or to name the source node that fails, run the kernel's source nodes (its `# Topologically Sorted
  Source Nodes` comment; the FX graph of them is `repro-kernel/source_graph.txt` when the capture job
  packaged it) eagerly and in fp64 on the capsule's inputs -- a C2 candidate built from the
  capsule, not from a guess; get the inputs with `replay_inductor_kernel.build_args(capsule, "cpu")`
  (it rebuilds sparse ones), never from the capsule's fields directly: fp64 non-finite too = `NATIVE_MATH`; finite = the kernel or its lowering
  (C2 item 3's table).
* **0 hits**: an `in_out` argument (post-call values: the log says so), or the NaN needs the
  neighbouring kernels -- fall back to C2 below.
* No capsule though C1 asserted: the assert was on a graph input or output, not a kernel output (the
  log says which) -- the NaN came from upstream eager code: plain 5N.

Size: a big input (>= 16 MB, `INDUCTOR_CAPSULE_SPARSE_MIN_BYTES`) indexed row-wise by an integer input
of the kernel -- an embedding table by token ids -- is saved as the indexed rows only, **after a proof at
capture time**: the kernel is relaunched in the model's process on the full input and on the rebuilt one
(saved rows in place, every other row NaN), and any output differing bit for bit saves it whole (the log:
`sparse <arg>: <n> of <N> rows -- verified` or `NOT identical -> saved whole`). 2026-11-20: 256 of
151,936 rows, 1.4 MB instead of 273 MB, replay `matches_model=True`; the self-test
(`INDUCTOR_CAPSULE_SPARSE_SELFTEST=drop`, one row lost) fell back to the whole table. Everything else is
saved whole up to `INDUCTOR_CAPSULE_MAX_BYTES` (4 GiB); `INDUCTOR_CAPSULE_SPARSE=0` saves all whole. The
saved values are the model's data -- the same rule as `NAN_CAPTURE_DATA=1`: sparse means less of it, not
none.

## C2. Candidate from the kernel (N3 with a known kernel)

1. **The compiled module alone**: the module (or the smallest function holding the
   kernel's source nodes) under `torch.compile`, the model's shapes, dtypes, autocast and
   the input *structure* N1 recorded (zero / padded rows, masks, magnitudes from
   `capsule_info.py` of the consumer capsule). Variants on one job (N5): compiled vs
   eager, the suspect constant (eps, scale) changed, the structure removed.
2. **Optional**: the dynamo minifier (`TORCHDYNAMO_REPRO_AFTER=aot
   TORCHDYNAMO_REPRO_LEVEL=4`) shrinks a graph to the smallest one whose compiled result
   differs from eager; `torch._inductor.compiler_bisector` names the backend stage or
   lowering. Neither tested on ROCm here -- one attempt, then by hand.
3. **Reduce downward**: the fused function -> one compiled op over a value ladder (zero,
   subnormals, `finfo.tiny`, normals, huge, inf) vs eager and fp64 -> the same op as a
   hand-written Triton kernel (`tl.*`) without Inductor, dumping `h.asm["amdgcn"]`. The
   level where it still fails names the owner:

   | fails as | owner |
   |---|---|
   | raw Triton `tl.<op>` | Triton / its LLVM AMDGPU lowering (read the ISA: special-value class masks, prescaling, denorm mode) |
   | compiled op only, raw Triton fine | Inductor's lowering (the `output_code` line differs from the hand-written one) |
   | compiled module only, single op fine | fusion / layout / reduction order -- keep the smallest fused function |

Verdict by N6. A compiled result non-finite where eager and fp64 are finite is
`NATIVE_KERNEL_BUG` (the README names the owner from the table). Package the lowest level
that fails as `repro.py` and the compiled-op and model-module scripts beside it (step 6
*Native reproducer*).
