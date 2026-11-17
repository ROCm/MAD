# Step 5N, compiled track: a NaN born inside torch.compile'd code

Read from `step5n-native.md` N1 when the NaN may be born inside a compiled region. The
capture is blind there: torch.compile refuses to trace under its dispatch mode, so
`NAN_CAPTURE_SUSPEND=...OptimizedModule.__call__` runs every compiled region unchecked
(`capture suspended inside (compiled regions, unchecked)` in the digest). The
first capsule is then the first *eager* op that reads the NaN -- `PROPAGATED`, chain
stopped at `outside_window` -- and names the consumer, never the producer. This
track replaces N2 (locate) and the start of N3 (the first candidate) with Inductor's own
instruments. N3-N6 (candidate contract, load, reduction, verdict) still apply.

## Compiled track or plain 5N?

| evidence (from N1) | use |
|---|---|
| capsule `PROPAGATED`, chain `outside_window`, frames at the first eager op after an `OptimizedModule` (or its dirty operand is a compiled module's output) | **compiled track** -- C0 first |
| no capsule / no replayable group while the model NaNs, and most of the step is compiled (whole-model `torch.compile`, `fullgraph`, a compiled loss or optimizer) | **compiled track** |
| C0: the compile-off model also NaNs | **plain 5N** -- model numerics or an eager kernel; compilation is a bystander (e.g. `0 * rsqrt(0)`, NaN eager too) |
| C1: the first assert is a *graph input* | **plain 5N** with the upstream eager producer as suspect |
| the faulting op is load- or timing-gated (needs a co-tenant or concurrency, 5c/5d reproduced) | **plain 5N** -- C1 adds a host sync after every kernel, which removes races; at most run C0. A NaN *step* that merely varies between runs (compiled kernels drift by 1e-5 and the trajectories part) is not this: every run still NaNs, alone |
| the compiled region is one or two blocks and N1 already names a plausible suspect (a norm, an eps, a softmax, zero/padded rows) | **plain 5N bottom-up** -- a direct candidate costs one run. Fall back to this track if the candidate stays clean |

Both apply (a big compiled region *and* a cheap obvious suspect): one 5N candidate run and
C0 as back-to-back `mad-run.sh` runs (one GPU lock); whichever localises first
wins. Ledger the choice: `5N track: compiled | plain -- <which row>`.

## C0. Compile-off control (one wave, the discriminator)

Same geometry and step count as the step-4 wave, no capture, compilation off: the
model's own switch (`--compile-layers none`, `compile: false`), else
`TORCHDYNAMO_DISABLE=1` in the trainer's environment. Clean on every run while the
compiled controls NaN'd at the ledgered step = born in compiled code (ledger `C0 clean
n/N`). Compile-off also NaNs: plain 5N. A different NaN step under compile-off is not
"clean": run it to the step-4 wave's length.

## C1. `nan_asserts` on the model (one wave, no capture)

Inductor checks, after every generated kernel, that its output buffers hold no NaN/Inf,
plus every graph input and output (`torch/_inductor/codegen/wrapper.py`,
`codegen/simd.py`). Env on the trainer, via the same route as `NAN_*`:

```
TORCHINDUCTOR_NAN_ASSERTS=1  TORCH_LOGS=output_code  TORCH_LOGS_OUT=<ARTIFACT_DIR_C>/native/output_code.<run>.log
```

Check `output_code.<run>.log` holds `assert not ... isnan()` lines (no
lines = the flag did not reach the trainer, or a cached graph was reused: give each
run its own `TORCHINDUCTOR_CACHE_DIR`). Stop at the first
`AssertionError`. Read the traceback's last frame -- a line in the generated wrapper
(`/tmp/inductor/.../*.py`) -- and the kernel call just above it in `output_code.<run>.log`:

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

C0 and C1 are two `madengine run` invocations (a "wave" of sequential `mad-run.sh` runs,
`env-madengine.md`): C0 with the compile-off flag, C1 with the env above and
`TORCH_LOGS_OUT` per run; each run gets its own inductor cache. The traceback's wrapper
line maps into the log: the module's source starts on the line after its `Output code:`
header, so wrapper line N is log line header + N.

## C2. Candidate from the kernel (N3 with a known kernel)

1. **The compiled module alone**: the module (or the smallest function holding the
   kernel's source nodes) under `torch.compile`, the model's shapes, dtypes, autocast and
   the input *structure* N1 recorded (zero / padded rows, masks, magnitudes from
   `capsule_info` of the consumer capsule). Variants in one run (N5): compiled vs
   eager, the suspect constant (eps, scale) changed, the structure removed.
2. **Optional**: the dynamo minifier (`TORCHDYNAMO_REPRO_AFTER=aot
   TORCHDYNAMO_REPRO_LEVEL=4`) shrinks a graph to the smallest one whose compiled result
   differs from eager; `torch._inductor.compiler_bisector` names the backend stage or
   lowering. Neither is tested on ROCm -- one attempt, then by hand.
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
