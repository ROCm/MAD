# Step 4, wrong FINITE values: exploding gradients, loss spikes, NaN as a rare tail

Read when `<NAN_SIGNAL>` is a magnitude (grad norm, loss jump), or the NaN is rare but
the model shows finite spikes before it (the 2026-11-09 Qwen3 MoE: grad norm 48.6,
1671, 39540, 183628 on steps 2-15, NaN only at 19). The non-finite capture fires on the
tail; on a spike it fires on nothing -- or on the first op that *overflows*, often a
correct reduction (`_foreach_norm`) far downstream of the cause.

Also when there is **no NaN at all**: the model's own check reports wrong finite output -- parity
against a reference, or outputs that differ between identical passes (a fused attention kernel that
returns plausible numbers, `RESULT ... wrong=yes bad_calls=<n>/<m>`, never a NaN). Its
`<NAN_SIGNAL>` is that line (`wrong=yes`); the capture has to look for the wrong value itself
(*The checks*), never for a non-finite one. A model that prints nothing gets its line from differential
mode's check mode (`step2-diff.md`): `<NAN_SIGNAL>` = `ND_DIFF_HIT`, the diff_tap environment on every arm,
and the switch the bisection named picks the twin / op filter.

## The signal (step 1)

`<NAN_SIGNAL>` may be a threshold written as a regex on the model's log line, e.g.
`grad_norm: +([1-9][0-9]{2,}|nan|inf)` (>= 100 or non-finite); `wave_summary.py` reads it
directly as `--signal-above grad_norm=100` (strips log colours). Ledger the threshold and
the model's normal range (from a clean run's log or a known-good build). A trainer that
does not log the grad norm: `after_backward()` from `scripts/grad_trajectory.py`
(`NUMERIC_DOCTOR_GRADLOG`), whose summary lists the jump steps.

## Determinism: assume NOT (rule from 2026-11-09)

Assume the step a spike or NaN appears on is **non-deterministic** -- for MoE always
(routing sets per-expert token counts; any upstream difference moves the spikes) --
unless proven for this case: >= 2 runs on different nodes, same image / seed checkpoint
/ flags, **bit-identical** per-step values (TensorBoard full precision, or a parameter
digest), not the printed digits. The proof is environment-bound like a rate (rule 5)
and covers only its flags (e.g. `--debug.deterministic`). Ledger it.

## The checks (op_nan_capture, GEMM family: mm, addmm, bmm, baddbmm, _grouped_mm, ...; and fused ops with a value check)

| knob | check | cost | false positives |
|---|---|---|---|
| `NAN_CAPTURE_BOUNDS=1` | Cauchy-Schwarz: \|C_ij\| <= \|\|A_i\|\| \|\|B_j\|\| (+ bias); mm/addmm/bmm/baddbmm. Softmax (`aten._softmax`, `aten.softmax`): rows sum to 1, values in [0, 1]. Attention (aten SDPA flash / efficient / cuDNN): each output value within its V column's [min, max] over the keys (0 allowed: a fully masked row); not with dropout | ~3 elementwise passes over C | none by construction (rounding slack only) |
| `NAN_CAPTURE_VERIFY_EVERY=N` | recompute every N-th call of each site in higher precision (16-bit -> fp32, fp32 -> fp64); flag max\|C-ref\|/max\|ref\| above max(dtype floor, 50 x the site's median error) | one higher-precision GEMM per checked call | low; floor bf16 3e-2, fp16 5e-3, fp32 1e-4 |

**Fused ops** -- `NAN_CAPTURE_VERIFY_EVERY` also checks every op with an entry in
`scripts/math_refs.py` `VALUE_CHECKS`: an fp32 recompute, from the op's inputs, of what its
consumer reads (bounds do not apply); which ops have one, and the variants each models, is the
`VALUE_CHECKS` table itself (an unmodelled variant runs unchecked); floor 3e-2 relative for
16-bit inputs. The suspect op has no entry (the signal names an op the capture cannot judge):
add one -- the op's schema, its math in fp32, what it writes, `NotImplementedError` for variants
it does not model -- and check it first on clean and failing shapes against the library's own
output; else step 5N with a value check in the candidate (`step5n-native.md` N3).

**Fused ops with a second implementation: the twin check** (`NAN_CAPTURE_TWIN_EVERY=N`) -- every N-th call of
each op registered in `scripts/twins.py` runs the stack's own second implementation on the op's pre-call inputs
and is judged like a value check (dtype floor, 5x the site's median error -- `NAN_CAPTURE_TWIN_MEDIAN_X`; a hand-written check uses 50x); a hit is origin `wrong_value`,
verify kind `twin`, with `twin_source` / `twin_kind`. No hand-written reference needed. The registry covers:

| twin kind | ops | independence |
|---|---|---|
| `native` | vLLM `_C.rms_norm`, `_C.fused_add_rms_norm`, `_C.silu_and_mul`, any `vllm_ir.*` op (vLLM's IR `native` implementation); aiter `rmsnorm2d_fwd`, `rms_norm`, `rmsnorm2d_fwd_with_add` (SGLang's `RMSNorm.forward_native`, else vLLM's) | pure PyTorch: fully independent |
| `alt_kernel` | aiter `mla_decode_stage1_asm_fwd` (aiter's Triton MLA decode) | another kernel of the same library: shares its conventions |
| `fp64_twin` | every Inductor kernel (`inductor_ref.py`) | the same algorithm in fp64: catches lowering / precision faults, not a fusion computing the wrong thing |

A hand-written `math_refs` value check wins for its op while `NAN_CAPTURE_VERIFY_EVERY` is on. A twin compares
finite positions only, so the NaN path still scans the op (except a twin that owns scratch outputs: MLA). Calls a
twin could not judge (variant not modelled, framework absent, twin non-finite where the kernel is finite) are
listed at `close()` (`twin check: calls not judged`) -- read it before trusting a clean run. Knobs:
`NAN_CAPTURE_TWIN_OPS` (which ops), `NAN_CAPTURE_TWIN_MAX_BYTES` (1 GiB), `NAN_CAPTURE_TWIN_MEDIAN_X` (5: a twin's
own gap on correct calls is larger than a recompute's, so 50x the median would hide a race). Cost measured
2026-11-24 (torchcompile example, 24 Inductor kernels, 100 steps): capture alone 45 s, twins every call 2 m 21 s,
every 4th 1 m 19 s, every 16th 1 m 05 s (no capture: 26 s) -- `=1` for a small model, `=4..16` for a big one;
a framework twin (native / alt_kernel) costs one extra op call per checked call. An op with no twin: add one in
`twins.py` -- an argument mapping onto the stack's second implementation, `NotImplementedError` for variants it
does not model, checked against the kernel on random inputs (`tests/twins/test_adapters.py`).

**Any op, any library: the rerun check** (`NAN_CAPTURE_RERUN_EVERY=N`) -- every N-th call of each op
is run a second time on identical copies of its pre-call operands (and the same random-number
state), and the op is captured when the two outputs differ by more than rounding noise (relative to
max|output|: bf16 2e-2, fp16 5e-3, fp32 1e-4). It needs no reference, so it covers fused ops nobody
wrote a value check for: a kernel race returns another answer on the same inputs. Blind to a kernel
that is wrong the same way every time (that needs a reference: the checks above), and to a race
that fires only under the original call's load (the rerun runs right after it). Ops torch tags
`nondeterministic_bitwise` (atomics) may differ below the floor; `close()` prints a census of every
op re-run and how far below its floor it came -- read it before trusting a clean run. Costs one
extra launch (and operand copies, <= `NAN_CAPTURE_RERUN_MAX_BYTES`, 4 GiB) per checked call:
`=1` for a model of a few hundred ops a step, `=8..64` for a big one. **A finite signal with no
value check for the suspect op: this first** (`NAN_CAPTURE_RERUN_EVERY=1 NAN_CAPTURE_DATA=1`), with
`NAN_CAPTURE_VERIFY_EVERY` beside it when the op has one. `NUMERIC_DOCTOR_VALUE_CHECKS=0` turns the
`math_refs` value checks off (to measure the generic checks alone). The rerun covers every op by
default, **independent of `NAN_CAPTURE_OP_CLASSES`** (which narrows only the NaN scan: a capture
narrowed to `gemm` for speed once left the rerun 2 op types, 2026-11-21); `NAN_CAPTURE_RERUN_OPS`
narrows the rerun itself (classes, namespaces such as `aiter`, op names). Cost seen on the
examples: about 1.5-2x a step at `=4`.

A hit is captured at once (no settling: a wrong value is not masked later) with origin
`wrong_value` and `capsule["verify"]` (`kind`, `rel_err`/`ratio`, `threshold`, site
calls). The NaN path stays on: a non-finite output is still the NaN path's.

**Strategy.** A finite signal: `NAN_CAPTURE_TWIN_EVERY=1` beside the checks below (`=4..16` on a big model) -- the
suspect is often a fused op with a twin. A finite signal with a suspect fused op: `NAN_CAPTURE_VERIFY_EVERY=1` (a decode
step calls it a few times) and `NAN_CAPTURE_DATA=1` for synthetic or public data -- the capsule then
replays the race on the real operands. For GEMMs: default (not proven deterministic): catch it in the run where it happens --
`NAN_CAPTURE_BOUNDS=1 NAN_CAPTURE_VERIFY_EVERY=1` for a small model, `=8..64` for a big
one, with `NAN_CAPTURE_OP_CLASSES=gemm` to cut the capture's own perturbation. Proven
deterministic: re-run with `NAN_CAPTURE_ARM_AFTER_STEPS=<S-1>` and `VERIFY_EVERY=1`, where S
is the first spike step; no hit at S means the proof was wrong -> default.

**Control (always).** Run the same capture on a known-good build or configuration (the
fixed library, fp32 GEMMs, the other attention backend): it must finish its steps with
no `WRONG VALUE` line. A hit there is a tolerance problem, not the bug -- raise the floor
for that dtype, never drop the check.

## The capture can change the fault

The checks add kernels and allocations. A fault that depends on memory contents or
timing (an uninitialized or out-of-bounds read, a race) can change shape under the
capture -- on the 2026-11-09 MoE the expert GEMM went NaN at step 2 under the capture
(48.6, finite, without it), while the good build's numerics stayed bit-identical. That
is evidence, not noise: ledger both, and let the replay (5b) decide.

## Replay (5b)

`replay_nan_op.py --chain` on a `wrong_value` capsule counts a run when the terminal
op fails the capsule's own check again (the bound, or the reference at the capsule's
threshold); the loop line says `[wrong-value capsule: ...]`. Deterministic faults need few
iterations (`CAPSULE_ITER_COUNT=200..2000`). Replay the same capsule in the known-good
image too: reproducing in the bad one and not the good one is the verdict.

**A NaN capsule can replay as a wrong FINITE value.** The [C1] walk compares every
GEMM-family row with a higher-precision recompute on the operands it bound; a wrong row is
marked `NO` / `WRONG VALUE: <rel err>` and becomes the culprit, and the [C2] loop then
counts runs that repeat it (`--wrong-value` forces it). The 2026-11-09 MoE capsule: in the
model the expert GEMM's garbage overflowed to NaN, in replay the same garbage stayed finite
-- judged by finiteness alone it read `NOT_REPRODUCED` (0/2000); judged by value, 200/200.
Before calling a GEMM capsule clean, read its walk's `WRONG VALUE` notes.

**Reduce to one op shape (5N/N5).** A deterministic wrong GEMM usually depends only on the
problem shape: sweep the varying dimension (a short script with three modes: `sweep` calls the
op with a new random size each call and checks it against fp32, `replay` reruns given sizes on
fresh data, `pairs` runs size A right before size B), check whether the failing sizes form bands (kernel
selection), whether a failing size fails on fresh random data every time, and whether the
previous call's shape matters. A control that keeps shapes fixed must use a FAILING shape,
not an arbitrary one (2026-11-09: uniform 5037-row experts missed every band).
