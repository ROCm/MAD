# Step 4, wrong FINITE values: exploding gradients, loss spikes, NaN as a rare tail

Read when `<NAN_SIGNAL>` is a magnitude (grad norm, loss jump), or the NaN is rare but
the model shows finite spikes before it (grad norm climbing by orders of magnitude over
a few steps, NaN only later). The non-finite capture fires on the tail; on a spike it
fires on nothing -- or on the first op that *overflows*, often a correct reduction
(`_foreach_norm`) far downstream of the cause.

## The signal (step 1)

`<NAN_SIGNAL>` may be a threshold written as a regex on the model's log line, e.g.
`grad_norm: +([1-9][0-9]{2,}|nan|inf)` (>= 100 or non-finite); `madengine nan waves`
reads it directly as `--signal-above grad_norm=100` (strips log colours). Ledger the
threshold and the model's normal range (from a clean run's log or a known-good build). A
trainer that does not log the grad norm: `after_backward()` from
`nan_doctor_rt.grad_trajectory` (`NAN_DOCTOR_GRADLOG`), whose summary lists the jump
steps.

## Determinism: assume NOT

Assume the step a spike or NaN appears on is **non-deterministic** -- for MoE always
(routing sets per-expert token counts; any upstream difference moves the spikes) --
unless proven for this case: >= 2 runs on different nodes, same image / seed checkpoint
/ flags, **bit-identical** per-step values (TensorBoard full precision, or a parameter
digest), not the printed digits. The proof is environment-bound like a rate (rule 5)
and covers only its flags (e.g. `--debug.deterministic`). Ledger it.

## The checks (op_nan_capture, GEMM family: mm, addmm, bmm, baddbmm, _grouped_mm, ...)

| knob | check | cost | false positives |
|---|---|---|---|
| `NAN_CAPTURE_BOUNDS=1` | Cauchy-Schwarz: \|C_ij\| <= \|\|A_i\|\| \|\|B_j\|\| (+ bias); mm/addmm/bmm/baddbmm | ~3 elementwise passes over C | none by construction (rounding slack only) |
| `NAN_CAPTURE_VERIFY_EVERY=N` | recompute every N-th call of each site in higher precision (16-bit -> fp32, fp32 -> fp64); flag max\|C-ref\|/max\|ref\| above max(dtype floor, 50 x the site's median error) | one higher-precision GEMM per checked call | low; floor bf16 3e-2, fp16 5e-3, fp32 1e-4 |

A hit is captured at once (no settling: a wrong value is not masked later) with origin
`wrong_value` and `capsule["verify"]` (`kind`, `rel_err`/`ratio`, `threshold`, site
calls). The NaN path stays on: a non-finite output is still the NaN path's.

**Strategy.** Default (not proven deterministic): catch it in the run where it happens --
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
capture -- an expert GEMM can go NaN under the capture where the uncaptured run only
spiked finite, while the good build's numerics stay bit-identical. That is evidence, not
noise: ledger both, and let the replay (5b) decide.

## Replay (5b)

`madengine nan replay <CAPSULE> --mode chain ...` on a `wrong_value` capsule counts a run
when the terminal op fails the capsule's own check again (the bound, or the reference at
the capsule's threshold); the loop line says `[wrong-value capsule: ...]`. Deterministic
faults need few iterations (`--iters 200` to `2000`). Replay the same capsule in the
known-good image too: reproducing in the bad one and not the good one is the verdict.

**A NaN capsule can replay as a wrong FINITE value.** The [C1] walk compares every
GEMM-family row with a higher-precision recompute on the operands it bound; a wrong row is
marked `NO` / `WRONG VALUE: <rel err>` and becomes the culprit, and the [C2] loop then
counts runs that repeat it (`--wrong-value` forces it, via `--extra-args`). In the model
the GEMM's garbage can overflow to NaN while in replay the same garbage stays finite
-- judged by finiteness alone it reads `NOT_REPRODUCED`; judged by value, it reproduces.
Before calling a GEMM capsule clean, read its walk's `WRONG VALUE` notes.

**Reduce to one op shape (5N/N5).** A deterministic wrong GEMM usually depends only on the
problem shape: sweep the varying dimension (a minimal standalone `torch.mm` program, see
`step5n-native.md`), check whether the failing sizes form bands (kernel
selection), whether a failing size fails on fresh random data every time, and whether the
previous call's shape matters. A control that keeps shapes fixed must use a FAILING shape,
not an arbitrary one (uniform sizes can miss every failing band).
