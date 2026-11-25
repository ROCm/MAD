# Step 5b, conditional: deterministic-fault guardrail

**You are here if** the `[C1]` walk has **no `DIVERGED` row** and the terminal op is
`finite = NO`, or 5b reproduces substantially at concurrency 1.

`[C1]` names the first `DIVERGED` row, else the first non-finite one. On an exactly
reproducing fault a wrong upstream op re-runs to the same wrong value, nothing
diverges, and `[C1]` names **the last op** -- with no `UPSTREAM` hedge (bare
`CHAIN_CULPRIT <op> @ depth N`); `fp64 ref` only checks finiteness. So do not sweep
the named op on that alone. Cheapest first:

0. **Read the walk's `terminal fp64 ref on replay operands` line** first: `finite`
   with the terminal op non-finite already makes it `KERNEL_BUG` (the operands
   cannot excuse it). It is printed only when the terminal op has a reference: a
   CPU kernel, or a math reference (`flash_attn._flash_attn_backward`,
   `aten._scaled_dot_product_flash_attention_backward`,
   `flash_attn._flash_attn_varlen_backward` per sequence; dropout, softcap and alibi
   are refused as `unavailable`; the fused norms and gated activations of aiter,
   sgl_kernel and vLLM `_C` -- `nan_doctor_rt.math_refs`, each checked against its kernel;
   a missing one is one entry there, from the op's schema). On synthetic operands it is meaningful only when
   the derived ones were recomputed (`operands: SYNTHETIC (derived ... recomputed)`,
   `step5-read.md`).
1. **Check by value**: each op's GPU output vs an fp64 recomputation (tolerance
   scaling with accumulation depth); the earliest disagreeing op outranks the
   non-finite one.
2. **Second implementation**: `TORCH_BLAS_PREFER_HIPBLASLT=0` (GEMM),
   `TORCH_ROCM_FA_PREFER_CK=0` (SDPA), passed as `--env K=V`. NaN stops -> the fault is
   in whichever op's dispatch changed, not necessarily `[C1]`'s.
3. **Repair and retest**: overwrite one op's output with a reference-correct value,
   chain intact; NaN gone -> it depends on that op. (Removing an op doesn't work.)
   For out-buffers the capsule's op writes (e.g. an attention backward's `dq`/`dk`/`dv`):
   `madengine nan exec --image <IMAGE> -- python3 -m nan_doctor_rt.replay_nan_op
   --written-operands zero` and `random` (or `madengine nan replay --extra-args`) -- the
   same NaN under all three means the op creates it, not the uninitialised buffer.
   Destroys the operand of an in-place op; don't use it there.
4. Sweeping anyway: the chain sweep `[S3]`, not `[S1]`/`[S2]` on the named op.

Report deterministic vs contention-gated and how you know; `[C1]`'s op as
**unverified attribution**; ops without a reference as **unverifiable**.
