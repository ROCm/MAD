# Expected results of the examples

One result card per example under `examples/`: what a correct debug-numeric run finds -- the bug
(verdict, kernel, shape), the root cause and the fix -- in the `result.json` format of
`scripts/result_card.py`. Compare a new run against it:

    python3 scripts/compare_results.py validation/expected/<example>.json artifacts/<run>

`RESULT_COMPARE=SAME` is a pass. `REFINED` names the narrower side: the expected card narrower = the
run found the right thing but a less minimal answer (e.g. qwen_hf_generate_nan patching all 14
relaxed waits instead of the 3 that matter); the run narrower = a better answer than this card --
update the card. `DIFFERENT` is a failure to explain. Rates, hosts, call sites and the mechanism's
wording are printed as notes and never judged.

These are the answers: they live outside `examples/` on purpose. The skill never reads this
directory, and a `PRIORS=none` run must not be pointed at it (step1-ask.md Q7). Built 2026-11-20
from two validation rounds per example (each card's `evidence`), skill 15442b0.

`deepseek_mla_decode_wrong.json` (no NaN: wrong finite values) was added from two blind `PRIORS=none`
runs, skill d9801c0. Its `fix.alternatives` are accepted answers: a run that ships one of them compares
`REFINED` (`vmcnt0@0x2848` where the card has the minimal `vmcnt(18)@0x2848`).

`deepseek_sparse_mla_decode_nan.json` (an uninitialized read: a `torch.empty` split count no gfx950 kernel
writes) is blind run 6's card (skill 327c0ab, `PRIORS=none`); blind run 5 (c419f5b) reached the same root cause and
fix through 5N. Its `bug.kernel` is the Triton reduce; a run drafted before `result_card.py` named `triton.*` /
`inductor.*` kernels (2026-11-21) has `kernel: null` -- re-draft it before comparing.

`softmoe_train_nan.json` was updated 2026-11-23 (skill 47936cc): since the capture watches Inductor kernels
(2026-11-21) it catches the kernel directly as G1, so the verdict is `KERNEL_BUG` / `op_capsule`, not the earlier
`NATIVE_KERNEL_BUG` / `kernel_capsule` reached through 5N. A run from before 2026-11-21 compares DIFFERENT on the verdict only.
`root_cause.location.source` names the faulty code (or, when that is not Python one can name, the nearest Python that
selects it), never the model's call site -- that is a `condition` (step9-report.md, 2026-11-23). torchcompile's card
moved from `train.py:SoftMoEMLP.forward` to `torch/_inductor/codegen/triton.py:TritonOverrides.sqrt` accordingly.
