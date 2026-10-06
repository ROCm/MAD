#!/usr/bin/env python3
"""Guard gather_initial_states against out-of-range KDA state indices.

PROBLEM (the 2P/2D disagg producer GPU memory fault)
  kimi_gdn_linear_attn calls gather_initial_states(recurrent_state, state_indices,
  has_initial_state) at prefill. The Triton kernel forms a GPU address
  state_ptr + state_idx*stride even where the value load is has_initial_state-masked;
  an out-of-range state_idx therefore faults the device ("Memory access fault by GPU
  ... Reason: Unknown"). Standalone prefill is fine (has_initial_state=False / valid
  idx); only the disagg producer prefill faults -> it carries a state index that is
  >= recurrent_state.shape[0] (or a stray value) for a request the scheduler treated
  as having a prior state.

FIX (surgical, in the op wrapper so it covers every caller)
  Before launching the kernel, compute a safe index tensor:
    - where has_initial_state is False -> 0 (a fresh prefill gathers nothing anyway)
    - where True -> clamp into [0, n_state_blocks-1]
  and log once if any index was out of range (so the upstream metadata bug is
  visible). This keeps the GPU address valid; masked-off rows still contribute zeros.

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_kda_gather_guard.py <vllm_install_dir>
"""
import os
import sys

REL = "model_executor/layers/mamba/ops/gather_initial_states.py"


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[k3-kda-gather] {REL} not found -- skip.")
        return 0
    src = open(path).read()
    orig = src

    if "k3-kda guard" in src:
        print("[k3-kda-gather] already applied.")
        return 0

    anchor = (
        "    row_size = state[0].numel()\n"
        "    # Mamba pages may pad stride(0), but each state row remains dense.\n"
        "    assert state[0].is_contiguous()\n"
        "    output = torch.empty(\n"
    )
    repl = (
        "    row_size = state[0].numel()\n"
        "    # Mamba pages may pad stride(0), but each state row remains dense.\n"
        "    assert state[0].is_contiguous()\n"
        "    # k3-kda guard: an out-of-range state index makes the kernel form an\n"
        "    # OOB GPU address (state_ptr + idx*stride) -> Memory access fault, even\n"
        "    # where the value load is has_initial_state-masked. Under 2P/2D disagg the\n"
        "    # producer prefill has been observed to carry indices >= state.shape[0];\n"
        "    # zero the effective index where has_initial_state is False and clamp any\n"
        "    # stray index into range so the address stays valid.\n"
        "    _n_state_blocks = int(state.shape[0])\n"
        "    _safe_idx = torch.where(\n"
        "        has_initial_state,\n"
        "        indices.to(torch.int64).clamp(0, _n_state_blocks - 1),\n"
        "        torch.zeros_like(indices, dtype=torch.int64),\n"
        "    )\n"
        "    if bool((indices >= _n_state_blocks).any()) or bool((indices < 0).any()):\n"
        "        import logging as _lg\n"
        "        _bad = indices[(indices >= _n_state_blocks) | (indices < 0)]\n"
        "        _lg.getLogger(__name__).warning(\n"
        '            \"[k3-kda gather] clamped %d out-of-range state idx (n_blocks=%d, \"\n'
        '            \"sample=%s); disagg producer prefill likely mis-flagged initial state.\",\n'
        "            int(_bad.numel()), _n_state_blocks, _bad[:8].tolist(),\n"
        "        )\n"
        "    indices = _safe_idx\n"
        "    output = torch.empty(\n"
    )
    if anchor not in src:
        print("[k3-kda-gather] WARN: anchor not found -- gather guard NOT applied.")
        return 0
    src = src.replace(anchor, repl, 1)

    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        print(f"[k3-kda-gather] ERROR: compile failed: {e}", file=sys.stderr)
        open(path, "w").write(orig)
        return 1
    print("[k3-kda-gather] guarded gather_initial_states against OOB state idx.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
