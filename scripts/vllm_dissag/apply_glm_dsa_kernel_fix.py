#!/usr/bin/env python3
"""Apply the GLM-5.1 DSA sparse-attention invalid-token kernel fix (vllm #45324).

The DSA indexer kernel `_convert_req_index_to_global_index_kernel` in
  vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
maps invalid token slots to 0 instead of -1. With block-size 1 + DSA sparse MLA
that corrupts KV reads and the model emits `!!!` for every prompt.

Upstream fix: vllm-project/vllm #45324 -- flip the 0 to -1 in the tl.where call:
  is_invalid_tok | (~valid_block), 0,  base * BLOCK_SIZE + inblock_off
  is_invalid_tok | (~valid_block), -1, base * BLOCK_SIZE + inblock_off

Design (matches launcher contract -- runs unconditionally for GLM, aborts on real
failure):
  * IDEMPOTENT  : if already -1, report and exit 0 (no-op).
  * SELF-SKIPPING: if the file/anchor is absent (refactored or the rebase already
    fixed it differently), report and exit 0 -- do NOT abort, because b10a9f7a may
    carry the fix natively. We only fail on the one unambiguous bad state we can
    fix and didn't, or on write failure.
  * VERIFIES the post-write state.

Usage: apply_glm_dsa_kernel_fix.py <vllm_install_dir>
"""
import os
import re
import sys

REL = "v1/attention/backends/mla/rocm_aiter_mla_sparse.py"

# Anchor is the stable right-hand side of the tl.where; the middle operand is the
# 0 (buggy) / -1 (fixed) we toggle. Whitespace-tolerant.
RE_ANY = re.compile(
    r"(is_invalid_tok\s*\|\s*\(~valid_block\)\s*,\s*)(-?\d+)(\s*,\s*base\s*\*\s*BLOCK_SIZE\s*\+\s*inblock_off)"
)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    vllm_dir = sys.argv[1]
    path = os.path.join(vllm_dir, REL)

    if not os.path.isfile(path):
        # File not present on this build -> nothing we can or should do. The
        # rebase may use a different sparse backend layout. Do not block launch.
        print(f"[glm-dsa] {REL} not found under {vllm_dir} -- skipping (assuming native/refactored).")
        return 0

    src = open(path).read()
    m = RE_ANY.search(src)
    if not m:
        # Anchor gone (refactored / already fixed differently). Don't block.
        print(f"[glm-dsa] invalid-token kernel anchor not found in {path} -- skipping (assuming native fix).")
        return 0

    cur = m.group(2)
    if cur == "-1":
        print(f"[glm-dsa] already fixed (kernel returns -1) in {path} -- no-op.")
        return 0
    if cur != "0":
        # Unexpected value -- surface it but don't guess. Treat as needs-attention.
        print(f"[glm-dsa] ERROR: unexpected invalid-token return value '{cur}' (expected 0 or -1) in {path}.",
              file=sys.stderr)
        return 1

    # cur == "0" : the known bug. Flip to -1.
    new_src = src[:m.start(2)] + "-1" + src[m.end(2):]
    try:
        open(path, "w").write(new_src)
    except OSError as e:
        print(f"[glm-dsa] ERROR: failed to write patched {path}: {e}", file=sys.stderr)
        return 1

    # Verify.
    chk = RE_ANY.search(open(path).read())
    if not chk or chk.group(2) != "-1":
        print(f"[glm-dsa] ERROR: post-write verification failed in {path}.", file=sys.stderr)
        return 1
    print(f"[glm-dsa] patched: invalid-token kernel now returns -1 (vllm #45324) in {path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
