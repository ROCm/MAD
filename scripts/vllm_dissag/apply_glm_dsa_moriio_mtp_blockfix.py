#!/usr/bin/env python3
"""Make MoRIIO KV transfer tolerate MTP / speculative-decode extra blocks.

ROOT CAUSE (proven on GLM-5.2 EP8 1P/1D, job set 2026-08-29):
  With MTP (speculative decoding, --speculative-config method=mtp) enabled, the PREFILL
  side allocates one extra trailing KV block per speculative token for the lookahead /
  draft slot. The MoRIIO KV connector's block-transfer offset math
  (moriio_layout.py::compute_block_transfer_offsets) asserted len(local) <= len(remote)
  and raised

      ValueError: local_block_ids longer than remote_block_ids: 6 > 5   (also 5 > 4)

  on the first WRITE task for any request whose block count crosses that boundary. The
  KV write aborts -> decode never receives the prompt KV -> the router drops the prefill
  ("No prefill servers available"). Short requests where the extra block doesn't tip a
  block boundary return 200; 50K/128K requests fail. The mismatch is ALWAYS
  local = remote + N_spec (the draft block(s)).

  This affects ALL three disagg configs (TP8 / EP8 / EP16) because all use MoRIIO for KV
  transfer. It is the MoRIIO analogue of the NIXL-connector fix in upstream vLLM PR
  #46694 (which deferred decode spec-slot allocation); the MoRIIO connector never got the
  equivalent change.

FIX (surgical, connector-side):
  When local_block_ids is longer than remote_block_ids, the surplus is always the
  TRAILING draft block(s); the leading blocks are the real prompt-prefix KV that must
  transfer. Truncate local to len(remote) (keep leading, drop trailing) with a
  warning_once, instead of raising. The decode side recomputes the dropped draft block on
  its first step. Correctness is gated by NIAH: with this patch EP8-MTP retrieval holds at
  128K = 9/10 (a wrong-end truncation collapses NIAH), and KV write failures drop to 0.

  Non-speculative runs are unaffected: without MTP, len(local) <= len(remote) always, so
  the new branch never fires.

Idempotent + anchor-based. A missing anchor warns-and-skips; a found OLD anchor that
fails to apply is a hard error (would silently keep MTP broken under disagg).

Usage: apply_glm_dsa_moriio_mtp_blockfix.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py"

OLD = """    if len(local_block_ids) > len(remote_block_ids):
        raise ValueError(
            "local_block_ids longer than remote_block_ids: "
            f"{len(local_block_ids)} > {len(remote_block_ids)}"
        )"""

NEW = """    if len(local_block_ids) > len(remote_block_ids):
        # Speculative decoding (MTP): the prefill side holds extra TRAILING draft
        # block(s) for the lookahead token that the decode side has not allocated a
        # slot for. Truncate to the remote length (keep leading prompt-prefix KV,
        # drop the trailing draft block); decode recomputes it. See vLLM PR #46694
        # (NIXL connector analogue). NIAH gates the alignment correctness.
        logger.warning_once(
            "local_block_ids longer than remote_block_ids: %d > %d "
            "(speculative-decode lookahead blocks); truncating trailing surplus "
            "to align with remote KV transfer.",
            len(local_block_ids),
            len(remote_block_ids),
        )
        local_block_ids = local_block_ids[: len(remote_block_ids)]"""

LOGGER_ANCHOR = "import torch\n"
LOGGER_ADD = (
    "import torch\n\n"
    "from vllm.logger import init_logger\n\n"
    "logger = init_logger(__name__)\n"
)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[glm-mtp] {REL} not found -- skipping (connector layout differs).")
        return 0

    src = open(path).read()

    if "truncating trailing surplus" in src:
        print("[glm-mtp] already applied -- skipping.")
        return 0

    if OLD not in src:
        print(
            "[glm-mtp] ERROR: expected raise-anchor not found in "
            f"{REL}; connector changed -- refusing to apply blindly.",
            file=sys.stderr,
        )
        return 1

    # Ensure a module logger exists (the file ships without one).
    if "logger = init_logger(__name__)" not in src:
        if LOGGER_ANCHOR not in src:
            print("[glm-mtp] ERROR: import anchor not found.", file=sys.stderr)
            return 1
        src = src.replace(LOGGER_ANCHOR, LOGGER_ADD, 1)

    src = src.replace(OLD, NEW, 1)
    open(path, "w").write(src)
    print(f"[glm-mtp] applied MoRIIO MTP block-transfer fix to {REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
