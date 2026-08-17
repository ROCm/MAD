#!/usr/bin/env python3
"""Guard the MoRIIO connector abort path against a None peer_zmq (mori v1.2.1).

ROOT CAUSE (observed on the router image, job 199144 decode crash):
  When a request is ABORTED before its KV-transfer peer handshake completes,
  the connector's release path runs:

    moriio_connector.py::_release_write_prefill_blocks
      peer_zmq = get_peer_zmq_from_request_id(request_id, is_producer=False)  # -> None
      remote_host, _, remote_notify_port = parse_moriio_zmq_address(peer_zmq) # None.split(",")
    -> AttributeError: 'NoneType' object has no attribute 'split'

  This only catches ValueError, not the AttributeError from a None peer_zmq, so
  the EngineCore dies -> cascades to all decode workers (EngineDeadError) -> decode
  is dead. Triggered by any request aborted before the peer handshake (e.g. a
  canary/curl that times out during first-token cold JIT).

  The SAME FILE already guards this correctly at the other call site
  (request_finished / _should_notify path): `if peer_zmq is not None:` then parse,
  else fall back to params. The release path just missed the guard — an
  inconsistent-guard bug in the connector.

FIX (surgical, matches the file's own existing pattern):
  In _release_write_prefill_blocks, when the params don't already carry
  remote_host/remote_notify_port, guard the peer_zmq lookup: if it is None, log
  and return (same graceful bail the existing `except ValueError` already does for
  the "missing remote notify address" case). No behavior change when peer_zmq is
  valid; single-geometry / non-aborted requests are unaffected.

Idempotent + anchor-based + self-skipping (no-ops if the anchor is absent/already
guarded, so it is safe across connector revisions and other images). A found-old
anchor that fails to apply is a hard error (would leave the crash).

Usage: apply_glm_moriio_abort_guard_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"

# The buggy two lines: fetch peer_zmq (may be None) then parse it unguarded.
OLD = """                peer_zmq = get_peer_zmq_from_request_id(request_id, is_producer=False)
                remote_host, _, remote_notify_port = parse_moriio_zmq_address(peer_zmq)"""

NEW = """                peer_zmq = get_peer_zmq_from_request_id(request_id, is_producer=False)
                # Abort-path guard: a request aborted before the KV peer handshake
                # has peer_zmq=None; parse_moriio_zmq_address(None) would raise
                # AttributeError and kill the EngineCore. Bail gracefully like the
                # ValueError case below (matches the guarded call site elsewhere).
                if peer_zmq is None:
                    logger.warning(
                        "Cannot release WRITE prefill blocks for request %s: "
                        "no peer zmq address (aborted before peer handshake)",
                        request_id,
                    )
                    return
                remote_host, _, remote_notify_port = parse_moriio_zmq_address(peer_zmq)"""


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[glm-abort] {REL} not found -- skipping (connector layout differs).")
        return 0

    src = open(path).read()
    if "no peer zmq address (aborted before peer handshake)" in src:
        print("[glm-abort] already patched -- no-op.")
        return 0
    if OLD not in src:
        # Anchor absent: either the release path was refactored or this image
        # already guards it. Do not block launch.
        print("[glm-abort] release-path anchor not found -- skipping (assuming "
              "native guard / refactored).")
        return 0

    src = src.replace(OLD, NEW, 1)
    try:
        open(path, "w").write(src)
    except OSError as e:
        print(f"[glm-abort] ERROR: write failed for {path}: {e}", file=sys.stderr)
        return 1

    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:  # noqa: BLE001
        print(f"[glm-abort] ERROR: patched file fails to compile: {e}", file=sys.stderr)
        return 1
    print(f"[glm-abort] patched _release_write_prefill_blocks None-guard in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
