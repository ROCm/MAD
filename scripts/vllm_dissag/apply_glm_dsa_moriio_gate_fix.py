#!/usr/bin/env python3
"""Fix the MoRIIO transfer-completion gate for GLM-5.1 DSA dual KV cache.

ROOT CAUSE (PROVEN, job 37615 instrumentation):
  GLM-5.1 registers num_layers=156 KV caches = 78 main MLA (model.layers.N.self_attn.attn)
  + 78 DSA indexer caches (model.layers.N.self_attn.indexer.k_cache). BUT only the 78
  main-MLA layers ever go through the KV-connector save_kv_layer hook -> only they write
  -> writes_done caps at 78. The indexer caches are registered (counted in num_layers)
  but vLLM NEVER calls save_kv_layer for them (the DSA indexer is a separate attention
  component / DeepseekV32IndexerBackend that doesn't use the connector save path; decode
  recomputes indexer state from the transferred main latent KV).

  The producer completion gate (moriio_engine.py):
      request_info.writes_done += 1
      if request_info.writes_done >= self.worker.num_layers:  # 156, never reached
          send_notify(...)
  caps at writes_done=78 < num_layers=156 -> send_notify NEVER fires -> decode never
  gets completion -> "Reaped deferred sends / no finished_sending after 60s" -> stall.

FIX:
  Add self.num_transfer_layers = count of caches that actually transfer (exclude
  '.indexer.' caches), with a fallback to num_layers (so single-geometry models -
  DeepSeek-V3 / Hunyuan, no indexer - are bit-identical). Gate completion on
  num_transfer_layers instead of num_layers. num_layers itself is left unchanged
  (it is also used by the Llama-4 per-layer block-window loop, which needs all caches).

Companion to apply_glm_dsa_moriio_engine_fix.py (per-layer offset caching). This gate
fix is the primary unblocker; the offset fix is correctness insurance for the layers
that DO write (all same geometry here, but harmless).

Idempotent + anchor-based. Patches BOTH files (connector: define the field; engine:
use it). A found-old-anchor that fails is a hard error.

Usage: apply_glm_dsa_moriio_gate_fix.py <vllm_install_dir>
"""
import os
import sys

CONN_REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
ENG_REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py"


def patch_connector(path: str) -> int:
    src = open(path).read()
    old = "        self.num_layers = len(self.kv_caches.keys())"
    new = """        self.num_layers = len(self.kv_caches.keys())
        # DSA dual-KV fix: the producer completion gate must count only layers that
        # actually transfer via save_kv_layer. GLM-5.1 registers 2 caches/layer (main
        # MLA + DSA indexer), but only the main-MLA caches go through save_kv_layer; the
        # '.indexer.' caches are registered yet never written. Gating on len(kv_caches)
        # would never be reached. Exclude indexer caches; fall back to num_layers for
        # single-geometry models (DeepSeek/Hunyuan have no indexer -> identical).
        self.num_transfer_layers = (
            len([k for k in self.kv_caches.keys() if ".indexer." not in k])
            or self.num_layers
        )
        logger.info(
            "[moriio] completion gate: num_transfer_layers=%d (num_layers=%d)",
            self.num_transfer_layers, self.num_layers,
        )"""
    if "self.num_transfer_layers" in src:
        print(f"[glm-gate] connector already patched -- no-op.")
        return 0
    if old not in src:
        print(f"[glm-gate] WARN: connector anchor (num_layers=) not found -- skipping.")
        return 0
    src = src.replace(old, new, 1)
    open(path, "w").write(src)
    print(f"[glm-gate] patched connector: defined num_transfer_layers in {path}")
    return 0


def patch_engine(path: str) -> int:
    src = open(path).read()
    old = "        if request_info.writes_done >= self.worker.num_layers:"
    new = """        if request_info.writes_done >= getattr(
            self.worker, "num_transfer_layers", self.worker.num_layers
        ):"""
    if 'getattr(\n            self.worker, "num_transfer_layers"' in src or "num_transfer_layers" in src:
        print(f"[glm-gate] engine already patched -- no-op.")
        return 0
    if old not in src:
        print(f"[glm-gate] WARN: engine anchor (writes_done gate) not found -- skipping.")
        return 0
    src = src.replace(old, new, 1)
    open(path, "w").write(src)
    print(f"[glm-gate] patched engine: gate on num_transfer_layers in {path}")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    base = sys.argv[1]
    conn = os.path.join(base, CONN_REL)
    eng = os.path.join(base, ENG_REL)
    if not os.path.isfile(conn) or not os.path.isfile(eng):
        print("[glm-gate] connector/engine not found -- skipping (layout differs).")
        return 0

    # ATOMIC: both halves (connector defines num_transfer_layers, engine gates on
    # it) are needed together or not at all. On a restructured image (e.g. mori
    # v1.2.1, whose engine replaced the writes_done>=num_layers gate with a sealed
    # writes_expected mechanism that already handles hybrid/DSA dual-KV natively),
    # the engine anchor is gone. Applying only the connector half would inject a
    # dead num_transfer_layers into restructured internals. So if the engine anchor
    # is absent, skip BOTH — the native gate already does the right thing.
    eng_src = open(eng).read()
    eng_gate_present = "        if request_info.writes_done >= self.worker.num_layers:" in eng_src
    eng_already = "num_transfer_layers" in eng_src
    if not eng_gate_present and not eng_already:
        print("[glm-gate] engine gate anchor absent (image restructured, e.g. mori "
              "v1.2.1 sealed writes_expected) -- skipping BOTH halves (native gate handles DSA).")
        return 0

    rc = patch_connector(conn) or patch_engine(eng)
    if rc:
        return rc

    try:
        import py_compile
        py_compile.compile(conn, doraise=True)
        py_compile.compile(eng, doraise=True)
        print("[glm-gate] py_compile OK (both files)")
    except Exception as e:  # noqa: BLE001
        print(f"[glm-gate] ERROR: compile failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
