#!/usr/bin/env python3
"""TEMPORARY instrumentation: does the DSA indexer cache reach the MoRIIO save path?

Adds logging at two points in moriio_connector.py to settle the dual-KV RCA:
  1. register_kv_caches: log all kv_caches layer names + their shapes + num_layers.
     -> shows whether the DeepseekV32IndexerCache is even registered, and its geometry.
  2. _write_blocks_for_req: log each distinct layer_name that actually triggers a write.
     -> compare the COUNT/SET of written layers vs num_layers. If indexer layers are in
        kv_caches (counted in num_layers) but never written, writes_done can never reach
        num_layers -> send_notify never fires -> the stall.

This is diagnostic only (no behavior change). Remove before any production use.
Idempotent + anchor-safe.

Usage: apply_glm_dsa_moriio_instrument.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[glm-instr] {REL} not found -- skipping.")
        return 0

    src = open(path).read()
    orig = src

    # --- Point 1: log kv_caches inventory at num_layers assignment -------------
    a1 = "        self.num_layers = len(self.kv_caches.keys())"
    b1 = """        self.num_layers = len(self.kv_caches.keys())
        # [glm-instr] kv-cache inventory (dual-KV diagnosis)
        try:
            for _ln, _kv in self.kv_caches.items():
                logger.info("[glm-instr][register] layer=%s shape=%s dtype=%s",
                            _ln, tuple(_kv.shape), _kv.dtype)
            logger.info("[glm-instr][register] num_layers=%d total_kv_caches=%d",
                        self.num_layers, len(self.kv_caches))
        except Exception as _e:  # noqa: BLE001
            logger.info("[glm-instr][register] inventory log failed: %s", _e)"""
    if "[glm-instr][register]" in src:
        pass
    elif a1 in src:
        src = src.replace(a1, b1, 1)
    else:
        print("[glm-instr] WARN: register anchor (num_layers=) not found.")

    # --- Point 2: log each written layer in _write_blocks_for_req -------------
    a2 = "    def _write_blocks_for_req(self, req_id: ReqId, meta: ReqMeta, layer_name, kv_layer):"
    b2 = (a2 + "\n"
          '        # [glm-instr] record which layers actually trigger a KV write\n'
          '        try:\n'
          '            _seen = getattr(self, "_glm_instr_written_layers", None)\n'
          '            if _seen is None:\n'
          '                _seen = set(); self._glm_instr_written_layers = _seen\n'
          '            if layer_name not in _seen:\n'
          '                _seen.add(layer_name)\n'
          '                logger.info("[glm-instr][write] NEW layer=%s total_written=%d/%d",\n'
          '                            layer_name, len(_seen), getattr(self, "num_layers", -1))\n'
          '        except Exception as _e:  # noqa: BLE001\n'
          '            logger.info("[glm-instr][write] log failed: %s", _e)')
    if "[glm-instr][write]" in src:
        pass
    elif a2 in src:
        src = src.replace(a2, b2, 1)
    else:
        print("[glm-instr] WARN: _write_blocks_for_req anchor not found.")

    if src != orig:
        open(path, "w").write(src)
        print(f"[glm-instr] instrumented {path}")
    else:
        print(f"[glm-instr] already instrumented / nothing to do for {path}")

    try:
        import py_compile
        py_compile.compile(path, doraise=True)
        print("[glm-instr] py_compile OK")
    except Exception as e:  # noqa: BLE001
        print(f"[glm-instr] ERROR: compile failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
