#!/usr/bin/env python3
"""Fix MoRIIO self.block_size reconciliation for hybrid K3 (KDA + MLA).

PROBLEM
  register_kv_caches() reconciles self.block_size from `first_layer_name`'s
  geometry. `first_layer_name` is chosen (moriio_connector.py:~1707) as the first
  NON-MLA 5-D KV layer, falling back to the very first layer. For hybrid Kimi-K3
  the full-attention layers are MLA (excluded), so `first_layer_name` resolves to
  a KDA *mamba* layer whose block_size is 1 (indivisible conv/ssm state page).
  self.block_size then becomes 1. Every real attention layer has block_size 1536
  (the mamba-page padding raises the attention page to match), so the per-layer
  guard trips:
    ValueError: MoRIIO KV cache block size mismatch for layer
      language_model.model.layers.3.self_attn.attn: 1536 != 1

FIX (surgical)
  Reconcile self.block_size from the first NON-mamba layer's geometry, not from
  whichever layer happens to be first. Mamba layers keep block_size 1 and are
  already exempted from the guard (the `_k3_is_mamba` branch a few lines below).

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_moriio_block_size_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[k3-blocksize] {REL} not found -- skip.")
        return 0
    src = open(path).read()
    orig = src

    if "k3-kda: block_size must reflect the ATTENTION layers" in src:
        print("[k3-blocksize] already applied.")
        return 0

    anchor = (
        "        self.num_blocks = first_geometry.num_blocks\n"
        "        self.slot_size_bytes = first_geometry.slot_size_bytes\n"
        "        if first_geometry.block_size != self.block_size:\n"
    )
    repl = (
        "        self.num_blocks = first_geometry.num_blocks\n"
        "        self.slot_size_bytes = first_geometry.slot_size_bytes\n"
        "        # k3-kda: block_size must reflect the ATTENTION layers, not mamba.\n"
        "        # first_layer_name can be a KDA mamba layer (block_size=1); using\n"
        "        # it would make every real attention layer (block_size 1536 after\n"
        "        # mamba-page padding) trip the guard below. Reconcile from the\n"
        "        # first non-mamba layer's geometry instead.\n"
        "        from vllm.v1.kv_cache_interface import MambaSpec as _K3MambaSpecBS\n"
        "        _attn_block_size = None\n"
        "        for _ln in kv_caches:\n"
        "            if isinstance(self.layer_to_spec.get(_ln), _K3MambaSpecBS):\n"
        "                continue\n"
        "            _attn_block_size = self._get_layer_transfer_geometry(_ln).block_size\n"
        "            break\n"
        "        if _attn_block_size is None:\n"
        "            _attn_block_size = first_geometry.block_size\n"
        "        if _attn_block_size != self.block_size:\n"
    )
    if anchor in src:
        src = src.replace(anchor, repl, 1)
    else:
        print("[k3-blocksize] WARN: anchor not found -- reconcile block may still "
              "use first_geometry. Review register_kv_caches().")
        return 0

    # The reconcile body still references first_geometry.block_size in the log +
    # assignment; rewrite those two to _attn_block_size within the reconcile block.
    old_body = (
        "            logger.info(\n"
        '                "KV cache block_size=%d differs from config block_size=%d; "\n'
        '                "using actual tensor shape (attention backend override).",\n'
        "                first_geometry.block_size,\n"
        "                self.block_size,\n"
        "            )\n"
        "            self.block_size = first_geometry.block_size\n"
    )
    new_body = (
        "            logger.info(\n"
        '                "KV cache block_size=%d differs from config block_size=%d; "\n'
        '                "using actual tensor shape (attention backend override).",\n'
        "                _attn_block_size,\n"
        "                self.block_size,\n"
        "            )\n"
        "            self.block_size = _attn_block_size\n"
    )
    if old_body in src:
        src = src.replace(old_body, new_body, 1)
    else:
        print("[k3-blocksize] WARN: reconcile body not matched; the new block_size "
              "guard var may be unused. Review manually.")

    if src != orig:
        open(path, "w").write(src)
        try:
            import py_compile
            py_compile.compile(path, doraise=True)
        except Exception as e:
            print(f"[k3-blocksize] ERROR: compile failed: {e}", file=sys.stderr)
            open(path, "w").write(orig)
            return 1
        print("[k3-blocksize] reconcile self.block_size from first non-mamba "
              "layer (fixes MLA vs KDA block_size guard).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
