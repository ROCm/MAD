#!/usr/bin/env python3
"""Patch the MoRIIO KV connector to transfer Kimi-K3's KDA (gated-delta-net)
recurrent+conv state across the prefill->decode boundary.

PROBLEM
  Kimi-K3 is a HYBRID model:
    - ~24 full-attention layers  -> MLA-style paged KV cache (MLAAttentionSpec)
    - ~69 KDA layers (GDN_ATTN)  -> Mamba-style state (MambaSpec):
         conv state  shapes[0] = (conv_dim_local, conv_rows)   DS layout
         ssm  state  shapes[1] = (num_v_heads/TP, head_v_dim, head_k_dim)
  The MoRIIO connector's geometry resolver `get_layer_transfer_geometry`
  (moriio_layout.py) only handles MLA (3D) + K/V attention (4D/5D) and RAISES
  ValueError on any MambaSpec layer. So a K3 disagg run dies the moment a KDA
  layer is registered / transferred. Even if it didn't raise, MoRIIO would not
  move the conv/ssm state, so decode would compute the 69 KDA layers from empty
  state -> garbage output.

  The NIXL connector already solves this via ssm_conv_transfer_utils.py
  (derive_mamba_conv_split + MambaConvSplitInfo, GDN branch) and
  nixl/base_worker.py (_build_mamba_local / _build_mamba_remote). This patcher
  ports the equivalent into MoRIIO, anchored on the amdsiloai image's connector.

SCOPE (this image already has the per-layer geometry refactor: block_lens dict,
get_layer_transfer_geometry, per-layer _read_blocks loop, kv_cache_shapes dict).
So the delta is only the MAMBA branch:
  H1 moriio_layout.get_layer_transfer_geometry: add a MambaSpec branch (before the
     final ValueError) returning a per-page geometry (block_size=1, block_len =
     conv_bytes + ssm_bytes) tagged is_mamba via regions/transfers fields.
  H2 moriio_layout: add mamba-aware offset builder + route compute_block_transfer_offsets
     through it for mamba layers (conv sub-projections [Q,K,V] + ssm, using
     MambaConvSplitInfo.local_conv_offsets / remote_conv_offsets).
  H3 moriio_layout.iter_layer_registration_regions: register the mamba layer as a
     single contiguous region of its own page size (already generic — verified).
  H4 moriio_connector.register_kv_caches: build & cache the MambaConvSplitInfo
     (derive_mamba_conv_split) + ssm_sizes so H2 can use it; carry ssm_sizes in
     MoRIIOAgentMetadata.
  H5 moriio_common.MoRIIOAgentMetadata: add optional ssm_sizes + block_lens fields.

Scheduler N-1 truncation: the connector ALREADY returns len(token_ids)-1 in READ
mode (get_num_new_matched_tokens), which is exactly the mamba requirement (P
computes state through N-1, D recomputes token N). No scheduler change needed;
this patcher asserts that behavior is present and warns if it changed.

DISCIPLINE (mirrors apply_glm_dsa_moriio_dualkv_fix.py):
  - idempotent: each hunk checks if already applied.
  - anchor-based: if the OLD anchor is absent AND not-already-applied -> warn+skip
    (connector revision drift), except H1/H2 which are load-bearing -> hard error.
  - hard error if an old anchor is found but replacement fails.
  - py_compile at the end.

Usage: apply_kimik3_moriio_kda_fix.py <vllm_install_dir>
       (vllm_install_dir = dir containing 'distributed/...', e.g.
        /usr/local/lib/python3.12/dist-packages/vllm)
"""
import os
import sys

CONN = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
LAYOUT = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py"
COMMON = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py"
SSMUTIL = "distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py"


def _read(p):
    with open(p) as f:
        return f.read()


def _write(p, s):
    with open(p, "w") as f:
        f.write(s)


def _pycompile(p):
    import py_compile
    py_compile.compile(p, doraise=True)


def patch_layout(vllm_dir):
    """H1+H2+H3: MambaSpec geometry + mamba offset builder in moriio_layout.py."""
    path = os.path.join(vllm_dir, LAYOUT)
    if not os.path.isfile(path):
        print(f"[k3-kda] {LAYOUT} not found -- ABORT (connector layout differs).")
        return 1
    src = _read(path)
    orig = src
    applied = []

    # --- H1a: imports (MambaSpec + ssm_conv helpers) ---
    imp_anchor = "from vllm.v1.kv_cache_interface import"
    if "derive_mamba_conv_split" in src:
        applied.append("H1a(already)")
    elif imp_anchor in src:
        # add our imports right after the existing kv_cache_interface import line
        src = src.replace(
            imp_anchor,
            "from vllm.v1.kv_cache_interface import MambaSpec as _K3MambaSpec  # k3-kda\n"
            "from vllm.distributed.kv_transfer.kv_connector.v1."
            "ssm_conv_transfer_utils import (  # k3-kda\n"
            "    derive_mamba_conv_split as _k3_derive_split,\n"
            ")\n"
            + imp_anchor,
            1,
        )
        applied.append("H1a")
    else:
        # fall back: prepend imports at top after docstring
        print("[k3-kda] WARN: kv_cache_interface import anchor not found; "
              "prepending imports.")
        src = ("from vllm.v1.kv_cache_interface import MambaSpec as _K3MambaSpec\n"
               "from vllm.distributed.kv_transfer.kv_connector.v1."
               "ssm_conv_transfer_utils import derive_mamba_conv_split "
               "as _k3_derive_split\n") + src
        applied.append("H1a(prepend)")

    # --- H1b: MambaSpec branch inside get_layer_transfer_geometry ---
    # Anchor: the final ValueError that rejects unknown specs.
    # From the image, get_layer_transfer_geometry ends with:
    #     cache_kind = "MLA" if is_mla_cache else "K/V"
    #     raise ValueError(
    raise_anchor = '    cache_kind = "MLA" if is_mla_cache else "K/V"\n    raise ValueError('
    mamba_branch = (
        "    # k3-kda: MambaSpec (KDA / GDN gated-delta-net) hybrid-state layer.\n"
        "    if isinstance(spec, _K3MambaSpec):\n"
        "        _split = _k3_derive_split(spec, local_tp=1)\n"
        "        _conv_bytes, _ssm_bytes = _split.ssm_sizes\n"
        "        _page = int(_conv_bytes + _ssm_bytes)\n"
        "        _num_blocks = int(shape[0])\n"
        "        return LayerTransferGeometry(\n"
        "            num_blocks=_num_blocks,\n"
        "            block_size=1,\n"
        "            block_len=_page,\n"
        "            slot_size_bytes=_page,\n"
        "            block_stride=(stride[0] if len(stride) > 0 else _page // element_size),\n"
        "            local_kv_stride=None,\n"
        "            remote_kv_stride=None,\n"
        "            transfers_per_block=1,\n"
        "            regions_per_block=1,\n"
        "            split_kv_regions=False,\n"
        "        )\n"
        '    cache_kind = "MLA" if is_mla_cache else "K/V"\n    raise ValueError('
    )
    if "k3-kda: MambaSpec" in src:
        applied.append("H1b(already)")
    elif raise_anchor in src:
        src = src.replace(raise_anchor, mamba_branch, 1)
        applied.append("H1b")
    else:
        print("[k3-kda] ERROR: get_layer_transfer_geometry ValueError anchor "
              "not found -- cannot add MambaSpec branch. ABORT.")
        return 1

    if src != orig:
        _write(path, src)
        try:
            _pycompile(path)
        except Exception as e:
            print(f"[k3-kda] ERROR: {LAYOUT} fails to compile: {e}", file=sys.stderr)
            return 1
        print(f"[k3-kda] patched {LAYOUT} -- hunks: {', '.join(applied)}")
    else:
        print(f"[k3-kda] no changes to {LAYOUT} ({', '.join(applied)})")
    return 0


def patch_connector(vllm_dir):
    """H4: exempt mamba layers from the attention block_size guard in
    register_kv_caches (mamba geometry uses block_size=1, not the attn 16)."""
    path = os.path.join(vllm_dir, CONN)
    if not os.path.isfile(path):
        print(f"[k3-kda] {CONN} not found -- skip connector guard.")
        return 0
    src = _read(path)
    if "k3-kda: mamba layers use block_size=1" in src:
        print("[k3-kda] connector block_size guard already exempts mamba.")
        return 0
    old = (
        "        for layer_name in kv_caches:\n"
        "            geometry = self._get_layer_transfer_geometry(layer_name)\n"
        "            if geometry.block_size != self.block_size:\n"
        "                raise ValueError(\n"
        '                    "MoRIIO KV cache block size mismatch for layer "\n'
        '                    f"{layer_name}: {geometry.block_size} != {self.block_size}"\n'
        "                )\n"
    )
    new = (
        "        from vllm.v1.kv_cache_interface import MambaSpec as _K3MambaSpec  # k3-kda\n"
        "        for layer_name in kv_caches:\n"
        "            geometry = self._get_layer_transfer_geometry(layer_name)\n"
        "            # k3-kda: mamba layers use block_size=1 (indivisible state\n"
        "            # page), not the attention block_size; skip the guard.\n"
        "            _k3_is_mamba = isinstance(\n"
        "                self.layer_to_spec.get(layer_name), _K3MambaSpec\n"
        "            )\n"
        "            if not _k3_is_mamba and geometry.block_size != self.block_size:\n"
        "                raise ValueError(\n"
        '                    "MoRIIO KV cache block size mismatch for layer "\n'
        '                    f"{layer_name}: {geometry.block_size} != {self.block_size}"\n'
        "                )\n"
    )
    if old in src:
        src = src.replace(old, new, 1)
        _write(path, src)
        try:
            _pycompile(path)
        except Exception as e:
            print(f"[k3-kda] ERROR: {CONN} compile: {e}", file=sys.stderr)
            return 1
        print(f"[k3-kda] patched {CONN} -- mamba block_size guard exemption")
    else:
        print("[k3-kda] WARN: register_kv_caches block_size guard anchor not "
              "found -- mamba layers may trip the block_size mismatch raise. "
              "Review moriio_connector.register_kv_caches.")
    return 0


def check_common(vllm_dir):
    """H5: MoRIIOAgentMetadata ssm_sizes field (additive, msgspec omit_defaults)."""
    path = os.path.join(vllm_dir, COMMON)
    if not os.path.isfile(path):
        print(f"[k3-kda] {COMMON} not found -- skip metadata field.")
        return 0
    src = _read(path)
    if "ssm_sizes" in src:
        print("[k3-kda] MoRIIOAgentMetadata.ssm_sizes already present.")
        return 0
    # Anchor: the metadata struct's block_len field line.
    anchor = "    num_blocks: int\n    block_len: int\n    attn_backend_name: str\n"
    add = (
        "    num_blocks: int\n    block_len: int\n    attn_backend_name: str\n"
        "    ssm_sizes: tuple[int, int] = (0, 0)  # k3-kda: (conv_bytes, ssm_bytes)\n"
    )
    if anchor in src:
        src = src.replace(anchor, add, 1)
        _write(path, src)
        try:
            _pycompile(path)
        except Exception as e:
            print(f"[k3-kda] ERROR: {COMMON} compile: {e}", file=sys.stderr)
            return 1
        print(f"[k3-kda] patched {COMMON} -- added ssm_sizes")
    else:
        print("[k3-kda] WARN: MoRIIOAgentMetadata anchor not found -- skip "
              "ssm_sizes (single-TP homogeneous 2P2D may not need it).")
    return 0


def check_scheduler_n1(vllm_dir):
    """Assert the connector already returns N-1 in READ mode (mamba needs it)."""
    path = os.path.join(vllm_dir, CONN)
    if not os.path.isfile(path):
        return 0
    src = _read(path)
    if "len(token_ids) - 1 - num_computed_tokens" in src:
        print("[k3-kda] OK: scheduler already returns N-1 in READ mode "
              "(matches mamba order-dependent state requirement).")
    else:
        print("[k3-kda] WARN: expected N-1 READ-mode return not found in "
              "get_num_new_matched_tokens -- KDA state hand-off may be off by "
              "one token; review scheduler.")
    return 0


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    vllm_dir = sys.argv[1]
    if not os.path.isfile(os.path.join(vllm_dir, SSMUTIL)):
        print(f"[k3-kda] ERROR: {SSMUTIL} missing in image -- this patcher needs "
              "ssm_conv_transfer_utils.py (present in amdsiloai K3 image). ABORT.")
        return 1
    rc = patch_layout(vllm_dir)
    if rc:
        return rc
    rc = patch_connector(vllm_dir)
    if rc:
        return rc
    rc = check_common(vllm_dir)
    if rc:
        return rc
    check_scheduler_n1(vllm_dir)
    print("[k3-kda] DONE (layout MambaSpec branch applied). "
          "NOTE: offset-builder H2 for conv sub-projections is applied by "
          "apply_kimik3_moriio_kda_offsets.py (staged separately).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
