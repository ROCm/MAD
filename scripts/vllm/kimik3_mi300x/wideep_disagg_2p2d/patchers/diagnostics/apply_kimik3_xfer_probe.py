#!/usr/bin/env python3
"""RUNTIME PROBE (K3_XFER_PROBE=1): log MoRIIO transfer offsets + source KV checksum
on the PRODUCER (compute_block_transfer_offsets, MLA branch) so we can see the
ACTUAL local/remote byte offsets and a checksum of the local KV being sent.

Gated by K3_XFER_PROBE=1 (default off => byte-identical). Logs at most the FIRST
few transfers per process to avoid flooding. Correlate producer log (this) with
decode-side KV checksum to determine if offsets are wrong vs decode reads wrong
blocks. Anchor-based, idempotent, py_compile-checked.
Usage: apply_kimik3_xfer_probe.py <vllm_install_dir>
"""
import os, sys

MARK = "k3-xfer-probe"
REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py"
REL2 = "v1/attention/backends/mla/triton_mla.py"
REL3 = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"

OLD = "    return merge_fn(offset_local, offset_remote, sizes)\n"
# NOTE: this OLD appears twice (mamba fn + MLA fn). We target the MLA one via a
# larger unique anchor including the preceding zip loop tail.
OLD_CTX = (
    "            offset_remote[w] = element_size * (\n"
    "                geometry.remote_kv_stride + rb * geometry.block_stride\n"
    "            )\n"
    "            w += 1\n"
    "\n"
    "    return merge_fn(offset_local, offset_remote, sizes)\n"
)
NEW_CTX = (
    "            offset_remote[w] = element_size * (\n"
    "                geometry.remote_kv_stride + rb * geometry.block_stride\n"
    "            )\n"
    "            w += 1\n"
    "\n"
    "    import os as _k3xp_os  # " + MARK + "\n"
    "    if _k3xp_os.environ.get('K3_XFER_PROBE', '0') == '1':\n"
    "        try:\n"
    "            import logging as _k3xp_lg\n"
    "            _k3xp_L = _k3xp_lg.getLogger(__name__)\n"
    "            _k3xp_n = getattr(compute_block_transfer_offsets, '_k3xp_count', 0)\n"
    "            if _k3xp_n < 6 and local_block_ids:\n"
    "                compute_block_transfer_offsets._k3xp_count = _k3xp_n + 1\n"
    "                _k3xp_lb0 = int(local_block_ids[0])\n"
    "                _k3xp_rb0 = int(remote_block_ids[0])\n"
    "                _k3xp_flat = kv_cache.flatten()\n"
    "                _k3xp_o = offset_local[0] // element_size\n"
    "                _k3xp_slice = _k3xp_flat[_k3xp_o:_k3xp_o + 64]\n"
    "                _k3xp_cs = float(_k3xp_slice.detach().to('cpu').double().sum().item())\n"
    "                _k3xp_L.warning(\n"
    "                    '[" + MARK + "] layer=%s nblk=%d lb0=%d rb0=%d block_stride=%d elt=%d '\n"
    "                    'off_local0=%d off_remote0=%d size0=%d src_cs=%.6f',\n"
    "                    layer_name, len(local_block_ids), _k3xp_lb0, _k3xp_rb0,\n"
    "                    int(geometry.block_stride), int(element_size),\n"
    "                    int(offset_local[0]), int(offset_remote[0]), int(sizes[0]), _k3xp_cs,\n"
    "                )\n"
    "        except Exception as _k3xp_e:\n"
    "            import logging as _k3xp_lg2\n"
    "            _k3xp_lg2.getLogger(__name__).warning('[" + MARK + "] err %s', _k3xp_e)\n"
    "\n"
    "    return merge_fn(offset_local, offset_remote, sizes)\n"
)


DEC_OLD = '        block_table = attn_metadata.decode.block_table\n        seq_lens = attn_metadata.decode.seq_lens\n'
DEC_NEW = "        block_table = attn_metadata.decode.block_table\n        seq_lens = attn_metadata.decode.seq_lens\n        import os as _k3dp_os  # k3-xfer-probe\n        if _k3dp_os.environ.get('K3_XFER_PROBE', '0') == '1':\n            try:\n                import logging as _k3dp_lg\n                _k3dp_n = getattr(decode_attention_fwd, '_k3dp_count', 0)\n                if _k3dp_n < 8 and block_table is not None and block_table.numel() > 0:\n                    decode_attention_fwd._k3dp_count = _k3dp_n + 1\n                    _k3dp_b0 = int(block_table.flatten()[0].item())\n                    _k3dp_sl = int(seq_lens.flatten()[0].item()) if seq_lens is not None and seq_lens.numel() else -1\n                    _k3dp_blk = kv_c_and_k_pe_cache[_k3dp_b0].flatten()\n                    _k3dp_cs = float(_k3dp_blk[:64].detach().to('cpu').double().sum().item())\n                    _k3dp_nz = int((_k3dp_blk != 0).sum().item())\n                    _k3dp_lg.getLogger(__name__).warning(\n                        '[k3-xfer-probe DECODE] read_block0=%d seq_len0=%d dst_cs=%.6f nonzero=%d/%d',\n                        _k3dp_b0, _k3dp_sl, _k3dp_cs, _k3dp_nz, int(_k3dp_blk.numel()),\n                    )\n            except Exception as _k3dp_e:\n                import logging as _k3dp_lg2\n                _k3dp_lg2.getLogger(__name__).warning('[k3-xfer-probe DECODE] err %s', _k3dp_e)\n"


GBI_OLD = '                    _k3_gbi_d = blocks.get_block_ids()  # k3-mamba-blockids\n                    block_notify_list = (\n                        _k3_gbi_d[0] if num_external_tokens > 0 else []\n                    )\n'
GBI_NEW = "                    _k3_gbi_d = blocks.get_block_ids()  # k3-mamba-blockids\n                    import os as _k3gp_os  # k3-xfer-probe\n                    if _k3gp_os.environ.get('K3_XFER_PROBE','0')=='1':\n                        try:\n                            import logging as _k3gp_lg\n                            _k3gp_lg.getLogger(__name__).warning(\n                                '[k3-xfer-probe GBI] ngroups=%d group0=%s group1=%s num_ext=%d',\n                                len(_k3_gbi_d),\n                                str([int(x) for x in _k3_gbi_d[0][:6]]) if len(_k3_gbi_d)>0 else '[]',\n                                str([int(x) for x in _k3_gbi_d[1][:6]]) if len(_k3_gbi_d)>1 else '[]',\n                                int(num_external_tokens),\n                            )\n                        except Exception as _k3gp_e:\n                            import logging as _k3gp_lg2\n                            _k3gp_lg2.getLogger(__name__).warning('[k3-xfer-probe GBI] err %s', _k3gp_e)\n                    block_notify_list = (\n                        _k3_gbi_d[0] if num_external_tokens > 0 else []\n                    )\n"


GD_OLD = '        self.layer_to_spec = build_layer_to_spec(kv_cache_config)\n'
GD_NEW = "        self.layer_to_spec = build_layer_to_spec(kv_cache_config)\n        import os as _k3grp_os  # k3-xfer-probe\n        if _k3grp_os.environ.get('K3_XFER_PROBE','0')=='1':\n            try:\n                import logging as _k3grp_lg\n                for _gi, _g in enumerate(kv_cache_config.kv_cache_groups):\n                    _sp = _g.kv_cache_spec\n                    _lns = list(getattr(_g, 'layer_names', []) or [])\n                    _k3grp_lg.getLogger(__name__).warning(\n                        '[k3-xfer-probe GROUP] idx=%d spec=%s nlayers=%d layers0_3=%s',\n                        _gi, type(_sp).__name__, len(_lns), str(_lns[:4]),\n                    )\n            except Exception as _k3grp_e:\n                import logging as _k3grp_lg2\n                _k3grp_lg2.getLogger(__name__).warning('[k3-xfer-probe GROUP] err %s', _k3grp_e)\n"


MB_OLD = '            w += 1\n    return merge_fn(offset_local, offset_remote, sizes)\n'
MB_NEW = "            w += 1\n    import os as _k3mp_os  # k3-xfer-probe MAMBA\n    if _k3mp_os.environ.get('K3_XFER_PROBE','0')=='1':\n        try:\n            import logging as _k3mp_lg\n            _k3mp_n = getattr(compute_mamba_block_transfer_offsets,'_k3mp_c',0)\n            if _k3mp_n < 6 and local_block_ids:\n                compute_mamba_block_transfer_offsets._k3mp_c = _k3mp_n+1\n                _k3mp_lg.getLogger(__name__).warning(\n                    '[k3-xfer-probe MAMBA] layer=%s nblk=%d lb=%s rb=%s stride=%d nsub=%d off0=%d sz0=%d',\n                    layer_name, len(local_block_ids),\n                    str([int(x) for x in local_block_ids[:6]]),\n                    str([int(x) for x in remote_block_ids[:6]]),\n                    int(stride), len(subregions), int(offset_local[0]) if offset_local else -1,\n                    int(sizes[0]) if sizes else -1,\n                )\n        except Exception as _k3mp_e:\n            import logging as _k3mp_l2\n            _k3mp_l2.getLogger(__name__).warning('[k3-xfer-probe MAMBA] err %s', _k3mp_e)\n    return merge_fn(offset_local, offset_remote, sizes)\n"


def main():
    if len(sys.argv) < 2:
        print(f"[{MARK}] usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 1
    base = sys.argv[1]
    edits = [(REL, OLD_CTX, NEW_CTX, "producer-offsets"),
             (REL2, DEC_OLD, DEC_NEW, "decode-read"),
             (REL3, GBI_OLD, GBI_NEW, "gbi-groups"),
             (REL3, GD_OLD, GD_NEW, "group-dump"),
             (REL, MB_OLD, MB_NEW, "mamba-offsets")]
    for rel, old, new, tag in edits:
        path = os.path.join(base, rel)
        if not os.path.isfile(path):
            print(f"[{MARK}] {tag}: not found {path}", file=sys.stderr); return 1
        src = open(path).read()
        # per-edit idempotency: skip only if THIS edit's unique tag marker present
        _tagmark = "k3-xfer-probe " + tag.upper().split("-")[0]
        if old not in src:
            if _tagmark in src or new[:60] in src:
                print(f"[{MARK}] {tag}: already applied."); continue
            print(f"[{MARK}] {tag}: ANCHOR MISSING", file=sys.stderr); return 1
        src = src.replace(old, new, 1)
        open(path, "w").write(src)
        try:
            import py_compile; py_compile.compile(path, doraise=True)
        except Exception as e:
            print(f"[{MARK}] {tag}: compile FAIL {e}", file=sys.stderr); return 1
        print(f"[{MARK}] {tag}: applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
