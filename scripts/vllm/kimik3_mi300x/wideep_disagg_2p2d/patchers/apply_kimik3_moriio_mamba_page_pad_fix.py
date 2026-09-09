#!/usr/bin/env python3
"""Use the PADDED physical mamba page for MoRIIO block stride/geometry.

PROBLEM (the GPU memory fault on the disagg producer path)
  The hybrid KV allocator pads the mamba page so it matches the attention page
  ("Padding mamba page size by N%"). The mamba KV cache tensor is therefore
  [num_blocks, 1, 1, spec.page_size_bytes] (gpu_model_runner._reshape_kv_cache_tensors)
  with block stride = spec.page_size_bytes (padded).

  But moriio_layout computed the mamba block stride / geometry page as the
  UNPADDED live-data size (conv_bytes + ssm_bytes) from derive_mamba_conv_split:
    - get_layer_transfer_geometry MambaSpec branch: _page = conv+ssm
    - compute_mamba_block_transfer_offsets: page = conv+ssm; stride = page

  Since padded > unpadded, `lbase = block_id * unpadded_page` drifts below the true
  block address by (padded-unpadded) per block. Low block IDs are ~ok, higher ones
  land out of bounds. On the producer's first real disagg prefill the MoRIIO RDMA
  write to these addresses touches unmapped GPU memory ->
    "Memory access fault by GPU node-N on address 0x...". All ranks fault together.
  (Standalone prefill/decode never hit this: the fault is only on the KV-transfer
  write path.)

FIX (surgical, 2 sites in moriio_layout.py)
  1) compute_mamba_block_transfer_offsets: stride = int(spec.page_size_bytes)
     (keep conv/ssm sub-region offsets+sizes as the live payload within the page).
  2) get_layer_transfer_geometry MambaSpec branch: _page = int(spec.page_size_bytes)
     so region_len = num_blocks * padded_page matches the real tensor extent.

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_moriio_mamba_page_pad_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py"


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[k3-mamba-pad] {REL} not found -- skip.")
        return 0
    src = open(path).read()
    orig = src

    if "k3-mamba-pad: physical padded page" in src:
        print("[k3-mamba-pad] already applied.")
        return 0

    changed = 0

    # --- site 1: compute_mamba_block_transfer_offsets stride ---
    # Handle the two shapes this may exist in (with or without the interim
    # `page = int(conv_bytes + ssm_bytes)` line).
    off_variants = [
        (
            "    conv_bytes, ssm_bytes = split.ssm_sizes\n"
            "    page = int(conv_bytes + ssm_bytes)\n"
            "    # Sub-regions within one page: conv sub-projections, then ssm.\n"
            "    subregions = list(split.local_conv_offsets)  # [(off,size), ...] for Q,K,V\n"
            "    subregions.append((int(conv_bytes), int(ssm_bytes)))  # ssm follows conv\n"
            "\n"
            "    # Byte stride between blocks = full page (state blocks are indivisible; one\n"
            "    # logical block == one physical page for mamba).\n"
            "    stride = page\n"
        ),
    ]
    off_new = (
        "    conv_bytes, ssm_bytes = split.ssm_sizes\n"
        "    # Sub-regions within one page: conv sub-projections, then ssm.\n"
        "    subregions = list(split.local_conv_offsets)  # [(off,size), ...] for Q,K,V\n"
        "    subregions.append((int(conv_bytes), int(ssm_bytes)))  # ssm follows conv\n"
        "\n"
        "    # k3-mamba-pad: physical padded page as block stride (the tensor is\n"
        "    # [num_blocks,1,1,spec.page_size_bytes]; unpadded conv+ssm drifts OOB).\n"
        "    stride = int(spec.page_size_bytes)\n"
    )
    for v in off_variants:
        if v in src:
            src = src.replace(v, off_new, 1)
            changed += 1
            break
    else:
        # Fallback: just rewrite the bare `stride = page` if present.
        if "\n    stride = page\n" in src:
            src = src.replace(
                "\n    stride = page\n",
                "\n    # k3-mamba-pad: physical padded page (was conv+ssm; drifted OOB)\n"
                "    stride = int(spec.page_size_bytes)\n",
                1,
            )
            changed += 1

    # --- site 2: geometry MambaSpec branch _page ---
    geom_old = (
        "        _split = _k3_derive_split(spec, local_tp=1)\n"
        "        _conv_bytes, _ssm_bytes = _split.ssm_sizes\n"
        "        _page = int(_conv_bytes + _ssm_bytes)\n"
    )
    geom_new = (
        "        # k3-mamba-pad: physical padded page (matches the reshaped tensor\n"
        "        # [num_blocks,1,1,page_size_bytes]); unpadded conv+ssm under-registers.\n"
        "        _page = int(spec.page_size_bytes)\n"
    )
    if geom_old in src:
        src = src.replace(geom_old, geom_new, 1)
        changed += 1

    if changed == 0:
        print("[k3-mamba-pad] WARN: no anchors matched -- mamba page may still use "
              "unpadded size. Review moriio_layout.py.")
        return 0

    if src != orig:
        open(path, "w").write(src)
        try:
            import py_compile
            py_compile.compile(path, doraise=True)
        except Exception as e:
            print(f"[k3-mamba-pad] ERROR: compile failed: {e}", file=sys.stderr)
            open(path, "w").write(orig)
            return 1
        print(f"[k3-mamba-pad] applied {changed} site(s): mamba block stride/page "
              "-> spec.page_size_bytes (padded).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
