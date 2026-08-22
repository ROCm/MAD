#!/usr/bin/env python3
"""ROOT-CAUSE FIX: Kimi-K3 MLA prefill/decode last-block boundary garbage.

Disagg (P/D) serving of Kimi-K3 uses a "mamba N-1" scheme: prefill computes
only the first N-1 of N prompt tokens (KDA recurrence needs h(N-1); decode
recomputes token N locally). The MLA/attention KV, however, is still copied
one WHOLE block at a time (full geometry.block_len bytes) by the MoRIIO RDMA
writer. Because prefill only wrote N-1 token slots, the final (partial) MLA
block's slot for the boundary token (position N-1) is UNINITIALIZED producer
HBM. The whole-block copy ships that garbage into decode's paged slot N-1,
racing + clobbering decode's own correct recompute -> nondeterministic wrong
exact-recall.

FIX: thread valid_tokens (= req.num_prompt_tokens) from the scheduler write
leg down to the layout offset builder, and clamp the FINAL MLA block's copy
size to only the VALID slots. The stale boundary slot is then never
transferred and decode's local recompute is the sole writer. The MambaSpec
branch returns BEFORE the clamp, so KDA/mamba transfers are untouched. When
valid_tokens is None (non-mamba / missing) there is no clamp -> byte-identical
to current behavior. Idempotent, anchor-based, two-pass (verify-all-then-write),
py_compile-checked. Usage: apply_kimik3_moriio_mla_boundary_clamp.py <vllm_dir>
"""
import os
import sys

MARK = "k3-mla-boundary"

_MORIIO = "distributed/kv_transfer/kv_connector/v1/moriio"
COMMON = _MORIIO + "/moriio_common.py"
CONN = _MORIIO + "/moriio_connector.py"
ENGINE = _MORIIO + "/moriio_engine.py"
LAYOUT = _MORIIO + "/moriio_layout.py"

# ----------------------------------------------------------------------------
# moriio_common.py
# ----------------------------------------------------------------------------
C1_old = '''    mamba_local_block_ids: list[int] | None = None  # k3-mamba-blockids
    enqueue_time: float = field(default_factory=time.perf_counter)
    retried: int = 0'''
C1_new = '''    mamba_local_block_ids: list[int] | None = None  # k3-mamba-blockids
    valid_tokens: int | None = None  # k3-mla-boundary
    enqueue_time: float = field(default_factory=time.perf_counter)
    retried: int = 0'''

C2_old = '''    # k3-mamba-blockids: mamba KV-group [1] local slot id(s) for this req.
    mamba_local_block_ids: list[int] = field(default_factory=list)'''
C2_new = '''    # k3-mamba-blockids: mamba KV-group [1] local slot id(s) for this req.
    mamba_local_block_ids: list[int] = field(default_factory=list)
    # k3-mla-boundary: valid prompt-token count (num_prompt_tokens) for the
    # write leg; clamps the final MLA block's RDMA copy to valid slots only.
    valid_tokens: int | None = None'''

C3_old = '''        write_mode=False,
        mamba_local_block_ids: list[int] | None = None,  # k3-mamba-blockids
    ):'''
C3_new = '''        write_mode=False,
        mamba_local_block_ids: list[int] | None = None,  # k3-mamba-blockids
        valid_tokens: int | None = None,  # k3-mla-boundary
    ):'''

C4_old = '''        _req.mamba_local_block_ids = list(mamba_local_block_ids or [])  # k3-mamba-blockids
        if write_mode:'''
C4_new = '''        _req.mamba_local_block_ids = list(mamba_local_block_ids or [])  # k3-mamba-blockids
        _req.valid_tokens = valid_tokens  # k3-mla-boundary
        if write_mode:'''

# ----------------------------------------------------------------------------
# moriio_connector.py
# ----------------------------------------------------------------------------
N1_old = '''            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=kv_params,
                write_mode=True,
                mamba_local_block_ids=self._reqs_save_mamba.get(req_id, []),  # k3-mamba-blockids
            )'''
N1_new = '''            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=kv_params,
                write_mode=True,
                mamba_local_block_ids=self._reqs_save_mamba.get(req_id, []),  # k3-mamba-blockids
                valid_tokens=req.num_prompt_tokens,  # k3-mla-boundary
            )'''

N2_old = '''        remote_ip: str,
        mamba_local_block_ids: list[int] | None = None,  # k3-mamba-blockids
    ) -> None:
        """Schedule a block write operation.'''
N2_new = '''        remote_ip: str,
        mamba_local_block_ids: list[int] | None = None,  # k3-mamba-blockids
        valid_tokens: int | None = None,  # k3-mla-boundary
    ) -> None:
        """Schedule a block write operation.'''

N3_old = '''            mamba_local_block_ids=mamba_local_block_ids,  # k3-mamba-blockids
            layer_name=layer_name,'''
N3_new = '''            mamba_local_block_ids=mamba_local_block_ids,  # k3-mamba-blockids
            valid_tokens=valid_tokens,  # k3-mla-boundary
            layer_name=layer_name,'''

N4_old = '''            remote_ip=meta.remote_host,
            mamba_local_block_ids=meta.mamba_local_block_ids,  # k3-mamba-blockids
        )'''
N4_new = '''            remote_ip=meta.remote_host,
            mamba_local_block_ids=meta.mamba_local_block_ids,  # k3-mamba-blockids
            valid_tokens=getattr(meta, "valid_tokens", None),  # k3-mla-boundary
        )'''

N5_old = '''        remote_moriio_meta: MoRIIOAgentMetadata,
        remote_tp_size: int | None = None,
    ) -> tuple[list[int], list[int], list[int]]:'''
N5_new = '''        remote_moriio_meta: MoRIIOAgentMetadata,
        remote_tp_size: int | None = None,
        valid_tokens: int | None = None,  # k3-mla-boundary
    ) -> tuple[list[int], list[int], list[int]]:'''

N6_old = '''            remote_num_blocks=remote_moriio_meta.num_blocks,
            merge_fn=lambda local, remote, sizes: self.merge_contiguous_blocks('''
N6_new = '''            remote_num_blocks=remote_moriio_meta.num_blocks,
            valid_tokens=valid_tokens,  # k3-mla-boundary
            merge_fn=lambda local, remote, sizes: self.merge_contiguous_blocks('''

# ----------------------------------------------------------------------------
# moriio_engine.py
# ----------------------------------------------------------------------------
E1_old = '''            offsets = self.worker._compute_block_transfer_offsets(
                task.layer_name,
                _k3_local,
                _k3_remote,
                remote_moriio_meta,
            )'''
E1_new = '''            offsets = self.worker._compute_block_transfer_offsets(
                task.layer_name,
                _k3_local,
                _k3_remote,
                remote_moriio_meta,
                valid_tokens=getattr(task, "valid_tokens", None),  # k3-mla-boundary
            )'''

# ----------------------------------------------------------------------------
# moriio_layout.py
# ----------------------------------------------------------------------------
L1_old = '''    remote_num_blocks: int,
    merge_fn: Callable[
        [list[int], list[int], list[int]], tuple[list[int], list[int], list[int]]
    ] = merge_contiguous_offsets,
) -> tuple[list[int], list[int], list[int]]:'''
L1_new = '''    remote_num_blocks: int,
    merge_fn: Callable[
        [list[int], list[int], list[int]], tuple[list[int], list[int], list[int]]
    ] = merge_contiguous_offsets,
    valid_tokens: int | None = None,  # k3-mla-boundary
) -> tuple[list[int], list[int], list[int]]:'''

L2_old = '''    sizes = [transfer_size_byte] * total

    w = 0'''
L2_new = '''    sizes = [transfer_size_byte] * total
    # k3-mla-boundary: prefill wrote only `valid_tokens` slots; the last partial
    # block's tail (past the last valid token) is uninitialized producer HBM.
    # Clamp the final block's copy to valid slots so the stale boundary slot is
    # never transferred (decode's local recompute is then the sole writer).
    if valid_tokens is not None and local_block_ids:
        _bs = geometry.block_size
        _valid_in_last = valid_tokens - (len(local_block_ids) - 1) * _bs
        if 0 < _valid_in_last < _bs:
            _clamped = _valid_in_last * geometry.slot_size_bytes
            _last = (len(local_block_ids) - 1) * per_block
            for _j in range(per_block):
                sizes[_last + _j] = _clamped
            import logging as _k3lg
            _k3lg.getLogger(__name__).info(
                "[k3-mla-boundary] clamped last block to %d/%d slots (%d B)",
                _valid_in_last, _bs, _clamped,
            )

    w = 0'''

EDITS = {
    COMMON: [
        (C1_old, C1_new, "C1 WriteTask.valid_tokens field"),
        (C2_old, C2_new, "C2 ReqMeta.valid_tokens field"),
        (C3_old, C3_new, "C3 add_new_req signature"),
        (C4_old, C4_new, "C4 add_new_req body setter"),
    ],
    CONN: [
        (N1_old, N1_new, "N1 build_connector_meta write-mode call"),
        (N2_old, N2_new, "N2 schedule_write_blocks signature"),
        (N3_old, N3_new, "N3 WriteTask construction"),
        (N4_old, N4_new, "N4 _write_blocks_for_req call"),
        (N5_old, N5_new, "N5 _compute_block_transfer_offsets signature"),
        (N6_old, N6_new, "N6 forward into compute_block_transfer_offsets"),
    ],
    ENGINE: [
        (E1_old, E1_new, "E1 _prepare_transfer_plan call"),
    ],
    LAYOUT: [
        (L1_old, L1_new, "L1 compute_block_transfer_offsets signature"),
        (L2_old, L2_new, "L2 MLA last-block clamp"),
    ],
}


def main():
    if len(sys.argv) < 2:
        print(f"[{MARK}] usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 1
    base = sys.argv[1]

    # Pass 1: load every target, skip already-applied files, and verify that
    # EVERY anchor is present BEFORE writing anything. Any missing anchor aborts
    # with return 1 and zero partial edits (safe-by-construction).
    plans = []
    for rel, edits in EDITS.items():
        path = os.path.join(base, rel)
        if not os.path.isfile(path):
            print(f"[{MARK}] not found {path}", file=sys.stderr)
            return 1
        src = open(path).read()
        if MARK in src:
            print(f"[{MARK}] {rel}: already applied, skipping.")
            continue
        for old, new, tag in edits:
            if old not in src:
                print(f"[{MARK}] {rel}: {tag}: ANCHOR MISSING", file=sys.stderr)
                return 1
        plans.append((path, rel, src, edits))

    if not plans:
        print(f"[{MARK}] all target files already applied; nothing to do.")
        return 0

    # Pass 2: all anchors verified -> apply + write + py_compile each file.
    for path, rel, src, edits in plans:
        for old, new, tag in edits:
            src = src.replace(old, new, 1)
        open(path, "w").write(src)
        try:
            import py_compile
            py_compile.compile(path, doraise=True)
        except Exception as e:
            print(f"[{MARK}] compile FAIL {path}: {e}", file=sys.stderr)
            return 1
        print(f"[{MARK}] applied {rel}")

    print(f"[{MARK}] applied to {len(plans)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
