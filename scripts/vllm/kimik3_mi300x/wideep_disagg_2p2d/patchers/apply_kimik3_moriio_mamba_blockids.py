#!/usr/bin/env python3
"""ROOT-CAUSE FIX: route Kimi-K3 KDA/mamba state transfer by the MAMBA KV-cache
group's block ids, not the attention group's.

BUG (confirmed by code + prefill-direct-correct + decode-context-free):
  Kimi-K3 is HYBRID: KV-cache group [0] = MLA attention (paged blocks),
  group [1] = KDA/mamba (per-request MambaSpec state slot, block_size=1).
  These groups allocate slots INDEPENDENTLY -> for the same request the
  attention block id != the mamba slot id.

  The MoRIIO connector treats the model as attention-only:
    * decode advertises only get_block_ids()[0] (attention) as block_notify_list
      (moriio_connector.py ~768); the mamba group [1] slot id is NEVER sent.
    * prefill's per-layer write task uses those attention block ids as BOTH
      source and dest offsets for EVERY layer, including the ~69 KDA layers.
  But decode's KDA forward reads its recurrent/conv state from the MAMBA group
  slot (kda_metadata.py:283 non_spec_state_indices_tensor = block_table[:,0] of
  the mamba group). So the KDA state is written to an ATTENTION block offset that
  decode never reads -> decode's KDA state stays zero -> 69/93 layers blank ->
  fluent but CONTEXT-FREE output. (Attention KV works because its block ids DO
  match on both ends -- which is why prefill-direct is correct and attention-only
  looked byte-perfect.)

FIX (thread the mamba group [1] block ids end-to-end, use them ONLY for mamba
layers; attention path unchanged -> byte-identical for attention):
  1. moriio_common: additive fields
       - WriteTask.mamba_local_block_ids
       - RemoteAllocInfo.mamba_block_ids
       - ReqMeta.mamba_local_block_ids
       - add_new_req(mamba_local_block_ids=None) sets ReqMeta field
  2. connector (decode advertise): also compute mamba_block_notify_list =
     get_block_ids()[1] and send it via send_notify_block.
  3. connector send_notify_block: carry mamba_block_notify_list in the msg.
  4. engine _handle_remote_blocks_message: store mamba_block_ids in RemoteAllocInfo.
  5. connector (prefill save capture in update_state_after_alloc do_remote_decode):
     capture prefill's own mamba slot get_block_ids()[1] into a side dict
     self._reqs_save_mamba (init in scheduler __init__).
  6. connector build_connector_meta save add_new_req calls: pass mamba_local.
  7. connector _write_blocks_for_req -> schedule_write_blocks -> WriteTask: carry
     mamba_local_block_ids.
  8. engine _prepare_transfer_plan: for a MambaSpec layer, compute offsets from
     the mamba local/remote block ids instead of the attention ones.

All hunks idempotent + anchor-based + py_compile-checked. Non-hybrid models keep
group [1] empty -> mamba lists are [] -> zero behavior change.

Usage: apply_kimik3_moriio_mamba_blockids.py <vllm_install_dir>
"""
import os
import sys

CONN = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
ENG = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py"
COMMON = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py"

MARK = "k3-mamba-blockids"


def _apply(path, subs, tag):
    """subs: list of (old, new, guard_substr). If guard_substr in src, skip that
    sub (idempotent). Hard-error if an old anchor is missing and not applied."""
    if not os.path.isfile(path):
        print(f"[{MARK}] {tag}: FILE NOT FOUND {path}", file=sys.stderr)
        return False
    src = open(path).read()
    orig = src
    for old, new, guard in subs:
        if guard in src:
            continue
        if old not in src:
            print(f"[{MARK}] {tag}: ANCHOR MISSING (guard={guard!r})", file=sys.stderr)
            return False
        src = src.replace(old, new, 1)
    if src == orig:
        print(f"[{MARK}] {tag}: already applied.")
        return True
    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        open(path, "w").write(orig)
        print(f"[{MARK}] {tag}: COMPILE FAIL, rolled back: {e}", file=sys.stderr)
        return False
    print(f"[{MARK}] {tag}: applied.")
    return True


def patch_common(base):
    path = os.path.join(base, COMMON)
    subs = [
        # WriteTask field: MUST go after the last non-default field (remote_ip)
        # and before the first field with a default (enqueue_time), else
        # dataclass raises "non-default argument follows default argument".
        (
            "    remote_ip: str\n"
            "    enqueue_time: float = field(default_factory=time.perf_counter)\n",
            "    remote_ip: str\n"
            "    mamba_local_block_ids: list[int] | None = None  # " + MARK + "\n"
            "    enqueue_time: float = field(default_factory=time.perf_counter)\n",
            "mamba_local_block_ids: list[int] | None = None  # " + MARK,
        ),
        # RemoteAllocInfo field
        (
            "    block_ids: list[int]\n"
            "    writes_done: int = 0\n",
            "    block_ids: list[int]\n"
            "    mamba_block_ids: list[int] | None = None  # " + MARK + "\n"
            "    writes_done: int = 0\n",
            "mamba_block_ids: list[int] | None = None  # " + MARK,
        ),
        # ReqMeta field (append after remote_dp_size_local default)
        (
            "    # Per-pod DP size; 0 means fallback to remote_dp_size.\n"
            "    remote_dp_size_local: int = 0\n",
            "    # Per-pod DP size; 0 means fallback to remote_dp_size.\n"
            "    remote_dp_size_local: int = 0\n"
            "    # " + MARK + ": mamba KV-group [1] local slot id(s) for this req.\n"
            "    mamba_local_block_ids: list[int] = field(default_factory=list)\n",
            MARK + ": mamba KV-group [1] local slot",
        ),
        # add_new_req signature: add optional param
        (
            "    def add_new_req(\n"
            "        self,\n"
            "        request_id: ReqId,\n"
            "        local_block_ids: list[int],\n"
            "        kv_transfer_params: dict[str, Any],\n"
            "        write_mode=False,\n"
            "    ):",
            "    def add_new_req(\n"
            "        self,\n"
            "        request_id: ReqId,\n"
            "        local_block_ids: list[int],\n"
            "        kv_transfer_params: dict[str, Any],\n"
            "        write_mode=False,\n"
            "        mamba_local_block_ids: list[int] | None = None,  # " + MARK + "\n"
            "    ):",
            "mamba_local_block_ids: list[int] | None = None,  # " + MARK,
        ),
        # add_new_req body: set ReqMeta field before dispatch to reqs_to_save/recv
        (
            "        if write_mode:\n"
            "            self.reqs_to_save[request_id] = _req\n"
            "        else:\n"
            "            self.reqs_to_recv[request_id] = _req\n",
            "        _req.mamba_local_block_ids = list(mamba_local_block_ids or [])  # " + MARK + "\n"
            "        if write_mode:\n"
            "            self.reqs_to_save[request_id] = _req\n"
            "        else:\n"
            "            self.reqs_to_recv[request_id] = _req\n",
            "_req.mamba_local_block_ids = list(mamba_local_block_ids or [])  # " + MARK,
        ),
    ]
    return _apply(path, subs, "moriio_common.py")


def patch_engine(base):
    path = os.path.join(base, ENG)
    subs = [
        # _handle_remote_blocks_message: capture mamba_block_notify_list
        (
            "        block_notify_list = data.get(\"block_notify_list\", [])\n"
            "        decode_dp_rank = data.get(\"decode_rank\", 0)\n",
            "        block_notify_list = data.get(\"block_notify_list\", [])\n"
            "        mamba_block_notify_list = data.get(\"mamba_block_notify_list\", [])  # " + MARK + "\n"
            "        decode_dp_rank = data.get(\"decode_rank\", 0)\n",
            "mamba_block_notify_list = data.get(\"mamba_block_notify_list\", [])  # " + MARK,
        ),
        (
            "            self.done_remote_allocate_req_dict[transfer_id] = RemoteAllocInfo(\n"
            "                block_ids=block_notify_list, decode_dp_rank=decode_dp_rank\n"
            "            )\n",
            "            self.done_remote_allocate_req_dict[transfer_id] = RemoteAllocInfo(\n"
            "                block_ids=block_notify_list, decode_dp_rank=decode_dp_rank,\n"
            "                mamba_block_ids=list(mamba_block_notify_list or []),  # " + MARK + "\n"
            "            )\n",
            "mamba_block_ids=list(mamba_block_notify_list or []),  # " + MARK,
        ),
        # _prepare_transfer_plan: mamba layers use mamba block ids
        (
            "        offsets = request_info.transfer_offsets.get(geometry_key)\n"
            "        if offsets is None:\n"
            "            offsets = self.worker._compute_block_transfer_offsets(\n"
            "                task.layer_name,\n"
            "                task.local_block_ids,\n"
            "                request_info.block_ids,\n"
            "                remote_moriio_meta,\n"
            "            )\n"
            "            request_info.transfer_offsets[geometry_key] = offsets\n",
            "        offsets = request_info.transfer_offsets.get(geometry_key)\n"
            "        if offsets is None:\n"
            "            from vllm.v1.kv_cache_interface import MambaSpec as _K3MS_BL  # " + MARK + "\n"
            "            _k3_mamba = isinstance(\n"
            "                self.worker.layer_to_spec.get(task.layer_name), _K3MS_BL\n"
            "            )\n"
            "            if _k3_mamba:\n"
            "                _k3_local = task.mamba_local_block_ids or task.local_block_ids\n"
            "                _k3_remote = request_info.mamba_block_ids or request_info.block_ids\n"
            "            else:\n"
            "                _k3_local = task.local_block_ids\n"
            "                _k3_remote = request_info.block_ids\n"
            "            import os as _k3dbgos\n"
            "            if _k3dbgos.environ.get('K3_MAMBA_BC', '0') == '1':\n"
            "                _k3dbg = getattr(self, '_k3_mb_seen', None)\n"
            "                if _k3dbg is None:\n"
            "                    _k3dbg = set(); self._k3_mb_seen = _k3dbg\n"
            "                _k3k = ('M' if _k3_mamba else 'A')\n"
            "                if _k3k not in _k3dbg:\n"
            "                    _k3dbg.add(_k3k)\n"
            "                    _k3srcnorm = -1.0; _k3el = []\n"
            "                    try:\n"
            "                        _k3t = self.worker.kv_caches[task.layer_name]\n"
            "                        if _k3_local:\n"
            "                            _k3slot = _k3t[int(_k3_local[0])].flatten()\n"
            "                            _k3srcnorm = float(_k3slot.float().norm())\n"
            "                            _k3el = [round(float(x),3) for x in _k3slot[:6].float().tolist()]\n"
            "                    except Exception as _k3ee:\n"
            "                        _k3srcnorm = -2.0\n"
            "                    import logging as _k3dbglg\n"
            "                    _k3dbglg.getLogger('vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine').info(\n"
            "                        '[k3-mamba-bc] layer=%s mamba=%s used_local=%s used_remote=%s src_slot_norm=%.4e src_el=%s tshape=%s',\n"
            "                        task.layer_name, _k3_mamba,\n"
            "                        (_k3_local or [])[:3], (_k3_remote or [])[:3],\n"
            "                        _k3srcnorm, _k3el, tuple(self.worker.kv_caches[task.layer_name].shape),\n"
            "                    )\n"
            "            offsets = self.worker._compute_block_transfer_offsets(\n"
            "                task.layer_name,\n"
            "                _k3_local,\n"
            "                _k3_remote,\n"
            "                remote_moriio_meta,\n"
            "            )\n"
            "            request_info.transfer_offsets[geometry_key] = offsets\n",
            "_k3_mamba = isinstance(",
        ),
    ]
    return _apply(path, subs, "moriio_engine.py")


def patch_connector(base):
    path = os.path.join(base, CONN)
    subs = [
        # scheduler __init__: side dict for prefill's own mamba slot per req
        (
            "        self._reqs_need_save: dict[ReqId, tuple[Request, list[int]]] = {}\n",
            "        self._reqs_need_save: dict[ReqId, tuple[Request, list[int]]] = {}\n"
            "        self._reqs_save_mamba: dict[ReqId, list[int]] = {}  # " + MARK + "\n",
            "self._reqs_save_mamba: dict[ReqId, list[int]] = {}  # " + MARK,
        ),
        # prefill save capture (do_remote_decode branch)
        (
            "        if params.get(\"do_remote_decode\"):\n"
            "            local_block_ids = blocks.get_block_ids()[0]\n"
            "            self._reqs_need_save[request.request_id] = (request, local_block_ids)\n",
            "        if params.get(\"do_remote_decode\"):\n"
            "            _k3_gbi = blocks.get_block_ids()\n"
            "            local_block_ids = _k3_gbi[0]\n"
            "            self._reqs_save_mamba[request.request_id] = (  # " + MARK + "\n"
            "                list(_k3_gbi[1]) if len(_k3_gbi) > 1 else []\n"
            "            )\n"
            "            self._reqs_need_save[request.request_id] = (request, local_block_ids)\n",
            "self._reqs_save_mamba[request.request_id] = (  # " + MARK,
        ),
        # send_notify_block signature
        (
            "    def send_notify_block(\n"
            "        self,\n"
            "        req_id: ReqId,\n"
            "        transfer_id: TransferId,\n"
            "        block_notify_list: list[int],\n"
            "        host=None,\n"
            "        port=None,\n"
            "    ):",
            "    def send_notify_block(\n"
            "        self,\n"
            "        req_id: ReqId,\n"
            "        transfer_id: TransferId,\n"
            "        block_notify_list: list[int],\n"
            "        host=None,\n"
            "        port=None,\n"
            "        mamba_block_notify_list: list[int] | None = None,  # " + MARK + "\n"
            "    ):",
            "mamba_block_notify_list: list[int] | None = None,  # " + MARK,
        ),
        # send_notify_block body: include mamba list in message
        (
            "            \"block_notify_list\": block_notify_list or [],\n",
            "            \"block_notify_list\": block_notify_list or [],\n"
            "            \"mamba_block_notify_list\": mamba_block_notify_list or [],  # " + MARK + "\n",
            "\"mamba_block_notify_list\": mamba_block_notify_list or [],  # " + MARK,
        ),
        # decode advertise: build mamba_block_notify_list
        (
            "                    block_notify_list = (\n"
            "                        blocks.get_block_ids()[0] if num_external_tokens > 0 else []\n"
            "                    )\n",
            "                    _k3_gbi_d = blocks.get_block_ids()  # " + MARK + "\n"
            "                    block_notify_list = (\n"
            "                        _k3_gbi_d[0] if num_external_tokens > 0 else []\n"
            "                    )\n"
            "                    mamba_block_notify_list = (\n"
            "                        (list(_k3_gbi_d[1]) if len(_k3_gbi_d) > 1 else [])\n"
            "                        if num_external_tokens > 0 else []\n"
            "                    )\n",
            "_k3_gbi_d = blocks.get_block_ids()  # " + MARK,
        ),
        # decode advertise: pass mamba list to send_notify_block
        (
            "                        self.send_notify_block(\n"
            "                            req_id=request.request_id,\n"
            "                            transfer_id=request.kv_transfer_params[\"transfer_id\"],\n"
            "                            block_notify_list=block_notify_list,\n"
            "                            host=_notify_host,\n"
            "                            port=target_port,\n"
            "                        )\n",
            "                        self.send_notify_block(\n"
            "                            req_id=request.request_id,\n"
            "                            transfer_id=request.kv_transfer_params[\"transfer_id\"],\n"
            "                            block_notify_list=block_notify_list,\n"
            "                            host=_notify_host,\n"
            "                            port=target_port,\n"
            "                            mamba_block_notify_list=mamba_block_notify_list,  # " + MARK + "\n"
            "                        )\n",
            "mamba_block_notify_list=mamba_block_notify_list,  # " + MARK,
        ),
        # build_connector_meta: chunked-prefill final-chunk save add_new_req
        (
            "                        meta.add_new_req(\n"
            "                            request_id=req_id,\n"
            "                            local_block_ids=self._reqs_need_pending_save[req_id][1],\n"
            "                            kv_transfer_params=kv_params,\n"
            "                            write_mode=True,\n"
            "                        )\n",
            "                        meta.add_new_req(\n"
            "                            request_id=req_id,\n"
            "                            local_block_ids=self._reqs_need_pending_save[req_id][1],\n"
            "                            kv_transfer_params=kv_params,\n"
            "                            write_mode=True,\n"
            "                            mamba_local_block_ids=self._reqs_save_mamba.get(req_id, []),  # " + MARK + "\n"
            "                        )\n",
            "mamba_local_block_ids=self._reqs_save_mamba.get(req_id, []),  # " + MARK + "\n"
            "                        )\n",
        ),
        # build_connector_meta: single-chunk save add_new_req
        (
            "            meta.add_new_req(\n"
            "                request_id=req_id,\n"
            "                local_block_ids=block_ids,\n"
            "                kv_transfer_params=kv_params,\n"
            "                write_mode=True,\n"
            "            )\n",
            "            meta.add_new_req(\n"
            "                request_id=req_id,\n"
            "                local_block_ids=block_ids,\n"
            "                kv_transfer_params=kv_params,\n"
            "                write_mode=True,\n"
            "                mamba_local_block_ids=self._reqs_save_mamba.get(req_id, []),  # " + MARK + "\n"
            "            )\n",
            "mamba_local_block_ids=self._reqs_save_mamba.get(req_id, []),  # " + MARK + "\n"
            "            )\n",
        ),
        # schedule_write_blocks signature
        (
            "        layer_name: str,\n"
            "        kv_layer: torch.Tensor,\n"
            "        remote_notify_port: int,\n"
            "        remote_ip: str,\n"
            "    ) -> None:\n"
            "        \"\"\"Schedule a block write operation.",
            "        layer_name: str,\n"
            "        kv_layer: torch.Tensor,\n"
            "        remote_notify_port: int,\n"
            "        remote_ip: str,\n"
            "        mamba_local_block_ids: list[int] | None = None,  # " + MARK + "\n"
            "    ) -> None:\n"
            "        \"\"\"Schedule a block write operation.",
            "mamba_local_block_ids: list[int] | None = None,  # " + MARK,
        ),
        # schedule_write_blocks body: WriteTask carries mamba local
        (
            "        task = WriteTask(\n"
            "            request_id=request_id,\n"
            "            transfer_id=transfer_id,\n"
            "            dst_engine_id=dst_engine_id,\n"
            "            local_block_ids=local_block_ids,\n"
            "            remote_block_ids_hint=remote_block_ids,\n"
            "            layer_name=layer_name,\n",
            "        task = WriteTask(\n"
            "            request_id=request_id,\n"
            "            transfer_id=transfer_id,\n"
            "            dst_engine_id=dst_engine_id,\n"
            "            local_block_ids=local_block_ids,\n"
            "            remote_block_ids_hint=remote_block_ids,\n"
            "            mamba_local_block_ids=mamba_local_block_ids,  # " + MARK + "\n"
            "            layer_name=layer_name,\n",
            "mamba_local_block_ids=mamba_local_block_ids,  # " + MARK,
        ),
        # _write_blocks_for_req: pass mamba local into schedule_write_blocks
        (
            "        self.schedule_write_blocks(\n"
            "            request_id=req_id,\n"
            "            transfer_id=meta.transfer_id,\n"
            "            dst_engine_id=meta.remote_engine_id,\n"
            "            local_block_ids=meta.local_block_ids,\n"
            "            remote_block_ids=meta.remote_block_ids,\n"
            "            layer_name=layer_name,\n"
            "            kv_layer=kv_layer,\n"
            "            remote_notify_port=meta.remote_notify_port,\n"
            "            remote_ip=meta.remote_host,\n"
            "        )\n",
            "        self.schedule_write_blocks(\n"
            "            request_id=req_id,\n"
            "            transfer_id=meta.transfer_id,\n"
            "            dst_engine_id=meta.remote_engine_id,\n"
            "            local_block_ids=meta.local_block_ids,\n"
            "            remote_block_ids=meta.remote_block_ids,\n"
            "            layer_name=layer_name,\n"
            "            kv_layer=kv_layer,\n"
            "            remote_notify_port=meta.remote_notify_port,\n"
            "            remote_ip=meta.remote_host,\n"
            "            mamba_local_block_ids=meta.mamba_local_block_ids,  # " + MARK + "\n"
            "        )\n",
            "mamba_local_block_ids=meta.mamba_local_block_ids,  # " + MARK,
        ),
    ]
    return _apply(path, subs, "moriio_connector.py")


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    base = sys.argv[1]
    ok = True
    ok = patch_common(base) and ok
    ok = patch_engine(base) and ok
    ok = patch_connector(base) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
