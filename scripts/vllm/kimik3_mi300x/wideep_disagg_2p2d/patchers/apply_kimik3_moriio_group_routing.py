#!/usr/bin/env python3
"""ROOT-CAUSE FIX: per-kv-cache-group block-id routing for Kimi-K3 disagg.

Kimi-K3 is a hybrid model with FOUR kv-cache groups (confirmed at runtime):
  idx 0: MambaSpec (23 KDA layers)
  idx 1: MambaSpec (23 KDA layers)
  idx 2: MambaSpec (23 KDA layers)
  idx 3: MLAAttentionSpec (24 full-attention MLA layers)  <- ATTENTION GROUP
``blocks.get_block_ids()`` returns a tuple of 4 lists, one per group, in this
index order.

The MoRIIO connector was written for a 2-group model. It hardcodes:
  * attention blocks = get_block_ids()[0]   (WRONG: idx 0 is a Mamba group)
  * mamba blocks     = get_block_ids()[1]   (WRONG: mamba is spread over 0/1/2)
Result: the producer RDMA-writes MLA KV using group-0 (mamba) block ids while
decode's MLA attention reads group-3 blocks -> decode reads NEVER-WRITTEN blocks
-> garbage exact-recall. This is THE root cause of the disagg decode-recall bug.

THE FIX (this patch): carry ALL groups' block-id lists end-to-end as a
list-of-lists keyed by group index, and route EVERY layer's KV transfer to ITS
OWN group's list, indexed by the layer's kv-cache-group index. Concretely, in
MoRIIOWriter._prepare_transfer_plan, per layer:
    gi        = worker._layer_group_idx.get(layer_name, worker._attn_group_idx)
    _k3_local  = task.all_group_block_ids[gi]        # producer's own blocks
    _k3_remote = request_info.all_group_block_ids[gi] # decode's blocks
with a safe fallback to the legacy mamba/attn ([1]/[0]) behavior when
all_group_block_ids is absent/short.

Wire/dataclass plumbing (all trailing-defaulted -> backward-compatible):
  moriio_common.py : WriteTask + ReqMeta + RemoteAllocInfo gain
                     ``all_group_block_ids``; add_new_req gains the kwarg.
  moriio_connector.py:
    - scheduler producer-save (update_state_after_alloc): stash full group tuple
      in self._reqs_save_allgrp; pass it through add_new_req.
    - scheduler decode-advertise: send the full list-of-lists in the notify
      (key "all_group_block_notify"); send_notify_block wire dict carries it.
    - worker __init__: compute self._layer_group_idx (layer_name -> group idx)
      and self._attn_group_idx (first non-Mamba group) from
      kv_cache_config.kv_cache_groups; self._k3_group_routing from env.
    - schedule_write_blocks / WriteTask / _write_blocks_for_req: plumb the field.
  moriio_engine.py:
    - _handle_remote_blocks_message: store all_group_block_ids into
      RemoteAllocInfo.
    - _prepare_transfer_plan: the group-indexed routing (see above).

Gated behind env K3_GROUP_ROUTING (default "1" = on; "0" restores legacy
[0]/[1] routing for A/B). Non-K3 / 2-group models NEVER populate
all_group_block_ids, so they always take the legacy fallback -> zero behavior
change. Idempotent (MARK guard per file), TWO-PASS (verify every anchor across
every file BEFORE any write; any miss => zero writes, return 1), py_compile of
each edited file.

Usage: apply_kimik3_moriio_group_routing.py <vllm_install_dir>
"""
import os
import sys

MARK = "k3-group-routing"

COMMON = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py"
CONN = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
ENGINE = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py"


# --------------------------------------------------------------------------- #
# moriio_common.py
# --------------------------------------------------------------------------- #
COMMON_EDITS = [
    (
        "C1 WriteTask field",
        "    mamba_local_block_ids: list[int] | None = None  # k3-mamba-blockids\n"
        "    enqueue_time: float = field(default_factory=time.perf_counter)\n",
        "    mamba_local_block_ids: list[int] | None = None  # k3-mamba-blockids\n"
        "    all_group_block_ids: list[list[int]] | None = None  # " + MARK + "\n"
        "    enqueue_time: float = field(default_factory=time.perf_counter)\n",
    ),
    (
        "C2 RemoteAllocInfo field",
        "    mamba_block_ids: list[int] | None = None  # k3-mamba-blockids\n"
        "    writes_done: int = 0\n",
        "    mamba_block_ids: list[int] | None = None  # k3-mamba-blockids\n"
        "    all_group_block_ids: list[list[int]] | None = None  # " + MARK + "\n"
        "    writes_done: int = 0\n",
    ),
    (
        "C3 ReqMeta field",
        "    # k3-mamba-blockids: mamba KV-group [1] local slot id(s) for this req.\n"
        "    mamba_local_block_ids: list[int] = field(default_factory=list)\n",
        "    # k3-mamba-blockids: mamba KV-group [1] local slot id(s) for this req.\n"
        "    mamba_local_block_ids: list[int] = field(default_factory=list)\n"
        "    # " + MARK + ": ALL kv-cache-groups' local block-id lists (list of\n"
        "    # lists, indexed by group index) so each layer routes to its own group.\n"
        "    all_group_block_ids: list[list[int]] | None = None\n",
    ),
    (
        "C4 add_new_req signature",
        "        write_mode=False,\n"
        "        mamba_local_block_ids: list[int] | None = None,  # k3-mamba-blockids\n"
        "    ):\n",
        "        write_mode=False,\n"
        "        mamba_local_block_ids: list[int] | None = None,  # k3-mamba-blockids\n"
        "        all_group_block_ids: list[list[int]] | None = None,  # " + MARK + "\n"
        "    ):\n",
    ),
    (
        "C5 add_new_req set field",
        "        _req.mamba_local_block_ids = list(mamba_local_block_ids or [])  # k3-mamba-blockids\n"
        "        if write_mode:\n",
        "        _req.mamba_local_block_ids = list(mamba_local_block_ids or [])  # k3-mamba-blockids\n"
        "        _req.all_group_block_ids = (  # " + MARK + "\n"
        "            [list(g) for g in all_group_block_ids]\n"
        "            if all_group_block_ids is not None else None\n"
        "        )\n"
        "        if write_mode:\n",
    ),
]


# --------------------------------------------------------------------------- #
# moriio_connector.py
# --------------------------------------------------------------------------- #
CONN_EDITS = [
    (
        "N3 scheduler init dict",
        "        self._reqs_save_mamba: dict[ReqId, list[int]] = {}  # k3-mamba-blockids\n",
        "        self._reqs_save_mamba: dict[ReqId, list[int]] = {}  # k3-mamba-blockids\n"
        "        self._reqs_save_allgrp: dict[ReqId, list[list[int]]] = {}  # " + MARK + "\n",
    ),
    (
        "N6 send_notify_block signature",
        "        mamba_block_notify_list: list[int] | None = None,  # k3-mamba-blockids\n"
        "    ):\n"
        "        path = make_zmq_path(\"tcp\", host, port)\n",
        "        mamba_block_notify_list: list[int] | None = None,  # k3-mamba-blockids\n"
        "        all_group_block_notify: list[list[int]] | None = None,  # " + MARK + "\n"
        "    ):\n"
        "        path = make_zmq_path(\"tcp\", host, port)\n",
    ),
    (
        "N7 send_notify_block wire dict",
        "            \"mamba_block_notify_list\": mamba_block_notify_list or [],  # k3-mamba-blockids\n",
        "            \"mamba_block_notify_list\": mamba_block_notify_list or [],  # k3-mamba-blockids\n"
        "            \"all_group_block_notify\": all_group_block_notify or [],  # " + MARK + "\n",
    ),
    (
        "N2 scheduler producer-save capture",
        "            self._reqs_save_mamba[request.request_id] = (  # k3-mamba-blockids\n"
        "                list(_k3_gbi[1]) if len(_k3_gbi) > 1 else []\n"
        "            )\n"
        "            self._reqs_need_save[request.request_id] = (request, local_block_ids)\n",
        "            self._reqs_save_mamba[request.request_id] = (  # k3-mamba-blockids\n"
        "                list(_k3_gbi[1]) if len(_k3_gbi) > 1 else []\n"
        "            )\n"
        "            self._reqs_save_allgrp[request.request_id] = [  # " + MARK + "\n"
        "                list(g) for g in _k3_gbi\n"
        "            ]\n"
        "            self._reqs_need_save[request.request_id] = (request, local_block_ids)\n",
    ),
    (
        "N4 scheduler decode-advertise compute",
        "                    mamba_block_notify_list = (\n"
        "                        (list(_k3_gbi_d[1]) if len(_k3_gbi_d) > 1 else [])\n"
        "                        if num_external_tokens > 0 else []\n"
        "                    )\n",
        "                    mamba_block_notify_list = (\n"
        "                        (list(_k3_gbi_d[1]) if len(_k3_gbi_d) > 1 else [])\n"
        "                        if num_external_tokens > 0 else []\n"
        "                    )\n"
        "                    all_group_block_notify = (  # " + MARK + "\n"
        "                        [list(g) for g in _k3_gbi_d]\n"
        "                        if num_external_tokens > 0 else []\n"
        "                    )\n",
    ),
    (
        "N5 scheduler send_notify_block call",
        "                        self.send_notify_block(\n"
        "                            req_id=request.request_id,\n"
        "                            transfer_id=request.kv_transfer_params[\"transfer_id\"],\n"
        "                            block_notify_list=block_notify_list,\n"
        "                            host=_notify_host,\n"
        "                            port=target_port,\n"
        "                            mamba_block_notify_list=mamba_block_notify_list,  # k3-mamba-blockids\n"
        "                        )\n",
        "                        self.send_notify_block(\n"
        "                            req_id=request.request_id,\n"
        "                            transfer_id=request.kv_transfer_params[\"transfer_id\"],\n"
        "                            block_notify_list=block_notify_list,\n"
        "                            host=_notify_host,\n"
        "                            port=target_port,\n"
        "                            mamba_block_notify_list=mamba_block_notify_list,  # k3-mamba-blockids\n"
        "                            all_group_block_notify=all_group_block_notify,  # " + MARK + "\n"
        "                        )\n",
    ),
    (
        "N8 build_connector_meta add_new_req (pending final chunk)",
        "                        meta.add_new_req(\n"
        "                            request_id=req_id,\n"
        "                            local_block_ids=self._reqs_need_pending_save[req_id][1],\n"
        "                            kv_transfer_params=kv_params,\n"
        "                            write_mode=True,\n"
        "                            mamba_local_block_ids=self._reqs_save_mamba.get(req_id, []),  # k3-mamba-blockids\n"
        "                        )\n",
        "                        meta.add_new_req(\n"
        "                            request_id=req_id,\n"
        "                            local_block_ids=self._reqs_need_pending_save[req_id][1],\n"
        "                            kv_transfer_params=kv_params,\n"
        "                            write_mode=True,\n"
        "                            mamba_local_block_ids=self._reqs_save_mamba.get(req_id, []),  # k3-mamba-blockids\n"
        "                            all_group_block_ids=self._reqs_save_allgrp.get(req_id, None),  # " + MARK + "\n"
        "                        )\n",
    ),
    (
        "N9 build_connector_meta add_new_req (direct save)",
        "            meta.add_new_req(\n"
        "                request_id=req_id,\n"
        "                local_block_ids=block_ids,\n"
        "                kv_transfer_params=kv_params,\n"
        "                write_mode=True,\n"
        "                mamba_local_block_ids=self._reqs_save_mamba.get(req_id, []),  # k3-mamba-blockids\n"
        "            )\n",
        "            meta.add_new_req(\n"
        "                request_id=req_id,\n"
        "                local_block_ids=block_ids,\n"
        "                kv_transfer_params=kv_params,\n"
        "                write_mode=True,\n"
        "                mamba_local_block_ids=self._reqs_save_mamba.get(req_id, []),  # k3-mamba-blockids\n"
        "                all_group_block_ids=self._reqs_save_allgrp.get(req_id, None),  # " + MARK + "\n"
        "            )\n",
    ),
    (
        "N11 schedule_write_blocks signature",
        "        mamba_local_block_ids: list[int] | None = None,  # k3-mamba-blockids\n"
        "    ) -> None:\n"
        "        \"\"\"Schedule a block write operation.\n",
        "        mamba_local_block_ids: list[int] | None = None,  # k3-mamba-blockids\n"
        "        all_group_block_ids: list[list[int]] | None = None,  # " + MARK + "\n"
        "    ) -> None:\n"
        "        \"\"\"Schedule a block write operation.\n",
    ),
    (
        "N12 WriteTask construction",
        "            remote_block_ids_hint=remote_block_ids,\n"
        "            mamba_local_block_ids=mamba_local_block_ids,  # k3-mamba-blockids\n"
        "            layer_name=layer_name,\n",
        "            remote_block_ids_hint=remote_block_ids,\n"
        "            mamba_local_block_ids=mamba_local_block_ids,  # k3-mamba-blockids\n"
        "            all_group_block_ids=all_group_block_ids,  # " + MARK + "\n"
        "            layer_name=layer_name,\n",
    ),
    (
        "N10 _write_blocks_for_req schedule_write_blocks call",
        "            remote_notify_port=meta.remote_notify_port,\n"
        "            remote_ip=meta.remote_host,\n"
        "            mamba_local_block_ids=meta.mamba_local_block_ids,  # k3-mamba-blockids\n"
        "        )\n",
        "            remote_notify_port=meta.remote_notify_port,\n"
        "            remote_ip=meta.remote_host,\n"
        "            mamba_local_block_ids=meta.mamba_local_block_ids,  # k3-mamba-blockids\n"
        "            all_group_block_ids=meta.all_group_block_ids,  # " + MARK + "\n"
        "        )\n",
    ),
    (
        "N1 worker __init__ compute group indices",
        "        self.layer_to_spec = build_layer_to_spec(kv_cache_config)\n",
        "        self.layer_to_spec = build_layer_to_spec(kv_cache_config)\n"
        "        # " + MARK + ": Kimi-K3 has 4 kv-cache groups (0/1/2 mamba, 3 MLA).\n"
        "        # Map each layer to ITS OWN group index and find the attention\n"
        "        # (non-Mamba) group so every layer routes to its own block-id list.\n"
        "        self._layer_group_idx: dict[str, int] = {}\n"
        "        self._attn_group_idx = 0\n"
        "        try:\n"
        "            from vllm.v1.kv_cache_interface import MambaSpec as _K3GR_Mamba\n"
        "            _k3gr_groups = getattr(kv_cache_config, \"kv_cache_groups\", []) or []\n"
        "            for _k3gr_gi, _k3gr_grp in enumerate(_k3gr_groups):\n"
        "                for _k3gr_ln in getattr(_k3gr_grp, \"layer_names\", []) or []:\n"
        "                    self._layer_group_idx[_k3gr_ln] = _k3gr_gi\n"
        "            for _k3gr_gi, _k3gr_grp in enumerate(_k3gr_groups):\n"
        "                _k3gr_lns = getattr(_k3gr_grp, \"layer_names\", []) or []\n"
        "                if _k3gr_lns and not isinstance(\n"
        "                    self.layer_to_spec.get(_k3gr_lns[0]), _K3GR_Mamba\n"
        "                ):\n"
        "                    self._attn_group_idx = _k3gr_gi\n"
        "                    break\n"
        "        except Exception:\n"
        "            self._layer_group_idx = {}\n"
        "            self._attn_group_idx = 0\n"
        "        import os as _k3gros\n"
        "        self._k3_group_routing = (\n"
        "            _k3gros.environ.get(\"K3_GROUP_ROUTING\", \"1\") == \"1\"\n"
        "        )\n"
        "        logger.info(\n"
        "            \"[" + MARK + "] enabled=%s attn_group_idx=%s n_groups=%s \"\n"
        "            \"sample_layer_group_idx=%s\",\n"
        "            self._k3_group_routing,\n"
        "            self._attn_group_idx,\n"
        "            len(getattr(kv_cache_config, \"kv_cache_groups\", []) or []),\n"
        "            dict(list(self._layer_group_idx.items())[:4]),\n"
        "        )\n",
    ),
]


# --------------------------------------------------------------------------- #
# moriio_engine.py
# --------------------------------------------------------------------------- #
ENGINE_EDITS = [
    (
        "E1 _handle_remote_blocks_message capture",
        "        block_notify_list = data.get(\"block_notify_list\", [])\n"
        "        mamba_block_notify_list = data.get(\"mamba_block_notify_list\", [])  # k3-mamba-blockids\n"
        "        decode_dp_rank = data.get(\"decode_rank\", 0)\n",
        "        block_notify_list = data.get(\"block_notify_list\", [])\n"
        "        mamba_block_notify_list = data.get(\"mamba_block_notify_list\", [])  # k3-mamba-blockids\n"
        "        all_group_block_notify = data.get(\"all_group_block_notify\", [])  # " + MARK + "\n"
        "        decode_dp_rank = data.get(\"decode_rank\", 0)\n",
    ),
    (
        "E2 RemoteAllocInfo construction",
        "            self.done_remote_allocate_req_dict[transfer_id] = RemoteAllocInfo(\n"
        "                block_ids=block_notify_list, decode_dp_rank=decode_dp_rank,\n"
        "                mamba_block_ids=list(mamba_block_notify_list or []),  # k3-mamba-blockids\n"
        "            )\n",
        "            self.done_remote_allocate_req_dict[transfer_id] = RemoteAllocInfo(\n"
        "                block_ids=block_notify_list, decode_dp_rank=decode_dp_rank,\n"
        "                mamba_block_ids=list(mamba_block_notify_list or []),  # k3-mamba-blockids\n"
        "                all_group_block_ids=(  # " + MARK + "\n"
        "                    [list(g) for g in all_group_block_notify]\n"
        "                    if all_group_block_notify else None\n"
        "                ),\n"
        "            )\n",
    ),
    (
        "E3 _prepare_transfer_plan group routing",
        "            from vllm.v1.kv_cache_interface import MambaSpec as _K3MS_BL  # k3-mamba-blockids\n"
        "            _k3_mamba = isinstance(\n"
        "                self.worker.layer_to_spec.get(task.layer_name), _K3MS_BL\n"
        "            )\n"
        "            if _k3_mamba:\n"
        "                # k3-mamba-blockids: mamba/KDA state lives in a SEPARATE KV-cache\n"
        "                # group whose slot ids differ from the attention group's block ids.\n"
        "                # Route the mamba-layer transfer by the mamba group's ids (falling\n"
        "                # back to attention ids for non-hybrid models).\n"
        "                _k3_local = task.mamba_local_block_ids or task.local_block_ids\n"
        "                _k3_remote = request_info.mamba_block_ids or request_info.block_ids\n"
        "            else:\n"
        "                _k3_local = task.local_block_ids\n"
        "                _k3_remote = request_info.block_ids\n",
        "            from vllm.v1.kv_cache_interface import MambaSpec as _K3MS_BL  # k3-mamba-blockids\n"
        "            # " + MARK + ": route EVERY layer by ITS OWN kv-cache-group index.\n"
        "            # Kimi-K3 has 4 groups (0/1/2 mamba, 3 MLA); the legacy code below\n"
        "            # hardcoded [0]/[1] and sent MLA KV to mamba block ids. When all\n"
        "            # groups' block ids are carried end-to-end (K3_GROUP_ROUTING=1) use\n"
        "            # the per-layer group index; otherwise fall back to legacy behavior.\n"
        "            _k3_gr_local = getattr(task, \"all_group_block_ids\", None)\n"
        "            _k3_gr_remote = getattr(request_info, \"all_group_block_ids\", None)\n"
        "            _k3_gr_on = getattr(self.worker, \"_k3_group_routing\", False)\n"
        "            if (\n"
        "                _k3_gr_on\n"
        "                and _k3_gr_local is not None\n"
        "                and _k3_gr_remote is not None\n"
        "            ):\n"
        "                _k3_gi = self.worker._layer_group_idx.get(\n"
        "                    task.layer_name, self.worker._attn_group_idx\n"
        "                )\n"
        "                if _k3_gi < len(_k3_gr_local) and _k3_gi < len(_k3_gr_remote):\n"
        "                    _k3_local = _k3_gr_local[_k3_gi]\n"
        "                    _k3_remote = _k3_gr_remote[_k3_gi]\n"
        "                else:\n"
        "                    _k3_local = task.local_block_ids\n"
        "                    _k3_remote = request_info.block_ids\n"
        "            elif isinstance(\n"
        "                self.worker.layer_to_spec.get(task.layer_name), _K3MS_BL\n"
        "            ):\n"
        "                # k3-mamba-blockids (legacy 2-group fallback): mamba/KDA state\n"
        "                # lives in a SEPARATE KV-cache group whose slot ids differ from\n"
        "                # the attention group's block ids.\n"
        "                _k3_local = task.mamba_local_block_ids or task.local_block_ids\n"
        "                _k3_remote = request_info.mamba_block_ids or request_info.block_ids\n"
        "            else:\n"
        "                _k3_local = task.local_block_ids\n"
        "                _k3_remote = request_info.block_ids\n",
    ),
]


FILES = [
    (COMMON, COMMON_EDITS),
    (CONN, CONN_EDITS),
    (ENGINE, ENGINE_EDITS),
]


def main():
    if len(sys.argv) < 2:
        print(f"[{MARK}] usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 1
    base = sys.argv[1]

    resolved = []  # (path, src, edits, already)
    # ---- PASS 1: verify EVERY anchor across EVERY file BEFORE any write ---- #
    missing = False
    for rel, edits in FILES:
        path = os.path.join(base, rel)
        if not os.path.isfile(path):
            print(f"[{MARK}] not found {path}", file=sys.stderr)
            return 1
        src = open(path).read()
        already = MARK in src
        resolved.append((path, src, edits, already))
        if already:
            print(f"[{MARK}] {rel}: MARK already present -> will skip.")
            continue
        for tag, old, _new in edits:
            cnt = src.count(old)
            if cnt != 1:
                print(
                    f"[{MARK}] {rel}: ANCHOR MISSING/NON-UNIQUE ({cnt}) :: {tag}",
                    file=sys.stderr,
                )
                missing = True
    if missing:
        print(f"[{MARK}] aborting: unmatched anchors, ZERO writes.", file=sys.stderr)
        return 1

    # ---- PASS 2: apply + py_compile ---- #
    import py_compile

    for path, src, edits, already in resolved:
        if already:
            continue
        for _tag, old, new in edits:
            src = src.replace(old, new, 1)
        open(path, "w").write(src)
        try:
            py_compile.compile(path, doraise=True)
        except Exception as e:
            print(f"[{MARK}] compile FAIL {path}: {e}", file=sys.stderr)
            return 1
        print(f"[{MARK}] applied + compiled: {path}")

    print(f"[{MARK}] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
