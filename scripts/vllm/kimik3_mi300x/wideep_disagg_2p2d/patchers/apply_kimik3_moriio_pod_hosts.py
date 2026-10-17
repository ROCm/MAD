#!/usr/bin/env python3
"""Advertise the peer pool's per-DP-pod node IPs so multi-NODE decode/prefill DP works.

ROOT CAUSE (multi-node disagg KV transfer, traced 2026-08-04):
  In 2P/2D wide-EP the decode (and prefill) pool spans >1 physical node: a master
  (rank 0, has the api-server) + a `--headless` worker (rank 1) that joins the
  master's DP group via --data-parallel-address. The headless worker is INVISIBLE
  to router service-discovery (only the master's api-server registers) and its IP
  is never conveyed to the peer pool.

  prefill's all-to-all handshake (_background_moriio_handshake) resolves each remote
  decode rank's host via `pod_hosts[pod_index]`, where
    pod_hosts = meta.multi_pod_hosts if set else [meta.remote_host]   (line ~1624)
  and multi_pod_hosts is populated ONLY from kv_transfer_params["remote_hosts"],
  which the router never sets in WRITE mode. So pod_hosts collapses to the single
  master host, and prefill dials EVERY decode rank at the master IP. The rank(s)
  living on the worker node are never reached -> their KV is never written -> those
  decode ranks generate WITHOUT the prompt context. Symptom: clean alternation
  (DP2: 50% wrong; DP8: ~88% wrong) while standalone forward is perfectly coherent.

FIX (static, launcher-advertised):
  The launcher already knows every node IP. It passes the peer pool's per-pod host
  list (ordered by pod index = global_dp_rank // dp_size_local) into
  kv_connector_extra_config["moriio_pod_hosts"] (comma-separated). This patcher makes
  the handshake fall back to THAT list when meta.multi_pod_hosts is empty, instead of
  the single master host. Topology-independent: fixes TP8/DP2 AND TP2/DP8.

  Precedence unchanged: an explicit meta.multi_pod_hosts (router-provided) still wins;
  the env list is only the fallback that replaces the single-host default.

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_moriio_pod_hosts.py <vllm_install_dir>
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
        print(f"[k3-podhosts] {REL} not found -- skip.")
        return 0
    src = open(path).read()
    orig = src

    if "k3-podhosts" in src:
        print("[k3-podhosts] already applied.")
        return 0

    anchor = (
        "            pod_hosts = list(meta.multi_pod_hosts) if meta.multi_pod_hosts else [host]\n"
        "            remote_dp_size_local = int(meta.remote_dp_size_local) or remote_dp_size\n"
    )
    repl = (
        "            # k3-podhosts: fall back to the launcher-advertised peer pod-host\n"
        "            # list (kv_connector_extra_config['moriio_pod_hosts'], ordered by\n"
        "            # pod index) instead of the single master host, so prefill can reach\n"
        "            # decode ranks that live on a HEADLESS worker node (invisible to the\n"
        "            # router). meta.multi_pod_hosts is [remote_host] (a single-host DEFAULT)\n"
        "            # when the router omits remote_hosts, so we must prefer the launcher list\n"
        "            # whenever it advertises MORE pods than meta -- not only when meta is empty.\n"
        "            _k3_meta_hosts = list(meta.multi_pod_hosts) if meta.multi_pod_hosts else []\n"
        "            _k3_ph = (\n"
        "                self.kv_transfer_config.kv_connector_extra_config.get(\n"
        "                    \"moriio_pod_hosts\", \"\"\n"
        "                )\n"
        "                if hasattr(self, \"kv_transfer_config\")\n"
        "                else \"\"\n"
        "            )\n"
        "            _k3_ph = [h.strip() for h in str(_k3_ph).split(\",\") if h.strip()]\n"
        "            if len(_k3_ph) > len(_k3_meta_hosts):\n"
        "                pod_hosts = _k3_ph\n"
        "                logger.info(\n"
        "                    \"[k3-podhosts] using launcher pod_hosts=%s (meta had %s)\",\n"
        "                    pod_hosts, _k3_meta_hosts,\n"
        "                )\n"
        "            elif _k3_meta_hosts:\n"
        "                pod_hosts = _k3_meta_hosts\n"
        "            else:\n"
        "                pod_hosts = [host]\n"
        "            # k3-podhosts: remote_dp_size_local (DP ranks PER POD) drives\n"
        "            # pod_index = global_dp_rank // dp_local. The router never sends it, so\n"
        "            # meta.remote_dp_size_local defaults to remote_dp_size (the GLOBAL size)\n"
        "            # -> pod_index collapses to 0 -> every rank resolves to pod 0 (master),\n"
        "            # so rank1's KV is written to the master not the worker. Derive the true\n"
        "            # per-pod local size from len(pod_hosts): dp_local = remote_dp_size //\n"
        "            # num_pods. With 2 pods and remote_dp_size 2 -> dp_local 1 -> pod_index\n"
        "            # (1,1)=1 -> pod_hosts[1]=worker. Honor an explicit meta value if the\n"
        "            # router ever sets one (< remote_dp_size).\n"
        "            _k3_npods = max(1, len(pod_hosts))\n"
        "            if int(meta.remote_dp_size_local) and int(meta.remote_dp_size_local) < remote_dp_size:\n"
        "                remote_dp_size_local = int(meta.remote_dp_size_local)\n"
        "            elif remote_dp_size % _k3_npods == 0:\n"
        "                remote_dp_size_local = remote_dp_size // _k3_npods\n"
        "                logger.info(\n"
        "                    \"[k3-podhosts] derived remote_dp_size_local=%d (dp_size=%d, pods=%d)\",\n"
        "                    remote_dp_size_local, remote_dp_size, _k3_npods,\n"
        "                )\n"
    )
    if anchor not in src:
        print("[k3-podhosts] WARN anchor not found -- not applied.", file=sys.stderr)
        return 0
    src = src.replace(anchor, repl, 1)

    # --- SYMMETRIC FIX on the decode->prefill NOTIFY path (send_notify_block) ---
    # Same two bugs: _dp_local and _remote_hosts come from kv_transfer_params, which
    # the router never sets -> _dp_local=0 (no per-pod host resolution) and
    # _notify_host stays the single prefill master -> notify for prefill rank1 lands
    # on the master -> prefill rank1's write_ready_flags never set -> prefill times
    # out ("Timed out waiting for write_ready_flags") -> EngineDead. Feed the
    # launcher-advertised prefill pod-hosts (extra_config['moriio_pod_hosts']) and a
    # derived _dp_local here too.
    notify_anchor = (
        "                    _notify_host = remote_host\n"
        "                    _kvp = request.kv_transfer_params or {}\n"
        "                    _remote_hosts = _kvp.get(\"remote_hosts\") or []\n"
        "                    if _dp_local > 0 and _remote_hosts:\n"
        "                        _pod_idx = pod_index(remote_dp_rank, _dp_local)\n"
        "                        if 0 <= _pod_idx < len(_remote_hosts):\n"
        "                            _notify_host = _remote_hosts[_pod_idx]\n"
    )
    notify_repl = (
        "                    _notify_host = remote_host\n"
        "                    _kvp = request.kv_transfer_params or {}\n"
        "                    _remote_hosts = _kvp.get(\"remote_hosts\") or []\n"
        "                    # k3-podhosts: fall back to launcher pod-hosts + derived dp_local\n"
        "                    if not _remote_hosts and hasattr(self, \"kv_transfer_config\"):\n"
        "                        _k3n = self.kv_transfer_config.kv_connector_extra_config.get(\n"
        "                            \"moriio_pod_hosts\", \"\"\n"
        "                        )\n"
        "                        _remote_hosts = [h.strip() for h in str(_k3n).split(\",\") if h.strip()]\n"
        "                    _k3_dpl = _dp_local\n"
        "                    if (not _k3_dpl) and _remote_hosts and _dp_size % len(_remote_hosts) == 0:\n"
        "                        _k3_dpl = _dp_size // len(_remote_hosts)\n"
        "                    if _k3_dpl > 0 and _remote_hosts:\n"
        "                        _pod_idx = pod_index(remote_dp_rank, _k3_dpl)\n"
        "                        if 0 <= _pod_idx < len(_remote_hosts):\n"
        "                            _notify_host = _remote_hosts[_pod_idx]\n"
        "                        _remote_dp_rank_for_port = fold_local_rank(remote_dp_rank, _k3_dpl)\n"
        "                        logger.info(\n"
        "                            \"[k3-podhosts] notify prefill rank=%d -> host=%s (dp_local=%d)\",\n"
        "                            remote_dp_rank, _notify_host, _k3_dpl,\n"
        "                        )\n"
    )
    if notify_anchor in src:
        src = src.replace(notify_anchor, notify_repl, 1)
    else:
        print("[k3-podhosts] WARN notify anchor not found -- notify path unpatched.",
              file=sys.stderr)

    # --- 3rd site: _write_blocks_for_req stashes multi_pod_hosts + remote_dp_size_local
    # on the WORKER; MoRIIOEngine._execute_write_task/_finalize_if_complete read these
    # to target the "write_done" completion notify back to the correct PREFILL pod.
    # Same wrong meta defaults ([single host] / global dp_size) -> completion notify
    # for prefill rank1 lands on the master -> prefill rank1 write_ready never set ->
    # "Timed out waiting for write_ready_flags" -> EngineDead. Prefer the launcher
    # list here too (worker HAS self.kv_transfer_config.kv_connector_extra_config).
    stash_anchor = (
        "        if meta.multi_pod_hosts:\n"
        "            self.multi_pod_hosts = list(meta.multi_pod_hosts)\n"
        "        else:\n"
        "            self.multi_pod_hosts = [meta.remote_host]\n"
        "        if meta.remote_dp_size_local:\n"
        "            self.remote_dp_size_local = int(meta.remote_dp_size_local)\n"
        "        else:\n"
        "            self.remote_dp_size_local = int(meta.remote_dp_size)\n"
    )
    stash_repl = (
        "        # k3-podhosts: prefer launcher-advertised peer pod-hosts + derived\n"
        "        # dp_local over the single-host / global-dp_size meta defaults, so the\n"
        "        # write_done completion notify targets the right prefill pod/rank.\n"
        "        _k3_meta_ph = list(meta.multi_pod_hosts) if meta.multi_pod_hosts else []\n"
        "        _k3_lp = (\n"
        "            self.kv_transfer_config.kv_connector_extra_config.get(\n"
        "                \"moriio_pod_hosts\", \"\"\n"
        "            )\n"
        "            if hasattr(self, \"kv_transfer_config\")\n"
        "            else \"\"\n"
        "        )\n"
        "        _k3_lp = [h.strip() for h in str(_k3_lp).split(\",\") if h.strip()]\n"
        "        if len(_k3_lp) > len(_k3_meta_ph):\n"
        "            self.multi_pod_hosts = _k3_lp\n"
        "        elif _k3_meta_ph:\n"
        "            self.multi_pod_hosts = _k3_meta_ph\n"
        "        else:\n"
        "            self.multi_pod_hosts = [meta.remote_host]\n"
        "        _k3_gdp = int(meta.remote_dp_size)\n"
        "        _k3_npods = max(1, len(self.multi_pod_hosts))\n"
        "        if 0 < int(meta.remote_dp_size_local) < _k3_gdp:\n"
        "            self.remote_dp_size_local = int(meta.remote_dp_size_local)\n"
        "        elif _k3_gdp % _k3_npods == 0:\n"
        "            self.remote_dp_size_local = _k3_gdp // _k3_npods\n"
        "        else:\n"
        "            self.remote_dp_size_local = _k3_gdp\n"
        "        logger.info(\n"
        "            \"[k3-podhosts] write-stash multi_pod_hosts=%s remote_dp_size_local=%d\",\n"
        "            self.multi_pod_hosts, self.remote_dp_size_local,\n"
        "        )\n"
    )
    if stash_anchor in src:
        src = src.replace(stash_anchor, stash_repl, 1)
    else:
        print("[k3-podhosts] WARN write-stash anchor not found -- completion path unpatched.",
              file=sys.stderr)

    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        open(path, "w").write(orig)
        print(f"[k3-podhosts] ERROR compile: {e}", file=sys.stderr)
        return 1
    print("[k3-podhosts] handshake falls back to launcher-advertised peer pod_hosts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
