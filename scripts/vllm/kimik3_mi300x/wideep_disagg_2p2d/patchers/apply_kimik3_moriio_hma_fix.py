#!/usr/bin/env python3
"""Declare MoRIIOConnector as SupportsHMA so K3's Hybrid Memory Allocator stays ON.

PROBLEM
  K3 is a hybrid model (MLA attention + KDA/mamba state) => vLLM needs the Hybrid
  Memory Allocator (HMA / hybrid kv cache manager) to lay out the two cache kinds.
  vLLM (config/vllm.py:1643) FORCE-DISABLES HMA when the selected KV connector is
  not declared `SupportsHMA`. MoRIIOConnector subclasses only KVConnectorBase_V1,
  NOT SupportsHMA -> HMA turned off -> "hybrid SSM models require HMA and will
  fail at startup" (exactly K3). NIXL works because NixlBaseConnector subclasses
  (KVConnectorBase_V1, SupportsHMA).

FIX (mirror NIXL, minimal)
  H1  add SupportsHMA to the MoRIIOConnector import + class bases.
  H2  implement the single abstract method `request_finished_all_groups(request,
      block_ids: tuple[list[int], ...])` by flattening the per-group block-id
      tuple into one list and delegating to the existing single-group
      `request_finished` (which delegates to connector_scheduler.request_finished).
      This matches NIXL's own impl (it just forwards to the scheduler).

  Flattening rationale: MoRIIO's scheduler.request_finished treats block_ids as a
  flat list of this request's blocks (for producer notify / consumer unmap). For a
  hybrid model HMA passes one list per kv-cache group (attention group + mamba
  group); concatenating them preserves the full block set the connector must
  track. Order within the flattened list does not matter for MoRIIO's use (it maps
  by request_id + notifies block ids), unlike the transfer offset path which is
  computed per-layer elsewhere.

Idempotent, anchor-based, py_compile-checked. Usage:
    apply_kimik3_moriio_hma_fix.py <vllm_install_dir>
"""
import os
import sys

CONN = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], CONN)
    if not os.path.isfile(path):
        print(f"[k3-hma] {CONN} not found -- ABORT.")
        return 1
    src = open(path).read()
    orig = src
    applied = []

    # --- H1a: import SupportsHMA (added to the existing base import block) ---
    if "SupportsHMA" in src:
        applied.append("import(already)")
    else:
        # The connector imports KVConnectorBase_V1 from the v1 base module.
        imp_anchor = "    KVConnectorBase_V1,\n"
        if imp_anchor in src:
            src = src.replace(imp_anchor, imp_anchor + "    SupportsHMA,\n", 1)
            applied.append("import")
        else:
            print("[k3-hma] ERROR: KVConnectorBase_V1 import anchor not found. ABORT.")
            return 1

    # --- H1b: add SupportsHMA to the class bases ---
    if "class MoRIIOConnector(KVConnectorBase_V1, SupportsHMA)" in src:
        applied.append("class(already)")
    else:
        cls_anchor = "class MoRIIOConnector(KVConnectorBase_V1):"
        if cls_anchor in src:
            src = src.replace(
                cls_anchor,
                "class MoRIIOConnector(KVConnectorBase_V1, SupportsHMA):",
                1,
            )
            applied.append("class")
        else:
            print("[k3-hma] ERROR: MoRIIOConnector class def anchor not found. ABORT.")
            return 1

    # --- H2: add request_finished_all_groups method (delegates to request_finished) ---
    if "def request_finished_all_groups" in src:
        applied.append("method(already)")
    else:
        # Anchor: the connector-level request_finished (line ~266), which itself
        # delegates to the scheduler. Insert our method right before it.
        m_anchor = (
            "    def request_finished(\n"
            "        self,\n"
            "        request: \"Request\",\n"
            "        block_ids: list[int],\n"
            "    ) -> tuple[bool, dict[str, Any] | None]:\n"
            "        assert self.connector_scheduler is not None\n"
            "        return self.connector_scheduler.request_finished(request, block_ids)\n"
        )
        m_new = (
            "    def request_finished_all_groups(\n"
            "        self,\n"
            "        request: \"Request\",\n"
            "        block_ids: tuple[list[int], ...],\n"
            "    ) -> tuple[bool, dict[str, Any] | None]:\n"
            "        # k3-kda/HMA: hybrid models pass one block-id list per kv-cache\n"
            "        # group (attention + mamba). Flatten to the single-group list\n"
            "        # the MoRIIO scheduler expects and delegate.\n"
            "        flat: list[int] = []\n"
            "        for grp in block_ids:\n"
            "            flat.extend(grp)\n"
            "        assert self.connector_scheduler is not None\n"
            "        return self.connector_scheduler.request_finished(request, flat)\n"
            "\n"
            + m_anchor
        )
        if m_anchor in src:
            src = src.replace(m_anchor, m_new, 1)
            applied.append("method")
        else:
            print("[k3-hma] ERROR: connector-level request_finished anchor not "
                  "found -- cannot add request_finished_all_groups. ABORT.")
            return 1

    if src != orig:
        open(path, "w").write(src)
        try:
            import py_compile
            py_compile.compile(path, doraise=True)
        except Exception as e:
            print(f"[k3-hma] ERROR: compile failed: {e}", file=sys.stderr)
            return 1
        print(f"[k3-hma] patched {CONN} -- {', '.join(applied)}")
    else:
        print(f"[k3-hma] no changes ({', '.join(applied)})")

    # runtime sanity: is it now recognized as SupportsHMA?
    return 0


if __name__ == "__main__":
    sys.exit(main())
