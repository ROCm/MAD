#!/usr/bin/env python3
"""
Patch vLLM's MoRI all2all kernel selection to use the AsyncLL (WRITE+poll) kernel
for multi-node expert parallelism on Broadcom Thor2 (BCM57608) NICs.

WHY: vLLM's MoriAll2AllManager._make_all2all_kwargs picks InterNodeV1 (mori_high_throughput)
or InterNodeV1LL (mori_low_latency) for multi-node. BOTH compile from internode_v1.cpp, which
posts RDMA AMO_ADD atomics into GPU VRAM. The Thor2 NIC has no PCIe atomic-completer capability
(AtomicOpsCap: 32bit- 64bit-), so those atomics fault (res_rx_pci_err) -> QP ERROR -> dispatch
hang. The AsyncLL kernel (ep_async_ll) uses RDMA WRITE + poll signalling instead of atomics, and
is validated at EP2 and EP16 (500/500 rounds, 0 errors) on this MI300X + Thor2 stack.

WHAT: when env MORI_EP_FORCE_ASYNC_LL=1 (default ON here), the multi-node branch selects
AsyncLL regardless of --all2all-backend. Idempotent + anchor-based + reversible.

Requirement: AsyncLL asserts numExpertPerToken < warpSize(64) (GLM-5.2 top-k=8, OK) and prefers
SDMA (set MORI_ENABLE_SDMA=1 in the pod env; without it AsyncLL still works but uses CUs).
"""
import os
import re
import sys
import py_compile

VLLM = os.environ.get(
    "VLLM_PKG",
    "/usr/local/lib/python3.12/dist-packages/vllm",
)
TARGET = os.path.join(VLLM, "distributed/device_communicators/all2all.py")

ANCHOR = "            if self._all2all_backend == \"mori_low_latency\":\n"
OLD = (
    "            if self._all2all_backend == \"mori_low_latency\":\n"
    "                kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1LL\n"
    "            else:\n"
    "                kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1\n"
)
NEW = (
    "            # [thor2-async_ll patch] Broadcom Thor2 (BCM57608) has no PCIe atomic-completer;\n"
    "            # InterNodeV1/V1LL post RDMA atomics into GPU VRAM and hang. AsyncLL uses WRITE+poll.\n"
    "            import os as _os\n"
    "            if _os.environ.get(\"MORI_EP_FORCE_ASYNC_LL\", \"1\") == \"1\":\n"
    "                kernel_type = mori.ops.EpDispatchCombineKernelType.AsyncLL\n"
    "            elif self._all2all_backend == \"mori_low_latency\":\n"
    "                kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1LL\n"
    "            else:\n"
    "                kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1\n"
)
MARKER = "[thor2-async_ll patch]"


def main():
    if not os.path.exists(TARGET):
        print(f"ERROR: target not found: {TARGET}", file=sys.stderr)
        return 2
    src = open(TARGET, encoding="utf-8").read()

    if MARKER in src:
        print("already patched (idempotent no-op)")
        return 0
    if OLD not in src:
        print("ERROR: anchor block not found — vLLM version differs; inspect "
              "_make_all2all_kwargs in all2all.py", file=sys.stderr)
        return 3

    # backup once
    bak = TARGET + ".orig"
    if not os.path.exists(bak):
        open(bak, "w", encoding="utf-8").write(src)

    src2 = src.replace(OLD, NEW, 1)
    open(TARGET, "w", encoding="utf-8").write(src2)

    # byte-compile to catch syntax errors early
    py_compile.compile(TARGET, doraise=True)
    print(f"patched {TARGET}")
    print(f"backup at {bak}")
    print("AsyncLL now selected for multi-node MoRI-EP when MORI_EP_FORCE_ASYNC_LL=1 (default).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
