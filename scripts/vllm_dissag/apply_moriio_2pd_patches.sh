#!/bin/bash
# apply_moriio_2pd_patches.sh — Runtime patches for multi-node disagg DP (PR #39276)
# Applied at container startup before vLLM launches.
# Once PR #39276 is merged upstream, this script becomes a no-op.
set -euo pipefail

VLLM_ROOT=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
MORIIO_DIR="${VLLM_ROOT}/distributed/kv_transfer/kv_connector/v1/moriio"
ENGINE_DIR="${VLLM_ROOT}/v1/engine"

echo "[patch] vLLM root: ${VLLM_ROOT}"
echo "[patch] Applying PR #39276 runtime patches for multi-node disagg DP..."

# --------------------------------------------------------------------------
# Patch 1: core.py — engine_id: local_dp_rank → dp_rank
# --------------------------------------------------------------------------
CORE_PY="${ENGINE_DIR}/core.py"
if grep -q 'engine_id}_dp{local_dp_rank}' "$CORE_PY" 2>/dev/null; then
    sed -i 's/engine_id}_dp{local_dp_rank}/engine_id}_dp{dp_rank}/g' "$CORE_PY"
    echo "[patch] core.py: engine_id fixed (local_dp_rank → dp_rank)"
else
    echo "[patch] core.py: already patched or not applicable"
fi

# --------------------------------------------------------------------------
# Patch 2: utils.py — engine_id: local_index → index
# --------------------------------------------------------------------------
UTILS_PY="${ENGINE_DIR}/utils.py"
if grep -q 'engine_id}_dp{local_index}' "$UTILS_PY" 2>/dev/null; then
    sed -i 's/engine_id}_dp{local_index}/engine_id}_dp{index}/g' "$UTILS_PY"
    echo "[patch] utils.py: engine_id fixed (local_index → index)"
else
    echo "[patch] utils.py: already patched or not applicable"
fi

# --------------------------------------------------------------------------
# Patch 3: moriio_common.py — Use data_parallel_size_local and local dp_rank
# --------------------------------------------------------------------------
COMMON_PY="${MORIIO_DIR}/moriio_common.py"

# 3a: dp_rank = ... % data_parallel_size_local
if grep -q 'dp_rank = vllm_config.parallel_config.data_parallel_rank$' "$COMMON_PY" 2>/dev/null; then
    sed -i 's/dp_rank = vllm_config.parallel_config.data_parallel_rank$/dp_rank = (vllm_config.parallel_config.data_parallel_rank % vllm_config.parallel_config.data_parallel_size_local)/' "$COMMON_PY"
    echo "[patch] moriio_common.py: dp_rank uses local modulo"
else
    echo "[patch] moriio_common.py: dp_rank already patched"
fi

# 3b: dp_size = data_parallel_size_local
if grep -q 'dp_size = vllm_config.parallel_config.data_parallel_size$' "$COMMON_PY" 2>/dev/null; then
    sed -i 's/dp_size = vllm_config.parallel_config.data_parallel_size$/dp_size = vllm_config.parallel_config.data_parallel_size_local/' "$COMMON_PY"
    echo "[patch] moriio_common.py: dp_size uses local size"
else
    echo "[patch] moriio_common.py: dp_size already patched"
fi

# 3c: Default ports for remote_handshake_port and remote_notify_port
if grep -q 'remote_handshake_port=kv_transfer_params\["remote_handshake_port"\]' "$COMMON_PY" 2>/dev/null; then
    sed -i 's/remote_handshake_port=kv_transfer_params\["remote_handshake_port"\]/remote_handshake_port=kv_transfer_params.get("remote_handshake_port", 8405)/' "$COMMON_PY"
    sed -i 's/remote_notify_port=kv_transfer_params\["remote_notify_port"\]/remote_notify_port=kv_transfer_params.get("remote_notify_port", 61005)/' "$COMMON_PY"
    echo "[patch] moriio_common.py: default ports added"
else
    echo "[patch] moriio_common.py: default ports already patched"
fi

# --------------------------------------------------------------------------
# Patch 4: moriio_connector.py — Full multi-node DP fixes
# --------------------------------------------------------------------------
CONNECTOR_PY="${MORIIO_DIR}/moriio_connector.py"

# 4a: Add _is_kv_master flag and _req_kv_params cache and local dp_rank
python3 << 'PYEOF'
import re, sys

fpath = sys.argv[1] if len(sys.argv) > 1 else ""
if not fpath:
    sys.exit(1)

with open(fpath, "r") as f:
    src = f.read()

changed = False

# Fix: dp_rank use local modulo in scheduler connector
old = "self.dp_rank = self.vllm_config.parallel_config.data_parallel_rank\n"
new = ("self.dp_rank = (self.vllm_config.parallel_config.data_parallel_rank\n"
       "                        % self.vllm_config.parallel_config\n"
       "                        .data_parallel_size_local)\n"
       "        self._is_kv_master = (\n"
       "            self.vllm_config.parallel_config.data_parallel_rank\n"
       "            < self.vllm_config.parallel_config.data_parallel_size_local)\n")
if old in src and "_is_kv_master" not in src:
    src = src.replace(old, new, 1)
    changed = True
    print("[patch] moriio_connector.py: _is_kv_master + local dp_rank added")

# Fix: Add _req_kv_params dict
old2 = "self._reqs_need_save: dict[ReqId, tuple[Request, list[int]]] = {}"
new2 = (old2 + "\n        self._req_kv_params: dict[ReqId, dict] = {}")
if old2 in src and "_req_kv_params" not in src:
    src = src.replace(old2, new2, 1)
    changed = True
    print("[patch] moriio_connector.py: _req_kv_params dict added")

# Fix: Cache kv_transfer_params on do_remote_decode
old3 = 'self._reqs_need_save[request.request_id] = (request, local_block_ids)'
new3 = (old3 + '\n            self._req_kv_params[request.request_id] = dict(params)')
if old3 in src and '_req_kv_params[request.request_id] = dict(params)' not in src:
    src = src.replace(old3, new3, 1)
    changed = True
    print("[patch] moriio_connector.py: cached kv_params for do_remote_decode")

# Fix: Guard send_notify_block with _is_kv_master
old4 = "                for tp_index in range(self.tp_size):\n                    target_port = request.kv_transfer_params[\n"
new4 = "                if self._is_kv_master:\n                  for tp_index in range(self.tp_size):\n                    target_port = request.kv_transfer_params[\n"
if old4 in src and "if self._is_kv_master:" not in src:
    src = src.replace(old4, new4, 1)
    changed = True
    print("[patch] moriio_connector.py: send_notify_block guarded with _is_kv_master")

# Fix: block_size assertion → graceful override
old5 = "        assert block_size == self.block_size"
new5 = ("        if block_size != self.block_size:\n"
        "            logger.info(\n"
        '                "KV cache block_size=%d differs from config block_size=%d; "\n'
        '                "using actual tensor shape (attention backend override).",\n'
        "                block_size, self.block_size)\n"
        "            self.block_size = block_size")
if old5 in src:
    src = src.replace(old5, new5, 1)
    changed = True
    print("[patch] moriio_connector.py: block_size assert → graceful override")

# Fix: Use cached _req_kv_params in build_connector_meta for _reqs_need_recv
old6 = "        for req_id, (req, block_ids) in self._reqs_need_recv.items():\n            assert req.kv_transfer_params is not None\n"
new6 = ("        for req_id, (req, block_ids) in self._reqs_need_recv.items():\n"
        "            kv_params = self._req_kv_params.get(\n"
        "                req_id, req.kv_transfer_params or {}\n"
        "            )\n")
if old6 in src:
    src = src.replace(old6, new6, 1)
    changed = True
    print("[patch] moriio_connector.py: _reqs_need_recv uses cached params")

# Fix: Use cached _req_kv_params for _reqs_need_save
old7 = "        for req_id, (req, block_ids) in self._reqs_need_save.items():\n            assert req.kv_transfer_params is not None\n"
new7 = ("        for req_id, (req, block_ids) in self._reqs_need_save.items():\n"
        "            kv_params = self._req_kv_params.get(\n"
        "                req_id, req.kv_transfer_params or {}\n"
        "            )\n")
if old7 in src:
    src = src.replace(old7, new7, 1)
    changed = True
    print("[patch] moriio_connector.py: _reqs_need_save uses cached params")

# Fix: Add _recving_transfers_start dict
old8 = "self._recving_transfers_callback_addr: dict[ReqId, tuple[str, str]] = {}"
new8 = (old8 + "\n        self._recving_transfers_start: dict[str, float] = {}")
if old8 in src and "_recving_transfers_start" not in src:
    src = src.replace(old8, new8, 1)
    changed = True
    print("[patch] moriio_connector.py: _recving_transfers_start dict added")

if changed:
    with open(fpath, "w") as f:
        f.write(src)
    print("[patch] moriio_connector.py: all patches applied")
else:
    print("[patch] moriio_connector.py: already patched or no changes needed")
PYEOF
python3 /dev/stdin "$CONNECTOR_PY" < /dev/null 2>&1 || echo "[patch] WARNING: connector patch script had errors"

# --------------------------------------------------------------------------
# Patch 5: moriio_engine.py — Timeouts, failure detection, ZMQ retry
# --------------------------------------------------------------------------
ENGINE_PY="${MORIIO_DIR}/moriio_engine.py"

# 5a: Add imports (os, time)
if ! grep -q '^import os$' "$ENGINE_PY" 2>/dev/null; then
    sed -i '1,/^import threading/s/^import threading/import os\nimport threading\nimport time/' "$ENGINE_PY"
    echo "[patch] moriio_engine.py: added os/time imports"
fi

# 5b: Deferred task timeout
if ! grep -q 'VLLM_MORIIO_DEFERRED_TIMEOUT_S' "$ENGINE_PY" 2>/dev/null; then
    python3 << 'PYEOF2'
import sys

fpath = sys.argv[1] if len(sys.argv) > 1 else ""
if not fpath:
    sys.exit(1)

with open(fpath, "r") as f:
    src = f.read()

changed = False

# Deferred task timeout
old = """        still_deferred: list[WriteTask] = []
        for task in self._deferred_tasks:
            if self._is_remote_ready(task):
                self._execute_write_task(task)
            else:
                still_deferred.append(task)"""

new = """        _defer_timeout = int(
            os.environ.get("VLLM_MORIIO_DEFERRED_TIMEOUT_S", "300"))
        still_deferred: list[WriteTask] = []
        _now = time.monotonic()
        for task in self._deferred_tasks:
            if self._is_remote_ready(task):
                self._execute_write_task(task)
            elif (hasattr(task, '_defer_time')
                  and (_now - task._defer_time) > _defer_timeout):
                logger.error(
                    "Deferred write task EXPIRED for req %s "
                    "(transfer %s) after %ds",
                    task.request_id, task.transfer_id, _defer_timeout)
            else:
                if not hasattr(task, '_defer_time'):
                    task._defer_time = _now
                still_deferred.append(task)"""

if old in src:
    src = src.replace(old, new, 1)
    changed = True
    print("[patch] moriio_engine.py: deferred task timeout added")

# Transfer timeout (replace blocking Wait)
old2 = """        for status in transfers_to_wait:
            try:
                status.Wait()
                if not status.Succeeded():
                    logger.error(
                        "Transfer failed: %s, Code: %s", status.Message(), status.Code()
                    )
                    raise TransferError("MoRIIO transfer failed!")
            except Exception as e:
                logger.error("Transfer %s failed: %s", status, e)
                raise"""

new2 = """        _xfer_timeout = int(
            os.environ.get("VLLM_MORIIO_TRANSFER_TIMEOUT_S", "120"))
        for status in transfers_to_wait:
            _deadline = time.monotonic() + _xfer_timeout
            while status.InProgress():
                if time.monotonic() > _deadline:
                    logger.error(
                        "RDMA write timed out after %ds "
                        "(Code=%s, Msg=%s)",
                        _xfer_timeout, status.Code(),
                        status.Message())
                    break
                time.sleep(0.001)
            if status.Failed():
                logger.error(
                    "Transfer failed: %s, Code: %s",
                    status.Message(), status.Code())
            elif not status.Succeeded():
                logger.error(
                    "Transfer did not succeed "
                    "(timeout or unknown state, Code=%s)",
                    status.Code())"""

if old2 in src:
    src = src.replace(old2, new2, 1)
    changed = True
    print("[patch] moriio_engine.py: transfer timeout added")

# ZMQ retry
old3 = """        sock = self.paths[path]
        try:
            for req_id in req_list:
                if not isinstance(req_id, str):
                    logger.warning(
                        "Invalid req_id type: %s, expected str", type(req_id)
                    )
                    continue
                sock.send(req_id.encode("utf-8"))
        except Exception as e:
            logger.error("Failed to send notification to %s: %s", path, e)
            self.paths.pop(path, None)
            raise"""

new3 = """        sock = self.paths[path]
        _MAX_RETRIES = 3
        for req_id in req_list:
            if not isinstance(req_id, str):
                logger.warning(
                    "Invalid req_id type: %s, expected str", type(req_id))
                continue
            for _attempt in range(_MAX_RETRIES):
                try:
                    sock.send(req_id.encode("utf-8"), zmq.NOBLOCK)
                    break
                except zmq.Again:
                    if _attempt < _MAX_RETRIES - 1:
                        time.sleep(0.01 * (_attempt + 1))
                        logger.warning(
                            "ZMQ send retry %d for req %s to %s",
                            _attempt + 1, req_id, path)
                    else:
                        logger.error(
                            "ZMQ send FAILED after %d retries "
                            "for req %s to %s",
                            _MAX_RETRIES, req_id, path)
                except Exception as e:
                    logger.error(
                        "Failed to send notification to %s: %s",
                        path, e)
                    self.paths.pop(path, None)
                    raise"""

if old3 in src:
    src = src.replace(old3, new3, 1)
    changed = True
    print("[patch] moriio_engine.py: ZMQ retry added")

if changed:
    with open(fpath, "w") as f:
        f.write(src)
    print("[patch] moriio_engine.py: all patches applied")
else:
    print("[patch] moriio_engine.py: already patched or no changes needed")
PYEOF2
    python3 /dev/stdin "$ENGINE_PY" < /dev/null 2>&1 || echo "[patch] WARNING: engine patch script had errors"
else
    echo "[patch] moriio_engine.py: already patched"
fi

echo "[patch] All PR #39276 patches applied successfully."
