#!/usr/bin/env python3
'Idempotently patch installed vLLM MoRIIO sources to log request and write UID mappings.'

import argparse
import importlib.util
import os
import sys

BACKUP_SUFFIX = ".orig_moriio_reqid_map"


_HELPER_BLOCK = '''
# --- BEGIN moriio_profiling: request_id <-> MoRI write_uid map -------------------
#     Added by moriio_profiling/patch_moriio_reqid_map.py. Gated on env
#     MORIIO_REQID_MAP=1; when off this is a single module-level bool check.
import os as _moriio_os

_MORIIO_REQID_MAP_ENABLED = _moriio_os.environ.get("MORIIO_REQID_MAP", "0") == "1"


def _moriio_log_reqid_map(request_id, transfer_id, layer_name, write_uid, direction):
    if not _MORIIO_REQID_MAP_ENABLED:
        return
    logger.info(
        "moriio_reqid_map dir=%s request_id=%s transfer_id=%s layer=%s write_uid=%s",
        direction,
        request_id,
        transfer_id,
        layer_name,
        write_uid,
    )
# --- END moriio_profiling --------------------------------------------------------
'''


# Anchors must be unique or patching aborts.

_ENGINE_EDITS = [
    (
        "engine: insert _moriio_log_reqid_map helper after logger",
        "logger = init_logger(__name__)\n",
        "logger = init_logger(__name__)\n" + _HELPER_BLOCK,
    ),
    (
        "engine: write_remote_data signature (+ reqid kwargs)",
        "    def write_remote_data(\n"
        "        self, transfer_size_byte, local_offset=0, remote_offset=0, session=None\n"
        "    ):\n",
        "    def write_remote_data(\n"
        "        self, transfer_size_byte, local_offset=0, remote_offset=0, session=None,\n"
        "        request_id=None, transfer_id=None, layer_name=None,\n"
        "    ):\n",
    ),
    (
        "engine: write_remote_data body (log after write_uid alloc)",
        "        write_uid = self.moriio_engine.allocate_transfer_uid()\n"
        "\n"
        "        transfer_status = session.batch_write(\n",
        "        write_uid = self.moriio_engine.allocate_transfer_uid()\n"
        '        _moriio_log_reqid_map(request_id, transfer_id, layer_name, write_uid, "write")\n'
        "\n"
        "        transfer_status = session.batch_write(\n",
    ),
    (
        "engine: write_remote_data_single signature (+ reqid kwargs)",
        "    def write_remote_data_single(\n"
        "        self, transfer_size_byte, local_offset=0, remote_offset=0, sess_idx=0\n"
        "    ):\n",
        "    def write_remote_data_single(\n"
        "        self, transfer_size_byte, local_offset=0, remote_offset=0, sess_idx=0,\n"
        "        request_id=None, transfer_id=None, layer_name=None,\n"
        "    ):\n",
    ),
    (
        "engine: write_remote_data_single body (hoist uid + log)",
        "        transfer_status = self.sessions[sess_idx].write(\n"
        "            local_offset,\n"
        "            remote_offset,\n"
        "            transfer_size_byte,\n"
        "            self.moriio_engine.allocate_transfer_uid(),\n"
        "        )\n",
        "        write_uid = self.moriio_engine.allocate_transfer_uid()\n"
        '        _moriio_log_reqid_map(request_id, transfer_id, layer_name, write_uid, "write_single")\n'
        "        transfer_status = self.sessions[sess_idx].write(\n"
        "            local_offset,\n"
        "            remote_offset,\n"
        "            transfer_size_byte,\n"
        "            write_uid,\n"
        "        )\n",
    ),
    (
        "engine: read_remote_data signature (+ reqid kwargs)",
        "    def read_remote_data(\n"
        "        self, transfer_size_byte, local_offset=0, remote_offset=0, session=None\n"
        "    ):\n",
        "    def read_remote_data(\n"
        "        self, transfer_size_byte, local_offset=0, remote_offset=0, session=None,\n"
        "        request_id=None, transfer_id=None, layer_name=None,\n"
        "    ):\n",
    ),
    (
        "engine: read_remote_data body (hoist uid + log)",
        "        transfer_status = session.batch_read(\n"
        "            local_offset,\n"
        "            remote_offset,\n"
        "            transfer_size_byte,\n"
        "            self.moriio_engine.allocate_transfer_uid(),\n"
        "        )\n",
        "        read_uid = self.moriio_engine.allocate_transfer_uid()\n"
        '        _moriio_log_reqid_map(request_id, transfer_id, layer_name, read_uid, "read")\n'
        "        transfer_status = session.batch_read(\n"
        "            local_offset,\n"
        "            remote_offset,\n"
        "            transfer_size_byte,\n"
        "            read_uid,\n"
        "        )\n",
    ),
    (
        "engine: _do_layer_write batch call (pass reqid through)",
        "                self.worker.moriio_wrapper.write_remote_data(\n"
        "                    plan.transfer_sizes,\n"
        "                    plan.transfer_local_offsets,\n"
        "                    plan.transfer_remote_offsets,\n"
        "                    sessions[plan.sess_idx],\n"
        "                )\n",
        "                self.worker.moriio_wrapper.write_remote_data(\n"
        "                    plan.transfer_sizes,\n"
        "                    plan.transfer_local_offsets,\n"
        "                    plan.transfer_remote_offsets,\n"
        "                    sessions[plan.sess_idx],\n"
        "                    request_id=plan.request_id,\n"
        "                    transfer_id=plan.transfer_id,\n"
        "                    layer_name=plan.layer_name,\n"
        "                )\n",
    ),
    (
        "engine: _do_layer_write single call (pass reqid through)",
        "                self.worker.moriio_wrapper.write_remote_data_single(\n"
        "                    plan.transfer_sizes[i],\n"
        "                    plan.transfer_local_offsets[i],\n"
        "                    plan.transfer_remote_offsets[i],\n"
        "                    plan.sess_idx,\n"
        "                )\n",
        "                self.worker.moriio_wrapper.write_remote_data_single(\n"
        "                    plan.transfer_sizes[i],\n"
        "                    plan.transfer_local_offsets[i],\n"
        "                    plan.transfer_remote_offsets[i],\n"
        "                    plan.sess_idx,\n"
        "                    request_id=plan.request_id,\n"
        "                    transfer_id=plan.transfer_id,\n"
        "                    layer_name=plan.layer_name,\n"
        "                )\n",
    ),
]

_CONNECTOR_EDITS = [
    (
        "connector: _read_blocks read_remote_data call (pass reqid through)",
        "            transfer_status = self.moriio_wrapper.read_remote_data(\n"
        "                offs[2], offs[0], offs[1], sessions[sess_idx]\n"
        "            )\n",
        "            transfer_status = self.moriio_wrapper.read_remote_data(\n"
        "                offs[2],\n"
        "                offs[0],\n"
        "                offs[1],\n"
        "                sessions[sess_idx],\n"
        "                request_id=request_id,\n"
        "                transfer_id=transfer_id,\n"
        "                layer_name=layer_name,\n"
        "            )\n",
    ),
]

# Per-file markers provide idempotency and status checks.


FILES = [
    ("moriio_engine.py", "def _moriio_log_reqid_map(", _ENGINE_EDITS),
    (
        "moriio_connector.py",
        "                sessions[sess_idx],\n                request_id=request_id,",
        _CONNECTOR_EDITS,
    ),
]


def locate_moriio_dir(override):
    if override:
        if not os.path.isdir(override):
            fail(f"--moriio-dir does not exist: {override}")
        return override
    try:
        spec = importlib.util.find_spec("vllm")
    except Exception as e:  # pragma: no cover - depends on runtime env
        fail(
            "could not locate the installed vllm package; pass "
            f"--moriio-dir explicitly. (discovery error: {e})"
        )
    if spec is None:
        fail(
            "could not locate the installed vllm package; pass "
            "--moriio-dir explicitly."
        )
    package_dirs = list(spec.submodule_search_locations or ())
    if not package_dirs and spec.origin:
        package_dirs.append(os.path.dirname(spec.origin))
    if not package_dirs:
        fail(
            "could not determine the installed vllm package directory; pass "
            "--moriio-dir explicitly."
        )
    d = os.path.join(
        package_dirs[0],
        "distributed", "kv_transfer", "kv_connector", "v1", "moriio",
    )
    if not os.path.isdir(d):
        fail(f"expected moriio dir not found under installed vllm: {d}")
    return d


def fail(msg):
    print(f"[patch_moriio_reqid_map] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def apply_edits(text, edits):
    'Apply unique-anchor edits or raise without writing.'
    problems = []
    out = text
    for label, old, new in edits:
        n = out.count(old)
        if n != 1:
            problems.append(f"  - {label}: anchor found {n} times (expected 1)")
            continue
        out = out.replace(old, new)
    if problems:
        raise ValueError("\n".join(problems))
    return out


def main():
    ap = argparse.ArgumentParser(description="Patch installed vLLM MoRIIO for reqid<->write_uid mapping.")
    ap.add_argument("--moriio-dir", default=None, help="path to .../kv_connector/v1/moriio")
    ap.add_argument("--check", action="store_true", help="report status only; change nothing")
    ap.add_argument("--revert", action="store_true", help="restore the .orig backups")
    args = ap.parse_args()

    moriio_dir = locate_moriio_dir(args.moriio_dir)
    print(f"[patch_moriio_reqid_map] target dir: {moriio_dir}")

    if args.revert:
        reverted = 0
        for fname, _marker, _ in FILES:
            path = os.path.join(moriio_dir, fname)
            bak = path + BACKUP_SUFFIX
            if os.path.exists(bak):
                with open(bak, "r", encoding="utf-8") as f:
                    orig = f.read()
                with open(path, "w", encoding="utf-8") as f:
                    f.write(orig)
                os.remove(bak)
                print(f"[patch_moriio_reqid_map] reverted {fname} from backup")
                reverted += 1
            else:
                print(f"[patch_moriio_reqid_map] no backup for {fname}, skipping")
        print(f"[patch_moriio_reqid_map] revert done ({reverted} file(s)).")
        return

    if args.check:
        for fname, marker, _ in FILES:
            path = os.path.join(moriio_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                patched = marker in f.read()
            print(f"[patch_moriio_reqid_map] {fname}: {'PATCHED' if patched else 'unpatched'}")
        return

    # Validate every edit before writing to avoid partial patches.

    staged = []
    for fname, marker, edits in FILES:
        path = os.path.join(moriio_dir, fname)
        if not os.path.exists(path):
            fail(f"file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if marker in text:
            print(f"[patch_moriio_reqid_map] {fname}: already patched, skipping")
            continue
        try:
            new_text = apply_edits(text, edits)
        except ValueError as e:
            fail(
                f"anchors did not match cleanly in {fname} (vLLM source may have "
                f"drifted from the version this patcher targets). Nothing written.\n{e}"
            )
        staged.append((path, text, new_text))

    if not staged:
        print("[patch_moriio_reqid_map] nothing to do (all files already patched).")
        return

    for path, orig, new_text in staged:
        bak = path + BACKUP_SUFFIX
        if not os.path.exists(bak):
            with open(bak, "w", encoding="utf-8") as f:
                f.write(orig)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_text)
        print(f"[patch_moriio_reqid_map] patched {os.path.basename(path)} (backup: {os.path.basename(bak)})")

    print("[patch_moriio_reqid_map] done. Enable at runtime with MORIIO_REQID_MAP=1.")


if __name__ == "__main__":
    main()
