#!/usr/bin/env python3
"""
Mooncake Benchmark - Target Node
Same approach as mori/rixl: shared-folder coordination, one port per size, per-size run then teardown.
Starts metadata server once; for each size: allocates VRAM, registers with Mooncake TransferEngine,
signals ready, waits for initiator_done, then cleans up.
CLI args match rixl/mori; other params (protocol, device, gpu_id, ports) are hardcoded or from env.
"""

import argparse
import gc
import os
import sys
import time
import socket as sock
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.utils import add_common_bench_args, generate_test_sizes, resolve_hostname

SHARED_DIR = None
_metadata_proc = None

# Mooncake VRAM/engine (optional; target runs without if unavailable)
try:
    import torch
    from mooncake.engine import TransferEngine
    _MOONCAKE_AVAILABLE = True
except Exception:
    _MOONCAKE_AVAILABLE = False
    torch = None
    TransferEngine = None


def delete_engine_metadata(metadata_url, session_id):
    """Delete all stale metadata keys for a session from the metadata server.

    Mooncake's initialize() registers multiple keys per session:
      - mooncake/rpc_meta/<session_id>  (RPC endpoint info)
      - mooncake/ram/<session_id>       (RDMA NIC topology)
    The C++ destructor sends DELETEs on teardown, but Python GC is unreliable for
    triggering it between loop iterations (internal refs may prevent collection).
    Worse, if the old destructor runs AFTER the new engine registers the same keys,
    it deletes the new engine's keys — causing 404s during transfers.
    This function manually sends DELETEs as a robust pre-cleanup step.
    """
    for prefix in ("mooncake/rpc_meta", "mooncake/ram"):
        key = f"{prefix}/{session_id}"
        url = f"{metadata_url}?key={urllib.parse.quote(key, safe='')}"
        try:
            req = urllib.request.Request(url, method="DELETE")
            urllib.request.urlopen(req, timeout=5)
            print(f"  Deleted stale metadata key: {key}")
        except Exception:
            pass  # Key may not exist yet, or already deleted — that's fine


def stop_metadata_server():
    global _metadata_proc
    if _metadata_proc is not None and _metadata_proc.poll() is None:
        _metadata_proc.terminate()
        try:
            _metadata_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _metadata_proc.kill()
        _metadata_proc = None
        print("✓ Metadata server stopped")


def main():
    global SHARED_DIR, _metadata_proc

    parser = argparse.ArgumentParser(description="Mooncake Benchmark Target (same args as rixl/mori)")
    add_common_bench_args(parser, default_shared="shared")
    args = parser.parse_args()

    SHARED_DIR = Path(args.shared_folder)
    test_sizes = generate_test_sizes(args.start_size, args.end_size)
    hostname = sock.gethostname()
    target_node = args.target_node or hostname
    # Hardcoded / from env (same style as rixl/mori)
    metadata_port = 8000
    port_base = 15000
    protocol = "rdma"
    device_name = os.environ.get("IBDEVICES", "mlx5_0")
    gpu_id = 0
    metadata_url = f"http://{target_node}:{metadata_port}/metadata"

    # Do not stop metadata server in atexit: Mooncake engine destructors run at
    # interpreter shutdown (after atexit), so we stop the server via a delayed killer.

    print("=" * 60)
    print("Mooncake Benchmark - TARGET")
    print("=" * 60)
    print(f"Hostname: {hostname}")
    print(f"Target Node: {target_node}")
    if args.initiator_node:
        print(f"Initiator Node: {args.initiator_node}")
    print(f"Start size: {args.start_size} bytes ({args.start_size / (1024*1024):.2f} MB)")
    print(f"End size: {args.end_size} bytes ({args.end_size / (1024*1024):.2f} MB)")
    print(f"Metadata port: {metadata_port}")
    print(f"Target port range: {port_base}..{port_base + len(test_sizes) - 1} (one per size)")
    print(f"GPU ID: {gpu_id}")
    print(f"Shared folder: {SHARED_DIR}")
    print("=" * 60 + "\n")

    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    from common.sync_socket import SocketSyncTarget

    print(f"Sync port: {args.sync_port}")
    sync = SocketSyncTarget(port=args.sync_port)
    sync.wait_for_connection()

    # If port is still in use (e.g. previous run's delayed killer hasn't run yet), wait for it
    for _ in range(35):
        try:
            with sock.socket(sock.AF_INET, sock.SOCK_STREAM) as probe:
                probe.settimeout(1)
                probe.connect(("127.0.0.1", metadata_port))
        except (OSError, sock.error):
            # Port not in use (connection refused) -> we can start
            break
        print("Metadata port in use; waiting 1s for previous run's server to stop...")
        time.sleep(1)
    else:
        print("WARNING: Metadata port still in use after 35s; starting server anyway (may fail).", file=sys.stderr)

    # Start metadata server once
    _metadata_proc = subprocess.Popen(
        [sys.executable, "-m", "mooncake.http_metadata_server", "--port", str(metadata_port)],
        env=os.environ,
    )
    time.sleep(2)
    if _metadata_proc.poll() is not None:
        print("ERROR: Metadata server exited early", file=sys.stderr)
        sys.exit(1)
    print(f"Metadata server started on port {metadata_port}\n")

    # Buffer size: 2x the largest test size, minimum 2 GB.  A large registered
    # RDMA memory region lets the NIC pipeline DMA from multiple initiator
    # threads instead of serializing them through a region equal to block_size.
    max_test_size = max(test_sizes)
    MIN_BUFFER_SIZE = 2_147_483_648  # 2 GB
    buffer_size = max(max_test_size * 2, MIN_BUFFER_SIZE)

    # Pre-allocate ONE large VRAM buffer that persists across all sizes.
    buffer_tensor = None
    buffer_ptr = 0
    buffer_len = 0
    if _MOONCAKE_AVAILABLE and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        buffer_tensor = torch.empty(buffer_size, dtype=torch.uint8, device=device)
        buffer_ptr = buffer_tensor.data_ptr()
        buffer_len = buffer_tensor.nbytes
        print(f"Pre-allocated {buffer_size} byte VRAM buffer on {device}: ptr={buffer_ptr}")

    print(f"Buffer size: {buffer_size} bytes ({buffer_size / (1024**3):.2f} GB)")
    print(f"Starting benchmark with {len(test_sizes)} sizes\n")

    for size_index, current_size in enumerate(test_sizes):
        target_port = port_base + size_index
        print("\n" + "=" * 60)
        print(f"Testing size: {current_size} bytes ({current_size / (1024*1024):.2f} MB)")
        print("=" * 60 + "\n")

        engine = None
        if buffer_tensor is not None:
            os.environ["MC_LEGACY_RPC_PORT_BINDING"] = "1"
            session_id = f"{hostname}:{target_port}"

            delete_engine_metadata(metadata_url, session_id)

            engine = TransferEngine()
            ret = engine.initialize(session_id, metadata_url, protocol, device_name)
            if ret != 0:
                print(f"ERROR: Failed to initialize TransferEngine: {ret}", file=sys.stderr)
                sync.signal_target_ready(current_size, port=target_port, metadata_port=metadata_port)
                sync.wait_for_initiator_done(current_size)
                sync.signal_target_done(current_size)
                continue
            ret = engine.register_memory(buffer_ptr, buffer_len)
            if ret != 0:
                print(f"ERROR: Failed to register VRAM buffer: {ret}", file=sys.stderr)
                sync.signal_target_ready(current_size, port=target_port, metadata_port=metadata_port)
                sync.wait_for_initiator_done(current_size)
                sync.signal_target_done(current_size)
                continue
            print(f"VRAM buffer registered: ptr={buffer_ptr}, length={buffer_len}")
            # Allow metadata server to process registration before initiator fetches
            time.sleep(2)

        sync.signal_target_ready(current_size, port=target_port, metadata_port=metadata_port)
        print(f"Target ready for size {current_size} (port={target_port}). Waiting for initiator...\n")

        sync.wait_for_initiator_done(current_size)

        if engine is not None:
            try:
                engine.unregister_memory(buffer_ptr)
            except Exception:
                pass
            del engine
            engine = None
            gc.collect()
            delete_engine_metadata(metadata_url, session_id)
            time.sleep(1)

        sync.signal_target_done(current_size)
        print(f"✓ Both nodes completed size {current_size}\n")
        time.sleep(2)

    if buffer_tensor is not None:
        del buffer_tensor
    gc.collect()
    meta_pid = _metadata_proc.pid if _metadata_proc is not None else None
    if meta_pid is not None:
        _DELAY_KILL_SEC = 10
        subprocess.Popen(
            ["sh", "-c", f"sleep {_DELAY_KILL_SEC} && kill {meta_pid} 2>/dev/null || true"],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"✓ Metadata server will be stopped in {_DELAY_KILL_SEC}s.")
        print(f"  If you re-run the benchmark before then, wait for port {metadata_port} to be free.")

    print("\n" + "=" * 60)
    print("All test sizes completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
