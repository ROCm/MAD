#!/usr/bin/env python3
"""
Mooncake Benchmark - Initiator Node
Same approach as mori/rixl: shared-folder coordination, read target port and metadata port from shared folder.
Per size: initializes Mooncake TransferEngine, allocates VRAM, runs benchmark (WorkerThreads), computes throughput,
writes per-size result and appends to results_mooncake.json.
CLI args match rixl/mori; other params (protocol, device, threads, etc.) are hardcoded or from env.
"""

import argparse
import gc
import os
import statistics
import sys
import time
import socket as sock
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

# Add parent to path for common imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.utils import (
    add_common_bench_args,
    append_result_to_file,
    collect_version_info,
    generate_test_sizes,
    resolve_hostname,
)

SHARED_DIR = None


class WorkerThread:
    """Worker thread for running Mooncake VRAM-to-VRAM transfers (inlined from former mooncake_bench_engine)."""

    def __init__(self, worker_id, engine, target_session_id, local_buffer_ptr,
                 remote_buffer_ptr, block_size, batch_size, operation,
                 warmup_iters=0, num_iters=None):
        self.worker_id = worker_id
        self.engine = engine
        self.target_session_id = target_session_id
        self.local_buffer_ptr = local_buffer_ptr
        self.remote_buffer_ptr = remote_buffer_ptr
        self.block_size = block_size
        self.batch_size = batch_size
        self.operation = operation
        self.warmup_iters = warmup_iters
        self.num_iters = num_iters
        self.batch_count = 0
        self.durations = []  # per-iteration durations (seconds)
        self.running = True

    def _do_one_write_sync(self):
        """Submit one write, wait for completion. Returns duration in seconds or None on error."""
        t0 = time.perf_counter()
        batch_id = self.engine.transfer_submit_write(
            self.target_session_id,
            self.local_buffer_ptr,
            self.remote_buffer_ptr,
            self.block_size,
        )
        if batch_id < 0:
            return None
        while True:
            status = self.engine.transfer_check_status(batch_id)
            if status != 0:
                break
            time.sleep(0.0001)
        t1 = time.perf_counter()
        return (t1 - t0) if status == 1 else None

    def run(self):
        use_iters = self.num_iters is not None and self.num_iters > 0
        num_to_run = self.num_iters if use_iters else 999999
        end_time = time.time() + 60 if not use_iters else 0

        # Warmup phase (sync: one at a time, wait for completion)
        for _ in range(self.warmup_iters):
            if self.operation == "read":
                self.engine.transfer_sync_read(
                    self.target_session_id,
                    self.local_buffer_ptr,
                    self.remote_buffer_ptr,
                    self.block_size,
                )
            else:
                self._do_one_write_sync()

        # Benchmark: one iteration, wait for completion, next (same as RIXL)
        iters_done = 0
        while iters_done < num_to_run and (use_iters or time.time() < end_time) and self.running:
            if self.operation == "read":
                t0 = time.perf_counter()
                ret = self.engine.transfer_sync_read(
                    self.target_session_id,
                    self.local_buffer_ptr,
                    self.remote_buffer_ptr,
                    self.block_size,
                )
                t1 = time.perf_counter()
                if ret >= 0:
                    self.batch_count += 1
                    self.durations.append(t1 - t0)
                    iters_done += 1
                else:
                    break
            else:
                d = self._do_one_write_sync()
                if d is not None:
                    self.batch_count += 1
                    self.durations.append(d)
                    iters_done += 1
                else:
                    break

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


def main():
    global SHARED_DIR

    parser = argparse.ArgumentParser(description="Mooncake Benchmark Initiator (same args as rixl/mori)")
    add_common_bench_args(parser, default_shared="shared", include_append=True)
    args = parser.parse_args()

    # Hardcoded / from env (same style as rixl/mori)
    protocol = "rdma"
    device_name = os.environ.get("IBDEVICES", "mlx5_0")
    gpu_id = 0
    threads = 1
    batch_size = 1
    operation = "write"
    warmup_iters = 100
    num_iters = 128

    SHARED_DIR = Path(args.shared_folder)
    target_node = args.target_node or os.environ.get("TARGET_IP", "localhost")
    test_sizes = generate_test_sizes(args.start_size, args.end_size)

    hostname = sock.gethostname()
    log_path = os.environ.get("LOG_PATH", ".")
    os.makedirs(log_path, exist_ok=True)

    print("=" * 60)
    print("Mooncake Benchmark - INITIATOR")
    print("=" * 60)
    print(f"Hostname: {hostname}")
    print(f"Node ID: {hostname}")
    print(f"Target: {target_node} (port per size from shared folder)")
    print(f"Start size: {args.start_size} bytes ({args.start_size / (1024*1024):.2f} MB)")
    print(f"End size: {args.end_size} bytes ({args.end_size / (1024*1024):.2f} MB)")
    print(f"GPU ID: {gpu_id}, Operation: {operation}, Threads: {threads}, Warmup: {warmup_iters}, Iterations: {num_iters}")
    print(f"Shared folder: {SHARED_DIR}")
    print("=" * 60 + "\n")

    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    from common.sync_socket import SocketSyncInitiator

    print(f"Sync port: {args.sync_port}")
    sync = SocketSyncInitiator(host=resolve_hostname(target_node), port=args.sync_port)
    sync.connect()

    # Collect version info for metadata
    metadata = collect_version_info(
        "mooncake",
        [
            lambda: __import__("importlib.metadata", fromlist=["version"]).version("mooncake-transfer-engine-non-cuda"),
            lambda: __import__("importlib.metadata", fromlist=["version"]).version("mooncake-transfer-engine"),
            lambda: __import__("importlib.metadata", fromlist=["version"]).version("mooncake"),
            lambda: __import__("mooncake").__version__,
            lambda: os.environ.get("MOONCAKE_VERSION"),
            lambda: open("/etc/mooncake_version").read().strip(),
        ],
    )

    results_path = SHARED_DIR / "results_mooncake.json"

    # Buffer size: 2x the largest test size, minimum 2 GB.  A large registered
    # RDMA memory region allows the NIC to pipeline DMA across 12 worker threads
    # instead of serializing them through a region equal to the block size.
    max_test_size = max(test_sizes)
    MIN_BUFFER_SIZE = 2_147_483_648  # 2 GB
    buffer_size = max(max_test_size * 2, MIN_BUFFER_SIZE)

    print(f"Test sizes: {test_sizes}")
    print(f"Number of test sizes: {len(test_sizes)}")
    print(f"Buffer size: {buffer_size} bytes ({buffer_size / (1024**3):.2f} GB)\n")

    print(f"Starting benchmark with {len(test_sizes)} sizes\n")

    # Keep references to old engines so their C++ destructors don't run during
    # the loop.  The initiator uses the bare hostname as session_id, so every
    # iteration registers the *same* metadata keys (mooncake/rpc_meta/<host>,
    # mooncake/ram/<host>).  If we let the old destructor fire while a new
    # engine owns those keys, the destructor deletes the NEW engine's keys and
    # transfers fail with 404.  Preventing collection avoids the race entirely.
    _old_engines = []

    # Pre-allocate ONE large VRAM buffer that persists across all sizes.
    buffer_tensor = None
    buffer_ptr = 0
    buffer_len = 0
    if _MOONCAKE_AVAILABLE and torch is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        buffer_tensor = torch.empty(buffer_size, dtype=torch.uint8, device=device)
        buffer_ptr = buffer_tensor.data_ptr()
        buffer_len = buffer_tensor.nbytes
        print(f"Pre-allocated {buffer_size} byte VRAM buffer on {device}: ptr={buffer_ptr}\n")

    for current_size in test_sizes:
        print("\n" + "=" * 60)
        print(f"Testing size: {current_size} bytes ({current_size / (1024*1024):.2f} MB)")
        print("=" * 60 + "\n")

        sync_info = sync.wait_for_target_ready(current_size)
        metadata_port = sync_info.get("metadata_port", 8000)
        target_port = sync_info["port"]
        metadata_url = f"http://{target_node}:{metadata_port}/metadata"
        target_session_id = f"{target_node}:{target_port}"
        # Allow target's metadata registration to be visible before fetching
        time.sleep(2)
        out_file = os.path.join(log_path, f"mooncake_result_{current_size}.json")
        throughput_gbs = 0.0
        all_durations = []
        avg_gbs = min_gbs = max_gbs = std_gbs = 0.0
        engine = None

        if buffer_tensor is not None:
            try:
                delete_engine_metadata(metadata_url, hostname)

                engine = TransferEngine()
                ret = engine.initialize(hostname, metadata_url, protocol, device_name)
                if ret != 0:
                    print(f"ERROR: Failed to initialize TransferEngine: {ret}", file=sys.stderr)
                else:
                    remote_buffer_ptr = 0
                    for _ in range(30):
                        remote_buffer_ptr = engine.get_first_buffer_address(target_session_id)
                        if remote_buffer_ptr != 0:
                            break
                        time.sleep(1)
                    if remote_buffer_ptr == 0:
                        print(f"ERROR: Target buffer not found for {target_session_id}", file=sys.stderr)
                    else:
                        ret = engine.register_memory(buffer_ptr, buffer_len)
                        if ret != 0:
                            print(f"ERROR: Failed to register local VRAM: {ret}", file=sys.stderr)
                        else:
                            workers = []
                            thread_objs = []
                            for i in range(threads):
                                worker = WorkerThread(
                                    worker_id=i,
                                    engine=engine,
                                    target_session_id=target_session_id,
                                    local_buffer_ptr=buffer_ptr,
                                    remote_buffer_ptr=remote_buffer_ptr,
                                    block_size=current_size,
                                    batch_size=batch_size,
                                    operation=operation,
                                    warmup_iters=warmup_iters,
                                    num_iters=num_iters,
                                )
                                workers.append(worker)
                                thread_objs.append(threading.Thread(target=worker.run))
                            for t in thread_objs:
                                t.start()
                            for t in thread_objs:
                                t.join()
                            all_durations = []
                            for w in workers:
                                all_durations.extend(w.durations)
                            total_batch_count = sum(w.batch_count for w in workers)
                            total_bytes = total_batch_count * current_size
                            bandwidths_gbs = [current_size / (d * 1e9) for d in all_durations] if all_durations else []
                            avg_gbs = statistics.mean(bandwidths_gbs) if len(bandwidths_gbs) >= 1 else 0.0
                            min_gbs = min(bandwidths_gbs) if bandwidths_gbs else 0.0
                            max_gbs = max(bandwidths_gbs) if bandwidths_gbs else 0.0
                            std_gbs = statistics.stdev(bandwidths_gbs) if len(bandwidths_gbs) >= 2 else 0.0
                            throughput_gbs = avg_gbs
                            print(f"Bandwidth (GB/s): avg={avg_gbs:.4f} min={min_gbs:.4f} max={max_gbs:.4f} std={std_gbs:.4f}")
                            engine.unregister_memory(buffer_ptr)
            except KeyboardInterrupt:
                sync.send_initiator_done(current_size)
                sys.exit(130)
            except Exception as e:
                print(f"ERROR: Benchmark failed for size {current_size}: {e}", file=sys.stderr)
            finally:
                if engine is not None:
                    _old_engines.append(engine)
                engine = None
                delete_engine_metadata(metadata_url, hostname)
                time.sleep(2)
        else:
            if not _MOONCAKE_AVAILABLE:
                print("Mooncake/CUDA not available; skipping VRAM benchmark.", file=sys.stderr)

        result_entry = {
            "backend": "mooncake",
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "test_parameters": {
                "size_bytes": current_size,
                "size_mb": current_size / (1024 * 1024),
                "warmup_iters": warmup_iters,
                "num_iters": num_iters,
                "operation": operation,
            },
            "configuration": {
                "protocol": protocol,
                "threads": threads,
                "batch_size": batch_size,
                "device_name": device_name,
                "buffer_size": buffer_size,
            },
            "results": {
                "durations_s": all_durations,
                "bandwidth_gbs_avg": round(avg_gbs, 4),
                "bandwidth_gbs_min": round(min_gbs, 4),
                "bandwidth_gbs_max": round(max_gbs, 4),
                "bandwidth_gbs_std": round(std_gbs, 4),
            },
            "success": throughput_gbs > 0,
        }
        append_result_to_file(results_path, metadata, result_entry)
        print(f"  Result appended to {results_path}")

        sync.send_initiator_done(current_size)
        sync.wait_for_target_done(current_size)
        print(f"✓ Both nodes completed size {current_size}\n")
        time.sleep(1)

    _old_engines.clear()
    if buffer_tensor is not None:
        del buffer_tensor
    gc.collect()

    print("\n" + "=" * 60)
    print("All test sizes completed successfully!")
    print(f"Results file: {results_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
