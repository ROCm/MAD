#!/usr/bin/env python3
"""
RIXL Benchmark - Initiator Node (Simplified)
Simplified initiator benchmark with hardcoded values
"""

import argparse
import os
import statistics
import sys
import time
import socket as sock
from pathlib import Path
from datetime import datetime, timezone

import torch
from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.utils import (
    add_common_bench_args,
    append_result_to_file,
    collect_version_info,
    generate_test_sizes,
    resolve_hostname,
)

logger = get_logger(__name__)

# Shared directory for coordination (set from --shared_folder in main)
SHARED_DIR = None


def main():
    global SHARED_DIR

    parser = argparse.ArgumentParser(description='RIXL Benchmark Initiator (Simplified)')
    add_common_bench_args(parser, default_shared="/shared", include_append=True)
    args = parser.parse_args()

    # Set shared directory from argument (must match target)
    SHARED_DIR = Path(args.shared_folder)

    # Target address: --target_node takes precedence over TARGET_IP env
    target_node = args.target_node or os.environ.get('TARGET_IP', 'localhost')

    # Target uses 15473+; initiator listener uses a separate range to avoid "Address already in use"
    INITIATOR_PORT_BASE = 15573
    GPU_ID = 0
    OPERATION = 'WRITE'
    NUM_ITERS = 128  # timed iterations per size
    WARMUP_ITERS = 100

    # etcd runs on target node; use same port as target_bench.py (2379)
    ETCD_PORT = 2379
    etcd_endpoint = f"http://{target_node}:{ETCD_PORT}"
    os.environ['NIXL_ETCD_ENDPOINTS'] = etcd_endpoint

    # Resolve target IP for connections
    target_ip = resolve_hostname(target_node)
    hostname = sock.gethostname()

    print(f"{'='*60}")
    print(f"RIXL Benchmark - INITIATOR (Simplified)")
    print(f"{'='*60}")
    print(f"Hostname: {hostname}")
    print(f"Node ID: {hostname}")
    print(f"Target: {target_node} (IP: {target_ip}, port per size from shared folder)")
    print(f"Start size: {args.start_size} bytes ({args.start_size / (1024*1024):.2f} MB)")
    print(f"End size: {args.end_size} bytes ({args.end_size / (1024*1024):.2f} MB)")
    print(f"GPU ID: {GPU_ID}")
    print(f"Operation: {OPERATION}")
    print(f"Warmup iterations: {WARMUP_ITERS}, Timed iterations: {NUM_ITERS}")
    print(f"etcd endpoint: {etcd_endpoint}")
    print(f"Shared folder: {SHARED_DIR}")
    print(f"{'='*60}\n")

    # Ensure shared directory exists
    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    from common.sync_socket import SocketSyncInitiator

    print(f"Sync port: {args.sync_port}")
    sync = SocketSyncInitiator(host=resolve_hostname(target_node), port=args.sync_port)
    sync.connect()

    # Collect version info for metadata
    metadata = collect_version_info(
        "rixl",
        [
            lambda: __import__("importlib.metadata", fromlist=["version"]).version("rixl"),
            lambda: __import__("importlib.metadata", fromlist=["version"]).version("nixl"),
            lambda: __import__("rixl").__version__,
            lambda: __import__("nixl").__version__,
            lambda: os.environ.get("RIXL_VERSION"),
        ],
        pytorch_version=torch.__version__,
    )

    results_path = SHARED_DIR / "results_rixl.json"

    # Generate test sizes
    test_sizes = generate_test_sizes(args.start_size, args.end_size)
    print(f"Test sizes: {test_sizes}")
    print(f"Number of test sizes: {len(test_sizes)}\n")


    # Check if CUDA/ROCm is available
    if not torch.cuda.is_available():
        logger.error("PyTorch CUDA/ROCm is not available!")
        logger.error("Make sure ROCm is installed and PyTorch is built with ROCm support.")
        sys.exit(1)

    # Set device to specified GPU
    device = torch.device(f'cuda:{GPU_ID}')
    torch.set_default_device(device)

    logger.info(f"Using GPU device: {device}")
    logger.info(f"GPU Name: {torch.cuda.get_device_name(GPU_ID)}")

    print(f"\n{'='*60}")
    print(f"Starting benchmark with {len(test_sizes)} sizes")
    print(f"{'='*60}\n")

    # Iterate through each test size (fresh listener port per size to avoid bind "Address already in use")
    for size_index, current_size in enumerate(test_sizes):
        initiator_port = INITIATOR_PORT_BASE + size_index

        print(f"\n{'='*60}")
        print(f"Testing size: {current_size} bytes ({current_size / (1024*1024):.2f} MB)")
        print(f"{'='*60}\n")

        # Wait for target to be ready for this size
        sync_info = sync.wait_for_target_ready(current_size)
        target_port = sync_info["port"]

        # Create nixl agent config for this size (unique listener port per size)
        config = nixl_agent_config(True, True, initiator_port)

        # Initialize agent in initiator mode
        logger.info("Initializing RIXL agent in initiator mode...")
        agent = nixl_agent("initiator", config)

        # Fetch remote metadata and send local metadata
        logger.info(f"Fetching remote metadata from target {target_node}:{target_port}...")
        agent.fetch_remote_metadata("target", target_ip, target_port)
        agent.send_local_metadata(target_ip, target_port)

        # Wait for target to send its descriptors
        logger.info("Waiting for target descriptors...")
        notifs = []
        max_wait = 30
        wait_time = 0
        while len(notifs) == 0 and wait_time < max_wait:
            notifs = agent.get_new_notifs()
            if len(notifs) == 0:
                time.sleep(0.5)
                wait_time += 0.5

        if len(notifs) == 0:
            logger.error(f"Timeout waiting for target descriptors after {max_wait}s")
            sync.send_initiator_done(current_size)
            continue

        logger.info("Received target descriptors.")
        target_descs = agent.deserialize_descs(notifs["target"][0])

        # Ensure remote metadata has arrived
        logger.info("Waiting for remote metadata to be ready...")
        ready = False
        wait_time = 0
        while not ready and wait_time < max_wait:
            ready = agent.check_remote_metadata("target")
            if not ready:
                time.sleep(0.5)
                wait_time += 0.5

        if not ready:
            logger.error(f"Remote metadata not ready after {max_wait}s")
            sync.send_initiator_done(current_size)
            continue

        logger.info("Remote metadata ready.")

        # Allocate local buffer for current size
        logger.info(f"Allocating {current_size} bytes in VRAM on GPU {GPU_ID}...")
        tensor = torch.zeros(current_size, dtype=torch.uint8, device=device)

        # Register GPU memory with RIXL
        logger.info("Registering GPU memory with RIXL agent...")
        reg_descs = agent.register_memory([tensor])

        if not reg_descs:
            logger.error("Memory registration failed.")
            sync.send_initiator_done(current_size)
            sys.exit(1)

        logger.info("GPU memory registered successfully.")

        # Prepare initiator descriptors
        initiator_descs = reg_descs.trim()

        # Create transfer handle
        logger.info(f"Creating transfer handle for {OPERATION} operation...")
        xfer_handle = agent.initialize_xfer(
            OPERATION,
            initiator_descs,
            target_descs,
            "target",
            b"UUID"
        )

        if not xfer_handle:
            logger.error("Failed to create transfer handle")
            agent.deregister_memory(reg_descs)
            del tensor
            sync.send_initiator_done(current_size)
            sys.exit(1)

        logger.info("Transfer handle created successfully.")

        # Warmup phase
        print(f"Warming up ({WARMUP_ITERS} iterations)...")
        state = "DONE"  # so post-loop "if state == ERR" is always defined
        for i in range(WARMUP_ITERS):
            state = agent.transfer(xfer_handle)
            if state == "ERR":
                logger.error(f"Transfer error during warmup iteration {i}")
                break

            # Wait for completion
            while True:
                state = agent.check_xfer_state(xfer_handle)
                if state == "ERR":
                    logger.error(f"Transfer error state during warmup")
                    break
                elif state == "DONE":
                    break

            if state == "ERR":
                break

        if state == "ERR":
            agent.release_xfer_handle(xfer_handle)
            agent.deregister_memory(reg_descs)
            del tensor
            sync.send_initiator_done(current_size)
            sys.exit(1)

        print(f"Starting benchmark ({NUM_ITERS} iterations)...\n")

        # Benchmark: one iteration, wait for completion, next (sync like RIXL)
        durations = []
        transfer_count = 0

        for _ in range(NUM_ITERS):
            t0 = time.perf_counter()
            state = agent.transfer(xfer_handle)

            if state == "ERR":
                logger.error("Transfer error")
                break

            # Wait for completion
            while True:
                state = agent.check_xfer_state(xfer_handle)
                if state == "ERR":
                    logger.error("Transfer error state")
                    break
                elif state == "DONE":
                    transfer_count += 1
                    break

            if state == "ERR":
                break

            t1 = time.perf_counter()
            durations.append(t1 - t0)

        # Compute bandwidth per iteration (GB/s = bytes / duration / 1e9)
        bandwidths_gbs = [current_size / (d * 1e9) for d in durations] if durations else []
        avg_gbs = statistics.mean(bandwidths_gbs) if len(bandwidths_gbs) >= 1 else 0.0
        min_gbs = min(bandwidths_gbs) if bandwidths_gbs else 0.0
        max_gbs = max(bandwidths_gbs) if bandwidths_gbs else 0.0
        std_gbs = statistics.stdev(bandwidths_gbs) if len(bandwidths_gbs) >= 2 else 0.0

        total_bytes = transfer_count * current_size
        actual_duration = sum(durations)

        print(f"\n{'='*60}")
        print(f"Size {current_size} bytes: {transfer_count} iterations")
        print(f"Total data: {total_bytes / (1024*1024*1024):.2f} GB")
        print(f"Bandwidth (GB/s): avg={avg_gbs:.4f} min={min_gbs:.4f} max={max_gbs:.4f} std={std_gbs:.4f}")
        print(f"{'='*60}\n")

        result_entry = {
            "backend": "rixl",
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "test_parameters": {
                "size_bytes": current_size,
                "size_mb": current_size / (1024 * 1024),
                "warmup_iters": WARMUP_ITERS,
                "num_iters": NUM_ITERS,
                "operation": OPERATION,
                "gpu_id": GPU_ID,
            },
            "configuration": {
                "etcd_endpoint": etcd_endpoint,
            },
            "results": {
                "total_operations": transfer_count,
                "durations_s": durations,
                "bandwidth_gbs_avg": round(avg_gbs, 4),
                "bandwidth_gbs_min": round(min_gbs, 4),
                "bandwidth_gbs_max": round(max_gbs, 4),
                "bandwidth_gbs_std": round(std_gbs, 4),
            },
            "success": transfer_count > 0,
        }
        append_result_to_file(results_path, metadata, result_entry)
        print(f"  Result appended to {results_path}")

        # Cleanup for this size
        agent.release_xfer_handle(xfer_handle)
        agent.deregister_memory(reg_descs)
        del tensor

        # Cleanup agent for this size
        agent.remove_remote_agent("target")
        agent.invalidate_local_metadata(target_ip, target_port)

        # Signal that initiator is done with this size
        sync.send_initiator_done(current_size)

        # Wait for target to finish this size
        sync.wait_for_target_done(current_size)

        print(f"✓ Both nodes completed size {current_size}\n")

        # Small delay before next size
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"All test sizes completed successfully!")
    print(f"Results file: {results_path}")
    print(f"{'='*60}\n")

    logger.info("Benchmark complete. Exiting.")


if __name__ == "__main__":
    main()
