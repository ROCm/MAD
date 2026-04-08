#!/usr/bin/env python3
"""
RIXL Benchmark - Target Node (Simplified)
Simplified target benchmark with hardcoded values and embedded etcd server
"""

import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
import socket as sock
from pathlib import Path

import torch
from nixl._api import nixl_agent, nixl_agent_config
from nixl.logging import get_logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.utils import add_common_bench_args, generate_test_sizes

logger = get_logger(__name__)

# Global etcd process reference
etcd_process = None

# Shared directory for coordination (will be set from arguments)
SHARED_DIR = None


def start_etcd_server(hostname, etcd_port, etcd_peer_port):
    """Start etcd server in background"""
    global etcd_process

    logger.info(f"Starting etcd server on {hostname}:{etcd_port}...")
    print(f"Starting etcd server...")
    print(f"  Client URL: http://{hostname}:{etcd_port}")
    print(f"  Peer URL: http://{hostname}:{etcd_peer_port}")

    # Create a temporary data directory for etcd
    import tempfile
    etcd_data_dir = tempfile.mkdtemp(prefix='etcd_data_')
    logger.info(f"etcd data directory: {etcd_data_dir}")

    try:
        # Use 0.0.0.0 to bind to all interfaces, advertise the actual hostname
        etcd_process = subprocess.Popen([
            'etcd',
            '--data-dir', etcd_data_dir,
            '--listen-client-urls', f'http://0.0.0.0:{etcd_port}',
            '--advertise-client-urls', f'http://{hostname}:{etcd_port}',
            '--listen-peer-urls', f'http://0.0.0.0:{etcd_peer_port}',
            '--initial-advertise-peer-urls', f'http://{hostname}:{etcd_peer_port}',
            '--initial-cluster', f'default=http://{hostname}:{etcd_peer_port}',
            '--log-level', 'error'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait for etcd to start
        time.sleep(5)

        if etcd_process.poll() is not None:
            # Process exited, capture error output
            stdout, stderr = etcd_process.communicate(timeout=1)
            logger.error("etcd process failed to start!")
            logger.error(f"etcd stdout: {stdout}")
            logger.error(f"etcd stderr: {stderr}")
            print(f"etcd error output:\n{stderr}")
            return False

        logger.info("✓ etcd server started successfully")
        print("✓ etcd server started successfully\n")
        return True

    except Exception as e:
        logger.error(f"Failed to start etcd: {e}")
        return False


def stop_etcd_server():
    """Stop etcd server and clear process reference."""
    global etcd_process
    if etcd_process:
        logger.info("Stopping etcd server...")
        print("Stopping etcd server...")
        etcd_process.terminate()
        try:
            etcd_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            etcd_process.kill()
        etcd_process = None
        logger.info("✓ etcd server stopped")
        print("✓ etcd server stopped")


def run_etcd_server(hostname, etcd_port=2379, etcd_peer_port=2380, register_atexit=True):
    """
    Configure environment for etcd and start the etcd server.
    When register_atexit is True, registers stop_etcd_server for process exit.
    Returns True if etcd started successfully, False otherwise.
    """
    etcd_endpoint = f"http://{hostname}:{etcd_port}"
    os.environ["NIXL_ETCD_ENDPOINTS"] = etcd_endpoint
    if register_atexit:
        atexit.register(stop_etcd_server)
    return start_etcd_server(hostname, etcd_port, etcd_peer_port)


def main():
    global SHARED_DIR

    parser = argparse.ArgumentParser(description='RIXL Benchmark Target (Simplified)')
    add_common_bench_args(parser, default_shared="/shared")
    args = parser.parse_args()

    # Set shared directory from argument
    SHARED_DIR = Path(args.shared_folder)
    # Generate test sizes
    test_sizes = generate_test_sizes(args.start_size, args.end_size)
    print(f"Test sizes: {test_sizes}")
    print(f"Number of test sizes: {len(test_sizes)}\n")

    # Base port; each size uses base + index to avoid "Address already in use" after restart
    PORT_BASE = 15473
    GPU_ID = 0
    DEVICE_NAME = None  # auto-discovery
    ETCD_PORT = 2379
    ETCD_PEER_PORT = 2380

    hostname = sock.gethostname()
    target_node = args.target_node if args.target_node else hostname
    etcd_endpoint = f"http://{target_node}:{ETCD_PORT}"

    print(f"{'='*60}")
    print(f"RIXL Benchmark - TARGET (Simplified)")
    print(f"{'='*60}")
    print(f"Hostname: {hostname}")
    print(f"Target Node: {target_node}")
    if args.initiator_node:
        print(f"Initiator Node: {args.initiator_node}")
    print(f"Start size: {args.start_size} bytes ({args.start_size / (1024*1024):.2f} MB)")
    print(f"End size: {args.end_size} bytes ({args.end_size / (1024*1024):.2f} MB)")
    print(f"Port range: {PORT_BASE}..{PORT_BASE + len(test_sizes) - 1} (one per size)")
    print(f"GPU ID: {GPU_ID}")
    print(f"RDMA Device: {DEVICE_NAME or 'auto-discovery'}")
    print(f"etcd endpoint: {etcd_endpoint}")
    print(f"Shared folder: {SHARED_DIR}")
    print(f"{'='*60}\n")

    # Ensure shared directory exists
    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    from common.sync_socket import SocketSyncTarget

    print(f"Sync port: {args.sync_port}")
    sync = SocketSyncTarget(port=args.sync_port)
    sync.wait_for_connection()

    # Check if CUDA/ROCm is available (once)
    if not torch.cuda.is_available():
        logger.error("PyTorch CUDA/ROCm is not available!")
        logger.error("Make sure ROCm is installed and PyTorch is built with ROCm support.")
        sys.exit(1)

    device = torch.device(f'cuda:{GPU_ID}')
    torch.set_default_device(device)
    logger.info(f"Using GPU device: {device}")
    logger.info(f"GPU Name: {torch.cuda.get_device_name(GPU_ID)}")

    # Start etcd once for the entire sweep (avoids per-size restart overhead / port TIME_WAIT)
    if not run_etcd_server(target_node, ETCD_PORT, ETCD_PEER_PORT):
        logger.error("Failed to start etcd server. Exiting.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Starting benchmark with {len(test_sizes)} sizes")
    print(f"{'='*60}\n")

    for size_index, current_size in enumerate(test_sizes):
        port = PORT_BASE + size_index
        print(f"\n{'='*60}")
        print(f"Testing size: {current_size} bytes ({current_size / (1024*1024):.2f} MB)")
        print(f"{'='*60}\n")

        config = nixl_agent_config(True, True, port)
        logger.info("Initializing RIXL agent in target mode...")
        agent = nixl_agent("target", config)

        # Allocate VRAM buffer for current size
        logger.info(f"Allocating {current_size} bytes in VRAM on GPU {GPU_ID}...")
        tensor = torch.zeros(current_size, dtype=torch.uint8, device=device)

        # Register GPU memory with RIXL
        logger.info("Registering GPU memory with RIXL agent...")
        reg_descs = agent.register_memory([tensor])

        if not reg_descs:
            logger.error("Memory registration failed.")
            sync.signal_target_done(current_size)
            sys.exit(1)

        logger.info("GPU memory registered successfully.")
        target_descs = reg_descs.trim()
        target_desc_str = agent.get_serialized_descs(target_descs)

        print(f"\nTarget ready for size {current_size}")
        print(f"Listening on: {target_node}:{port}")
        print(f"GPU buffer allocated on: cuda:{GPU_ID}")

        sync.signal_target_ready(current_size, port=port)
        print(f"Waiting for initiator to connect...\n")

        # Wait for transfer to complete for this size
        notif_sent = False
        transfer_complete = False
        try:
            while not transfer_complete:
                try:
                    if not notif_sent:
                        ready = agent.check_remote_metadata("initiator")
                        if ready:
                            logger.info("Initiator metadata detected. Sending target descriptors...")
                            agent.send_notif("initiator", target_desc_str)
                            notif_sent = True
                            print("Descriptors sent to initiator. Waiting for transfers...")

                    if notif_sent and not transfer_complete:
                        if agent.check_remote_xfer_done("initiator", b"UUID"):
                            logger.info("Transfer completed successfully!")
                            transfer_complete = True
                            print(f"\n✓ Transfer complete for size {current_size}")

                    time.sleep(0.1)

                except Exception as e:
                    error_str = str(e)
                    if "REMOTE_DISCONNECT" in error_str or "nixlRemoteDisconnectError" in str(type(e)):
                        logger.warning("Initiator disconnected. Resetting and waiting for next connection...")
                        print("\n⚠️  Initiator disconnected. Ready for new connection.\n")
                        notif_sent = False
                        time.sleep(1)
                    else:
                        logger.error(f"Error during transfer: {e}")
                        print(f"Error: {e}")
                        notif_sent = False
                        time.sleep(1)

        except KeyboardInterrupt:
            print("\n\nShutting down...")
            agent.deregister_memory(reg_descs)
            del tensor
            stop_etcd_server()
            sys.exit(0)

        agent.deregister_memory(reg_descs)
        del tensor

        sync.signal_target_done(current_size)
        sync.wait_for_initiator_done(current_size)

        print(f"✓ Both nodes completed size {current_size}\n")

        time.sleep(1)

    stop_etcd_server()

    print(f"\n{'='*60}")
    print(f"All test sizes completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
