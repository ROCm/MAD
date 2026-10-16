#!/usr/bin/env python3
"""
MORI Benchmark - Initiator Node (node_rank=1 = TARGET in mori terms)

Runs one torchrun per size (like NIXL/Mooncake): for each size, sync with
target, run MoRI's Python benchmark through run_benchmark.py with
--sweep-start-size=X --sweep-max-size=X and the workload shared with mori_cpp
(common/mori_settings.py). Target parses output and writes results.
"""

import argparse
import os
import sys
import socket as sock
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.mori_settings import NUM_ITERS, WARMUP_ITERS, WORKLOAD_FLAGS, clamp_end_size
from common.utils import add_common_bench_args, generate_test_sizes, resolve_hostname

SHARED_DIR = None
RUN_BENCHMARK = str(Path(__file__).resolve().parent / "run_benchmark.py")


def main():
    global SHARED_DIR

    parser = argparse.ArgumentParser(description="MORI Benchmark Initiator")
    add_common_bench_args(parser, default_shared="/shared", include_append=True)
    args = parser.parse_args()

    SHARED_DIR = Path(args.shared_folder)
    target_node = args.target_node or os.environ.get("TARGET_IP", "localhost")
    master_addr = resolve_hostname(target_node)
    mori_root = os.environ.get("MORI_ROOT", "/workspace/mori")
    mori_bench = os.path.join(mori_root, "tests", "python", "io", "benchmark.py")
    if not os.path.isfile(mori_bench):
        print(f"ERROR: MORI benchmark not found: {mori_bench}", file=sys.stderr)
        sys.exit(1)

    hostname = sock.gethostname()
    initiator_ip = resolve_hostname(os.environ.get("NODE2", hostname))

    args.end_size = clamp_end_size(args.end_size)

    test_sizes = generate_test_sizes(args.start_size, args.end_size)

    print(f"{'='*60}")
    print("MORI Benchmark - INITIATOR (per-size, warmup+active in MORI)")
    print(f"{'='*60}")
    print(f"Hostname:     {hostname}")
    print(f"Local IP:     {initiator_ip}")
    print(f"Target:       {target_node} (IP: {master_addr})")
    print(f"Sweep:        {args.start_size} .. {args.end_size} bytes")
    print(f"Test sizes:   {test_sizes}")
    print(f"Buffer alloc: {args.end_size} bytes (max size, reused for all)")
    print(f"Iterations:   {NUM_ITERS} per size (+{WARMUP_ITERS} warmup)")
    print(f"Op type:      write")
    print(f"Shared folder: {SHARED_DIR}")
    print(f"MORI_ROOT:    {mori_root}")
    print(f"{'='*60}\n")

    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    from common.sync_socket import SocketSyncInitiator

    print(f"Sync port:    {args.sync_port}")
    sync = SocketSyncInitiator(host=resolve_hostname(target_node), port=args.sync_port)
    sync.connect()

    env = os.environ.copy()
    env.setdefault("GLOO_SOCKET_IFNAME", "eth0")
    env["PYTHONPATH"] = f"{mori_root}:{env.get('PYTHONPATH', '')}"

    last_size = args.start_size
    try:
        for size_index, current_size in enumerate(test_sizes):
            master_port = 29500 + size_index

            print(f"\n{'='*60}")
            print(f"Testing size: {current_size} bytes ({current_size / (1024*1024):.2f} MB)")
            print(f"{'='*60}\n")

            sync_info = sync.wait_for_target_ready(current_size)
            master_port = sync_info.get("port", master_port)

            cmd = [
                "torchrun",
                "--nnodes=2",
                "--node_rank=1",
                "--nproc_per_node=1",
                f"--master_addr={master_addr}",
                f"--master_port={master_port}",
                RUN_BENCHMARK,
                f"--host={initiator_ip}",
                *WORKLOAD_FLAGS,
                f"--buffer-size={args.end_size}",
                "--all",
                f"--sweep-start-size={current_size}",
                f"--sweep-max-size={current_size}",
            ]
            print(f"Command: {' '.join(cmd)}\n")

            ret = subprocess.run(cmd, env=env, cwd=mori_root, capture_output=True, text=True)
            if ret.stdout:
                print(ret.stdout)
            if ret.returncode != 0:
                is_barrier_failure = ret.stderr and (
                    "dist.barrier" in ret.stderr or "Connection reset by peer" in ret.stderr
                )
                if is_barrier_failure:
                    print(f"  Warning: torchrun exited {ret.returncode} due to post-benchmark "
                          "barrier failure (benchmark data transfer likely succeeded)",
                          file=sys.stderr)
                else:
                    if ret.stderr:
                        print(ret.stderr, file=sys.stderr)
                    sync.send_initiator_done(current_size)
                    sys.exit(ret.returncode)

            last_size = current_size
            sync.send_initiator_done(current_size)
            sync.wait_for_target_done(current_size)

    except KeyboardInterrupt:
        try:
            sync.send_initiator_done(last_size)
        except Exception:
            pass
        sys.exit(130)

    print(f"\n{'='*60}")
    print("All sizes completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
