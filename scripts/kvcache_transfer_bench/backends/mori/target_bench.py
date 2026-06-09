#!/usr/bin/env python3
"""
MORI Benchmark - Target Node (node_rank=0 = INITIATOR in mori terms)

Runs one torchrun per size (like RIXL/Mooncake): for each size, signal ready,
run MORI benchmark with --iters=128 --warmup-iters=100, parse results, write
in RIXL-style format (bandwidth_gbs_avg) for merge_results.
"""

import argparse
import json
import os
import sys
import socket as sock
import subprocess
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.utils import add_common_bench_args, append_result_to_file, collect_version_info, generate_test_sizes, resolve_hostname

SHARED_DIR = None
NUM_ITERS = 128
MAX_BUFFER_SIZE = 1073741824  # 1 GB — 2 GB (2^31) overflows signed int32 in RDMA paths


def parse_mori_output_all(output: str) -> list:
    """Parse ALL rows from the MORI benchmark PrettyTable output."""
    results = []
    for line in (output or "").strip().split("\n"):
        if "+" in line or "MsgSize" in line or "Initiator Rank" in line or not line.strip():
            continue
        if "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 7:
                try:
                    results.append({
                        "msg_size_bytes": int(parts[0]),
                        "batch_size": int(parts[1]),
                        "total_size_mb": float(parts[2]),
                        "max_bandwidth_gbps": float(parts[3]),
                        "avg_bandwidth_gbps": float(parts[4]),
                        "min_latency_us": float(parts[5]),
                        "avg_latency_us": float(parts[6]),
                    })
                except (ValueError, IndexError):
                    continue
    return results


def main():
    global SHARED_DIR

    parser = argparse.ArgumentParser(description="MORI Benchmark Target")
    add_common_bench_args(parser, default_shared="/shared", include_append=True)
    args = parser.parse_args()

    SHARED_DIR = Path(args.shared_folder)
    mori_root = os.environ.get("MORI_ROOT", "/workspace/mori")
    mori_bench = os.path.join(mori_root, "tests", "python", "io", "benchmark.py")
    if not os.path.isfile(mori_bench):
        print(f"ERROR: MORI benchmark not found: {mori_bench}", file=sys.stderr)
        sys.exit(1)

    hostname = sock.gethostname()
    target_node = args.target_node or hostname
    master_addr = resolve_hostname(target_node)

    original_end_size = args.end_size
    if args.end_size > MAX_BUFFER_SIZE:
        print(f"WARNING: Clamping sweep to {MAX_BUFFER_SIZE} bytes "
              f"(2 GB overflows signed int32 in RDMA paths)")
        args.end_size = MAX_BUFFER_SIZE

    test_sizes = generate_test_sizes(args.start_size, args.end_size)

    print(f"{'='*60}")
    print("MORI Benchmark - TARGET (per-size, warmup+active in MORI)")
    print(f"{'='*60}")
    print(f"Hostname:     {hostname}")
    print(f"Target Node:  {target_node} (IP: {master_addr})")
    if args.initiator_node:
        print(f"Initiator:    {args.initiator_node}")
    print(f"Sweep:        {args.start_size} .. {args.end_size} bytes")
    print(f"Test sizes:   {test_sizes}")
    print(f"Iterations:   {NUM_ITERS} per size")
    print(f"Shared folder: {SHARED_DIR}")
    print(f"MORI_ROOT:    {mori_root}")
    print(f"{'='*60}\n")

    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    from common.sync_socket import SocketSyncTarget

    print(f"Sync port:    {args.sync_port}")
    sync = SocketSyncTarget(port=args.sync_port)
    sync.wait_for_connection()

    metadata = collect_version_info(
        "mori",
        [
            lambda: __import__("importlib.metadata", fromlist=["version"]).version("mori"),
            lambda: __import__("mori").__version__,
            lambda: os.environ.get("MORI_VERSION"),
        ],
    )
    results_path = SHARED_DIR / "results_mori.json"

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

            sync.signal_target_ready(current_size, port=master_port)
            print("Waiting for initiator to connect...\n")

            cmd = [
                "torchrun",
                "--nnodes=2",
                "--node_rank=0",
                "--nproc_per_node=1",
                f"--master_addr={master_addr}",
                f"--master_port={master_port}",
                mori_bench,
                f"--host={master_addr}",
                "--op-type=write",
                f"--buffer-size={args.end_size}",
                "--transfer-batch-size=1",
                "--enable-sess",
                "--enable-batch-transfer",
                "--all",
                f"--sweep-start-size={current_size}",
                f"--sweep-max-size={current_size}",
                "--num-initiator-dev=1",
                "--num-target-dev=1",
                "--num-qp-per-transfer=1",
                "--num-worker-threads=1",
                f"--iters={NUM_ITERS}",
                "--poll_cq_mode=polling",
                "--log-level=info",
            ]

            print(f"Command: {' '.join(cmd)}\n")

            ret = subprocess.run(cmd, env=env, cwd=mori_root, capture_output=True, text=True)
            if ret.stdout:
                print(ret.stdout)

            all_metrics = parse_mori_output_all(ret.stdout or "")

            if ret.returncode != 0:
                if all_metrics:
                    print(f"  Warning: torchrun exited {ret.returncode} but valid metrics found",
                          file=sys.stderr)
                else:
                    print(f"  ERROR: torchrun exited {ret.returncode}", file=sys.stderr)
                    if ret.stderr:
                        for line in ret.stderr.strip().splitlines()[-15:]:
                            print(f"    {line}", file=sys.stderr)
                    sync.signal_target_done(current_size)
                    sys.exit(ret.returncode)

            if all_metrics:
                m = all_metrics[0]
                avg_gbs = round(m["avg_bandwidth_gbps"], 4)
                print(f"  Size={m['msg_size_bytes']:>12}  Avg BW={m['avg_bandwidth_gbps']:.2f} GB/s  "
                      f"Max BW={m['max_bandwidth_gbps']:.2f} GB/s  Avg Lat={m['avg_latency_us']:.2f} us")

                result_entry = {
                    "backend": "mori",
                    "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "test_parameters": {
                        "size_bytes": m["msg_size_bytes"],
                        "size_mb": m["msg_size_bytes"] / (1024 * 1024),
                        "num_iters": NUM_ITERS,
                        "operation": "write",
                    },
                    "results": {
                        "bandwidth_gbs_avg": avg_gbs,
                        "bandwidth_gbs_min": avg_gbs,  # MORI doesn't report per-iter; use avg
                        "bandwidth_gbs_max": round(m["max_bandwidth_gbps"], 4),
                        "bandwidth_gbs_std": 0.0,  # MORI doesn't report std
                    },
                    "success": True,
                }
                append_result_to_file(results_path, metadata, result_entry)
                print(f"  Result appended to {results_path}")

            last_size = current_size
            sync.signal_target_done(current_size)
            sync.wait_for_initiator_done(current_size)

    except KeyboardInterrupt:
        try:
            sync.signal_target_done(last_size)
        except Exception:
            pass
        sys.exit(130)

    print(f"\n{'='*60}")
    print("All sizes completed successfully!")
    print(f"Results file: {results_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
