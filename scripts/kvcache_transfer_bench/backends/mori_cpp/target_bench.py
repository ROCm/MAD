#!/usr/bin/env python3
"""
MoRI-IO C++ Benchmark - Target Node (bench_engine rank 0 = INITIATOR in mori terms)

Same roles as the Python MoRI backend: this node drives the RDMA writes, parses
bench_engine's table and writes results_mori_cpp.json. One bench_engine run
covers the whole sweep (per size: warmup, then a single timer around the timed
iterations); see engine.py for the workload flags.
"""

import argparse
import os
import re
import socket as sock
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.utils import add_common_bench_args, append_result_to_file, collect_version_info

import engine

# MsgSize(B)  Batch  Iters  AvgBW(GB/s)  AvgLat(us)  TotalDur(us)
ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def parse_bench_engine_output(output: str) -> list:
    results = []
    for line in (output or "").splitlines():
        m = ROW_RE.match(line)
        if m:
            results.append({
                "msg_size_bytes": int(m.group(1)),
                "batch_size": int(m.group(2)),
                "iters": int(m.group(3)),
                "avg_bandwidth_gbps": float(m.group(4)),
                "avg_latency_us": float(m.group(5)),
                "total_duration_us": float(m.group(6)),
            })
    return results


def main():
    parser = argparse.ArgumentParser(description="MoRI-IO C++ Benchmark Target")
    add_common_bench_args(parser, default_shared="/shared", include_append=True)
    args = parser.parse_args()

    shared_dir = Path(args.shared_folder)
    engine.require_engine()
    hostname = sock.gethostname()
    master_ip = engine.local_ip()
    end_size = engine.clamp_end_size(args.end_size)

    print(f"{'='*60}")
    print("MoRI-IO C++ Benchmark - TARGET (bench_engine rank 0, one run for the sweep)")
    print(f"{'='*60}")
    print(f"Hostname:     {hostname}")
    print(f"Master IP:    {master_ip}")
    if args.initiator_node:
        print(f"Initiator:    {args.initiator_node}")
    print(f"Sweep:        {args.start_size} .. {end_size} bytes")
    print(f"Iterations:   {engine.NUM_ITERS} per size (+{engine.WARMUP_ITERS} warmup)")
    print(f"QPs/workers:  {engine.NUM_QP_PER_TRANSFER} QP per transfer, {engine.NUM_WORKER_THREADS} worker thread")
    print(f"Batch:        {engine.TRANSFER_BATCH_SIZE}, contiguous")
    print(f"bench_engine: {engine.engine_path()}")
    print(f"Shared folder: {shared_dir}")
    print(f"{'='*60}\n")

    shared_dir.mkdir(parents=True, exist_ok=True)

    from common.sync_socket import SocketSyncTarget

    print(f"Sync port:    {args.sync_port}")
    sync = SocketSyncTarget(port=args.sync_port)
    sync.wait_for_connection()

    metadata = collect_version_info(
        "mori_cpp",
        [
            lambda: __import__("importlib.metadata", fromlist=["version"]).version("mori"),
            lambda: __import__("mori").__version__,
            lambda: os.environ.get("MORI_VERSION"),
        ],
    )
    results_path = shared_dir / "results_mori_cpp.json"

    sync.signal_target_ready(args.start_size, port=engine.BOOTSTRAP_PORT, master_ip=master_ip)
    rc, out, _ = engine.run(engine.command(0, master_ip, master_ip, engine.BOOTSTRAP_PORT,
                                           args.start_size, end_size))
    rows = parse_bench_engine_output(out)

    if rc == 0 and rows:
        for m in rows:
            print(f"  Size={m['msg_size_bytes']:>12}  Avg BW={m['avg_bandwidth_gbps']:.2f} GB/s  "
                  f"Avg Lat={m['avg_latency_us']:.2f} us")
            append_result_to_file(results_path, metadata, {
                "backend": "mori_cpp",
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "test_parameters": {
                    "size_bytes": m["msg_size_bytes"],
                    "size_mb": m["msg_size_bytes"] / (1024 * 1024),
                    "num_iters": m["iters"],
                    "warmup_iters": engine.WARMUP_ITERS,
                    "operation": "write",
                    "transfer_batch_size": m["batch_size"],
                    "batch_contiguous": True,
                    "num_qp_per_transfer": engine.NUM_QP_PER_TRANSFER,
                    "num_worker_threads": engine.NUM_WORKER_THREADS,
                },
                "results": {
                    "bandwidth_gbs_avg": round(m["avg_bandwidth_gbps"], 4),
                    "latency_us_avg": m["avg_latency_us"],
                    "total_duration_us": m["total_duration_us"],
                },
                "success": True,
            })
        print(f"  {len(rows)} result(s) appended to {results_path}")
    else:
        print(f"  ERROR: bench_engine exited {rc} with {len(rows)} result row(s); nothing recorded",
              file=sys.stderr)

    sync.signal_target_done(args.start_size)
    sync.wait_for_initiator_done(args.start_size)

    if rc != 0 or not rows:
        sys.exit(rc or 1)

    print(f"\n{'='*60}")
    print("Sweep completed successfully!")
    print(f"Results file: {results_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
