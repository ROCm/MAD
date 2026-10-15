#!/usr/bin/env python3
"""
MoRI-IO C++ Benchmark - Initiator Node (bench_engine rank 1 = TARGET in mori terms)

Waits for the target node to start bench_engine rank 0, then runs rank 1 with
the same workload flags (engine.py). This node only registers memory and waits;
the target node records the results.
"""

import argparse
import os
import socket as sock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.utils import add_common_bench_args, resolve_hostname

import engine


def main():
    parser = argparse.ArgumentParser(description="MoRI-IO C++ Benchmark Initiator")
    add_common_bench_args(parser, default_shared="/shared", include_append=True)
    args = parser.parse_args()

    engine.require_engine()
    target_node = args.target_node or os.environ.get("TARGET_IP", "localhost")
    self_ip = engine.local_ip()
    end_size = engine.clamp_end_size(args.end_size)

    print(f"{'='*60}")
    print("MoRI-IO C++ Benchmark - INITIATOR (bench_engine rank 1)")
    print(f"{'='*60}")
    print(f"Hostname:     {sock.gethostname()}")
    print(f"Local IP:     {self_ip}")
    print(f"Target:       {target_node}")
    print(f"Sweep:        {args.start_size} .. {end_size} bytes")
    print(f"bench_engine: {engine.engine_path()}")
    print(f"{'='*60}\n")

    from common.sync_socket import SocketSyncInitiator

    print(f"Sync port:    {args.sync_port}")
    sync = SocketSyncInitiator(host=resolve_hostname(target_node), port=args.sync_port)
    sync.connect()

    sync_info = sync.wait_for_target_ready(args.start_size)
    master_ip = sync_info.get("master_ip") or resolve_hostname(target_node)
    port = sync_info.get("port", engine.BOOTSTRAP_PORT)

    rc, _, _ = engine.run(engine.command(1, master_ip, self_ip, port, args.start_size, end_size))
    if rc != 0:
        print(f"  ERROR: bench_engine exited {rc}", file=sys.stderr)

    sync.send_initiator_done(args.start_size)
    sync.wait_for_target_done(args.start_size)
    sys.exit(rc)


if __name__ == "__main__":
    main()
