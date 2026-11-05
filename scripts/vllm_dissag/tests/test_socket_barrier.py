#!/usr/bin/env python3
"""socket_barrier.py --timeout: a missing peer must fail fast and be named.

A job held eight nodes for its whole allocation because NODE4 never launched
its container and the barrier waited forever without saying which peer was
absent. No GPU, no slurm, no container -- binds a real socket on loopback.
"""
import socket
import subprocess
import sys
import time
from pathlib import Path

BARRIER = Path(__file__).resolve().parent.parent / "socket_barrier.py"
FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(ips, ports, extra, timeout=60):
    cmd = [sys.executable, str(BARRIER), "--node-ips", ips, "--node-ports", ports] + extra
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


print("socket_barrier --timeout")

# A peer that never opens: must exit 1, must name the peer, must not hang.
dead = free_port()
t0 = time.monotonic()
r = run("127.0.0.1", str(dead), ["--timeout", "2", "--report-every", "0"])
elapsed = time.monotonic() - t0
check("missing peer exits nonzero", r.returncode == 1, f"rc={r.returncode}")
check("missing peer is named", f"127.0.0.1:{dead}" in r.stdout, r.stdout[-200:])
check("says it timed out", "timed out" in r.stdout, r.stdout[-200:])
check("gives up promptly", elapsed < 30, f"{elapsed:.1f}s")
check("still prints the legacy line", "Waiting for nodes. . ." in r.stdout)

# A peer that is already listening: must return 0 without waiting.
srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", 0))
srv.listen(5)
live = srv.getsockname()[1]
try:
    r = run("127.0.0.1", str(live), ["--timeout", "30"])
    check("open peer succeeds", r.returncode == 0, f"rc={r.returncode} {r.stdout[-200:]}")
    check("open peer does not warn", "timed out" not in r.stdout)

    # Default (no --timeout) must keep the historical wait-forever contract.
    r = run("127.0.0.1", str(live), [])
    check("default still exits 0 when peers are up", r.returncode == 0, f"rc={r.returncode}")
finally:
    srv.close()

# One live and one dead peer: the report must name only the dead one.
srv2 = socket.socket()
srv2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv2.bind(("127.0.0.1", 0))
srv2.listen(5)
up = srv2.getsockname()[1]
try:
    r = run("127.0.0.1,127.0.0.1", f"{up},{dead}", ["--timeout", "2", "--report-every", "0"])
    check("partial: exits nonzero", r.returncode == 1, f"rc={r.returncode}")
    check("partial: names the dead peer", f"127.0.0.1:{dead}" in r.stdout)
    check("partial: reports 1/2 missing", "1/2" in r.stdout, r.stdout[-200:])
finally:
    srv2.close()

# Stale local listener: --enable-port must fail the bind, not treat the leftover
# as "we are ready".
stale = socket.socket()
stale.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
stale.bind(("127.0.0.1", 0))
stale.listen(5)
held = stale.getsockname()[1]
try:
    r = run(
        "127.0.0.1",
        str(held),
        ["--enable-port", "--local-ip", "127.0.0.1", "--local-port", str(held),
         "--timeout", "5"],
    )
    check("stale listener: enable-port exits nonzero", r.returncode == 1,
          f"rc={r.returncode} out={r.stdout[-200:]!r} err={r.stderr[-200:]!r}")
    check("stale listener: names the bind failure",
          "failed to bind" in r.stdout or "failed to bind" in r.stderr,
          f"out={r.stdout[-200:]!r}")
finally:
    stale.close()

print()
if FAILED:
    print(f"FAILED: {len(FAILED)} -> {', '.join(FAILED)}")
    sys.exit(1)
print("all socket-barrier checks passed")
