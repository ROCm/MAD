#!/usr/bin/env python3
"""tcp_listen.sh against real listeners: find, probe, kill, and protect.

The vLLM image has no ss/fuser, so the barrier-port cleanup used to be a
silent no-op and a stale listener survived into the next boot. These helpers
read /proc instead. No GPU, no slurm, no container -- loopback sockets only.
"""
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HELPERS = Path(__file__).resolve().parent.parent / "tcp_listen.sh"
FAILED = []

LISTENER = (
    "import socket, sys, time\n"
    "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
    "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
    "s.bind(('0.0.0.0', int(sys.argv[1])))\n"
    "s.listen(1)\n"
    "print('up', flush=True)\n"
    "time.sleep(120)\n"
)


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def sh(snippet):
    return subprocess.run(
        ["bash", "-c", f'source "{HELPERS}"; {snippet}'],
        capture_output=True, text=True, timeout=30,
    )


def free_port():
    s = socket.socket()
    s.bind(("0.0.0.0", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def listen(port, exe=sys.executable):
    p = subprocess.Popen([exe, "-c", LISTENER, str(port)],
                         stdout=subprocess.PIPE, text=True)
    p.stdout.readline()
    return p


def gone(proc, wait=2.0):
    deadline = time.time() + wait
    while time.time() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.05)
    return False


def main():
    port = free_port()
    print(f"free port {port}")
    check("free: busy is false", sh(f"tcp_listen_busy {port}").returncode != 0)
    check("free: no pids", sh(f"tcp_listen_pids {port}").stdout.strip() == "")
    r = sh(f"tcp_listen_bind_probe {port}")
    check("free: bind probe succeeds", r.returncode == 0, r.stdout + r.stderr)
    check("free: kill is a no-op success", sh(f"tcp_listen_kill {port}").returncode == 0)

    port = free_port()
    proc = listen(port)
    try:
        print(f"listener pid {proc.pid} on {port}")
        check("held: inode found", sh(f"tcp_listen_inodes {port}").stdout.strip().isdigit())
        check("held: busy is true", sh(f"tcp_listen_busy {port}").returncode == 0)
        pids = sh(f"tcp_listen_pids {port}").stdout.split()
        check("held: pids names the listener", str(proc.pid) in pids, f"got {pids}")
        r = sh(f"tcp_listen_bind_probe {port}")
        check("held: bind probe fails", r.returncode != 0, r.stdout)
        check("held: bind probe says bind_fail", "bind_fail 0.0.0.0" in r.stdout, r.stdout)
        r = sh(f"tcp_listen_describe {port}")
        check("held: describe names the pid", f"pid={proc.pid}" in r.stdout, r.stdout)
        r = sh(f"tcp_listen_dump_port {port}")
        check("held: dump shows a LISTEN row", "st=0A" in r.stdout, r.stdout)
        r = sh(f"tcp_listen_kill {port}")
        check("kill: returns 0", r.returncode == 0, r.stdout + r.stderr)
        check("kill: listener is dead", gone(proc))
        check("kill: port is free again", sh(f"tcp_listen_busy {port}").returncode != 0)
    finally:
        if proc.poll() is None:
            proc.kill()

    # protect_engine=1 must refuse to kill a vLLM process. comm comes from the
    # exec'd filename, so a symlink named vllm-* makes the listener look like one.
    with tempfile.TemporaryDirectory() as tmp:
        fake = os.path.join(tmp, "vllm-engine")
        os.symlink(os.path.realpath(sys.executable), fake)
        port = free_port()
        proc = listen(port, fake)
        try:
            r = sh(f"tcp_listen_kill {port} 1")
            check("protect: refuses to kill an engine", r.returncode != 0, r.stdout + r.stderr)
            check("protect: says why", "not killing" in r.stderr, r.stderr)
            check("protect: engine still alive", proc.poll() is None)
            r = sh(f"tcp_listen_kill {port} 0")
            check("no protect: kills the engine", r.returncode == 0 and gone(proc), r.stdout + r.stderr)
        finally:
            if proc.poll() is None:
                proc.kill()

    if FAILED:
        print(f"\n{len(FAILED)} tcp_listen check(s) failed")
        return 1
    print("\nall tcp_listen checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
