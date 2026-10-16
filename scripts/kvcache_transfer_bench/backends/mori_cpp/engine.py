"""
MoRI-IO C++ benchmark (bench_engine) command line, shared by both nodes.
"""

import fcntl
import os
import socket as sock
import struct
import subprocess
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.mori_settings import WORKLOAD_FLAGS
from common.utils import resolve_hostname

BOOTSTRAP_PORT = 18515  # bench_engine's engines also listen on the next two ports
TIMEOUT_S = 1800


def engine_path() -> str:
    default = os.path.join(os.environ.get("MORI_ROOT", "/sgl-workspace/mori"),
                           "build_bench", "tests", "cpp", "bench_engine")
    return os.environ.get("MORI_IO_BENCH_ENGINE_BIN", default)


def require_engine() -> str:
    path = engine_path()
    if not os.access(path, os.X_OK):
        print(f"ERROR: MoRI-IO bench_engine not found: {path} "
              "(build the image with INSTALL_MORI_CPP_BENCH=1)", file=sys.stderr)
        sys.exit(1)
    return path


def _ifname() -> str:
    return os.environ.get("MORI_SOCKET_IFNAME") or os.environ.get("GLOO_SOCKET_IFNAME", "")


def local_ip() -> str:
    """IPv4 address of the bootstrap interface, else of this host's name."""
    ifname = _ifname()
    if ifname:
        s = sock.socket(sock.AF_INET, sock.SOCK_DGRAM)
        try:
            packed = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", ifname[:15].encode()))  # SIOCGIFADDR
            return sock.inet_ntoa(packed[20:24])
        except OSError:
            pass
        finally:
            s.close()
    return resolve_hostname(sock.gethostname())


def engine_env() -> dict:
    env = os.environ.copy()
    if _ifname():
        env.setdefault("MORI_SOCKET_IFNAME", _ifname())
    return env


def command(rank: int, master_ip: str, self_ip: str, port: int,
            start_size: int, end_size: int) -> List[str]:
    """Rank 0 is the MoRI initiator (drives the writes, prints the table), rank 1 the target."""
    return [
        require_engine(),
        "--rank", str(rank),
        "--master-ip", master_ip,
        "--self-ip", self_ip,
        "--port", str(port),
        *WORKLOAD_FLAGS,
        "--all",
        "--sweep-start-size", str(start_size),
        "--sweep-max-size", str(end_size),
    ]


def _text(stream) -> str:
    if isinstance(stream, bytes):
        return stream.decode(errors="replace")
    return stream or ""


def run(cmd: List[str]):
    """Run bench_engine; returns (exit code, stdout, stderr). Exit code 124 on timeout."""
    print(f"Command: {' '.join(cmd)}\n", flush=True)
    try:
        ret = subprocess.run(cmd, env=engine_env(), capture_output=True, text=True, timeout=TIMEOUT_S)
        rc, out, err = ret.returncode, ret.stdout, ret.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = 124, _text(e.stdout), _text(e.stderr) + f"\nbench_engine timed out after {TIMEOUT_S}s"
    if out:
        print(out, flush=True)
    if err:
        print(err, file=sys.stderr, flush=True)
    return rc, out, err
