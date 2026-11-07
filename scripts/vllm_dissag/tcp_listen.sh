#!/bin/bash
# TCP LISTEN helpers using /proc only (no ss/fuser).
# The vLLM image has neither; cleanup was a no-op and ZMQ 36367
# stayed bound. Source this file; do not assume psmisc or iproute2.
#
# /proc/net/tcp is the host network when docker uses --network host, so we can
# *see* a leftover LISTEN. PIDs come from this mount's /proc/*/fd, so we can
# only *kill* occupants in this PID namespace (this container). A previous
# job's process is cleared by the host `docker stop` in run_xPyD_models.slurm
# before this container starts.

# /proc/net/tcp documented columns 5–8 (tx_queue, rx_queue, tr, tm->when) are
# colon-joined in the file, so awk fields are: $8=uid $9=timeout $10=inode.
# Documented column 12 is awk $10. $12 is the sk pointer (hex) — do not use it.
tcp_listen_inodes() {
    local hex
    hex=$(printf '%04X' "$1")
    awk -v p="$hex" 'NR > 1 {
        n = split($2, a, ":")
        if (n >= 2 && toupper(a[n]) == p && toupper($4) == "0A") print $10
    }' /proc/net/tcp /proc/net/tcp6 2>/dev/null | grep -E '^[0-9]+$' | sort -u
}

tcp_listen_busy() {
    [[ -n "$(tcp_listen_inodes "$1")" ]]
}

tcp_listen_pids() {
    local inodes
    inodes=$(tcp_listen_inodes "$1" | tr '\n' ' ')
    [[ -z "${inodes// }" ]] && return 0
    python3 - "$inodes" <<'PY'
import os, sys
want = {i for i in sys.argv[1].split() if i.isdigit()}
found = set()
try:
    pids = os.listdir("/proc")
except OSError:
    pids = []
for pid in pids:
    if not pid.isdigit():
        continue
    fd = f"/proc/{pid}/fd"
    try:
        names = os.listdir(fd)
    except OSError:
        continue
    for n in names:
        try:
            target = os.readlink(f"{fd}/{n}")
        except OSError:
            continue
        if target.startswith("socket:[") and target[8:-1] in want:
            found.add(int(pid))
            break
for pid in sorted(found):
    print(pid)
PY
}

tcp_listen_describe() {
    local port="$1" inode pid comm
    echo "[tcp_listen] port=${port} inodes=$(tcp_listen_inodes "$port" | tr '\n' ' ')"
    for pid in $(tcp_listen_pids "$port"); do
        comm=$(ps -p "${pid}" -o comm= 2>/dev/null || echo "?")
        echo "[tcp_listen]   pid=${pid} comm=${comm}"
    done
}

# All /proc/net/{tcp,tcp6} rows for this port (any state), not just LISTEN.
# ZMQ EADDRINUSE then tcp_listen_inodes empty — LISTEN-only scan missed it.
tcp_listen_dump_port() {
    local hex
    hex=$(printf '%04X' "$1")
    echo "[tcp_listen] dump port=$1 hex=$hex (/proc/net/tcp{,6} any state)"
    awk -v p="$hex" 'NR > 1 {
        n = split($2, a, ":")
        if (n >= 2 && toupper(a[n]) == p)
            print FILENAME, $2, "st="$4, "uid="$8, "inode="$10
    }' /proc/net/tcp /proc/net/tcp6 2>/dev/null || true
}

tcp_listen_bind_probe() {
    local port="$1"
    python3 - "$port" <<'PY'
import errno
import socket
import sys

port = int(sys.argv[1])
ok = True
# ZMQ discovery is IPv4 0.0.0.0 (IPV6 off). [::] EADDRINUSE still means the
# port is taken on dual-stack. Missing IPv6 is not a failure.
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    print(f"[tcp_listen] bind_ok 0.0.0.0:{port}")
    s.close()
except OSError as e:
    print(f"[tcp_listen] bind_fail 0.0.0.0:{port} {e}")
    ok = False
try:
    s6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    s6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    s6.bind(("::", port))
    print(f"[tcp_listen] bind_ok [::]:{port}")
    s6.close()
except OSError as e:
    if e.errno == errno.EADDRINUSE:
        print(f"[tcp_listen] bind_fail [::]:{port} {e}")
        ok = False
    else:
        print(f"[tcp_listen] bind_skip [::]:{port} {e}")
sys.exit(0 if ok else 1)
PY
}

# Kill LISTEN occupants. With protect_engine=1, refuse to kill vllm/EngineCore.
# Returns 0 if the port is no longer LISTEN, 1 if still busy.
tcp_listen_kill() {
    local port="$1"
    local protect="${2:-0}"
    local pid comm killed=0
    for pid in $(tcp_listen_pids "$port"); do
        comm=$(ps -p "${pid}" -o comm= 2>/dev/null || true)
        if [[ "$protect" == "1" ]]; then
            case " ${comm} " in
                *"vllm"*|*"EngineCore"*|*"VLLM"*)
                    echo "Error: tcp ${port} held by engine ${comm} pid=${pid}; not killing" >&2
                    tcp_listen_describe "$port" >&2
                    return 1
                    ;;
            esac
        fi
        echo "[tcp_listen] kill -9 pid=${pid} (${comm}) tcp ${port}"
        kill -9 "${pid}" 2>/dev/null || true
        killed=1
    done
    if [[ "$killed" == "1" ]]; then
        sleep 0.2
    fi
    if tcp_listen_busy "$port"; then
        echo "[tcp_listen] tcp ${port} still LISTEN after kill (occupant outside this PID ns?)" >&2
        tcp_listen_describe "$port" >&2
        return 1
    fi
    return 0
}
