#!/usr/bin/env python3
"""In-container node-IP rendezvous for disaggregated P/D launchers.

madengine forwards MASTER_ADDR/NODE_RANK/NNODES into each container but NOT the
rank-ordered node IP list. This helper rebuilds that list with stdlib only:

  rank 0  -> tiny TCP server: collects {rank: ip} from peers, then broadcasts
             the comma-separated, rank-ordered IP list back to everyone.
  rank>0  -> connects to MASTER_ADDR, reports its own IP, then blocks until
             rank 0 broadcasts the full list.

On success the rank-ordered "ip0,ip1,..." list is printed to stdout (exit 0);
on failure an empty line is printed (exit 2), so callers can use
`out="$(ip_rendezvous.py ... || true)"` and test for emptiness.

NOTE: this file is intentionally duplicated under scripts/sglang_disagg/ and
scripts/vllm_dissag/ (each model packages its own script dir into the image).
Keep both copies identical.

Usage:
  ip_rendezvous.py <rank> <nnodes> <host_ip> <master_addr> <port> <job_id>
Env:
  IP_SYNC_TIMEOUT  total rendezvous budget in seconds (default 1800).
"""
import json
import os
import socket
import sys
import time


def recv_line(conn):
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf.split(b"\n", 1)[0].decode("utf-8", "replace") if buf else None


def serve(nnodes, host_ip, port, token, deadline):
    """rank-0 path: collect peer IPs, broadcast the rank-ordered list."""
    by_rank = {0: host_ip}
    # rank -> connection. Workers reconnect while we wait for slow nodes, so we
    # keep only the latest socket per rank (closing the stale one) to avoid
    # broadcasting over half-open sockets.
    conns = {}
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(max(8, nnodes + 2))
    srv.settimeout(1.0)
    try:
        while len(by_rank) < nnodes and time.time() < deadline:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            conn.settimeout(5.0)
            line = recv_line(conn)
            if not line:
                conn.close()
                continue
            try:
                msg = json.loads(line)
            except Exception:
                conn.close()
                continue
            if msg.get("token") != token:
                conn.close()
                continue
            wrank = int(msg.get("rank", -1))
            wip = str(msg.get("ip", "")).strip()
            if 0 <= wrank < nnodes and wip:
                old = conns.get(wrank)
                if old is not None:
                    try:
                        old.close()
                    except Exception:
                        pass
                by_rank[wrank] = wip
                conns[wrank] = conn
            else:
                conn.close()
        if len(by_rank) != nnodes:
            return None
        ipaddrs = ",".join(by_rank[i] for i in range(nnodes))
        payload = (ipaddrs + "\n").encode()
        # Best-effort broadcast: a single dead/stale peer socket must not abort
        # the whole rendezvous, so swallow per-conn send errors.
        for conn in conns.values():
            try:
                conn.sendall(payload)
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        return ipaddrs
    finally:
        srv.close()


def report(rank, host_ip, master_addr, port, token, deadline):
    """rank>0 path: report own IP, block on recv until rank-0 broadcasts."""
    payload = (json.dumps({"token": token, "rank": rank, "ip": host_ip}) + "\n").encode()
    while time.time() < deadline:
        try:
            sock = socket.create_connection((master_addr, port), timeout=3.0)
            sock.sendall(payload)
            # rank-0 broadcasts once, only after every node has reported, which
            # can be far longer than a few seconds when peers load the image at
            # different times. Block on recv for the remaining deadline so we
            # stay connected and never miss that one-shot broadcast.
            remaining = max(1.0, deadline - time.time())
            sock.settimeout(remaining)
            line = recv_line(sock)
            sock.close()
            if line:
                return line.strip()
        except Exception:
            time.sleep(1.0)
    return None


def main(argv):
    rank = int(argv[1])
    nnodes = int(argv[2])
    host_ip = argv[3]
    master_addr = argv[4]
    port = int(argv[5])
    job_id = argv[6]
    token = f"JOB{job_id}"
    timeout_s = int(os.environ.get("IP_SYNC_TIMEOUT", "1800"))
    deadline = time.time() + timeout_s
    if rank == 0:
        result = serve(nnodes, host_ip, port, token, deadline)
    else:
        result = report(rank, host_ip, master_addr, port, token, deadline)
    if result:
        print(result)
        return 0
    print("")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
