"""
Socket-based synchronization for KV cache benchmarks.

Provides SocketSyncTarget (target/server side) and SocketSyncInitiator
(initiator/client side) that coordinate benchmark phases over a persistent
TCP connection using newline-delimited JSON messages.

Used by all backends for inter-node coordination during benchmarks.

The sync TCP port defaults to 9999. Override via the ``port`` argument or
``--sync-port`` on benchmark CLIs (RIXL/Mori/Mooncake target/initiator scripts).
"""

import json
import socket
import sys
import time
from typing import Optional

DEFAULT_SYNC_PORT = 9999


def get_sync_port(port: Optional[int] = None) -> int:
    """Return ``port`` if set, otherwise ``DEFAULT_SYNC_PORT``."""
    return DEFAULT_SYNC_PORT if port is None else port


class SocketSyncTarget:
    """Target-side sync: listens for one initiator, then exchanges messages."""

    def __init__(self, port: Optional[int] = None):
        self._port = get_sync_port(port)
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("0.0.0.0", self._port))
        self._server.listen(1)
        self._conn = None
        self._rfile = None
        self._wfile = None

    def wait_for_connection(self, timeout=180):
        """Block until the initiator connects."""
        print(f"[sync-socket] Target listening on port {self._port}, waiting for initiator...")
        self._server.settimeout(timeout)
        try:
            self._conn, addr = self._server.accept()
        except socket.timeout:
            print(f"[sync-socket] Timed out after {timeout}s waiting for initiator", file=sys.stderr)
            sys.exit(1)
        self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._rfile = self._conn.makefile("r")
        self._wfile = self._conn.makefile("w")
        print(f"[sync-socket] Initiator connected from {addr}")

    def signal_target_ready(self, size, port=None, **extra):
        msg = {"event": "target_ready", "size": size}
        if port is not None:
            msg["port"] = port
        msg.update(extra)
        self._send(msg)
        print(f"[sync-socket] Sent target_ready for size {size}")

    def wait_for_initiator_done(self, size, timeout=180):
        print(f"[sync-socket] Waiting for initiator_done (size {size})...")
        msg = self._recv(timeout)
        if msg is None:
            print(f"[sync-socket] Timed out waiting for initiator_done (size {size})", file=sys.stderr)
            sys.exit(1)
        if msg.get("event") != "initiator_done" or msg.get("size") != size:
            print(f"[sync-socket] Unexpected message: {msg}", file=sys.stderr)
            sys.exit(1)
        print(f"[sync-socket] Initiator done with size {size}")

    def signal_target_done(self, size):
        self._send({"event": "target_done", "size": size})
        print(f"[sync-socket] Sent target_done for size {size}")

    def cleanup(self, start_size, end_size):
        pass

    def close(self):
        for f in (self._rfile, self._wfile):
            if f:
                try:
                    f.close()
                except Exception:
                    pass
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        try:
            self._server.close()
        except Exception:
            pass

    def _send(self, msg):
        self._wfile.write(json.dumps(msg) + "\n")
        self._wfile.flush()

    def _recv(self, timeout=180):
        self._conn.settimeout(timeout)
        try:
            line = self._rfile.readline()
        except socket.timeout:
            return None
        if not line:
            return None
        return json.loads(line)


class SocketSyncInitiator:
    """Initiator-side sync: connects to target, then exchanges messages."""

    def __init__(self, host, port: Optional[int] = None):
        self._host = host
        self._port = get_sync_port(port)
        self._sock = None
        self._rfile = None
        self._wfile = None

    def connect(self, timeout=180):
        """Retry connecting to the target's sync server until it is up."""
        print(f"[sync-socket] Connecting to target {self._host}:{self._port}...")
        deadline = time.time() + timeout
        interval = 0.5
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5)
                s.connect((self._host, self._port))
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = s
                self._rfile = s.makefile("r")
                self._wfile = s.makefile("w")
                print(f"[sync-socket] Connected to target")
                return
            except (OSError, socket.error):
                s.close()
                if time.time() >= deadline:
                    print(f"[sync-socket] Timed out after {timeout}s connecting to target", file=sys.stderr)
                    sys.exit(1)
                time.sleep(interval)
                interval = min(interval * 1.5, 5)

    def wait_for_target_ready(self, size, timeout=180):
        """Block until target sends target_ready. Returns the full message dict."""
        print(f"[sync-socket] Waiting for target_ready (size {size})...")
        msg = self._recv(timeout)
        if msg is None:
            print(f"[sync-socket] Timed out waiting for target_ready (size {size})", file=sys.stderr)
            sys.exit(1)
        if msg.get("event") != "target_ready" or msg.get("size") != size:
            print(f"[sync-socket] Unexpected message: {msg}", file=sys.stderr)
            sys.exit(1)
        print(f"[sync-socket] Target ready for size {size}")
        return msg

    def send_initiator_done(self, size):
        self._send({"event": "initiator_done", "size": size})
        print(f"[sync-socket] Sent initiator_done for size {size}")

    def wait_for_target_done(self, size, timeout=180):
        print(f"[sync-socket] Waiting for target_done (size {size})...")
        msg = self._recv(timeout)
        if msg is None:
            print(f"[sync-socket] Timed out waiting for target_done (size {size})", file=sys.stderr)
            sys.exit(1)
        if msg.get("event") != "target_done" or msg.get("size") != size:
            print(f"[sync-socket] Unexpected message: {msg}", file=sys.stderr)
            sys.exit(1)
        print(f"[sync-socket] Target done with size {size}")

    def close(self):
        for f in (self._rfile, self._wfile):
            if f:
                try:
                    f.close()
                except Exception:
                    pass
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    def _send(self, msg):
        self._wfile.write(json.dumps(msg) + "\n")
        self._wfile.flush()

    def _recv(self, timeout=180):
        self._sock.settimeout(timeout)
        try:
            line = self._rfile.readline()
        except socket.timeout:
            return None
        if not line:
            return None
        return json.loads(line)
