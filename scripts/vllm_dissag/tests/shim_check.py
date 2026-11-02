#!/usr/bin/env python3
"""Offline checks for agentic_models_shim.py against stdlib fake router/prefill servers.

Usage: shim_check.py <path to agentic_models_shim.py>
Prints one PASS/FAIL line per assertion; exits 0 iff all pass.
"""
import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SHIM = sys.argv[1]
results = []


def check(ok, name, detail=""):
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" ({detail})" if detail else ""), flush=True)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


def fake(health=200, post=None):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(health)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            post(self)
    return serve(H)


def chunk(w, data):
    w.write(b"%x\r\n%s\r\n" % (len(data), data))
    w.flush()


def sse_headers(h):
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.send_header("Transfer-Encoding", "chunked")
    h.end_headers()


class Shim:
    def __init__(self, upstream, prefill=""):
        self.port = free_port()
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log")
        env = dict(os.environ, AGENTIC_SHIM_PORT=str(self.port), AGENTIC_SHIM_UPSTREAM=f"127.0.0.1:{upstream}",
                   AGENTIC_SHIM_MODEL="m", AGENTIC_SHIM_PREFILL=f"127.0.0.1:{prefill}" if prefill else "")
        self.p = subprocess.Popen([sys.executable, SHIM], env=env, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", self.port)).close()
                return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("shim did not start")

    def close(self):
        self.p.terminate()
        self.p.wait()

    def output(self):
        self.log.seek(0)
        return self.log.read()


def post_chat(port, timeout):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = json.dumps({"model": "m", "messages": [], "stream": True})
    c.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
    return c, c.getresponse()


# 1. SSE events are relayed as they arrive, not in 8 KB blocks.
N, ITL, MSG = 100, 0.012, 240


def stream_events(h):
    sse_headers(h)
    for i in range(N):
        d = json.dumps({"i": i, "t": time.perf_counter(), "pad": "x" * (MSG - 60)})
        chunk(h.wfile, f"data: {d}\n\n".encode())
        time.sleep(ITL)
    chunk(h.wfile, b"data: [DONE]\n\n")
    h.wfile.write(b"0\r\n\r\n")
    h.wfile.flush()


print("=== relay latency (100 SSE events, 240 B every 12 ms) ===")
shim = Shim(fake(post=stream_events))
t0 = time.perf_counter()
c, r = post_chat(shim.port, 30)
lags, first, buf = [], None, b""
while True:
    d = r.read1(65536)
    if not d:
        break
    now = time.perf_counter()
    buf += d
    while b"\n\n" in buf:
        ev, buf = buf.split(b"\n\n", 1)
        if b'"i":' in ev:
            m = json.loads(ev.split(b"data: ", 1)[1])
            lags.append(now - m["t"])
            first = first if first is not None else now - t0
shim.close()
lags.sort()
check(len(lags) == N, "all events relayed", f"{len(lags)}/{N}")
check(first is not None and first < 0.05, "first event < 50 ms", f"{first * 1e3:.0f} ms" if first else "none")
check(bool(lags) and lags[len(lags) // 2] < 0.005, "per-event lag p50 < 5 ms",
      f"{lags[len(lags) // 2] * 1e3:.1f} ms" if lags else "none")

# 2. An upstream 5xx reaches the client immediately (no replay to the prefill server).
print("=== upstream 500 relayed immediately ===")


def fail_500(h):
    b = b'{"error":"Decode request failed"}'
    h.send_response(500)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(b)))
    h.end_headers()
    h.wfile.write(b)


def slow_prefill(h):
    sse_headers(h)
    try:
        for _ in range(20):
            chunk(h.wfile, b"data: {}\n\n")
            time.sleep(0.5)
        h.wfile.write(b"0\r\n\r\n")
    except (BrokenPipeError, ConnectionResetError):
        pass


shim = Shim(fake(post=fail_500), prefill=fake(post=slow_prefill))
t0 = time.perf_counter()
try:
    c, r = post_chat(shim.port, 5)
    r.read()
    status, dt = r.status, time.perf_counter() - t0
except socket.timeout:
    status, dt = None, time.perf_counter() - t0
shim.close()
check(status == 500, "client gets the 500", f"status={status}")
check(dt < 1.0, "500 within 1 s (prefill reply would take 10 s)", f"{dt:.2f} s")

# 3. A stream that breaks off upstream is not delivered as a complete one.
print("=== truncated upstream stream ===")


def truncated(h):
    sse_headers(h)
    for i in range(3):
        chunk(h.wfile, f"data: {{\"i\": {i}}}\n\n".encode())
    h.wfile.flush()
    h.connection.shutdown(socket.SHUT_RDWR)


shim = Shim(fake(post=truncated))
try:
    c, r = post_chat(shim.port, 5)
    r.read()
    outcome = "complete"
except http.client.IncompleteRead:
    outcome = "incomplete"
except Exception as e:
    outcome = repr(e)
time.sleep(0.2)
shim.close()
check(outcome == "incomplete", "client sees an incomplete body", outcome)
check("aborted" in shim.output(), "shim logs the aborted relay")

# 4. Readiness follows the router; prefill fallback only when the router has no /health (404).
print("=== readiness (/v1/models) ===")
down = free_port()
for router_health, prefill_health, want in ((200, 200, 200), (200, None, 200), (503, 200, 503), (503, None, 503),
                                            (None, 200, 503), (None, None, 503), (404, 200, 200), (404, None, 503)):
    up = fake(health=router_health, post=fail_500) if router_health else down
    pf = fake(health=prefill_health, post=fail_500) if prefill_health else down
    shim = Shim(up, prefill=pf)
    c = http.client.HTTPConnection("127.0.0.1", shim.port, timeout=15)
    c.request("GET", "/v1/models")
    got = c.getresponse().status
    shim.close()
    check(got == want, f"router={router_health or 'down'} prefill={prefill_health or 'down'} -> {want}", f"got {got}")

sys.exit(0 if all(results) else 1)
