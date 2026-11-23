#!/usr/bin/env python3
"""A round-robin HTTP router over independent replicas (colocated REPLICAS=1 mode).

Listens on --port and forwards each request -- method, path, headers, body -- to the next
backend in turn, streaming the response back, so the benchmarks and the accuracy harness, which
talk to one endpoint, measure the whole replica fleet. GET /health answers 200 when every backend
does. Standard library only, as the containers have it.

    replica_router.py --port 8000 --backends http://10.0.0.1:8001,http://10.0.0.1:8002,...
"""
import argparse
import http.client
import http.server
import itertools
import threading
import urllib.parse

HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization"}


def make_handler(backends):
    cycle, lock = itertools.cycle(backends), threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _forward(self):
            if self.path == "/health":
                ok = all(self._ok(b) for b in backends)
                self.send_response(200 if ok else 503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            with lock:
                target = next(cycle)
            u = urllib.parse.urlsplit(target)
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            conn = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
            conn.request(self.command, self.path, body,
                         {k: v for k, v in self.headers.items() if k.lower() not in HOP | {"host"}})
            r = conn.getresponse()
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in HOP and k.lower() != "content-length":
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            conn.close()

        @staticmethod
        def _ok(b):
            u = urllib.parse.urlsplit(b)
            try:
                c = http.client.HTTPConnection(u.hostname, u.port, timeout=5)
                c.request("GET", "/health")
                return c.getresponse().status == 200
            except OSError:
                return False

        do_GET = do_POST = _forward

        def log_message(self, *a):
            pass

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--backends", required=True)
    a = ap.parse_args(argv)
    backends = [b for b in a.backends.split(",") if b]
    http.server.ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(backends)).serve_forever()


if __name__ == "__main__":
    main()
