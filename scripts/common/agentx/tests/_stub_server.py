#!/usr/bin/env python3
"""Minimal stdlib HTTP stub of an sglang API worker for offline ctx-resolve tests.

Usage: python3 _stub_server.py <port> [empty]

When the 2nd arg is "empty" the window fields are omitted so the resolver's
fail-fast path can be exercised.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

EMPTY = len(sys.argv) > 2 and sys.argv[2] == "empty"


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            if EMPTY:
                self._send({"data": [{"id": "stub"}]})
            else:
                self._send({"data": [{"id": "stub", "max_model_len": 131072}]})
        elif self.path == "/get_server_info":
            if EMPTY:
                self._send({"server_args": {}})
            else:
                self._send({"server_args": {"context_length": 131072}})
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1])
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
