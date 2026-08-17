#!/usr/bin/env python3
"""Minimal stdlib stub of a vLLM `vllm serve` OpenAI server for offline tests.

Usage: python3 _stub_server.py <port> [empty]

Serves GET /v1/models with a single ModelCard. Normally the card advertises
max_model_len=131072; when the 2nd arg is "empty" the field is omitted (to
exercise the resolver's fail-fast path). Any other path returns 404.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1])
EMPTY = len(sys.argv) > 2 and sys.argv[2] == "empty"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            card = {"id": "stub"}
            if not EMPTY:
                card["max_model_len"] = 131072
            body = json.dumps({"data": [card]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
