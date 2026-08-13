#!/usr/bin/env python3
"""Agentic models shim for the vLLM disaggregated PD router.

The vLLM production PD router (vllm-router) serves /v1/chat/completions and
/v1/completions but returns 503 "No prefill servers available" on /v1/models
when workers register via MoRIIO service discovery (the HTTP worker registry
stays empty). The shared agentic harness (scripts/common/agentic_lib.sh) gates
readiness + served-model resolution on GET /v1/models, so it never starts.

This tiny shim (stdlib only, agentic path only) sits on a side port and:
  * GET /v1/models  -> 200 with the served model id, ONLY once the upstream
                       router answers GET /health 200 (so it doubles as the
                       readiness gate the harness expects).
  * GET /health     -> mirror upstream /health.
  * everything else -> stream-proxied verbatim to the upstream router
                       (POST /v1/chat/completions etc., SSE-safe).

Env:
  AGENTIC_SHIM_PORT      listen port (required)
  AGENTIC_SHIM_UPSTREAM  upstream router host:port (default 127.0.0.1:30000)
  AGENTIC_SHIM_MODEL     served model id to advertise on /v1/models (required)
"""
import http.client
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN_PORT = int(os.environ["AGENTIC_SHIM_PORT"])
UPSTREAM = os.environ.get("AGENTIC_SHIM_UPSTREAM", "127.0.0.1:30000")
MODEL = os.environ.get("AGENTIC_SHIM_MODEL", "")
PREFILL = os.environ.get("AGENTIC_SHIM_PREFILL", "").strip()  # host:port of prefill backend (diag only)
UP_HOST, UP_PORT = UPSTREAM.split(":")
UP_PORT = int(UP_PORT)
_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


def _health_ok(host, port):
    try:
        c = http.client.HTTPConnection(host, port, timeout=5)
        c.request("GET", "/health")
        r = c.getresponse()
        r.read()
        c.close()
        return r.status == 200
    except Exception:
        return False


def _upstream_health_ok():
    # Readiness signal: the production vllm-router serves GET /health (200 when
    # workers are registered). The MoRIIO toy proxy does NOT implement /health
    # (404), so fall back to the prefill backend's /health (a real vLLM OpenAI
    # server, 200 when the engine is up) when a PREFILL backend is configured.
    if _health_ok(UP_HOST, UP_PORT):
        return True
    if PREFILL:
        try:
            h, pt = PREFILL.split(":")
            return _health_ok(h, int(pt))
        except Exception:
            return False
    return False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            if _upstream_health_ok():
                self._send_json(200, {"object": "list", "data": [
                    {"id": MODEL, "object": "model", "owned_by": "vllm"}]})
            else:
                self._send_json(503, {"error": "router not ready"})
            return
        if self.path.rstrip("/") == "/health":
            self._send_json(200 if _upstream_health_ok() else 503, {"status": "ok"})
            return
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")

    def _replay_backend(self, method, body, headers):
        if not PREFILL:
            return
        try:
            h, pt = PREFILL.split(":")
            c = http.client.HTTPConnection(h, int(pt), timeout=60)
            c.request(method, self.path, body=body, headers=headers)
            r = c.getresponse(); b = r.read(); c.close()
            print(f"[agentic-shim][diag] backend {PREFILL} {self.path} -> {r.status}: "
                  f"{b[:800].decode('utf-8','replace')}", flush=True)
        except Exception as e:
            print(f"[agentic-shim][diag] backend replay failed: {e}", flush=True)

    def _proxy(self, method):
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
        try:
            conn = http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=3600)
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as e:
            self._send_json(502, {"error": f"upstream proxy failed: {e}"})
            return
        # Surface upstream error bodies (e.g. backend 400s) for diagnosis; these are
        # small non-streaming JSON responses, so read fully, log, and relay verbatim.
        if resp.status >= 400:
            err = resp.read()
            print(f"[agentic-shim] upstream {method} {self.path} -> {resp.status} "
                  f"(req_bytes={length}): {err[:600].decode('utf-8', 'replace')}", flush=True)
            if resp.status >= 500 and self.path.rstrip('/').endswith('/chat/completions'):
                self._replay_backend(method, body, headers)
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() in _HOP:
                    continue
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(err)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(err)
            conn.close()
            return
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() in _HOP:
                continue
            self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception:
            pass
        finally:
            conn.close()


if __name__ == "__main__":
    if not MODEL:
        print("[agentic-shim][ERROR] AGENTIC_SHIM_MODEL must be set", file=sys.stderr)
        sys.exit(2)
    srv = ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Handler)
    print(f"[agentic-shim] listening :{LISTEN_PORT} -> {UPSTREAM} (model={MODEL})", flush=True)
    srv.serve_forever()
