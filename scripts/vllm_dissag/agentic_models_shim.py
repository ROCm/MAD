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
PREFILL = os.environ.get("AGENTIC_SHIM_PREFILL", "").strip()  # host:port of prefill backend (readiness fallback)
UP_HOST, UP_PORT = UPSTREAM.split(":")
UP_PORT = int(UP_PORT)
_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


def _health_status(host, port):
    try:
        c = http.client.HTTPConnection(host, port, timeout=5)
        c.request("GET", "/health")
        r = c.getresponse()
        r.read()
        c.close()
        return r.status
    except Exception:
        return None


def _upstream_health_ok():
    # Readiness signal: the production vllm-router serves GET /health (200 when
    # workers are registered). The MoRIIO toy proxy does NOT implement /health
    # (404), so only then fall back to the prefill backend's /health (a real vLLM
    # OpenAI server, 200 when the engine is up) when a PREFILL backend is configured.
    # Any other router answer, or no answer, means not ready.
    status = _health_status(UP_HOST, UP_PORT)
    if status != 404 or not PREFILL:
        return status == 200
    try:
        h, pt = PREFILL.split(":")
        return _health_status(h, int(pt)) == 200
    except Exception:
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
        # The NIXL toy_proxy relays streamed SSE chunks but labels them
        # Content-Type: application/json, so aiperf never strips the "data:" SSE
        # framing and marks every request invalid. Force text/event-stream when the
        # client asked for a stream (JSON body "stream": true) OR the upstream is
        # already SSE; leave non-streaming JSON responses untouched.
        want_stream = False
        try:
            want_stream = bool(json.loads(body or b"{}").get("stream"))
        except Exception:
            want_stream = False
        up_ctype = (resp.getheader("Content-Type", "") or "").lower()
        force_sse = want_stream or "text/event-stream" in up_ctype
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() in _HOP:
                continue
            if force_sse and k.lower() == "content-type":
                continue
            self.send_header(k, v)
        if force_sse:
            self.send_header("Content-Type", "text/event-stream")
        # Chunked, so a relay that breaks off is missing the terminating chunk and the client sees
        # an incomplete body instead of a clean end of stream.
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                # read1, not read: read(n) blocks until n bytes arrive, which holds back SSE events.
                chunk = resp.read1(8192)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception as e:
            print(f"[agentic-shim] relay {method} {self.path} aborted: {e!r}", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    if not MODEL:
        print("[agentic-shim][ERROR] AGENTIC_SHIM_MODEL must be set", file=sys.stderr)
        sys.exit(2)
    srv = ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Handler)
    print(f"[agentic-shim] listening :{LISTEN_PORT} -> {UPSTREAM} (model={MODEL})", flush=True)
    srv.serve_forever()
