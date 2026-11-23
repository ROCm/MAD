"""benchmark_lora.py against a stub chat endpoint, through parse_to_csv --lora."""
import csv
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEN = []


class Stub(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        model = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["model"]
        SEEN.append(model)
        code = 404 if model == "missing" else 200
        out = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        self.send_response(code); self.send_header("Content-Length", str(len(out))); self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class Lora(unittest.TestCase):
    def run_bench(self, adapters):
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            env = dict(os.environ, LORA_URL=f"http://127.0.0.1:{srv.server_port}/v1/chat/completions",
                       LORA_ADAPTERS=adapters, LORA_CONCURRENCY="2", LORA_REQUESTS="6")
            p = subprocess.run([sys.executable, os.path.join(DIR, "benchmark_lora.py")], env=env,
                               capture_output=True, text=True, timeout=60)
        finally:
            srv.shutdown()
        d = tempfile.mkdtemp()
        log, out = os.path.join(d, "l.log"), os.path.join(d, "perf.csv")
        open(log, "w").write(p.stdout)
        subprocess.run([sys.executable, os.path.join(DIR, "parse_to_csv.py"), log, "--lora", "--perf-csv", out,
                        "--model-name", "M"], check=True, capture_output=True)
        return p.returncode, list(csv.DictReader(open(out)))

    def test_requests_spread_over_the_adapters(self):
        SEEN.clear()
        rc, rows = self.run_bench("sql,chat,code")
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(SEEN), sorted(["sql", "chat", "code"] * 2))
        self.assertTrue(rows[0]["metric"].startswith("lora requests/s (adapters=3 concurrency=2"))
        self.assertEqual(rows[0]["status"], "SUCCESS")

    def test_an_adapter_the_server_lacks_is_a_failure(self):
        rc, rows = self.run_bench("sql,missing")
        self.assertEqual((rc, rows[0]["status"]), (1, "FAILURE"))


if __name__ == "__main__":
    unittest.main()
