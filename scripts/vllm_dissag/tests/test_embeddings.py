"""benchmark_embeddings.py against a stub /v1/embeddings, through parse_to_csv --embeddings."""
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


class Stub(http.server.BaseHTTPRequestHandler):
    short = False

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = len(body["input"]) - (1 if Stub.short else 0)
        out = json.dumps({"data": [{"embedding": [0.0, 1.0]} for _ in range(n)]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class Embeddings(unittest.TestCase):
    def run_bench(self):
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            env = dict(os.environ, EMB_URL=f"http://127.0.0.1:{srv.server_port}/v1/embeddings", EMB_MODEL="m",
                       EMB_CONCURRENCY="1,4", EMB_REQUESTS="12", EMB_BATCH="3", EMB_WORDS="8")
            p = subprocess.run([sys.executable, os.path.join(DIR, "benchmark_embeddings.py")], env=env,
                               capture_output=True, text=True, timeout=60)
        finally:
            srv.shutdown()
        d = tempfile.mkdtemp()
        log, out = os.path.join(d, "e.log"), os.path.join(d, "perf.csv")
        open(log, "w").write(p.stdout)
        subprocess.run([sys.executable, os.path.join(DIR, "parse_to_csv.py"), log, "--embeddings",
                        "--perf-csv", out, "--model-name", "M"], check=True, capture_output=True)
        return p.returncode, list(csv.DictReader(open(out)))

    def test_rows_per_concurrency(self):
        Stub.short = False
        rc, rows = self.run_bench()
        self.assertEqual(rc, 0)
        self.assertEqual([r["metric"].split(" p50")[0] for r in rows],
                         ["embeddings inputs/s (concurrency=1", "embeddings inputs/s (concurrency=4"])
        self.assertTrue(all(r["status"] == "SUCCESS" and float(r["performance"]) > 0 for r in rows))

    def test_a_wrong_number_of_embeddings_is_a_failure(self):
        Stub.short = True
        rc, rows = self.run_bench()
        self.assertEqual(rc, 1)
        self.assertTrue(all(r["status"] == "FAILURE" for r in rows))


if __name__ == "__main__":
    unittest.main()
