"""benchmark_accuracy.py against a stub OpenAI-compatible endpoint, through parse_to_csv --accuracy."""
import csv
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.dirname(HERE)


class Stub(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        if "boom" in prompt:
            self.send_response(500); self.end_headers(); return
        answer = "42" if "right" in prompt else "7"
        out = json.dumps({"choices": [{"message": {"content": f"The answer is {answer}"}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class Accuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def run_sets(self, sets):
        d = tempfile.mkdtemp()
        paths = []
        for name, items in sets.items():
            p = os.path.join(d, name + ".jsonl")
            with open(p, "w") as f:
                f.write("\n".join(json.dumps(i) for i in items))
            paths.append(p)
        env = dict(os.environ, ACC_URL=f"http://127.0.0.1:{self.srv.server_port}/v1/chat/completions",
                   ACC_MODEL="m", ACC_TASKS=",".join(paths), ACC_TIMEOUT="10")
        p = subprocess.run([sys.executable, os.path.join(DIR, "benchmark_accuracy.py")], env=env,
                           capture_output=True, text=True, timeout=60)
        log = os.path.join(d, "acc.log")
        open(log, "w").write(p.stdout)
        csvp = os.path.join(d, "perf.csv")
        subprocess.run([sys.executable, os.path.join(DIR, "parse_to_csv.py"), log, "--accuracy",
                        "--perf-csv", csvp, "--model-name", "M"], capture_output=True, check=True)
        return p.returncode, {r["metric"]: r for r in csv.DictReader(open(csvp))}

    def test_scores_and_failures(self):
        rc, rows = self.run_sets({
            "qa": [{"id": "1", "prompt": "right?", "answer": "42", "match": "contains"},
                   {"id": "2", "prompt": "wrong?", "answer": "42", "match": "contains"},
                   {"id": "3", "prompt": "right!", "answer": r"\b42\b", "match": "regex"},
                   {"id": "4", "prompt": "right.", "answer": "the answer is 42", "match": "exact"}],
            "broken": [{"id": "5", "prompt": "boom", "answer": "42"}]})
        self.assertEqual(rc, 1)                                           # a failed request fails the run
        qa = rows["accuracy (task=qa n=4)"]
        self.assertEqual((qa["performance"], qa["status"]), ("0.7500", "SUCCESS"))   # a low score is a measurement
        self.assertEqual(rows["accuracy (task=broken n=1)"]["status"], "FAILURE")

    def test_the_bundled_smoke_set_is_well_formed(self):
        with open(os.path.join(DIR, "accuracy_tasks", "arith_smoke.jsonl")) as f:
            items = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(items), 20)
        self.assertTrue(all({"id", "prompt", "answer", "match"} <= set(i) for i in items))


if __name__ == "__main__":
    unittest.main()
