"""Replica serving (REPLICAS=1): the per-node servers a node runs, and the router in front of them."""
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MULTI = os.path.dirname(HERE)
SHARED = os.path.join(os.path.dirname(MULTI), "vllm_dissag")


def backend(name):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            out = json.dumps({"from": name}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Router(unittest.TestCase):
    def test_round_robin_and_health(self):
        a, b = backend("a"), backend("b")
        port = free_port()
        p = subprocess.Popen([sys.executable, os.path.join(SHARED, "replica_router.py"), "--port", str(port),
                              "--backends", f"http://127.0.0.1:{a.server_port},http://127.0.0.1:{b.server_port}"])
        try:
            for _ in range(50):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); break
                except OSError:
                    time.sleep(0.1)
            seen = [json.load(urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/completions", b"{}", {"Content-Type": "application/json"})))["from"]
                for _ in range(4)]
            self.assertEqual(seen, ["a", "b", "a", "b"])
            self.assertEqual(urllib.request.urlopen(f"http://127.0.0.1:{port}/health").status, 200)
        finally:
            p.terminate(); a.shutdown(); b.shutdown()


class DryRun(unittest.TestCase):
    def test_a_node_runs_gpus_over_tp_servers_on_their_own_gpus_and_ports(self):
        env = {"PATH": os.environ["PATH"], "HOME": "/tmp", "DRY_RUN": "1", "REPLICAS": "1", "TP_SIZE": "2",
               "PP_SIZE": "1", "GPUS_PER_NODE": "8", "MODEL_PATH": "/m/M", "MODEL_NAME": "M", "NODE_RANK": "0",
               "NNODES": "1", "MASTER_ADDR": "10.0.0.1", "IPADDRS": "10.0.0.1", "SERVE_PORT": "8000",
               "SHARED_DIR": SHARED, "ENABLE_EP": "0"}
        out = subprocess.run(["bash", os.path.join(MULTI, "serve_colocated.sh")], env=env,
                             capture_output=True, text=True, timeout=60).stdout
        heads = [l for l in out.splitlines() if l.startswith("===DRYRUN replica")]
        self.assertEqual([h.split("devices=")[1] for h in heads], ["0,1", "2,3", "4,5", "6,7"])
        self.assertIn("8004", out)
        self.assertNotIn("--nnodes", out)


if __name__ == "__main__":
    unittest.main()
