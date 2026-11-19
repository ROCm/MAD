"""MAD's reader of workload files (config/workload.py): inheritance, interpolation, refusals.

    python3 -m unittest discover -s scripts/common/config/tests
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import workload as W  # noqa: E402


class Workload(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.d = Path(self._d.name)

    def tearDown(self):
        self._d.cleanup()

    def put(self, name, text):
        p = self.d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return str(p)

    def test_defaults_merge_in_order_the_file_on_top(self):
        self.put("a.yaml", "version: 2\nmodel: {env: {X: a, A: a, L: [1, 2]}}\n")
        self.put("b.yaml", "version: 2\nmodel: {env: {X: b, B: b, L: [3]}}\n")
        f = self.put("w/f.yaml", "version: 2\ndefaults: [../a.yaml, ../b.yaml]\nmodel: {env: {B: f}}\n")
        self.assertEqual(W.env_of(W.load(f)), {"X": "b", "A": "a", "B": "f", "L": "[3]"})

    def test_interpolation_after_the_merge(self):
        self.put("base.yaml", "version: 2\nmodel: {env: {TP: '8', TAG: 'tp${model.env.TP}'}}\n")
        f = self.put("f.yaml", "version: 2\ndefaults: [base.yaml]\nmodel: {env: {TP: '2'}}\n")
        self.assertEqual(W.env_of(W.load(f))["TAG"], "tp2")

    def test_environment_interpolation_and_its_default(self):
        f = self.put("f.yaml", "version: 2\nmodel: {env: {A: '${oc.env:WL_TEST_A,dflt}', B: 'x-${oc.env:WL_TEST_B}'}}\n")
        os.environ["WL_TEST_B"] = "b"
        try:
            self.assertEqual(W.env_of(W.load(f)), {"A": "dflt", "B": "x-b"})
        finally:
            del os.environ["WL_TEST_B"]
        with self.assertRaisesRegex(W.WorkloadError, "WL_TEST_B is not set"):
            W.load(f)

    def test_benchmark_map_by_kind(self):
        f = self.put("f.yaml", "version: 2\nmodel: {env: {BENCHMARK_SCRIPT: niah, X: m}}\n"
                               "benchmark:\n  default: {env: {X: all}}\n  niah: {env: {W: n}}\n  sweep: {env: {W: s}}\n")
        data = W.load(f)
        self.assertEqual(W.env_of(data), {"BENCHMARK_SCRIPT": "niah", "X": "all", "W": "n"})
        self.assertEqual(W.env_of(data, "sweep")["W"], "s")

    def test_version_1_has_no_inheritance(self):
        f = self.put("f.yaml", "version: 1\nmodel: {env: {X: '${model.env.Y}'}}\nbenchmark: [{env: {B: '1'}}]\n")
        self.assertEqual(W.env_of(W.load(f)), {"X": "${model.env.Y}", "B": "1"})

    def test_refusals(self):
        cases = {
            "cycle": ("version: 2\ndefaults: [cycle.yaml]\n", "defaults cycle"),
            "missing": ("version: 2\ndefaults: [nowhere.yaml]\n", "not found"),
            "version": ("version: 3\n", "not supported"),
            "dangling": ("version: 2\nmodel: {env: {X: '${model.env.NOPE}'}}\n", "refers to nothing"),
            "site": ("version: 2\nsite: {partition: p}\n", "site:"),
            "serve": ("version: 2\nmodel: {serve: {prefill: [--x]}}\n", "model.serve"),
        }
        for name, (text, msg) in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(W.WorkloadError, msg):
                    W.load(self.put(f"{name}.yaml", text))

    def test_find_as_madengine_locates_it(self):
        self.assertIsNone(W.find("", str(self.d)))
        self.put("mad-config.yaml", "version: 2\n")
        self.assertEqual(W.find("", str(self.d)), str(self.d / "mad-config.yaml"))
        self.assertEqual(W.find("w/x.yaml", str(self.d)), str(self.d / "w/x.yaml"))
        self.assertEqual(W.find("/abs/x.yaml", str(self.d)), "/abs/x.yaml")

    def test_cli_env_is_nul_separated(self):
        f = self.put("f.yaml", "version: 2\nmodel: {env: {A: '1', B: 'two words'}}\n")
        out = subprocess.run([sys.executable, str(HERE.parent / "workload.py"), "env", f],
                             capture_output=True, text=True, check=True).stdout
        self.assertEqual(out, "A=1\0B=two words\0")
        bad = subprocess.run([sys.executable, str(HERE.parent / "workload.py"), "env", self.put("v.yaml", "version: 9\n")],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 1)


if __name__ == "__main__":
    unittest.main()
