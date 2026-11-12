"""nan-doctor-run.sh starts only nan-doctor's two launchers, under the GPU lock."""

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / ".claude" / "skills" / "mad-common" / "nan-doctor-run.sh"

STUB = """#!/bin/bash
echo "STUB $(basename "$0") CMD=${CMD:-} IMAGE=${IMAGE:-} CAPSULE=${CAPSULE:-} CAPSULE_ITER_COUNT=${CAPSULE_ITER_COUNT:-}"
exit ${STUB_EXIT:-0}
"""


class NanDoctorRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.mad = t / "mad"
        self.mad.mkdir()
        (self.mad / "models.json").write_text("[]")
        self.home = t / "nan-doctor"
        (self.home / "scripts" / "internal").mkdir(parents=True)
        for name in ("docker_replay_nan_op.sh", "replay_capsule_node_docker.sh"):
            (self.home / "scripts" / "internal" / name).write_text(STUB)
        self.env = os.environ.copy()
        self.env["MAD_GPU_LOCK"] = str(t / "gpu.lock")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def helper(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(HELPER), *args],
            cwd=self.mad,
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )

    def wait_done(self, run_id: str) -> str:
        for _ in range(50):
            out = self.helper("status", run_id).stdout
            if "=succeeded" in out or "=failed" in out:
                return out
            time.sleep(0.1)
        self.fail("run did not finish")

    def test_replay_runs_the_launcher_with_its_env(self) -> None:
        r = self.helper("start", "r1", str(self.home), "replay", "CMD=python3 x.py --a 1", "IMAGE=ci-x")
        self.assertEqual(r.returncode, 0, r.stderr)
        out = self.wait_done("r1")
        self.assertIn("NAN_DOCTOR_RUN_STATUS=succeeded exit=0", out.splitlines()[0])
        self.assertIn("STUB docker_replay_nan_op.sh CMD=python3 x.py --a 1 IMAGE=ci-x", out)

    def test_node5b_accepts_capsule_knobs(self) -> None:
        r = self.helper("start", "r2", str(self.home), "node5b", "CAPSULE=a.pt", "OUT_DIR=o",
                        "IMAGE=i", "CAPSULE_ITER_COUNT=10")
        self.assertEqual(r.returncode, 0, r.stderr)
        out = self.wait_done("r2")
        self.assertIn("STUB replay_capsule_node_docker.sh", out)
        self.assertIn("CAPSULE=a.pt", out)
        self.assertIn("CAPSULE_ITER_COUNT=10", out)

    def test_a_failing_launcher_reports_failed(self) -> None:
        (self.home / "scripts" / "internal" / "docker_replay_nan_op.sh").write_text(STUB.replace("${STUB_EXIT:-0}", "3"))
        self.helper("start", "r3", str(self.home), "replay", "CMD=true")
        self.assertIn("NAN_DOCTOR_RUN_STATUS=failed exit=3", self.wait_done("r3").splitlines()[0])

    def test_refuses_other_jobs_keys_and_missing_cmd(self) -> None:
        cases = [
            (["start", "a", str(self.home), "shell", "CMD=true"], "job must be"),
            (["start", "b", str(self.home), "replay", "CMD=true", "LD_PRELOAD=x"], "not accepted"),
            (["start", "c", str(self.home), "node5b", "CMD=true"], "not accepted"),
            (["start", "d", str(self.home), "replay", "IMAGE=i"], "needs CMD"),
            (["start", "e", str(self.home), "replay", "notakv"], "KEY=VALUE"),
            (["start", "f", str(self.mad), "replay", "CMD=true"], "nan-doctor checkout"),
            (["start", "../g", str(self.home), "replay", "CMD=true"], "run-id"),
        ]
        for args, needle in cases:
            r = self.helper(*args)
            self.assertEqual(r.returncode, 2, args)
            self.assertIn(needle, r.stderr, args)
        self.assertFalse((self.mad / ".mad-runs").exists() and any((self.mad / ".mad-runs").iterdir()))

    def test_takes_the_shared_gpu_lock(self) -> None:
        lock = self.env["MAD_GPU_LOCK"]
        holder = subprocess.Popen(["flock", lock, "sleep", "3"])
        try:
            time.sleep(0.3)
            self.helper("start", "q1", str(self.home), "replay", "CMD=true")
            time.sleep(0.5)
            self.assertIn("NAN_DOCTOR_RUN_STATUS=queued", self.helper("status", "q1").stdout)
        finally:
            holder.wait()
        self.assertIn("=succeeded", self.wait_done("q1"))

    def test_must_run_from_the_mad_root(self) -> None:
        r = subprocess.run(["bash", str(HELPER), "start", "x", str(self.home), "replay", "CMD=true"],
                           cwd=self.home, env=self.env, text=True, capture_output=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("MAD repo root", r.stderr)


if __name__ == "__main__":
    unittest.main()
