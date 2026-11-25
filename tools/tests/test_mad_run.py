"""mad-run.sh starts only madengine run/build and the GPU-using `madengine nan` commands."""

import os
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / ".claude" / "skills" / "mad-common" / "mad-run.sh"


class MadRunStartTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.cwd = t / "mad"
        self.cwd.mkdir()
        (self.cwd / "models.json").write_text("[]")
        bin_dir = t / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "madengine"
        stub.write_text('#!/bin/bash\necho "STUB madengine $*"\nexit ${STUB_EXIT:-0}\n')
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        self.env = os.environ.copy()
        self.env["PATH"] = f"{bin_dir}{os.pathsep}{self.env['PATH']}"
        self.env["MAD_GPU_LOCK"] = str(t / "gpu.lock")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def helper(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(HELPER), *args], cwd=self.cwd, env=self.env,
                              text=True, capture_output=True, check=False)

    def finished(self, run_id: str) -> str:
        for _ in range(100):
            out = self.helper("status", run_id).stdout
            # `failed exit=unknown` is transient for a command that has just ended: the
            # helper sees the process gone a moment before it reads the recorded exit code.
            if ("=succeeded" in out or "=failed" in out) and "exit=unknown" not in out:
                return out
            time.sleep(0.1)
        self.fail("run did not finish")

    def test_madengine_run_and_build_still_start(self) -> None:
        for i, sub in enumerate(("run --tags x", "build --tags x")):
            r = self.helper("start", f"r{i}", "--", "madengine", *sub.split())
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("MAD_RUN_STATUS=succeeded", self.finished(f"r{i}"))

    def test_the_gpu_using_nan_commands_start_and_hold_the_lock_like_run(self) -> None:
        for sub in ("exec", "group", "replay", "package"):
            r = self.helper("start", f"n_{sub}", "--", "madengine", "nan", sub, "--image", "i")
            self.assertEqual(r.returncode, 0, f"{sub}: {r.stderr}")
            out = self.finished(f"n_{sub}")
            self.assertIn("MAD_RUN_STATUS=succeeded", out)
            self.assertIn(f"STUB madengine nan {sub} --image i", out)

    def test_a_failing_nan_command_reports_failed(self) -> None:
        self.env["STUB_EXIT"] = "3"
        self.helper("start", "f1", "--", "madengine", "nan", "replay", "x")
        self.assertIn("MAD_RUN_STATUS=failed exit=3", self.finished("f1").splitlines()[0])

    def test_other_nan_commands_and_other_programs_are_refused(self) -> None:
        cases = [
            (["madengine", "nan", "ledger", "x"], "only 'madengine nan exec|group|replay|package'"),
            (["madengine", "nan"], "only 'madengine nan exec|group|replay|package'"),
            (["madengine", "report", "x"], "can be started"),
            (["bash", "-c", "true"], "only madengine commands"),
            (["bash", "nan_hunts/h/verify-G1/repro-G1/run.sh", "extra"], "only madengine commands"),
        ]
        for cmd, needle in cases:
            r = self.helper("start", "x1", "--", *cmd)
            self.assertEqual(r.returncode, 2, cmd)
            self.assertIn(needle, r.stderr, cmd)
        self.assertFalse((self.cwd / ".mad-runs").exists() and any((self.cwd / ".mad-runs").iterdir()))

    def test_the_extracted_reproducer_run_sh_starts(self) -> None:
        script = self.cwd / "nan_hunts" / "h" / "verify-G1" / "repro-G1" / "run.sh"
        script.parent.mkdir(parents=True)
        script.write_text("#!/bin/bash\necho shipped-reproducer\n")
        r = self.helper("start", "v1", "--", "bash", str(script.relative_to(self.cwd)))
        self.assertEqual(r.returncode, 0, r.stderr)
        out = self.finished("v1")
        self.assertIn("MAD_RUN_STATUS=succeeded", out)
        self.assertIn("shipped-reproducer", out)

    def test_a_shipped_script_runs_from_its_dir_in_an_empty_environment(self) -> None:
        script = self.cwd / "nan_hunts" / "h" / "verify-fix" / "fix" / "validate.sh"
        script.parent.mkdir(parents=True)
        script.write_text('#!/bin/bash\necho "pwd=$PWD leak=${LEAK:-none} home=${HOME:+set}"\n')
        self.env["LEAK"] = "investigation"
        r = self.helper("start", "v2", "--", "bash", str(script.relative_to(self.cwd)))
        self.assertEqual(r.returncode, 0, r.stderr)
        out = self.finished("v2")
        self.assertIn("MAD_RUN_STATUS=succeeded", out)
        self.assertIn(f"pwd={script.parent.resolve()} leak=none home=set", out)


if __name__ == "__main__":
    unittest.main()
