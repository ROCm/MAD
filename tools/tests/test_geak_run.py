"""geak-run.sh treats a patch written beside eval_dir as the finished patch."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / ".claude" / "skills" / "mad-common" / "geak-run.sh"


class GeakRunPatchTest(unittest.TestCase):
    def test_status_copies_a_patch_from_the_parent_into_eval_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            ev = Path(tmp) / "work" / "geak_eval"
            ev.mkdir(parents=True)
            parent_diff = ev.parent / "final_patch.diff"
            parent_diff.write_text("diff --git a/x b/x\n")
            run = runs / "opt_geak"
            run.mkdir(parents=True)
            (run / "exit_code").write_text("0")
            (run / "args.json").write_text(json.dumps({"eval_dir": str(ev)}))
            env = os.environ.copy()
            env["MAD_RUNS_DIR"] = str(runs)
            out = subprocess.run(
                ["bash", str(HELPER), "status", "opt_geak"],
                cwd=tmp,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("GEAK_RUN_STATUS=succeeded", out.stdout.splitlines()[0])
            self.assertIn("final_patch=present", out.stdout.splitlines()[0])
            self.assertEqual((ev / "final_patch.diff").read_text(), parent_diff.read_text())
