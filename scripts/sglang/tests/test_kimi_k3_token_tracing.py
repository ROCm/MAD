import os
from pathlib import Path
import subprocess
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "run_kimi_k3.sh"


class TokenTracingTests(unittest.TestCase):
    def run_runner(self, traced, token, inherited_xtrace=False):
        with tempfile.TemporaryDirectory(prefix="mad-k3-token-test-") as directory:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            binaries = root / "bin"
            binaries.mkdir()

            # Fake the install and benchmark leaving a CSV for the runner to move
            stubs = {
                "pip": "#!/bin/sh\nexit 0\n",
                "python3": (
                    # Keep the stub from tracing the test token
                    "#!/bin/sh\n"
                    "set +x\n"
                    'printf "%s" "${HF_TOKEN-}" > "$MAD_TEST_TOKEN_FILE"\n'
                    'printf "%s\\n" "model,performance" > perf_Kimi-K3.csv\n'
                ),
            }
            for name, source in stubs.items():
                path = binaries / name
                path.write_text(source)
                path.chmod(0o755)

            token_file = root / "child-token"
            env = os.environ.copy()
            for variable in ("MAD_SECRETS_HFTOKEN", "CONFIG", "VARIANT"):
                env.pop(variable, None)

            env.update(
                {
                    "PATH": str(binaries) + os.pathsep + env.get("PATH", ""),
                    "HF_TOKEN": "hf_PREEXISTING_TEST_TOKEN",
                    "HF_HUB_CACHE": str(root / "cache"),
                    "MAD_TEST_TOKEN_FILE": str(token_file),
                }
            )

            if token is not None:
                env["MAD_SECRETS_HFTOKEN"] = token
            if inherited_xtrace:
                env["SHELLOPTS"] = "xtrace"

            command = (
                ["bash"]
                + (["-x"] if traced else [])
                + [
                    str(RUNNER),
                    "--model_repo",
                    "moonshotai/Kimi-K3",
                ]
            )
            result = subprocess.run(
                command,
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(token_file.read_text(), token or "")
            self.assertTrue((root / "perf_Kimi-K3.csv").is_file())
            self.assertIn("+ pip install", result.stderr)
            output = result.stdout + result.stderr

            self.assertNotIn("hf_PREEXISTING_TEST_TOKEN", output)

            if token:
                self.assertNotIn(token, output)

    def test_token_reaches_child_without_appearing_in_output(self):
        for traced, inherited_xtrace in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(traced=traced, inherited_xtrace=inherited_xtrace):
                self.run_runner(
                    traced,
                    "hf_FAKE_REGRESSION_TOKEN_12345",
                    inherited_xtrace=inherited_xtrace,
                )

    def test_unset_token_keeps_existing_empty_export_behavior(self):
        for traced, inherited_xtrace in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(traced=traced, inherited_xtrace=inherited_xtrace):
                self.run_runner(traced, None, inherited_xtrace=inherited_xtrace)


if __name__ == "__main__":
    unittest.main()
