"""MAD is a public repository: nothing that identifies a customer, a compute site, a person, an
unreleased product or a credential may ship with the NaN-debugging skill, agent and workflow.

Built-in patterns (credentials, home directories of people, evidence citations, e-mail addresses)
always run. A private list of extra terms -- kept OUT of the repository, so the list does not
publish what it protects -- is read from the file named by $PUBLIC_CONTENT_DENYLIST; CI sets it
from a secret. Without it only the built-in patterns run.
"""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "validation"))
import public_content_scan as scanner  # noqa: E402

PATHS = [
    ROOT / ".claude" / "skills" / "mad-debug-numeric",
    ROOT / ".claude" / "agents" / "mad-numeric-doctor.md",
    ROOT / ".claude" / "workflows" / "mad-nan-hunt.js",
    ROOT / "docs" / "mad-nan-hunt.md",
    ROOT / "scripts" / "dummy" / "run_nan.sh",
    ROOT / ".claude" / "skills" / "mad-validate-numeric",
    ROOT / ".claude" / "mods" / "numeric-doctor",
    ROOT / "docs" / "mad-numeric-doctor.md",
    ROOT / "scripts" / "numeric_softmoe_train_nan" / "run.sh",
    ROOT / "scripts" / "numeric_softmoe_train_nan" / "train.py",
    ROOT / "scripts" / "numeric_qwen_vllm_serve_nan" / "run.sh",
    ROOT / "scripts" / "numeric_qwen_vllm_serve_nan" / "qwen_vllm_serve.py",
]


def fmt(hits) -> str:
    return "\n".join(f"{p}:{n}: [{why}] {m!r}" for p, n, why, m, _ in hits)


class PublicContentTest(unittest.TestCase):
    def test_builtin_patterns(self) -> None:
        hits = scanner.scan([str(p) for p in PATHS if p.exists()])
        self.assertEqual(hits, [], "non-public content:\n" + fmt(hits))

    def test_private_list(self) -> None:
        listing = os.environ.get("PUBLIC_CONTENT_DENYLIST")
        if not listing:
            self.skipTest("PUBLIC_CONTENT_DENYLIST is not set (CI sets it from a secret)")
        hits = scanner.scan([str(p) for p in PATHS if p.exists()], extra=scanner.load_patterns(listing))
        self.assertEqual(hits, [], "non-public content:\n" + fmt(hits))


if __name__ == "__main__":
    unittest.main()
