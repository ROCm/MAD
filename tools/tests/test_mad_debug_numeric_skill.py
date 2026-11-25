"""The mad-debug-numeric skill only refers to files and `madengine nan` commands that exist, and
carries no leftovers of the environments it no longer supports."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / ".claude" / "skills" / "mad-debug-numeric"
DOCS = sorted(SKILL.glob("reference/*.md")) + [SKILL / "SKILL.md"]
LEFTOVERS = [r"REPO_ROOT", r"sbatch", r"apptainer", r"\.sif\b", r"evidence\.md", r"\[E\d+",
             r"scripts/(?:user|internal)", r"(?<!mad-)debug-nan", r"replay_capsule_node",
             r"capture_to_repro", r"next_groups", r"nan-doctor-run"]


def nan_commands():
    try:
        from madengine.nan_doctor.cli import COMMANDS
    except ImportError:
        return None
    return set(COMMANDS)


class SkillDocsTest(unittest.TestCase):
    def test_every_reference_file_the_docs_name_exists(self) -> None:
        have = {p.name for p in SKILL.glob("reference/*.md")}
        for doc in DOCS:
            for name in re.findall(r"`((?:step|env|contention|waves)[A-Za-z0-9_-]*\.md)`", doc.read_text()):
                self.assertIn(name, have, f"{doc.name} names {name}")

    def test_no_leftovers_of_dropped_environments(self) -> None:
        for doc in DOCS:
            text = doc.read_text()
            for pat in LEFTOVERS:
                self.assertIsNone(re.search(pat, text), f"{doc.name}: {pat}")

    def test_madengine_nan_commands_exist(self) -> None:
        names = nan_commands()
        if names is None:
            self.skipTest("this madengine has no `nan` command group yet")
        for doc in DOCS:
            for cmd in re.findall(r"madengine nan ([a-z][a-z-]*)", doc.read_text()):
                self.assertIn(cmd, names, f"{doc.name}: madengine nan {cmd}")

    def test_skill_map_lists_every_reference_file(self) -> None:
        text = (SKILL / "SKILL.md").read_text()
        for p in SKILL.glob("reference/*.md"):
            self.assertIn(p.name, text, f"SKILL.md does not mention {p.name}")


if __name__ == "__main__":
    unittest.main()
