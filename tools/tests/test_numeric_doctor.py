"""GPU-free checks that tools/numeric-doctor is self-contained, public-clean and
visible to Claude Code, Cursor and Codex from the MAD root.

    python3 -m unittest discover -s tools/tests -p 'test_numeric_doctor.py'

Set NUMERIC_DOCTOR_DENYLIST to a file with one extra term per line (site names,
user handles) to scan for terms that cannot be listed in a public repo.
"""

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ND = REPO / "tools" / "numeric-doctor"
SKILLS = ("debug-numeric", "validate-examples")
MANUAL_EXAMPLE = ND / "examples" / "training" / "qwen3_moe_gradient_nan_nohint"

PUBLIC_PATTERNS = {
    "evidence tag": re.compile(r"\[E\d+\]"),
    "personal TODO": re.compile(r"TODO\((?!agent\))[A-Za-z]+\)"),
    "personal path": re.compile(r"/(?:home|apps|users)/(?!user/|<|\$)[a-z][\w.-]*/"),
    "internal GitHub org": re.compile(r"github\.com[/:][\w.-]+-Internal\b", re.I),
    "Slurm partition": re.compile(r"(?:--partition[= ]|sbatch -p |srun -p )[a-z0-9]"),
    "Slurm node": re.compile(r"\s-w\s+[a-z0-9]+-[a-z0-9-]+"),
    "Slurm job id": re.compile(r"\bjob \d{6,}\b"),
}


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False)


def shipped_files() -> list:
    """Files under tools/numeric-doctor that a commit would include."""
    out = git("ls-files", "-co", "--exclude-standard", "--", "tools/numeric-doctor").stdout
    return [REPO / line for line in out.splitlines() if line]


def frontmatter(skill_md: Path) -> str:
    text = skill_md.read_text()
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    return match.group(1) if match else ""


class PackageLayoutTest(unittest.TestCase):
    def test_package_root_holds_the_tools_the_skills_call(self) -> None:
        for rel in ("scripts/nd", "scripts/replay_nan_op.py", "scripts/op_nan_capture.py",
                    "scripts/compare_results.py", "scripts/gpu_check.py",
                    "scripts/env/profiles/local-docker.env", ".gitignore", "README.md"):
            self.assertTrue((ND / rel).is_file(), rel)
        for skill in SKILLS:
            self.assertTrue((ND / "skills" / skill / "SKILL.md").is_file(), skill)

    def test_manual_example_and_expected_cards_are_present(self) -> None:
        for name in ("run.sh", "Dockerfile", "qwen3_moe.def", "README.md", "capture.spec"):
            self.assertTrue((MANUAL_EXAMPLE / name).is_file(), name)
        self.assertTrue(list((ND / "validation" / "expected").glob("*.json")))

    def test_internal_only_material_is_not_shipped(self) -> None:
        self.assertFalse((ND / ".git").exists(), "nested .git")
        for rel in ("mad-numeric-doctor", "validation/answers", "saved_results", "bug_digger",
                    "patches", "docs", ".claude", "open_bugs.md", "TODO"):
            self.assertFalse((ND / rel).exists(), rel)

    def test_no_cache_or_local_settings_is_shipped(self) -> None:
        files = shipped_files()
        self.assertTrue(files, "nothing under tools/numeric-doctor would be committed")
        for path in files:
            self.assertNotIn("__pycache__", path.parts, path)
            self.assertNotEqual(path.name, "settings.local.json", path)


class ProfileTest(unittest.TestCase):
    def test_every_profile_loads_through_nd(self) -> None:
        profiles = sorted((ND / "scripts" / "env" / "profiles").glob("*.env"))
        names = {p.stem for p in profiles}
        self.assertTrue({"local-docker", "slurm-docker", "slurm-apptainer"} <= names, names)
        for profile in profiles:
            with tempfile.TemporaryDirectory() as art:
                made = subprocess.run(["bash", str(ND / "scripts" / "nd"), "profile", profile.stem,
                                       "--art", art], capture_output=True, text=True, check=False)
                self.assertEqual(made.returncode, 0, made.stderr)
                info = subprocess.run(["bash", str(ND / "scripts" / "nd"), "info"], capture_output=True,
                                      text=True, check=False, env={**os.environ, "ND_ENV": f"{art}/env.sh"})
                self.assertEqual(info.returncode, 0, info.stderr)
                sched = re.search(r"(?m)^SCHED=(\w+)", (ND / "scripts" / "env" / "profiles" /
                                                        profile.name).read_text()).group(1)
                self.assertIn(f"SCHED={sched} ", info.stdout, profile.name)


class PublicContentTest(unittest.TestCase):
    def scan(self, patterns: dict) -> list:
        hits = []
        for path in shipped_files():
            if path.is_symlink() or not path.is_file():
                continue
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            for label, pattern in patterns.items():
                for lineno, line in enumerate(text.splitlines(), 1):
                    if pattern.search(line):
                        hits.append(f"{path.relative_to(REPO)}:{lineno}: {label}: {line.strip()[:120]}")
        return hits

    def test_no_internal_paths_tags_or_cluster_details(self) -> None:
        hits = self.scan(PUBLIC_PATTERNS)
        self.assertEqual(hits, [], "\n" + "\n".join(hits[:40]))

    def test_no_private_denylist_terms(self) -> None:
        denylist = os.environ.get("NUMERIC_DOCTOR_DENYLIST")
        if not denylist:
            self.skipTest("NUMERIC_DOCTOR_DENYLIST not set")
        terms = [t.strip() for t in Path(denylist).read_text().splitlines() if t.strip()]
        patterns = {t: re.compile(rf"(?<![a-z]){re.escape(t)}(?![a-z])", re.I) for t in terms}
        hits = self.scan(patterns)
        self.assertEqual(hits, [], "\n" + "\n".join(hits[:40]))


class GitTrackingTest(unittest.TestCase):
    def ignored(self, rel: str) -> bool:
        return git("check-ignore", "-q", rel).returncode == 0

    def test_capture_specs_and_env_profiles_are_tracked(self) -> None:
        for rel in ("examples/training/qwen3_moe_gradient_nan_nohint/capture.spec",
                    "scripts/env/profiles/local-docker.env", "scripts/env/lib.sh"):
            self.assertFalse(self.ignored(f"tools/numeric-doctor/{rel}"), rel)

    def test_no_package_file_is_ignored_by_a_rule_outside_the_package(self) -> None:
        out = git("ls-files", "-o", "-i", "--exclude-standard", "--", "tools/numeric-doctor").stdout
        files = [line for line in out.splitlines() if "__pycache__" not in line]
        if not files:
            return
        rules = subprocess.run(["git", "check-ignore", "-v", "--stdin"], cwd=REPO, input="\n".join(files),
                               capture_output=True, text=True).stdout
        ignored = [line for line in rules.splitlines() if not line.startswith("tools/numeric-doctor/")]
        self.assertEqual(ignored, [])

    def test_run_artifacts_are_ignored(self) -> None:
        self.assertTrue(self.ignored("tools/numeric-doctor/artifacts/2026-01-01-x/report.md"))


class PreflightTest(unittest.TestCase):
    def test_an_example_is_scanned_alone_not_the_whole_mad_repo(self) -> None:
        run = ND / "examples" / "training" / "softmoe_train_nan" / "run.sh"
        out = subprocess.run([sys.executable, str(ND / "scripts" / "capture_preflight.py"), str(run)],
                             cwd=ND, capture_output=True, text=True, env={**os.environ, "ND_ENV": ""}).stdout
        self.assertIn(f"root={ND} ", out)
        self.assertIn("scope:    numeric-doctor checkout", out)
        self.assertNotIn("scripts/Primus", out)


class SkillDiscoveryTest(unittest.TestCase):
    def test_claude_and_cursor_read_the_package_skills_from_the_repo_root(self) -> None:
        for skill in SKILLS:
            link = REPO / ".claude" / "skills" / skill
            self.assertTrue(link.is_symlink(), link)
            self.assertEqual(link.resolve(), (ND / "skills" / skill).resolve())

    def test_codex_links_point_back_to_claude_skills(self) -> None:
        for skill in SKILLS:
            link = REPO / ".agents" / "skills" / skill
            self.assertTrue(link.is_symlink(), link)
            self.assertEqual(os.readlink(link), f"../../.claude/skills/{skill}")

    def test_skill_descriptions_answer_to_numeric_doctor(self) -> None:
        for skill in SKILLS:
            fm = frontmatter(ND / "skills" / skill / "SKILL.md")
            self.assertRegex(fm, rf"(?m)^name: {skill}$")
            self.assertIn("numeric doctor", fm.lower(), skill)

    def test_debug_skill_step0_finds_the_package_root_through_the_symlink(self) -> None:
        text = (ND / "skills" / "debug-numeric" / "SKILL.md").read_text()
        block = re.search(r"```bash\n(S=<SKILL_DIR>.*?\ndone\n)", text, re.S)
        self.assertIsNotNone(block, "step 0 root-resolution block")
        script = block.group(1).replace("<SKILL_DIR>", ".claude/skills/debug-numeric") + 'echo "$R"\n'
        env = {k: v for k, v in os.environ.items() if k != "NUMERIC_DOCTOR_ROOT"}
        out = subprocess.run(["bash", "-c", script], cwd=REPO, env=env, capture_output=True,
                             text=True, check=True)
        self.assertEqual(out.stdout.strip(), str(ND.resolve()))

    def test_validate_skill_finds_the_package_root_through_the_symlink(self) -> None:
        out = subprocess.run(["bash", "-c", "cd -P .claude/skills/validate-examples/../.. && pwd -P"],
                             cwd=REPO, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), str(ND.resolve()))


class SkillValidatorTest(unittest.TestCase):
    def load_validator(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("pyyaml not installed")
        sys.path.insert(0, str(REPO / "tools" / "validation"))
        self.addCleanup(sys.path.remove, str(REPO / "tools" / "validation"))
        import skill_validator
        return skill_validator

    def test_accepts_a_skill_symlinked_from_a_tool_directory(self) -> None:
        skill_validator = self.load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "tools" / "demo" / "skills" / "demo-skill"
            real.mkdir(parents=True)
            (real / "SKILL.md").write_text("---\nname: demo-skill\ndescription: Demo.\n---\nBody\n")
            (root / ".claude" / "skills").mkdir(parents=True)
            (root / ".agents" / "skills").mkdir(parents=True)
            os.symlink("../../tools/demo/skills/demo-skill", root / ".claude" / "skills" / "demo-skill")
            os.symlink("../../.claude/skills/demo-skill", root / ".agents" / "skills" / "demo-skill")
            validator = skill_validator.SkillValidator(repo_root=str(root))
            validator.validate()
            self.assertEqual(validator.errors, [])

    def test_repo_skills_pass_the_validator(self) -> None:
        skill_validator = self.load_validator()
        validator = skill_validator.SkillValidator()
        validator.validate()
        self.assertEqual(validator.errors, [])


class DocsTest(unittest.TestCase):
    def test_readme_paths_exist(self) -> None:
        text = (ND / "README.md").read_text()
        paths = set(re.findall(r"tools/numeric-doctor/[\w./-]*[\w/]", text))
        self.assertTrue(paths, "README names no tools/numeric-doctor paths")
        for rel in sorted(paths):
            if "/artifacts" in rel:
                continue
            self.assertTrue((REPO / rel).exists(), rel)

    def test_readme_gives_the_manual_prompts(self) -> None:
        text = (ND / "README.md").read_text().lower()
        self.assertIn("validate numeric doctor", text)
        self.assertIn("debug tools/numeric-doctor/examples/training/qwen3_moe_gradient_nan_nohint "
                      "with numeric doctor", text)
        self.assertIn("validate mad-numeric-doctor", text)
        self.assertIn("cd mad/tools/numeric-doctor", text)

    def test_agents_md_lists_the_skills(self) -> None:
        text = (REPO / "AGENTS.md").read_text()
        for skill in SKILLS:
            self.assertTrue(f"`{skill}`" in text, f"AGENTS.md does not list `{skill}`")

    def test_codeowners_names_an_owner_for_the_tool(self) -> None:
        text = (REPO / ".github" / "CODEOWNERS").read_text()
        self.assertRegex(text, r"(?m)^/tools/numeric-doctor/\s+@\S+")


if __name__ == "__main__":
    unittest.main()
