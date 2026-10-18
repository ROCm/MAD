#!/usr/bin/env python3
"""Agent Skill and Subagent Validator

Lints the agent configuration shared by Claude Code, Cursor and Codex CLI:
- .claude/skills/<name>/SKILL.md: frontmatter parses, `name` matches the
  directory, `description` is present and short enough, no unknown keys
  (catches typos such as `disable-model-invokation`), and `agent:` names an
  existing subagent
- .claude/agents/<name>.md: frontmatter parses, `name` matches the file
- .agents/skills/<name>: a symlink to .claude/skills/<name> (what Codex reads)
- explicit-only skills are explicit-only everywhere: a linked skill with
  `disable-model-invocation: true` (Claude Code, Cursor) needs
  agents/openai.yaml with `policy.allow_implicit_invocation: false` (Codex),
  and the reverse
- .claude/workflows/*.js start with `export const meta`

Requires PyYAML. Run from anywhere:

    python3 tools/validation/skill_validator.py
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML is not installed. Install it with: pip install pyyaml")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Claude Code SKILL.md fields, plus the Cursor-only display fields.
SKILL_FIELDS = {
    'name', 'description', 'when_to_use', 'argument-hint', 'arguments',
    'disable-model-invocation', 'user-invocable', 'allowed-tools',
    'disallowed-tools', 'model', 'effort', 'context', 'agent', 'background',
    'hooks', 'paths', 'shell', 'metadata', 'license', 'compatibility',
    'icon', 'color',
}
AGENT_FIELDS = {
    'name', 'description', 'tools', 'disallowedTools', 'model', 'permissionMode',
    'maxTurns', 'skills', 'mcpServers', 'hooks', 'memory', 'background',
    'effort', 'isolation', 'color', 'initialPrompt',
}
BUILTIN_AGENTS = {'general-purpose', 'Explore', 'Plan'}
# Directories under .claude/skills/ that hold shared helpers, not a skill.
SHARED_DIRS = {'mad-common'}
NAME_RE = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')
MAX_DESCRIPTION = 1024


def read_frontmatter(path: str) -> Tuple[Optional[dict], Optional[str]]:
    """Return (frontmatter, error) for a Markdown file with YAML frontmatter."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError as e:
        return None, f"cannot read ({e})"
    match = re.match(r'^---\n(.*?)\n---\n', text, re.S)
    if not match:
        return None, "no YAML frontmatter (expected '---' on the first line)"
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        return None, f"frontmatter is not valid YAML ({e})"
    if not isinstance(data, dict):
        return None, "frontmatter is not a mapping"
    return data, None


def implicit_invocation_allowed(skill_dir: str) -> Tuple[Optional[bool], Optional[str]]:
    """Read agents/openai.yaml; return (allow_implicit_invocation or None, error)."""
    path = os.path.join(skill_dir, 'agents', 'openai.yaml')
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        return None, f"agents/openai.yaml is not valid YAML ({e})"
    policy = data.get('policy') if isinstance(data, dict) else None
    value = policy.get('allow_implicit_invocation') if isinstance(policy, dict) else None
    if not isinstance(value, bool):
        return None, "agents/openai.yaml needs a boolean policy.allow_implicit_invocation"
    return value, None


class SkillValidator:
    """Validator for the repo's skills, subagents, Codex links and workflows"""

    def __init__(self, repo_root: str = REPO_ROOT):
        self.repo_root = repo_root
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.counts: Dict[str, int] = {'skills': 0, 'agents': 0, 'links': 0, 'workflows': 0}

    def _path(self, *parts: str) -> str:
        return os.path.join(self.repo_root, *parts)

    def validate(self) -> bool:
        agents = self._check_agents()
        linked = self._check_codex_links()
        self._check_skills(agents, linked)
        self._check_workflows()
        return not self.errors

    def _check_agents(self) -> set:
        names = set()
        agents_dir = self._path('.claude', 'agents')
        if not os.path.isdir(agents_dir):
            return names
        for entry in sorted(os.listdir(agents_dir)):
            if not entry.endswith('.md'):
                continue
            rel = f".claude/agents/{entry}"
            self.counts['agents'] += 1
            data, error = read_frontmatter(os.path.join(agents_dir, entry))
            if error:
                self.errors.append(f"{rel}: {error}")
                continue
            name = data.get('name')
            if name != entry[:-3]:
                self.errors.append(f"{rel}: name '{name}' must match the file name '{entry[:-3]}'")
            if not data.get('description'):
                self.errors.append(f"{rel}: missing description")
            for key in sorted(set(data) - AGENT_FIELDS):
                self.warnings.append(f"{rel}: unknown frontmatter key '{key}'")
            names.add(entry[:-3])
        return names

    def _check_codex_links(self) -> set:
        linked = set()
        links_dir = self._path('.agents', 'skills')
        if not os.path.isdir(links_dir):
            return linked
        skills_dir = os.path.realpath(self._path('.claude', 'skills'))
        for entry in sorted(os.listdir(links_dir)):
            rel = f".agents/skills/{entry}"
            path = os.path.join(links_dir, entry)
            self.counts['links'] += 1
            if not os.path.islink(path):
                self.errors.append(f"{rel}: must be a symlink to ../../.claude/skills/{entry} "
                                   "(on Windows, clone with core.symlinks=true)")
                continue
            target = os.path.realpath(path)
            if target != os.path.join(skills_dir, entry):
                self.errors.append(f"{rel}: points to {os.readlink(path)}, expected ../../.claude/skills/{entry}")
            elif not os.path.isfile(os.path.join(target, 'SKILL.md')):
                self.errors.append(f"{rel}: target has no SKILL.md")
            else:
                linked.add(entry)
        return linked

    def _check_skills(self, agents: set, linked: set):
        skills_dir = self._path('.claude', 'skills')
        if not os.path.isdir(skills_dir):
            return
        for entry in sorted(os.listdir(skills_dir)):
            skill_dir = os.path.join(skills_dir, entry)
            if not os.path.isdir(skill_dir) or entry in SHARED_DIRS:
                continue
            rel = f".claude/skills/{entry}/SKILL.md"
            if not os.path.isfile(os.path.join(skill_dir, 'SKILL.md')):
                self.errors.append(f"{rel}: missing (add it, or list the directory in SHARED_DIRS)")
                continue
            self.counts['skills'] += 1
            data, error = read_frontmatter(os.path.join(skill_dir, 'SKILL.md'))
            if error:
                self.errors.append(f"{rel}: {error}")
                continue

            name = data.get('name')
            if name != entry:
                self.errors.append(f"{rel}: name '{name}' must match the directory name '{entry}'")
            elif not NAME_RE.match(name) or len(name) > 64:
                self.errors.append(f"{rel}: name must be lowercase letters, digits and hyphens (max 64)")
            description = data.get('description')
            if not isinstance(description, str) or not description.strip():
                self.errors.append(f"{rel}: missing description")
            elif len(description) > MAX_DESCRIPTION:
                self.errors.append(f"{rel}: description is {len(description)} chars (max {MAX_DESCRIPTION})")
            for key in sorted(set(data) - SKILL_FIELDS):
                self.errors.append(f"{rel}: unknown frontmatter key '{key}'")

            agent = data.get('agent')
            if agent is not None and agent not in agents | BUILTIN_AGENTS:
                self.errors.append(f"{rel}: agent '{agent}' is not in .claude/agents/ or a built-in agent")
            if agent is not None and data.get('context') != 'fork':
                self.warnings.append(f"{rel}: 'agent' has no effect without 'context: fork'")
            if 'background' in data and data.get('context') != 'fork':
                self.warnings.append(f"{rel}: 'background' has no effect without 'context: fork'")

            manual = data.get('disable-model-invocation') is True
            implicit, yaml_error = implicit_invocation_allowed(skill_dir)
            if yaml_error:
                self.errors.append(f".claude/skills/{entry}/{yaml_error}")
            elif entry in linked:
                if manual and implicit is not False:
                    self.errors.append(
                        f"{rel}: disable-model-invocation is true but Codex may still invoke it; add "
                        "agents/openai.yaml with policy.allow_implicit_invocation: false")
                if implicit is False and not manual:
                    self.errors.append(
                        f"{rel}: agents/openai.yaml blocks implicit invocation in Codex but Claude Code "
                        "and Cursor may still auto-invoke it; set disable-model-invocation: true")

    def _check_workflows(self):
        workflows_dir = self._path('.claude', 'workflows')
        if not os.path.isdir(workflows_dir):
            return
        for entry in sorted(os.listdir(workflows_dir)):
            if not entry.endswith('.js'):
                continue
            self.counts['workflows'] += 1
            with open(os.path.join(workflows_dir, entry)) as f:
                first = f.readline()
            if not first.startswith('export const meta'):
                self.errors.append(f".claude/workflows/{entry}: must start with 'export const meta = {{'")

    def print_report(self):
        print("\nSkill Validation Report")
        print("=" * 60)
        print("Checked {skills} skill(s), {agents} subagent(s), {links} Codex link(s), "
              "{workflows} workflow(s)".format(**self.counts))
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        if not self.errors:
            print(f"\n✅ No errors ({len(self.warnings)} warnings)")
        else:
            print(f"\n❌ Validation failed: {len(self.errors)} errors")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lint agent skills, subagents, Codex links and workflows')
    parser.add_argument('--repo-root', default=REPO_ROOT,
                        help='MAD repository root (default: derived from this file)')
    args = parser.parse_args()

    validator = SkillValidator(os.path.abspath(args.repo_root))
    valid = validator.validate()
    validator.print_report()
    sys.exit(0 if valid else 1)
