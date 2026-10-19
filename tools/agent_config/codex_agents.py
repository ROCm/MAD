#!/usr/bin/env python3
"""Generate Codex custom agents from the Claude Code / Cursor subagents.

.claude/agents/<name>.md is the single source of truth: Claude Code and Cursor
read it directly, but Codex only reads TOML from .codex/agents/. This writes
.codex/agents/<name_with_underscores>.toml with the same description and the
Markdown body as developer_instructions. Edit the Markdown, then rerun:

    python3 tools/agent_config/codex_agents.py           # write
    python3 tools/agent_config/codex_agents.py --check   # fail if out of date (CI)
"""

import argparse
import os
import re
import sys
from typing import Dict

try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML is not installed. Install it with: pip install pyyaml")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_DIR = os.path.join('.claude', 'agents')
TARGET_DIR = os.path.join('.codex', 'agents')


def toml_string(value: str) -> str:
    """Render a TOML basic string: one line, or multi-line when it holds newlines."""
    escaped = value.replace('\\', '\\\\')
    if '\n' not in value:
        return '"' + escaped.replace('"', '\\"') + '"'
    return '"""\n' + escaped.replace('"""', '""\\"') + '"""'


def codex_name(name: str) -> str:
    # Codex documents agent names with underscores (e.g. pr_explorer).
    return name.replace('-', '_')


def render(source_rel: str, text: str) -> Dict[str, str]:
    """Return {target_file_name: toml_text} for one .claude/agents/*.md file."""
    match = re.match(r'^---\n(.*?)\n---\n(.*)$', text, re.S)
    if not match:
        raise ValueError(f"{source_rel}: no YAML frontmatter")
    meta = yaml.safe_load(match.group(1)) or {}
    name, description = meta.get('name'), meta.get('description')
    if not name or not description:
        raise ValueError(f"{source_rel}: frontmatter needs name and description")
    body = match.group(2).strip() + '\n'
    toml = (
        f"# Generated from {source_rel} by tools/agent_config/codex_agents.py.\n"
        "# Do not edit: change the Markdown file and rerun the script.\n"
        f'name = "{codex_name(name)}"\n'
        f"description = {toml_string(description.strip())}\n"
        f"developer_instructions = {toml_string(body)}\n"
    )
    return {f"{codex_name(name)}.toml": toml}


def generate(repo_root: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    source_dir = os.path.join(repo_root, SOURCE_DIR)
    for entry in sorted(os.listdir(source_dir)):
        if entry.endswith('.md'):
            with open(os.path.join(source_dir, entry)) as f:
                files.update(render(f"{SOURCE_DIR}/{entry}", f.read()))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate .codex/agents/*.toml from .claude/agents/*.md')
    parser.add_argument('--check', action='store_true',
                        help='Report out-of-date or stale files and exit 1 instead of writing')
    parser.add_argument('--repo-root', default=REPO_ROOT,
                        help='MAD repository root (default: derived from this file)')
    args = parser.parse_args()

    target_dir = os.path.join(args.repo_root, TARGET_DIR)
    try:
        wanted = generate(args.repo_root)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    existing = set(f for f in os.listdir(target_dir) if f.endswith('.toml')) if os.path.isdir(target_dir) else set()

    problems = []
    for file_name, text in wanted.items():
        path = os.path.join(target_dir, file_name)
        current = open(path).read() if os.path.exists(path) else None
        if current != text:
            problems.append(f"{TARGET_DIR}/{file_name} is {'missing' if current is None else 'out of date'}")
            if not args.check:
                os.makedirs(target_dir, exist_ok=True)
                with open(path, 'w') as f:
                    f.write(text)
    for file_name in sorted(existing - set(wanted)):
        problems.append(f"{TARGET_DIR}/{file_name} has no source in {SOURCE_DIR}/")
        if not args.check:
            os.remove(os.path.join(target_dir, file_name))

    if args.check:
        for problem in problems:
            print(f"❌ {problem}")
        if problems:
            print("Run: python3 tools/agent_config/codex_agents.py")
            return 1
        print(f"✅ {len(wanted)} Codex agent(s) match {SOURCE_DIR}/")
        return 0
    for problem in problems:
        print(f"Updated: {problem}")
    print(f"{len(wanted)} Codex agent(s) in {TARGET_DIR}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
