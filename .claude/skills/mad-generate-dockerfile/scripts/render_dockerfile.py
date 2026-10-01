#!/usr/bin/env python3
"""Render a MAD Dockerfile from a framework template.

Deterministic renderer shared by the mad-generate-dockerfile and mad-add-model
skills, so every coding agent (Claude Code, Cursor, Codex) produces the same
output instead of hand-rendering the Jinja2 templates.

Run from the repository root:

    # List frameworks and the context variables each template accepts
    python3 .claude/skills/mad-generate-dockerfile/scripts/render_dockerfile.py --list

    # Preview on stdout
    python3 .claude/skills/mad-generate-dockerfile/scripts/render_dockerfile.py \\
        --framework vllm --context '{"pip_packages": ["foo==1.0"]}'

    # Write the file (context may also be read from a JSON file with @path)
    python3 .claude/skills/mad-generate-dockerfile/scripts/render_dockerfile.py \\
        --framework vllm --context @/tmp/ctx.json \\
        --output docker/pyt_vllm_mymodel.ubuntu.amd.Dockerfile
"""

import argparse
import datetime
import json
import os
import re
import shlex
import sys
from typing import Dict, List

TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "framework_templates"
)
TEMPLATE_SUFFIX = "_base.Dockerfile.jinja"


def available_frameworks() -> List[str]:
    return sorted(
        f[: -len(TEMPLATE_SUFFIX)]
        for f in os.listdir(TEMPLATES_DIR)
        if f.endswith(TEMPLATE_SUFFIX)
    )


def make_env():
    try:
        import jinja2
    except ImportError:
        sys.exit("ERROR: jinja2 is not installed. Install it with: pip install jinja2")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    # Values spliced into RUN lines must be shell-quoted, or a spec like
    # "foo>=2" is parsed as a redirection and ";" can inject commands.
    env.filters["shquote"] = lambda v: shlex.quote(str(v))
    return env


def template_variables(env, framework: str) -> List[str]:
    import jinja2.meta

    source = env.loader.get_source(env, framework + TEMPLATE_SUFFIX)[0]
    return sorted(jinja2.meta.find_undeclared_variables(env.parse(source)))


def load_context(raw: str) -> Dict:
    if raw.startswith("@"):
        with open(raw[1:]) as f:
            raw = f.read()
    try:
        context = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"ERROR: --context is not valid JSON: {e}")
    if not isinstance(context, dict):
        sys.exit("ERROR: --context must be a JSON object")
    return context


def check_context(context: Dict, known: List[str]) -> List[str]:
    """Return errors for malformed values; warn about keys the template ignores."""
    errors = []
    for key in ("apt_packages", "pip_packages"):
        value = context.get(key)
        if value is not None and not (
            isinstance(value, list) and all(isinstance(v, str) for v in value)
        ):
            errors.append(f"'{key}' must be a list of strings")
    repos = context.get("git_repos")
    if repos is not None:
        if not isinstance(repos, list) or not all(
            isinstance(r, dict) and r.get("url") and r.get("path") for r in repos
        ):
            errors.append("'git_repos' must be a list of objects with 'url' and 'path'")
    for key in sorted(set(context) - set(known)):
        print(f"WARNING: context key '{key}' is not used by this template", file=sys.stderr)
    return errors


def render(framework: str, context: Dict) -> str:
    env = make_env()
    errors = check_context(context, template_variables(env, framework))
    if errors:
        sys.exit("ERROR: " + "; ".join(errors))
    context.setdefault("year", datetime.date.today().year)
    output = env.get_template(framework + TEMPLATE_SUFFIX).render(**context)
    # Skipped optional blocks leave runs of blank lines behind; collapse them.
    return re.sub(r"\n{3,}", "\n\n", output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a MAD Dockerfile from a framework template")
    parser.add_argument("--framework", choices=available_frameworks(), help="Template to render")
    parser.add_argument("--context", default="{}", help="JSON object, or @path to a JSON file")
    parser.add_argument("--output", help="Write to this path instead of stdout")
    parser.add_argument("--force", action="store_true", help="Overwrite --output if it exists")
    parser.add_argument("--list", action="store_true", help="List frameworks and their context variables")
    args = parser.parse_args()

    if args.list:
        env = make_env()
        for framework in available_frameworks():
            print(f"{framework}: {', '.join(template_variables(env, framework))}")
        return 0
    if not args.framework:
        parser.error("--framework is required (or use --list)")

    output = render(args.framework, load_context(args.context))
    if not args.output:
        sys.stdout.write(output)
        return 0
    if os.path.exists(args.output) and not args.force:
        sys.exit(f"ERROR: {args.output} already exists (use --force to overwrite)")
    with open(args.output, "w") as f:
        f.write(output)
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
