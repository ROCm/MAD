# MAD Agent Skills

This directory contains agent skills for streamlining MAD (Model Automation and Dashboarding) development workflows. They work in Claude Code, Cursor and Codex CLI — see [Claude Code, Cursor and Codex](#claude-code-cursor-and-codex). Shared repo guidance for all three tools is in [`AGENTS.md`](../AGENTS.md).

## Available Skills

### mad-add-model
Interactive wizard to add new AI models to the MAD platform. Guides you through:
- Model name and framework selection
- Dockerfile generation from templates
- Run script creation with GPU architecture detection
- models.json configuration
- Validation and testing

**Usage:** `/mad-add-model` (Codex: `$mad-add-model`). Explicit only: it writes files, so agents don't start it on their own.

**Time savings:** Reduces model addition from 30-60 minutes to <10 minutes

### mad-validate-model
Comprehensive multi-level validation for model configurations including:
- JSON schema validation
- Naming convention checks
- Cross-field consistency
- Framework-specific requirements
- File reference validation

**Usage:** `/mad-validate-model [model_name]` (Codex: `$mad-validate-model`)

### mad-generate-dockerfile
Standalone Dockerfile generator using framework-specific templates.

**Usage:** `/mad-generate-dockerfile` (Codex: `$mad-generate-dockerfile`). Explicit only.

### Benchmark, profile, tune and report skills
`mad-benchmark`, `mad-profile`, `mad-tune`, `primus-projection` (explicit only)
and `mad-report` (automatic, read-only) are linked into `.agents/skills/` too.
Claude Code runs their `` ```! `` pre-flight block before the skill loads
(Cursor and Codex are told to run it first) and forks them into the subagents
in `.claude/agents/`. Cursor reads the same subagents and the skills tell it to
delegate to them. Codex gets generated copies in `.codex/agents/` (see
[Subagents](#subagents)). Their GPU runs go through
`.claude/skills/mad-common/mad-run.sh`, which detaches each run and holds a
host-wide GPU lock.

### Claude Code-only workflows
The `mad-benchmark-sweep` and `mad-tune-search` workflows in
`.claude/workflows/` need Claude Code's workflow runtime. Both plan by default
and run only with `--execute`; `CLAUDE.md` describes them, and
`node --test tools/tests/*.test.mjs` tests them against a mock agent.

## Quick Start

1. **Add a new model:**
   ```
   User: I want to add a new vLLM model for Llama-3.2-8B
   Agent: [launches mad-add-model skill with guided workflow]
   ```

2. **Validate existing models:**
   ```
   User: Validate pyt_vllm_llama-3.1-8b
   Agent: [launches mad-validate-model skill]
   ```

## Directory Structure

```
.claude/
├── README.md                           # This file
├── settings.json                       # Shared permissions + models.json hook
├── agents/                             # Subagents the skills fork into (Claude Code, Cursor)
├── hooks/validate-models-json.sh       # Validates models.json edits
├── workflows/                          # mad-benchmark-sweep, mad-tune-search (Claude Code)
├── skills/
│   ├── mad-add-model/                  # Model addition wizard
│   │   ├── SKILL.md
│   │   └── agents/openai.yaml          # Codex: explicit invocation only
│   ├── mad-validate-model/             # Configuration validator
│   ├── mad-generate-dockerfile/        # Dockerfile generator
│   │   ├── scripts/render_dockerfile.py  # Template renderer
│   │   └── framework_templates/        # Jinja2 templates
│   ├── mad-benchmark/, mad-profile/, mad-tune/, mad-report/, primus-projection/
│   └── mad-common/                     # preflight.sh, mad-run.sh (shared helpers)
└── prompts/
    └── model_addition_guide.md         # Detailed guide

.agents/skills/                         # Symlinks to .claude/skills/* (Codex)
.codex/agents/                          # Codex copies of .claude/agents/*, generated
```

## Subagents

`.claude/agents/*.md` is the source for the three subagents. Claude Code and
Cursor read those files directly (Cursor also honors `readonly: true` on
`mad-perf-analyst`). Codex only reads TOML from `.codex/agents/`, so
`tools/agent_config/codex_agents.py` generates `mad_benchmark_runner.toml`,
`mad_tuner.toml` and `mad_perf_analyst.toml` from them. After editing an agent,
run `python3 tools/agent_config/codex_agents.py`; CI runs it with `--check`.

## Claude Code, Cursor and Codex

These skills follow the open Agent Skills format (YAML frontmatter + Markdown
body), so the same files work in all three tools. Each tool looks for project
skills in different places:

| Tool | Reads skills from | Invoke explicitly |
|---|---|---|
| Claude Code | `.claude/skills/` | `/mad-add-model` |
| Cursor | `.claude/skills/` and `.agents/skills/` (also `.cursor/skills/`) | describe the task, or name the skill |
| Codex CLI | `.agents/skills/` only | `$mad-add-model`, or pick from `/skills` |

All three also pick a skill automatically when the request matches its
`description`, except the explicit-only skills: those set
`disable-model-invocation: true` (read by Claude Code and Cursor) and ship
`agents/openai.yaml` with `policy.allow_implicit_invocation: false` (read by
Codex). Every skill that builds, runs or writes files is explicit only.

`.claude/skills/` holds the real files; `.agents/skills/*` are symlinks back
to them, so there's a single source of truth: edit the files under
`.claude/skills/` and every tool picks up the change. Don't add
`.cursor/skills/` — Cursor already reads `.claude/skills/`.

When writing or editing a skill, keep it tool-neutral:

- Write paths relative to the repository root (e.g.
  `.claude/skills/mad-generate-dockerfile/framework_templates/...`), not
  relative to the skill folder.
- Say "ask the user" rather than naming a tool-specific tool such as
  `AskUserQuestion`; Codex and Cursor don't have it.
- Put anything that must produce exact output in a script (like
  `render_dockerfile.py`) rather than prose instructions.
- Claude Code-only frontmatter (`context`, `agent`, `background`,
  `allowed-tools`) is ignored elsewhere, so the body must still work inline.
  Only Claude Code substitutes `$ARGUMENTS` and runs `` ```! `` blocks; say
  what other agents should do instead.

On native Windows, git checks out symlinks as plain text files unless
symlinks are enabled, and then Codex won't see the skills (Claude Code and
Cursor are unaffected because they read `.claude/skills/` directly). To use
the skills with Codex on Windows, turn on Developer Mode and clone with:

```
git clone -c core.symlinks=true <repo-url>
```

For an existing clone, run `git config core.symlinks true`, delete the
`.agents` folder, then run `git checkout -- .agents`. WSL clones need nothing
extra.

## Development

Skills are implemented using:
- **Jinja2** templates for code generation
- **Python** validation scripts
- **Agent-driven** interactive workflows (Claude Code, Cursor, Codex)
- Integration with existing MAD tools (tools/utils.py, tools/logger.py)

Checks (all GPU-free, all run in CI):
- `python3 tools/validation/skill_validator.py` — skill and subagent
  frontmatter, `.agents/skills` links, and explicit-only consistency
- `python3 tools/validation/model_config_validator.py` — every model registry
- `node --test tools/tests/*.test.mjs` — the workflows, against a mock agent
- `python3 tools/agent_config/codex_agents.py --check` — the Codex agents match
  `.claude/agents/`

## Contributing

When adding new skills:
1. Create skill directory under `.claude/skills/`
2. Add a `SKILL.md` with YAML frontmatter (`name`, `description`) and the workflow instructions;
   `name` must match the directory name
3. If it builds, runs or writes files, make it explicit only: add
   `disable-model-invocation: true` and an `agents/openai.yaml` with
   `policy.allow_implicit_invocation: false`
4. Add templates and validators as needed
5. Link it for Codex: `ln -s ../../.claude/skills/<name> .agents/skills/<name>`
6. Run `python3 tools/validation/skill_validator.py`
7. Update this README

## References

- [Main MAD README](../README.md)
