# MAD Agent Skills

This directory contains agent skills for streamlining MAD (Model Automation and Dashboarding) development workflows. They work in Claude Code, Cursor and Codex CLI — see [Claude Code, Cursor and Codex](#claude-code-cursor-and-codex).

## Available Skills

### mad-add-model
Interactive wizard to add new AI models to the MAD platform. Guides you through:
- Model name and framework selection
- Dockerfile generation from templates
- Run script creation with GPU architecture detection
- models.json configuration
- Validation and testing

**Usage:** `/mad-add-model` (Codex: `$mad-add-model`) or naturally describe adding a model

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

**Usage:** `/mad-generate-dockerfile` (Codex: `$mad-generate-dockerfile`)

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
├── skills/
│   ├── mad-add-model/                  # Model addition wizard
│   │   └── SKILL.md
│   ├── mad-validate-model/             # Configuration validator
│   │   └── SKILL.md
│   └── mad-generate-dockerfile/        # Dockerfile generator
│       ├── SKILL.md
│       ├── scripts/render_dockerfile.py  # Template renderer
│       └── framework_templates/        # Jinja2 templates
└── prompts/
    └── model_addition_guide.md         # Detailed guide

.agents/skills/                         # Symlinks to .claude/skills/* (Codex)
```

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
`description`.

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

## Contributing

When adding new skills:
1. Create skill directory under `.claude/skills/`
2. Add a `SKILL.md` with YAML frontmatter (`name`, `description`) and the workflow instructions;
   `name` must match the directory name
3. Add templates and validators as needed
4. Link it for Codex: `ln -s ../../.claude/skills/<name> .agents/skills/<name>`
5. Update this README

## References

- [Main MAD README](../README.md)
