---
name: mad-validate-model
description: Comprehensive multi-level validation for MAD model configurations (models.json entries, Dockerfiles, run scripts). Use when the user asks to validate, check, or verify a model configuration, or after adding/editing a models.json entry.
---

# mad-validate-model Skill Instructions

## Purpose
Perform comprehensive multi-level validation on MAD model configurations to catch errors before deployment.

## Usage

### Validate All Models
```
User: Validate all models
Agent: [Runs `python3 tools/validation/model_config_validator.py --all`,
        which validates the root models.json plus every scripts/*/models.json]
```

### Validate Specific Model
```
User: Validate pyt_vllm_llama-3.1-8b
Agent: [Runs the validator with --models-json pointing at the registry that
        contains the model, plus --model pyt_vllm_llama-3.1-8b]
```

## Validation Checks

These are the checks `tools/validation/model_config_validator.py` actually
performs. Errors fail validation; warnings are reported but pass.

### Registry File
- Valid JSON whose top level is an array of model objects (error)

### Required Fields and Types
- Required, non-empty: `name`, `dockerfile`, `scripts`, `n_gpus`, `owner`, `tags` (error)
- `args` must be present and a string; an empty string is allowed (error)
- `name`, `dockerfile`, `scripts`, `owner` are strings; `tags` is a list;
  `n_gpus` is a string; `timeout` is a number (error)
- Empty `tags` list (warning)

### Naming Convention
- Matches `^[a-z]+_[a-z0-9_.\-]+$`, i.e. `{framework}_{project}_{workload}`
  in lowercase letters, digits, `.`, `_` and `-` (error)

### Cross-Field Consistency (warnings)
- `tags` contains the framework prefix of the name (e.g. `pyt` for `pyt_*`)
- `dockerfile` contains the first two name components (e.g. `pyt_vllm`)

### Framework-Specific Rules
| Name prefix | Required tags (error) | Recommended `args` flags (warning) |
|---|---|---|
| `pyt_vllm` | `pyt`, `vllm` | `--config`, `--model_repo` |
| `pyt_sglang` | `pyt`, `sglang` | `--model_repo`, `--test_option` |
| `jax_maxtext` | `jax` | — |

### File References
- `{dockerfile}.ubuntu.amd.Dockerfile` and `scripts` exist, resolved relative
  to the directory containing the `models.json` (error; a warning instead
  with `--allow-missing-files`, for models still being generated)

### Duplicate Names
- No name repeated within the file, or registered in any other registry file
  (root `models.json` plus every `scripts/*/models.json`) (error)

Not checked: `args` values beyond the recommended flags, config files
referenced from `args`, `multiple_results` naming, and `training_precision`.
Review those by hand.

## Implementation

Run the validator CLI from the repository root (see Integration below). To
use it from Python instead, also run from the repository root:

```python
from tools.validation import ModelConfigValidator

# base_dir = directory of the models.json the entry belongs to
validator = ModelConfigValidator(verbose=True, base_dir='scripts/vllm')
valid = validator.validate(model_entry)
```

## Output Format

Real output for a vLLM entry missing the `pyt` tag and `--config`:

```
Model Validation Report for: pyt_vllm_newmodel
============================================================

❌ ERRORS (1):
  1. Missing required tag for pyt_vllm: pyt

⚠️  WARNINGS (2):
  1. Framework tag 'pyt' not in tags list. Tags: ['vllm', 'inference']
  2. Recommended argument missing: --config

❌ Validation failed: 1 errors, 2 warnings
```

Duplicate-name errors are printed after the per-model reports, e.g.
`❌ Duplicate model name across registries: 'pyt_vllm_x' also registered in: scripts/vllm/models.json`.

When reporting to the user, summarize the errors first, then the warnings,
and suggest a concrete fix for each.

## Integration

- Can be called standalone: `python3 tools/validation/model_config_validator.py --models-json scripts/{model_dir}/models.json --model MODEL_NAME`
  (models are registered per-directory; `--models-json` must point at the
  file that actually contains the model, not the root `models.json`)
- Use `--all` instead of `--models-json`/`--model` to validate every registry
  file in the repo in one pass, including cross-registry duplicate names
- Integrated into mad-add-model skill for automatic validation
- Can be run in CI/CD pipelines
