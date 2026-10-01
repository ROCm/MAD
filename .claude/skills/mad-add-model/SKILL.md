---
name: mad-add-model
description: Guided workflow to add a new AI model to the MAD platform (Dockerfile, run script, per-directory models.json entry, docs, and validation). Use when the user asks to add, create, or register a new model for MAD, or names "add model"/"new model".
---

# mad-add-model Skill Instructions

## Purpose
Guide users through adding a new AI model to the MAD platform with minimal manual work and maximum consistency.

## Workflow

All paths are relative to the **repository root**; run every command from
there. Steps that say **ask the user** mean: ask in the conversation (using
your structured-question tool if you have one) and wait for the answers — do
not assume defaults and continue on your own.

### Step 1: Gather Information (Interactive)

**Ask the user** for the following information, and do not generate any
files until you have the answers:

1. **Model Name Components:**
   - Framework (PyTorch, JAX)
   - Project (vllm, sglang, huggingface, maxtext, etc.)
   - Workload identifier (model name, e.g., llama-3.2-8b)
   
   Generate full name following convention: `{framework}_{project}_{workload}`
   Example: `pyt_vllm_llama-3.2-8b`

2. **Repository URL:**
   - Git repository to clone (can be empty for HuggingFace models)
   
3. **Model Type:**
   - Training or Inference

4. **GPU Requirements:**
   - Number of GPUs needed (-1 for all available)
   - Specific GPU architecture requirements (gfx942, gfx90a, etc.) if any

5. **Precision:**
   - fp16, fp32, fp8, int8, etc.

6. **Framework-Specific Details:**
   
   **For vLLM:**
   - HuggingFace model repository
   - Config type (default, extended, accuracy, unittest)
   - Benchmark type (throughput, latency, accuracy, all)
   - Multiple results CSV name (optional)
   
   **For SGLang:**
   - Model repository
   - Test options (latency, throughput, accuracy)
   - Data types (fp16, fp8, etc.)
   
   **For PyTorch Training:**
   - Training dataset
   - Batch size
   - Number of epochs
   
   **For JAX MaxText:**
   - Model variant
   - Environment YAML file

7. **Dependencies:**
   - Additional pip packages needed
   - System packages (apt) if any

8. **Owner/Contact:**
   - Email address (default: mad.support@amd.com)

### Step 2: Validate Name

Before generating files, validate:

1. **Check for duplicates** across **all** registered model files, not just
   the destination directory: search the root `models.json` and every
   `scripts/*/models.json` (models are registered per-directory; see Step 5)
   for an existing entry with the same name, e.g.
   `grep -l '"name": "{model_name}"' models.json scripts/*/models.json`
2. **Verify naming convention:** Must match `^[a-z]+_[a-z0-9_.\-]+$`
3. **Suggest corrections** if needed
4. **Ask the user** to confirm the final name

### Step 2b: Choose the File Layout

Decide which layout the model uses and fix two placeholders that every later
step uses:

| Layout | When | `{script_dir}` | `{dockerfile_name}` |
|---|---|---|---|
| **Shared framework** | A vLLM/SGLang model that runs with the existing shared Dockerfile and `run.sh`, needing only new `args`/configs | `scripts/{project}` (e.g. `scripts/vllm`) | the existing shared one (e.g. `pyt_vllm`) |
| **Workload-specific** | Anything needing its own Dockerfile or run script | `scripts/{model_name}` | `{model_name}` |

For the shared-framework layout, skip Steps 3, 4 and 6: the Dockerfile,
`run.sh` and README already exist in `{script_dir}`.

### Step 3: Generate Dockerfile

1. **Select the template** (`--framework` value for the renderer; templates
   live in `.claude/skills/mad-generate-dockerfile/framework_templates/`):
   - `pytorch` for PyTorch
     (set `build_flash_attention: true` for diffusion models that need ROCm flash-attention)
   - `jax` for JAX
   - `vllm` for vLLM
   - `sglang` for SGLang
   - `atom` for ATOM
   - `xdit` for xDiT
   - `sglang_disagg` for SGLang disaggregated inference
   - `vllm_disagg` for vLLM disaggregated inference

   See `.claude/skills/mad-generate-dockerfile/SKILL.md` for each template's
   context variables.

2. **Read existing similar Dockerfiles** for base image references:
   - For PyTorch: Read `docker/pyt_wan2.1_inference.ubuntu.amd.Dockerfile`
   - For JAX: Read `docker/primus_maxtext.ubuntu.amd.Dockerfile`
   - For vLLM: Read `docker/pyt_vllm.ubuntu.amd.Dockerfile`
   - For SGLang: Read `docker/pyt_sglang.ubuntu.amd.Dockerfile`
   - For ATOM: Read `docker/pyt_atom.ubuntu.amd.Dockerfile`
   - For xDiT: Read `docker/pyt_xdit.ubuntu.amd.Dockerfile`
   - For SGLang disagg: Read `docker/sglang_disagg_inference.ubuntu.amd.Dockerfile`
   - For vLLM disagg: Read `docker/vllm_disagg_inference.ubuntu.amd.Dockerfile`
   - Extract base image ARG value

3. **Render a preview** with the renderer script (never fill in the template
   by hand), passing the context as a JSON object:
   ```bash
   python3 .claude/skills/mad-generate-dockerfile/scripts/render_dockerfile.py \
     --framework {framework} \
     --context '{"base_image": "<extracted_base_image>", "apt_packages": [...], "pip_packages": [...], "git_repos": [{"url": "...", "path": "...", "checkout": "..."}]}'
   ```
   Omit keys the user did not provide; `year` defaults to the current year.
   It needs `jinja2` (`pip install jinja2` if missing).

4. **Show the preview** to the user and **ask the user** to confirm.

5. **Write the Dockerfile** by re-running the same command with
   `--output docker/{dockerfile_name}.ubuntu.amd.Dockerfile`
   (workload-specific layout only; see Step 2b)

### Step 4: Generate Run Script

1. **Script path:** `{script_dir}/run.sh` (workload-specific layout only;
   the shared-framework layout reuses the existing script, see Step 2b)

2. **Generate script content:**
   - Include MIT license header
   - Add GPU architecture detection if needed
   - Parse framework-specific arguments
   - Execute framework command
   - Output performance metrics (`echo "performance: <value> <unit>"`)
   - Handle errors

3. **For vLLM models:**
   ```bash
   #!/bin/bash
   set -ex
   
   export HF_HUB_CACHE="/myworkspace"
   
   # Parse arguments
   while [[ "$#" -gt 0 ]]; do
       case $1 in
           --model_repo) MODEL="$2"; shift ;;
           --config) CONFIG="$2"; shift ;;
           --benchmark) BENCHMARK="$2"; shift ;;
           *) echo "Unknown parameter: $1"; exit 1 ;;
       esac
       shift
   done
   
   # Run vLLM benchmark
   python3 -u run_vllm.py --config $CONFIG --model $MODEL --benchmark $BENCHMARK
   ```

4. **Save** it and make it executable (`chmod +x {script_dir}/run.sh`)

### Step 5: Register in models.json

MAD models are registered via **per-directory `models.json` files** in
`{script_dir}` (see Step 2b) — not a single root `models.json`. The root
`models.json` is reserved for models whose directory has a
`get_models_json.py` instead.

1. **Read the existing `{script_dir}/models.json`** if it exists (always true
   for the shared-framework layout), otherwise it will be created.

2. **Create new entry** with paths relative to `{script_dir}/`:
   ```json
   {
     "name": "{model_name}",
     "dockerfile": "../../docker/{dockerfile_name}",
     "scripts": "run.sh",
     "url": "{user_provided_url}",
     "n_gpus": "{user_provided_gpu_count}",
     "owner": "{user_provided_email}",
     "training_precision": "{user_provided_precision}",
     "tags": ["{framework}", "{project}", "{model_type}", ...],
     "args": "{framework_specific_args}"
   }
   ```
   - In both layouts the run script sits next to `models.json`, so
     `"scripts": "run.sh"` (as in `scripts/vllm/models.json`).
   - `dockerfile` is relative to `{script_dir}` — a file under `docker/` is
     reached via `../../docker/...`, without the `.ubuntu.amd.Dockerfile`
     suffix.

3. **Add optional fields** if provided:
   - `data` - Data source identifier
   - `timeout` - Custom timeout
   - `multiple_results` - CSV filename for multi-model results

4. **Parse existing JSON** (or start a new array), append new entry, pretty-print

5. **Validate JSON syntax** before writing

6. **Write `{script_dir}/models.json`**, then confirm it still parses:
   `python3 -m json.tool {script_dir}/models.json > /dev/null`

### Step 6: Create Documentation

1. **Generate README.md** at `{script_dir}/README.md` (workload-specific
   layout only; see Step 2b):
   ```markdown
   # {Model Name}
   
   ## Description
   {Auto-generated description based on framework and model}
   
   ## Usage
   ```bash
   madengine run --tags {model_name}
   ```
   
   ## Configuration
   - Framework: {framework}
   - GPUs: {n_gpus}
   - Precision: {precision}
   
   ## Expected Output
   Performance metrics in CSV format: perf_{model_name}.csv
   
   ## Owner
   {owner_email}
   ```

2. **Save documentation**

### Step 7: Validate Configuration

1. **Run validation tools:**
   - `python3 tools/validation/model_config_validator.py --models-json {script_dir}/models.json --model {model_name}` on the new entry
   - `python3 tools/validation/dockerfile_validator.py docker/{dockerfile_name}.ubuntu.amd.Dockerfile` on the Dockerfile
   - `python3 tools/validation/script_validator.py {script_dir}/run.sh` on the run script

2. **Check for errors:**
   - Missing required fields
   - Invalid naming convention
   - Non-existent file references
   - Framework-specific issues

3. **Display validation report** to user

4. **Offer to fix issues** if any found

### Step 8: Optional Docker Build Test

**Ask the user:** "Would you like me to test building the Docker image?"

If yes, from the repository root:
```bash
docker build -f docker/{dockerfile_name}.ubuntu.amd.Dockerfile \
  -t ci-{model_name}:test .
```

Show build output and report success/failure. In a sandboxed agent (e.g.
Codex) this needs network and Docker-socket access, so it may need the user's
approval to run outside the sandbox; if it can't run, give the user the
command to run themselves.

## Error Handling

- If any step fails, clearly explain the error
- Offer to retry with corrections
- Don't leave partial files (clean up on failure)
- Provide helpful suggestions for common issues

## Success Message

```
✅ Model {model_name} added successfully!

Generated files:
- {script_dir}/models.json (new entry)
- docker/{dockerfile_name}.ubuntu.amd.Dockerfile   (workload-specific layout only)
- {script_dir}/run.sh                              (workload-specific layout only)
- {script_dir}/README.md                           (workload-specific layout only)

Validation: All checks passed ✓

Next steps:
1. Review the generated files
2. Test: madengine run --tags {model_name}
3. Commit changes to git
```

## Agent Notes

- This skill works in Claude Code, Cursor and Codex. It only needs file
  read/write/edit, a shell, and a way to ask the user questions; use whichever
  tools your agent provides for those.
- Dockerfiles are rendered by
  `.claude/skills/mad-generate-dockerfile/scripts/render_dockerfile.py`
  (Jinja2); validation uses the scripts in `tools/validation/`.
