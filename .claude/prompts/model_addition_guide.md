# MAD Model Addition Guide

## Quick Start with Claude Skills

The fastest way to add a new model is using the `mad-add-model` Claude skill:

```
User: I want to add a new vLLM model for Llama-3.2-8B inference
```

Claude will guide you through an interactive workflow that:
1. Gathers all required information
2. Generates Dockerfile from templates
3. Creates run scripts
4. Updates models.json
5. Validates the configuration
6. Optionally tests the Docker build

**Time required:** 5-10 minutes (vs 30-60 minutes manually)

## Manual Process (Without Skills)

If you prefer to add models manually, follow these steps:

### 1. Choose a Model Name

Follow the naming convention: `{framework}_{project}_{workload}`

Examples:
- `pyt_vllm_llama-3.2-8b` - PyTorch vLLM for Llama 3.2 8B
- `jax_maxtext_train_llama-2-7b` - JAX MaxText training for Llama 2 7B
- `pyt_huggingface_bert` - PyTorch HuggingFace for BERT

### 2. Create Dockerfile

Skip this step if the model reuses a shared framework Dockerfile (e.g.
`docker/pyt_vllm` for vLLM, as in the example below). Otherwise:

Location: `docker/{model_name}.ubuntu.amd.Dockerfile`

Template structure:
```dockerfile
# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
# MIT License header...
###############################################################################
ARG BASE_DOCKER=<base_image>
FROM $BASE_DOCKER

USER root
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

# Install dependencies
RUN pip install --no-cache-dir <packages>

# Record configuration
RUN pip3 list

ENTRYPOINT [""]
```

### 3. Create Run Script

For frameworks with existing scripts (vLLM, SGLang), you can reuse them.

For new models, create: `scripts/{model_name}/run.sh`

Template:
```bash
#!/bin/bash
###############################################################################
# MIT License header...
###############################################################################
set -ex

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --arg1) ARG1="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Execute model
python script.py

# Output performance
echo "performance: <value> <metric>"
```

### 4. Update models.json

MAD models are registered via **per-directory `models.json` files**, not a
single root `models.json` — the root file is reserved for directories that
provide a `get_models_json.py`. Paths inside an entry are relative to the
directory containing that `models.json`.

For a model reusing the shared vLLM Dockerfile and script, add the entry to
`scripts/vllm/models.json` (create `scripts/{model_name}/models.json` instead
for a model with its own Dockerfile/script):
```json
{
  "name": "pyt_vllm_llama-3.2-8b",
  "url": "",
  "dockerfile": "../../docker/pyt_vllm",
  "scripts": "run.sh",
  "data": "huggingface",
  "n_gpus": "-1",
  "owner": "mad.support@amd.com",
  "training_precision": "",
  "tags": ["pyt", "vllm", "vllm_default", "inference"],
  "timeout": -1,
  "args": "--model_repo meta-llama/Llama-3.2-8B --config configs/default.yaml"
}
```

### 5. Validate Configuration

```bash
python3 tools/validation/model_config_validator.py --models-json scripts/vllm/models.json --model pyt_vllm_llama-3.2-8b
```

### 6. Test

```bash
# Build Docker image (the shared Dockerfile registered in the entry above)
docker build -f docker/pyt_vllm.ubuntu.amd.Dockerfile -t ci-test .

# Run model
madengine run --tags pyt_vllm_llama-3.2-8b
```

## Framework-Specific Guides

### vLLM Models

**Dockerfile:** Reuse `docker/pyt_vllm`

**Script:** Reuse `scripts/vllm/run.sh`

**Args pattern:**
```
--model_repo <huggingface_model_id> --config configs/<config_type>.yaml
```

**Config types:**
- `default.yaml` - Standard throughput benchmarks
- `extended.yaml` - Extended configuration tests
- `accuracy.yaml` - Accuracy validation
- `unittest.yaml` - Unit tests

### SGLang Models

**Dockerfile:** Reuse `docker/pyt_sglang`

**Script:** Reuse `scripts/sglang/run.sh`

**Args pattern:**
```
--model_repo <model> --test_option <latency,throughput> --num_gpu <N> --datatype <fp16>
```

### PyTorch Training

**Dockerfile:** Use `docker/pyt_huggingface` or create custom

**Script:** Create in `scripts/{model_name}/run.sh`

**Required fields:**
- `training_precision` - fp16, fp32, etc.
- `data` - Dataset reference

### JAX MaxText

**Dockerfile:** Use `docker/jax_maxtext`

**Script:** Reuse `scripts/jax-maxtext/run.sh`

**Required:**
- Environment YAML file in `scripts/jax-maxtext/env_scripts/`

## Common Issues

### Issue: Duplicate model name
**Solution:** Check the root `models.json` and every `scripts/*/models.json` for an existing entry with the same name, then choose a unique one

### Issue: Dockerfile build fails
**Solution:** Verify base image exists, check package names

### Issue: Script execution fails
**Solution:** Check environment variables, GPU availability

### Issue: Performance metrics not captured
**Solution:** Ensure script outputs: `echo "performance: <value> <metric>"`

## Best Practices

1. **Naming:** Always follow `{framework}_{project}_{workload}` convention
2. **Reuse:** Use existing Dockerfiles and scripts when possible
3. **Tags:** Include framework, project, and model type tags
4. **Validation:** Always validate before committing
5. **Testing:** Test Docker build and execution locally
6. **Documentation:** Add comments explaining custom configurations

## Getting Help

- Use `/mad-add-model` skill for guided workflow
- Check existing similar models in the relevant `scripts/{dir}/models.json`
- Validate with `/mad-validate-model`
