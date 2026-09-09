"""
Discover Primus example configs as madengine models (optional).

Convention-based: globs examples/*/configs/**/*.yaml from the Primus submodule (scripts/Primus),
so all launchers (megatron, megatron_bridge, torchtitan, maxtext, moe_package, etc.) are
discovered. New launchers added under examples/<name>/configs/ are picked up automatically.
All discovered models use the same dockerfile and run.sh; args pass --config_path <relpath>.
For SLURM/K8s, supply distributed (launcher, nnodes, primus.config_path) via additional_context.
"""
import os
import glob

try:
    from madengine.utils.discover_models import CustomModel  # madengine v2
except ImportError:
    from madengine.tools.discover_models import CustomModel  # madengine v1

# This file lives in scripts/primus_train; Primus submodule is scripts/Primus
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PRIMUS_ROOT = os.path.normpath(os.path.join(THIS_DIR, "..", "Primus"))
# One glob for all launchers: examples/<launcher>/configs/**/*.yaml
CONFIGS_GLOB = os.path.join(PRIMUS_ROOT, "examples", "*", "configs", "**", "*.yaml")

# JAX backends have their own dedicated discovery (scripts/jax-maxtext/get_models_json.py,
# scripts/jax-maxdiffusion/get_models_json.py) with correct dockerfiles and arch filtering.
# Discovering them here too would create duplicates on the wrong base image.
JAX_BACKENDS = {"maxtext", "maxdiffusion"}

# Precision is encoded in the config file name (llama3.1_8B-MXFP4-pretrain.yaml,
# gdn_1B_BF16-pretrain.yaml). Longest token first, so MXFP8 is not matched as FP8.
# Configs that carry no precision token (mamba_130M_pretrain.yaml, the diffusion
# configs) report "" — madengine's convention for unknown — rather than a guess.
# extract_primus_perf.py keeps its own copy of this table: it runs inside the
# container, where this module's madengine import is not available.
_PRECISION_TOKENS = ("MXFP8", "MXFP4", "BF16", "FP16", "FP8", "FP4")


def precision_from_config_name(short_name: str) -> str:
    """Return the madengine training_precision for a Primus config basename."""
    upper = short_name.upper()
    for token in _PRECISION_TOKENS:
        if token in upper:
            return token.lower()
    return ""


def list_models():
    # Default/smoke-test entry -> "primus_train/default". Lives here (not root models.json)
    # so this directory has one registration file, per madengine's models.json vs.
    # get_models_json.py rule. HSA_NO_SCRATCH_RECLAIM and the other arch-specific perf env
    # are not modeled here: madengine reads no per-model env field on the local Docker path
    # (CustomModel has none, and get_env_arg only consumes context docker_env_vars), so
    # run.sh applies them itself from MAD_SYSTEM_GPU_ARCHITECTURE at launch time.
    models = [
        CustomModel(
            name="default",
            dockerfile="../../docker/primus",
            dockercontext=".",
            scripts="run.sh",
            n_gpus="-1",
            owner="mad.support@amd.com",
            tags=["training", "primus", "megatron", "pretrain"],
            args="",
        )
    ]
    if not os.path.isdir(PRIMUS_ROOT):
        return models
    # recursive=True is required for ** to span directories; without it configs nested one
    # level deeper (examples/<launcher>/configs/<arch>/diffusion/*.yaml) are silently skipped,
    # which hides all of the nemo_automodel backend added in Primus v26.6.
    for yaml_path in sorted(glob.glob(CONFIGS_GLOB, recursive=True)):
        rel_path = os.path.relpath(yaml_path, PRIMUS_ROOT)
        # Path shape: examples/<launcher>/configs/<arch>/<file>.yaml
        parts = rel_path.split(os.sep)
        if len(parts) < 5:
            continue
        launcher = parts[1]   # megatron, torchtitan, megatron_bridge, etc.
        if launcher in JAX_BACKENDS:
            continue
        arch = parts[3]       # MI300X, MI355X, etc.
        short_name = os.path.splitext(os.path.basename(yaml_path))[0]
        # discover_models prefixes with dirname (primus_train/), so no prefix here
        name = f"{launcher}_{arch}_{short_name}"
        tags = ["primus", launcher, arch, short_name]
        models.append(
            CustomModel(
                name=name,
                dockerfile="../../docker/primus",
                dockercontext=".",
                scripts="run.sh",
                data="",
                n_gpus="8",
                owner="mad.support@amd.com",
                timeout=86400,
                training_precision=precision_from_config_name(short_name),
                tags=tags,
                args=f"--config_path {rel_path}",
                multiple_results="primus_perf_output.csv",
            )
        )
    return models
