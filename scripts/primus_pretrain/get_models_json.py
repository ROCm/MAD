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

# This file lives in scripts/primus_pretrain; Primus submodule is scripts/Primus
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PRIMUS_ROOT = os.path.normpath(os.path.join(THIS_DIR, "..", "Primus"))
# One glob for all launchers: examples/<launcher>/configs/**/*.yaml
CONFIGS_GLOB = os.path.join(PRIMUS_ROOT, "examples", "*", "configs", "**", "*.yaml")


def list_models():
    models = []
    if not os.path.isdir(PRIMUS_ROOT):
        return models
    for yaml_path in sorted(glob.glob(CONFIGS_GLOB, recursive=True)):
        rel_path = os.path.relpath(yaml_path, PRIMUS_ROOT)
        # Path shape: examples/<launcher>/configs/<arch>/<file>.yaml
        parts = rel_path.split(os.sep)
        if len(parts) < 5:
            continue
        launcher = parts[1]   # megatron, torchtitan, megatron_bridge, etc.
        arch = parts[3]       # MI300X, MI355X, etc.
        short_name = os.path.splitext(os.path.basename(yaml_path))[0]
        # discover_models prefixes with dirname (primus_pretrain/), so no prefix here
        name = f"{launcher}_{arch}_{short_name}"
        tags = ["primus", launcher, arch, short_name]
        models.append(
            CustomModel(
                name=name,
                dockerfile="../../docker/primus",
                scripts="run.sh",
                data="",
                n_gpus="8",
                owner="mad.support@amd.com",
                timeout=86400,
                training_precision="bf16",
                tags=tags,
                args=f"--config_path {rel_path}",
                multiple_results="primus_perf_output.csv",
            )
        )
    return models
