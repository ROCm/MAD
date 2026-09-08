"""
Discover Primus JAX/MaxDiffusion example configs as madengine models.

MaxDiffusion-only: globs examples/maxdiffusion/configs/**/*.yaml from the Primus
submodule (scripts/Primus). These run through the Primus `maxdiffusion` (JAX)
backend — Google's MaxDiffusion WAN/FLUX trainers launched via primus/cli, the
same way jax-maxtext runs MaxText. New MaxDiffusion configs added under
examples/maxdiffusion/configs/<DEVICE>/ are picked up automatically.

All discovered models build docker/primus_maxdiffusion (rocm/jax-training based +
maxdiffusion installed) and run through run.sh; args pass --config_path <relpath>.
Mirrors scripts/jax-maxtext/get_models_json.py.
"""
import os
import glob
import subprocess
import sys

try:
    from madengine.utils.discover_models import CustomModel  # madengine v2
except ImportError:
    from madengine.tools.discover_models import CustomModel  # madengine v1

# This file lives in scripts/jax-maxdiffusion; Primus submodule is scripts/Primus.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PRIMUS_ROOT = os.path.normpath(os.path.join(THIS_DIR, "..", "Primus"))
FETCH_SCRIPT = os.path.normpath(os.path.join(THIS_DIR, "..", "..", "tools", "fetch_primus.sh"))
CONFIGS_GLOB = os.path.join(PRIMUS_ROOT, "examples", "maxdiffusion", "configs", "**", "*.yaml")

# JAX/MaxDiffusion image, relative to scripts/jax-maxdiffusion.
DOCKERFILE = "../../docker/primus_maxdiffusion"

# Multi-node-only models (matched against the base model token of a config
# filename, e.g. "wan2.1_14b-pretrain" -> "wan2.1_14b"). None yet: the current
# WAN/FLUX benchmark configs run single-node on 8 GPUs. Override with
# JAX_MAXDIFFUSION_INCLUDE_MULTINODE=1 to discover any listed here.
MULTINODE_MODELS = set()

# Device -> GPU arch that should SKIP that device's configs (madengine skip_gpu_arch).
# Mirrors jax-maxtext: a single discovery works on both host types; only the
# host-appropriate configs run, the others are recorded as SKIPPED.
ARCH_SKIP_GPU = {"MI300X": "gfx950", "MI355X": "gfx942"}


def _precision_from_name(short_name: str) -> str:
    """Infer training precision from a config filename (…-fp8-…, …-nanoo_fp8-…, else bf16)."""
    lowered = short_name.lower()
    if "nanoo_fp8" in lowered:
        return "nanoo_fp8"
    if "fp8" in lowered:
        return "fp8"
    return "bf16"


def _have_primus():
    """Report whether the Primus checkout these models come from is usable.

    Mirrors scripts/jax-maxtext/get_models_json.py. Discovery is the only host-side
    hook that runs before the image build, which is where the checkout has to exist
    (both primus_* dockerfiles COPY it from the build context). Fetching is opt-in
    even so: cloning over the network is a surprising side effect of listing models,
    and it would fire on every madengine invocation.
    """
    if os.path.isdir(PRIMUS_ROOT):
        return True
    if os.environ.get("MAD_AUTO_FETCH_PRIMUS", "") not in ("", "0"):
        print("MAD_AUTO_FETCH_PRIMUS is set: fetching Primus into %s" % PRIMUS_ROOT, file=sys.stderr)
        rc = subprocess.call(["bash", FETCH_SCRIPT], stdout=sys.stderr.fileno())
        if rc == 0 and os.path.isdir(PRIMUS_ROOT):
            return True
        print("ERROR: %s failed (exit %d); no JAX/MaxDiffusion models discovered." % (FETCH_SCRIPT, rc), file=sys.stderr)
        return False
    print(
        "WARNING: no Primus checkout at %s, so no JAX/MaxDiffusion models can be discovered. "
        "Run tools/fetch_primus.sh, or set MAD_AUTO_FETCH_PRIMUS=1 to fetch it here." % PRIMUS_ROOT,
        file=sys.stderr,
    )
    return False


def list_models():
    # Default/smoke-test entry -> "jax-maxdiffusion/default". Reachable only via the scoped
    # name (--tags jax-maxdiffusion/default); tags is ["default"] with no family/arch/
    # precision tags so it never appears in sweeps like --tags maxdiffusion or --tags jax
    # and cannot duplicate the per-yaml entry for the same config.
    models = [
        CustomModel(
            name="default",
            dockerfile=DOCKERFILE,
            dockercontext=".",
            scripts="run.sh",
            data="",
            n_gpus="8",
            owner="mad.support@amd.com",
            timeout=86400,
            training_precision="bf16",
            tags=["default"],
            args="--config_path examples/maxdiffusion/configs/MI355X/wan2.1_1.3b-pretrain.yaml",
            multiple_results="primus_perf_output.csv",
            skip_gpu_arch=ARCH_SKIP_GPU["MI355X"],
        )
    ]
    if not _have_primus():
        return models
    include_multinode = os.environ.get("JAX_MAXDIFFUSION_INCLUDE_MULTINODE", "") not in ("", "0")
    for yaml_path in sorted(glob.glob(CONFIGS_GLOB)):
        rel_path = os.path.relpath(yaml_path, PRIMUS_ROOT)
        # Path shape: examples/maxdiffusion/configs/<arch>/<file>.yaml
        parts = rel_path.split(os.sep)
        if len(parts) < 5:
            continue
        arch = parts[3]       # MI300X, MI355X, etc.
        short_name = os.path.splitext(os.path.basename(yaml_path))[0]
        base_model = short_name.split("-")[0]
        if base_model in MULTINODE_MODELS and not include_multinode:
            continue
        precision = _precision_from_name(short_name)
        # discover_models prefixes discovered names with this dir (jax-maxdiffusion/),
        # so the final madengine tag is jax-maxdiffusion/maxdiffusion_<arch>_<short_name>.
        name = f"maxdiffusion_{arch}_{short_name}"
        tags = ["maxdiffusion", "jax", arch, short_name, precision]
        models.append(
            CustomModel(
                name=name,
                dockerfile=DOCKERFILE,
                dockercontext=".",
                scripts="run.sh",
                data="",
                n_gpus="8",
                owner="mad.support@amd.com",
                timeout=86400,
                training_precision=precision,
                tags=tags,
                args=f"--config_path {rel_path}",
                multiple_results="primus_perf_output.csv",
                skip_gpu_arch=ARCH_SKIP_GPU.get(arch, ""),
            )
        )
    return models
