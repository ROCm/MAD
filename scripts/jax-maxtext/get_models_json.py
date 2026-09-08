"""
Discover Primus JAX/MaxText example configs as madengine models.

MaxText-only: globs examples/maxtext/configs/**/*.yaml from the Primus submodule
(scripts/Primus). No Megatron/TorchTitan configs are discovered here — those backends
use rocm/primus:* images and their own MAD integration. New MaxText configs added under
examples/maxtext/configs/<DEVICE>/ are picked up automatically.

All discovered models build docker/primus_maxtext (rocm/jax-training:maxtext-*, the only
image that ships JAX) and run through run.sh; args pass --config_path <relpath>.
For SLURM/K8s, supply distributed settings via additional_context.
"""
import os
import glob
import subprocess
import sys

try:
    from madengine.utils.discover_models import CustomModel  # madengine v2
except ImportError:
    from madengine.tools.discover_models import CustomModel  # madengine v1

# This file lives in scripts/jax-maxtext; Primus submodule is scripts/Primus
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PRIMUS_ROOT = os.path.normpath(os.path.join(THIS_DIR, "..", "Primus"))
FETCH_SCRIPT = os.path.normpath(os.path.join(THIS_DIR, "..", "..", "tools", "fetch_primus.sh"))
CONFIGS_GLOB = os.path.join(PRIMUS_ROOT, "examples", "maxtext", "configs", "**", "*.yaml")

# JAX/MaxText image (rocm/jax-training:maxtext-*), relative to scripts/jax-maxtext.
DOCKERFILE = "../../docker/primus_maxtext"

# Multi-node-only models. The MAD JAX/MaxText suite mirrors the single-node
# env_scripts set (see ROCm/MAD scripts/jax-maxtext/env_scripts); these large
# models require multiple nodes and are intentionally NOT discovered as
# single-node madengine models. Matched against the base model token of a config
# filename (the part before the first '-', e.g. "llama3.1_405B-fp8-pretrain" ->
# "llama3.1_405B"). Override with JAX_MAXTEXT_INCLUDE_MULTINODE=1 to discover them.
MULTINODE_MODELS = {"grok1", "llama3.1_405B", "mixtral_8x22B"}

# Device -> GPU arch that should SKIP that device's configs (madengine skip_gpu_arch).
# MI300X configs are tuned for gfx942 and skipped on gfx950; MI355X configs are tuned
# for gfx950 and skipped on gfx942. So a single discovery works on both host types:
# only the host-appropriate configs run, the others are recorded as SKIPPED.
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

    Discovery is the only host-side hook that runs before the image build, which is
    where the checkout has to exist (both primus_* dockerfiles COPY it from the build
    context). Fetching is opt-in even so: cloning over the network is a surprising
    side effect of listing models, and it would fire on every madengine invocation.
    Everything goes to stderr to keep discovery's stdout clean.
    """
    if os.path.isdir(PRIMUS_ROOT):
        return True
    if os.environ.get("MAD_AUTO_FETCH_PRIMUS", "") not in ("", "0"):
        print("MAD_AUTO_FETCH_PRIMUS is set: fetching Primus into %s" % PRIMUS_ROOT, file=sys.stderr)
        rc = subprocess.call(["bash", FETCH_SCRIPT], stdout=sys.stderr.fileno())
        if rc == 0 and os.path.isdir(PRIMUS_ROOT):
            return True
        print("ERROR: %s failed (exit %d); no JAX/MaxText models discovered." % (FETCH_SCRIPT, rc), file=sys.stderr)
        return False
    # Say something rather than returning an empty list, which reads as "no MaxText
    # models exist" instead of "the checkout they are discovered from is missing".
    print(
        "WARNING: no Primus checkout at %s, so no JAX/MaxText models can be discovered. "
        "Run tools/fetch_primus.sh, or set MAD_AUTO_FETCH_PRIMUS=1 to fetch it here." % PRIMUS_ROOT,
        file=sys.stderr,
    )
    return False


def list_models():
    # Default/smoke-test entry -> "jax-maxtext/default". Reachable only via the scoped
    # name (--tags jax-maxtext/default); tags is ["default"] with no family/arch/precision
    # tags so it never appears in sweeps like --tags maxtext or --tags jax and cannot
    # duplicate the per-yaml entry for the same config.
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
            args="--config_path examples/maxtext/configs/MI300X/llama2_7B-bf16-pretrain.yaml",
            multiple_results="primus_perf_output.csv",
            skip_gpu_arch=ARCH_SKIP_GPU["MI300X"],
        )
    ]
    if not _have_primus():
        return models
    include_multinode = os.environ.get("JAX_MAXTEXT_INCLUDE_MULTINODE", "") not in ("", "0")
    for yaml_path in sorted(glob.glob(CONFIGS_GLOB)):
        rel_path = os.path.relpath(yaml_path, PRIMUS_ROOT)
        # Path shape: examples/maxtext/configs/<arch>/<file>.yaml
        parts = rel_path.split(os.sep)
        if len(parts) < 5:
            continue
        arch = parts[3]       # MI300X, MI355X, etc.
        short_name = os.path.splitext(os.path.basename(yaml_path))[0]
        # Skip multi-node-only models unless explicitly requested.
        base_model = short_name.split("-")[0]
        if base_model in MULTINODE_MODELS and not include_multinode:
            continue
        precision = _precision_from_name(short_name)
        # discover_models prefixes discovered names with this dir (jax-maxtext/), so the
        # final madengine tag is jax-maxtext/maxtext_<arch>_<short_name>. No prefix here.
        name = f"maxtext_{arch}_{short_name}"
        tags = ["maxtext", "jax", arch, short_name, precision]
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
