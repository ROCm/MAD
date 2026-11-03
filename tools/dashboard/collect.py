"""
Collect a MAD status snapshot: model support, ROCm images, QA scope, CI coverage, roadmap.

Everything the repository already knows is derived here: the model registry comes from
madengine's own discovery (scripts/*/models.json and scripts/*/get_models_json.py), base
images from the Dockerfiles. ROCm versions come from image tags, else from the images' registry metadata
(image_registry.py). Facts the repository cannot know (CI matrix, QA statements, public
roadmap, ROCm versions neither source states) come from the curated dashboard/status.yaml. The resulting snapshot is plain JSON so a release can be frozen
into dashboard/releases/<version>.json and re-rendered later without the repo state.
"""
import argparse
import ast
import datetime
import glob
import os
import re
import subprocess
import sys

import yaml

import image_registry

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
STATUS_PATH = os.path.join(REPO_ROOT, "dashboard", "status.yaml")
INTERNAL_ROADMAP_PATH = os.path.join(REPO_ROOT, "dashboard", "roadmap.internal.yaml")
PRIMUS_ROOT = os.path.join(REPO_ROOT, "scripts", "Primus")

SCHEMA = 2

# Registries whose get_models_json.py hides multi-node-only configs unless this is set.
MULTINODE_SWITCHES = {
    "jax-maxtext": "JAX_MAXTEXT_INCLUDE_MULTINODE",
    "jax-maxdiffusion": "JAX_MAXDIFFUSION_INCLUDE_MULTINODE",
}

# skip_gpu_arch is ISA-level; these are the Instinct products behind each ISA.
ISA_SKUS = {
    "gfx942": ("MI300X", "MI325X"),
    "gfx950": ("MI350X", "MI355X"),
}
SKUS = tuple(sku for skus in ISA_SKUS.values() for sku in skus)

# Longest token first, so MXFP4 is not reported as FP4.
_PRECISION_TOKENS = ("mxfp8", "mxfp4", "nanoo_fp8", "bf16", "fp16", "fp32", "fp8", "fp4")

STATUS_KEYS = {
    "release": str,
    "frameworks": dict,
    "suite_tags": dict,
    "exclude": list,
    "rocm_overrides": dict,
    "roadmap": list,
    "roadmap_source": dict,
    "qa": dict,
    "ci": list,
}


def warn(message, warnings):
    warnings.append(message)
    print(f"WARNING: {message}", file=sys.stderr)


# --------------------------------------------------------------------------- status.yaml


def load_status(path=STATUS_PATH, internal_roadmap_path=None):
    """Load and validate the curated status file; optionally merge the internal roadmap."""
    with open(path) as f:
        status = yaml.safe_load(f) or {}
    unknown = set(status) - set(STATUS_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    for key, kind in STATUS_KEYS.items():
        status.setdefault(key, kind())
        if not isinstance(status[key], kind):
            raise ValueError(f"{path}: '{key}' must be a {kind.__name__}")
    if not status["release"]:
        raise ValueError(f"{path}: 'release' is required (e.g. v26.10)")
    version_key(status["release"])
    for key in ("covered", "not_covered"):
        status["qa"].setdefault(key, [])
    status["internal"] = False
    if internal_roadmap_path:
        with open(internal_roadmap_path) as f:
            items = yaml.safe_load(f) or []
        if not isinstance(items, list):
            raise ValueError(f"{internal_roadmap_path}: expected a list of roadmap items")
        status["roadmap"] = status["roadmap"] + [dict(item, internal=True) for item in items]
        status["internal"] = True
    return status


def version_key(version):
    """'v26.10' -> (26, 10), so releases sort numerically rather than as strings."""
    m = re.fullmatch(r"v(\d+)\.(\d+)", version)
    if not m:
        raise ValueError(f"release version must look like v26.10, got {version!r}")
    return int(m.group(1)), int(m.group(2))


# --------------------------------------------------------------------------- discovery


def discover_models(repo_root=REPO_ROOT):
    """Return every registered model as madengine sees it, names scoped as '<dir>/<name>'.

    Uses madengine's DiscoverModels so the registry rules (per-dir models.json,
    get_models_json.py, name scoping) are the engine's, not a copy of them.
    """
    from madengine.utils.discover_models import DiscoverModels

    # MODEL_DIR makes DiscoverModels copy another tree into the cwd; never wanted here.
    os.environ.pop("MODEL_DIR", None)
    # Discover every Primus config directory, not just the ones matching this host's GPU.
    os.environ["JAX_HOST_DEVICE"] = "all"
    # Include the multi-node-only configs; rows_from() marks them opt-in.
    for switch in MULTINODE_SWITCHES.values():
        os.environ[switch] = "1"
    cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        discovery = DiscoverModels(argparse.Namespace())
        discovery.discover_models()
    finally:
        os.chdir(cwd)

    models = [dict(m, registry=_registry_of(m), dynamic=False) for m in discovery.models]
    for custom in discovery.custom_models:
        entry = custom.to_dict()
        entry["dockerfile"] = os.path.normpath(os.path.join("scripts", custom._fs_rel_path, custom.dockerfile))
        # Static entries get this from discover_models(); custom ones only when selected for a run.
        entry["scripts"] = os.path.normpath(os.path.join("scripts", custom._fs_rel_path, custom.scripts))
        entry["registry"] = custom._fs_rel_path
        entry["dynamic"] = True
        models.append(entry)
    return models


def _registry_of(model):
    """Registry directory for a static model; root models.json entries have no scope."""
    return model["name"].rsplit("/", 1)[0] if "/" in model["name"] else "."


def primus_checked_out(primus_root=PRIMUS_ROOT):
    return bool(glob.glob(os.path.join(primus_root, "examples", "*", "configs")))


def multinode_only(repo_root=REPO_ROOT):
    """{registry: base models its get_models_json.py hides unless MULTINODE_SWITCHES[registry] is set}.

    Read as a literal (MULTINODE_MODELS = {...}) rather than imported, so the list stays
    the generator's own.
    """
    hidden = {}
    for registry in MULTINODE_SWITCHES:
        path = os.path.join(repo_root, "scripts", registry, "get_models_json.py")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            tree = ast.parse(f.read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "MULTINODE_MODELS" for t in node.targets):
                # set() has no literal form; anything else must be one.
                is_empty_set = isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "set"
                hidden[registry] = set() if is_empty_set else set(ast.literal_eval(node.value))
    return hidden


def is_opt_in(model, hidden):
    """True for a config its registry discovers only with the multi-node switch set."""
    config = arg_value(model.get("args", ""), "--config_path")
    if not config:
        return False
    base_model = os.path.splitext(os.path.basename(config))[0].split("-")[0]
    return base_model in hidden.get(model.get("registry"), set())


# --------------------------------------------------------------------------- Dockerfiles


def resolve_dockerfile(dockerfile, repo_root=REPO_ROOT):
    """'docker/pyt_vllm' -> 'docker/pyt_vllm.ubuntu.amd.Dockerfile' (repo-relative), or None.

    madengine appends '.<os>.<vendor>.Dockerfile'; anchoring on that suffix keeps
    'docker/vllm_disagg_inference' from also matching 'vllm_disagg_inference.glmv5.1.*'.
    """
    pattern = re.compile(re.escape(os.path.basename(dockerfile)) + r"\.[A-Za-z0-9]+\.amd\.Dockerfile$")
    directory = os.path.join(repo_root, os.path.dirname(dockerfile))
    if not os.path.isdir(directory):
        return None
    matches = sorted(f for f in os.listdir(directory) if pattern.fullmatch(f))
    if not matches:
        return None
    return os.path.join(os.path.dirname(dockerfile), matches[0])


def base_image(dockerfile_text):
    """Default base image: the BASE_DOCKER/BASE_IMAGE build arg, else the first literal FROM."""
    m = re.search(r"^\s*ARG\s+BASE_(?:DOCKER|IMAGE)\s*=\s*(\S+)", dockerfile_text, re.M)
    if m:
        return m.group(1).strip("\"'")
    for m in re.finditer(r"^\s*FROM\s+(\S+)", dockerfile_text, re.M):
        if "$" not in m.group(1):
            return m.group(1)
    return ""


def rocm_from_image(image):
    """ROCm version encoded in an image reference, or None.

    Handles 'rocm7.2', 'rocm7.2.4', 'rocm-7.2', the packed 'rocm720' / 'rocm630' forms and
    a bare major version ('rocm10').
    """
    m = re.search(r"rocm[-_]?(\d+\.\d+(?:\.\d+)?)", image, re.I)
    if m:
        return m.group(1)
    m = re.search(r"rocm(\d)(\d)(\d)(?!\d)", image, re.I)
    if m:
        return ".".join(m.groups())
    m = re.search(r"rocm(\d{1,2})(?![\d.])", image, re.I)
    if m:
        return m.group(1)
    return None


def image_tag(image):
    name = image.rsplit("/", 1)[-1]
    return name.split(":", 1)[1] if ":" in name else "latest"


def is_unpinned(image):
    """True for a moving tag: 'latest' and its variants ('vllm-latest')."""
    return re.search(r"(?:^|[-_.])latest$", image_tag(image)) is not None


def describe_image(dockerfile, overrides, warnings, repo_root=REPO_ROOT, lookup=None):
    """Base image and ROCm version for a madengine dockerfile reference.

    The version comes from rocm_overrides in status.yaml, else the image tag, else `lookup`
    (image_registry.rocm_version: the image's registry metadata). Unpinned images report
    "latest", since their ROCm moves with every rebuild; the source notes today's version.
    """
    path = resolve_dockerfile(dockerfile, repo_root)
    if path is None:
        warn(f"{dockerfile}: no matching .amd.Dockerfile found", warnings)
        return {"dockerfile": dockerfile, "path": "", "image": "", "rocm": "unknown", "rocm_source": "unknown"}
    with open(os.path.join(repo_root, path)) as f:
        image = base_image(f.read())
    rocm, source = rocm_from_image(image), "image tag"
    override = str(overrides.get(dockerfile) or "")
    if override or rocm is None:
        found, evidence = lookup(image) if lookup and image else (None, "registry not checked")
    if override:
        rocm, source = override, "status.yaml"
        if found and found != override and not found.startswith(override + "."):
            warn(f"{dockerfile}: status.yaml says ROCm {override} but '{image}' says {found} ({evidence}); "
                 "update or drop the override", warnings)
    elif rocm is None and is_unpinned(image):
        rocm, source = "latest", f"unpinned :{image_tag(image)}" + (f", ROCm {found} at build time" if found else "")
    elif rocm is None and found:
        rocm, source = found, evidence
    elif rocm is None:
        rocm, source = "unknown", "unknown"
        warn(f"{dockerfile}: ROCm version of '{image}' not in its tag ({evidence}); "
             "add it to rocm_overrides in status.yaml", warnings)
    return {"dockerfile": dockerfile, "path": path, "image": image, "rocm": rocm, "rocm_source": source}


# --------------------------------------------------------------------------- per-model fields


def skus_for(model):
    """Instinct products a model is registered for.

    Primus/JAX configs name their product in a tag (MI300X, MI355X, ...); everything else
    runs on every product except the ISAs listed in skip_gpu_arch.
    """
    tagged = [sku for sku in SKUS if sku in model.get("tags", [])]
    if tagged:
        return tagged
    skipped = set(re.split(r"[\s,]+", model.get("skip_gpu_arch", "") or "")) - {""}
    return [sku for isa, skus in ISA_SKUS.items() if isa not in skipped for sku in skus]


def arg_value(args, flag):
    m = re.search(rf"(?:^|\s){re.escape(flag)}[\s=]+(\S+)", args or "")
    return m.group(1) if m else None


def model_label(model):
    """(display name, Hugging Face repo or None)."""
    args = model.get("args", "")
    repo = arg_value(args, "--model_repo")
    if repo and "/" in repo:
        return repo.split("/", 1)[1], repo
    env_name = (model.get("env_vars") or {}).get("MODEL_NAME")
    if env_name:
        return env_name, None
    config = arg_value(args, "--config_path")
    if config:
        return os.path.splitext(os.path.basename(config))[0], None
    workload = arg_value(args, "--workload")
    if workload:
        return workload, None
    return re.sub(r"^pyt_", "", model["name"].rsplit("/", 1)[-1]), None


def precision_of(model):
    if model.get("training_precision"):
        return model["training_precision"].lower()
    # The Hugging Face repo is more specific than the MAD name (…-MXFP4-Preview vs …_fp4).
    for text in (arg_value(model.get("args", ""), "--model_repo") or "", model["name"]):
        for token in _PRECISION_TOKENS:
            if re.search(rf"(?:^|[_\-./]){token}(?:$|[_\-./])", text.lower()):
                return token
    return ""


def nodes_of(model):
    """Node count: the card's slurm block, else sbatch-style '-N <n>' in args, else 1."""
    nodes = (model.get("slurm") or {}).get("nodes")
    if nodes:
        return int(nodes)
    n = arg_value(model.get("args", ""), "-N")
    return int(n) if n and n.isdigit() else 1


def deployment_of(model, frameworks, opt_in=False):
    """Where a card runs: its scheduler, its Kubernetes path and whether discovery hides it.

    scheduler: "slurm" for slurm_multi cards (their launcher script needs an allocation, even
    on one node) and for opt-in multi-node configs; "local" for a plain madengine run.
    k8s: "manual" where the framework documents a per-node run outside SLURM (status.yaml);
    madengine's slurm_multi launcher itself is SLURM-only.
    status: "opt-in" for configs hidden unless `switch` (MULTINODE_SWITCHES[registry]) is set.
    """
    launcher = (model.get("distributed") or {}).get("launcher", "")
    return {
        "scheduler": "slurm" if launcher == "slurm_multi" or opt_in else "local",
        "k8s": (frameworks.get(model["registry"]) or {}).get("k8s", ""),
        "status": "opt-in" if opt_in else "registered",
        "switch": MULTINODE_SWITCHES[model["registry"]] if opt_in else "",
    }


def deployment_from_row(row):
    """Deployment of a row from a schema-1 snapshot, which predates the field."""
    multi = row["run"]["launcher"] == "slurm_multi"
    return {"scheduler": "slurm" if multi else "local", "k8s": "", "status": "registered", "switch": ""}


# Precision/quantization tokens that distinguish variants of one model, not the model itself.
_VARIANT_TOKENS = r"(?:w-[a-z0-9]+-a-[a-z0-9]+|mxfp8|mxfp4|nanoo_fp8|bf16|fp16|fp32|fp8|fp4|kv|preview)"


def family_of(label):
    """'Llama-3.1-405B-Instruct-MXFP4-Preview' -> 'Llama-3.1-405B-Instruct', so variants share a page."""
    name = re.sub(rf"[-_.]{_VARIANT_TOKENS}(?=$|[-_.])", "", label, flags=re.I)
    return re.sub(r"^amd[-_]", "", name, flags=re.I) or label


def slugify(text):
    return re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-")


def docker_context(model):
    """Build context madengine uses: dockercontext, else repo root for Primus images, else ./docker."""
    if model.get("dockercontext"):
        return model["dockercontext"]
    return "." if "primus" in model["dockerfile"].lower() else "./docker"


def framework_of(model, frameworks):
    """(framework display name, workload) from status.yaml; Primus splits by launcher tag."""
    entry = frameworks.get(model["registry"])
    if entry is None:
        return model["registry"], "other"
    name = entry["name"]
    if entry.get("by_launcher") and len(model.get("tags", [])) > 1:
        name = f"{name} {model['tags'][1].replace('_', ' ').title()}"
    return name, entry.get("workload", "other")


def describe_model(model, frameworks, suite_tags, images, opt_in=False):
    label, hf_repo = model_label(model)
    framework, workload = framework_of(model, frameworks)
    image = images[model["dockerfile"]]
    family = family_of(label)
    # Opt-in configs size their data-center mesh from the allocation (dcn_* = -1): two or more.
    nodes = 2 if opt_in else nodes_of(model)
    return {
        "name": model["name"],
        "registry": model["registry"],
        "framework": framework,
        "workload": workload,
        "model": label,
        "hf_repo": hf_repo,
        "precision": precision_of(model),
        "skus": skus_for(model),
        "n_gpus": "all" if str(model.get("n_gpus", "-1")) == "-1" else str(model["n_gpus"]),
        "nodes": nodes,
        "deployment": deployment_of(model, frameworks, opt_in),
        "suites": [t for t in model.get("tags", []) if t in suite_tags],
        "dockerfile": model["dockerfile"],
        "image": image["image"],
        "rocm": image["rocm"],
        "family": family,
        "page": f"{slugify(framework)}/{slugify(family)}",
        # Variant label on its page when the family is a whole registry (one_page); else derived.
        "variant": "",
        "readme": (frameworks.get(model["registry"]) or {}).get("readme", ""),
        # Everything a recipe needs to reproduce what madengine runs.
        "run": {
            "dockerfile_path": image["path"],
            "context": docker_context(model),
            "build_args": model.get("docker_build_arg") or {},
            "scripts": model.get("scripts", ""),
            "args": model.get("args", ""),
            "env_vars": {k: str(v) for k, v in (model.get("env_vars") or {}).items()},
            "multiple_results": model.get("multiple_results", ""),
            "url": model.get("url", ""),
            "data": model.get("data", ""),
            "launcher": (model.get("distributed") or {}).get("launcher", ""),
            "n_gpus": str(model.get("n_gpus", "-1")),
        },
    }


def merge_one_page(rows, frameworks):
    """Registries marked one_page in status.yaml become one recipe page, one variant per card.

    The page is named after the cards' common prefix (large_ep_bench_1n, large_ep_bench_2n_mori_only
    -> large_ep_bench); the rest of each name, without its node-count token, labels its variant,
    so the recipe page can offer node counts and variants as separate choices.
    """
    by_registry = {}
    for r in rows:
        if (frameworks.get(r["registry"]) or {}).get("one_page"):
            by_registry.setdefault(r["registry"], []).append(r)
    for registry, regrows in by_registry.items():
        labels = [r["model"] for r in regrows]
        prefix = os.path.commonprefix(labels)
        if prefix not in labels:
            # Cut back to a whole token: x_mori / x_mooncake share 'x_mo', not a name.
            prefix = re.sub(r"[^_\-.]*$", "", prefix)
        name = prefix.rstrip("_-.") or registry
        for r in regrows:
            rest = re.sub(r"^\d+n(?:[_\-.]|$)", "", r["model"][len(prefix):].strip("_-."))
            r["family"] = name
            r["page"] = f"{r['page'].split('/', 1)[0]}/{slugify(name)}"
            r["variant"] = rest.replace("_", " ") or "Default"


# --------------------------------------------------------------------------- snapshot


def git_commit(repo_root=REPO_ROOT):
    try:
        return subprocess.check_output(["git", "-C", repo_root, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def derived_qa(rows, images):
    """QA statements the registry itself proves; merged ahead of the curated ones."""
    covered = [
        {
            "area": "Performance benchmark",
            "scope": f"{len(rows)} registered models across {len({r['framework'] for r in rows})} frameworks",
            "note": "Every run reports throughput/latency through madengine (performance line or multiple_results CSV).",
        }
    ]
    registered = [r for r in rows if r["deployment"]["status"] == "registered"]
    multi = [r for r in registered if r["nodes"] > 1]
    if multi:
        covered.append({
            "area": "Multi-node runs (SLURM)",
            "scope": f"{len(multi)} models on {min(r['nodes'] for r in multi)}–{max(r['nodes'] for r in multi)} nodes "
                     f"({', '.join(sorted({r['framework'] for r in multi}))})",
            "note": "slurm_multi model cards: the launcher script runs on the SLURM head node and starts one "
                    "container per node.",
        })
    niah = [r for r in rows if "niah" in r["suites"]]
    if niah:
        covered.append({
            "area": "Long-context accuracy (needle-in-a-haystack)",
            "scope": ", ".join(r["model"] for r in niah),
            "note": "Models tagged niah.",
        })
    not_covered = []
    manual = sorted({r["framework"] for r in rows if r["deployment"]["k8s"] == "manual"})
    not_covered.append({
        "area": "Kubernetes",
        "scope": "All registered models",
        "note": "No model card targets Kubernetes, and madengine's slurm_multi launcher is SLURM-only."
                + (f" {', '.join(manual)} documents a manual per-node run that works on Kubernetes." if manual else ""),
    })
    opt_in = [r for r in rows if r["deployment"]["status"] == "opt-in"]
    if opt_in:
        switches = sorted({r["deployment"]["switch"] for r in opt_in})
        not_covered.append({
            "area": "Multi-node training models",
            "scope": f"{', '.join(sorted({r['family'] for r in opt_in}))} ({', '.join(sorted({r['framework'] for r in opt_in}))})",
            "note": f"Discovered only with {' / '.join(switches)}=1. The recipe uses madengine's primus launcher "
                    "on SLURM and is not validated by MAD.",
        })
    unpinned = [i for i in images if i["rocm"] == "latest"]
    if unpinned:
        not_covered.append({
            "area": "Pinned base images",
            "scope": f"{len(unpinned)} Dockerfiles build from an unpinned latest tag",
            "note": ", ".join(i["dockerfile"] for i in unpinned) + " — results are not reproducible across rebuilds.",
        })
    return covered, not_covered


def source_repo_of(repo_root):
    """'owner/name' of the checkout's origin on GitHub ('ROCm/MAD' when it cannot be told)."""
    try:
        url = subprocess.check_output(["git", "-C", repo_root, "remote", "get-url", "origin"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "ROCm/MAD"
    m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
    return m.group(1) if m else "ROCm/MAD"


def roadmap_items(status, rows, warnings, github=True):
    """Hand-written roadmap items plus, when configured, the ones read from GitHub issues and PRs."""
    manual = [dict(item, source=item.get("source", "status.yaml")) for item in status["roadmap"]]
    if not github or not status["roadmap_source"].get("repo"):
        return manual, []
    import github_roadmap  # imports this module; only needed when a GitHub source is configured
    oncoming, landed = github_roadmap.collect_roadmap(status["roadmap_source"], rows, status["frameworks"], warnings)
    return manual + oncoming, landed


def build_snapshot(status, version=None, state="live", repo_root=REPO_ROOT, source_repo=None, github=True,
                   inspect_images=True):
    """Everything the site shows. `repo_root` is the MAD tree to describe (models, docker/, Primus);
    `status` (the curated data) can come from a different checkout. `inspect_images` reads base
    images' ROCm versions from their registries where the tag does not encode one."""
    warnings = []
    excluded = set(status["exclude"])
    models = [m for m in discover_models(repo_root) if m["name"] not in excluded]

    registries = {}
    for m in models:
        reg = registries.setdefault(m["registry"], {"dir": m["registry"], "kind": "dynamic" if m["dynamic"] else "static", "models": 0, "note": ""})
        reg["models"] += 1
        if m["registry"] not in status["frameworks"]:
            warn(f"scripts/{m['registry']}: not in status.yaml frameworks; shown under its directory name", warnings)
            status["frameworks"][m["registry"]] = {"name": m["registry"], "workload": "other"}
    if not primus_checked_out(os.path.join(repo_root, "scripts", "Primus")):
        for d in sorted(glob.glob(os.path.join(repo_root, "scripts", "*", "get_models_json.py"))):
            reg_dir = os.path.basename(os.path.dirname(d))
            registries.setdefault(reg_dir, {"dir": reg_dir, "kind": "dynamic", "models": 0, "note": ""})
            registries[reg_dir]["note"] = "Primus checkout missing: configs not discovered"
        warn("scripts/Primus is not checked out; Primus/JAX configs are missing from this snapshot", warnings)

    images = {}
    lookup = image_registry.rocm_version if inspect_images else None
    for m in models:
        if m["dockerfile"] not in images:
            images[m["dockerfile"]] = describe_image(m["dockerfile"], status["rocm_overrides"], warnings, repo_root, lookup)

    hidden = multinode_only(repo_root)
    rows = [describe_model(m, status["frameworks"], status["suite_tags"], images, is_opt_in(m, hidden)) for m in models]
    merge_one_page(rows, status["frameworks"])
    order = {name: i for i, name in enumerate(status["frameworks"])}
    rows.sort(key=lambda r: (order.get(r["registry"], len(order)), r["framework"], r["model"].lower(), r["name"]))

    image_rows = []
    for dockerfile, image in images.items():
        users = [r for r in rows if r["dockerfile"] == dockerfile]
        image_rows.append(dict(
            image,
            frameworks=sorted({r["framework"] for r in users}),
            skus=[s for s in SKUS if any(s in r["skus"] for r in users)],
            models=len(users),
        ))
    first_use = {}
    for i, r in enumerate(rows):
        first_use.setdefault(r["dockerfile"], i)
    image_rows.sort(key=lambda i: first_use[i["dockerfile"]])

    covered, not_covered = derived_qa(rows, image_rows)
    roadmap, landed = roadmap_items(status, rows, warnings, github)
    return {
        "schema": SCHEMA,
        "version": version or status["release"],
        "release": status["release"],
        "state": state,
        "internal": status["internal"],
        "commit": git_commit(repo_root),
        "source_repo": source_repo or source_repo_of(repo_root),
        "date": datetime.date.today().isoformat(),
        "skus": list(SKUS),
        "isa_skus": {isa: list(skus) for isa, skus in ISA_SKUS.items()},
        "models": rows,
        "images": image_rows,
        "registries": sorted(registries.values(), key=lambda r: r["dir"]),
        "suite_tags": status["suite_tags"],
        "roadmap": roadmap,
        "roadmap_landed": landed,
        "roadmap_source": status["roadmap_source"],
        "qa": {
            "covered": covered + status["qa"]["covered"],
            "not_covered": not_covered + status["qa"]["not_covered"],
        },
        "ci": status["ci"],
        "warnings": warnings,
    }


def diff_snapshots(prev, cur):
    """What changed between two snapshots: models added/removed, base image or ROCm moves."""
    if prev is None:
        return None
    prev_models = {m["name"]: m for m in prev["models"]}
    cur_models = {m["name"]: m for m in cur["models"]}
    prev_images = {i["dockerfile"]: i for i in prev["images"]}
    image_changes = []
    for image in cur["images"]:
        old = prev_images.get(image["dockerfile"])
        if old and (old["image"], old["rocm"]) != (image["image"], image["rocm"]):
            image_changes.append({
                "dockerfile": image["dockerfile"],
                "frameworks": image["frameworks"],
                "from_image": old["image"], "to_image": image["image"],
                "from_rocm": old["rocm"], "to_rocm": image["rocm"],
            })
    return {
        "since": prev["version"],
        "added": [cur_models[n] for n in sorted(cur_models.keys() - prev_models.keys())],
        "removed": [prev_models[n] for n in sorted(prev_models.keys() - cur_models.keys())],
        "image_changes": image_changes,
    }
