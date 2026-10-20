#!/usr/bin/env python3
"""Translate an InferenceX e2e scenario into the artifacts madengine runs.

Emits, per scenario:

  1. <model-prefix>_infx.yaml   config for the MAD runner that already serves
                                this scenario's framework
  2. <scenario>.manifest.json   the madengine manifest (built_images /
                                built_models / context / deployment_config)
  3. <model-prefix>_models.json the same model in models.json shape, for a
                                plain `madengine run --tags` invocation

Why a translator and not a new harness: 49 of AMD's 51 InferenceX scenarios map
onto a MAD runner that already exists (scripts/sglang, scripts/vllm,
scripts/atom, scripts/sglang_disagg), and madengine already runs in the RCCL CI
arm. So this is a ROUTER: read the scenario, pick the runner that serves its
framework, emit that runner's config. Adding a scenario is config, not code.

Four invariants every emitter inherits, each forced by something concrete:

  1. One config block per (tp, concurrency) point. The MAD runners expand their
     list args with itertools.product, i.e. a CARTESIAN product, while
     InferenceX zips tp against its own concurrency range. Collapsing them
     invents points that were never measured.
  2. Per-point server args. A recipe override carries args that belong to one
     concurrency only (cuda-graph-max-bs, max-running-requests, the whole
     hicache group), which a cartesian expander cannot represent at all.
  3. Drop false-valued boolean args. The runners render any bool as a bare flag,
     so a recipe's `enable-metrics: false` would be emitted as the flag and turn
     the option ON.
  4. Record every deviation in _infx_provenance.fidelity_gaps. A translated row
     that silently differs from the InferenceX recipe is worse than no row.

Usage:
  infx2mad.py --infx <inferencex-e2e> --scenario dsr1-fp8-mi355x-sglang \
              --cluster ruby --weights-dir <checkpoint or HF snapshot>
"""

import argparse
import json
import os
import re
import sys

import yaml

# Server args the MAD runners emit themselves. Leaving them in extra_args would
# pass them twice; the engine takes the last one, which hides the duplication.
RUNNER_OWNED_ARGS = {
    "tensor-parallel-size",
    "tp-size",
    "model-path",
    "served-model-name",
    "host",
    "port",
    "trust-remote-code",
    "dtype",
}

TP_KEYS = ("tensor-parallel-size", "tp-size")

GPU_ARCH = {"mi355x": "gfx950", "mi350x": "gfx950", "mi325x": "gfx942", "mi300x": "gfx942"}

# Which MAD runner serves each InferenceX framework, and the parts of that
# runner's config contract that differ between them. Verified against
# scripts/<dir>/run_<dir>.py at ROCm/MAD mad-rccl.
RUNNERS = {
    "sglang": {
        "dir": "sglang",
        "launcher": "sglang",
        # run_sglang.py filters configs by `variant`; run_vllm.py and
        # run_atom.py have no such key and would carry it into the CSV row,
        # whose columns are fixed by their CSV_HEADER.
        "supports_variant": True,
        "dtype_key": "dtype",
        "benchmarks": {"fixed-seq-len": "serving", "agentic-coding": "agentic"},
    },
    "vllm": {
        "dir": "vllm",
        "launcher": "vllm",
        "supports_variant": False,
        "dtype_key": "dtype",
        "benchmarks": {"fixed-seq-len": "serving", "agentic-coding": "agentic"},
    },
    "atom": {
        "dir": "atom",
        "launcher": "atom",
        "supports_variant": False,
        # run_atom.py's CSV has kv_cache_dtype, not dtype.
        "dtype_key": None,
        "benchmarks": {"fixed-seq-len": "serving", "agentic-coding": "agentic"},
    },
}

# Frameworks with an InferenceX scenario but no MAD runner. Named explicitly so
# a missing row reads as a known gap rather than a silent omission.
NO_RUNNER = {
    "atom-disagg": "no MAD runner",
    "tilert": "no MAD runner",
}

# Disaggregated prefill/decode is driven by scripts/sglang_disagg, which is env-
# and arg-driven rather than config-file driven: xP prefill nodes, yD decode
# nodes, one TP for both roles. So it gets its own emitter, and a scenario there
# produces one manifest per search-space point rather than one config file.
DISAGG_RUNNERS = {
    "sglang-disagg": {"dir": "sglang_disagg", "launcher": "sglang-disagg", "backend": "sglang"},
}

# Cluster facts that are not in the InferenceX scenario and cannot be derived
# from it. Mirrors CLUSTER_CONFIGS in rccl/ci/scripts/test_madengine.py.
CLUSTERS = {
    "ruby": {
        "partition": "meta64",
        "network_interface": "fenic0",
        "gpus_per_node": 8,
        "arch": "gfx950",
        # GPUs on Ruby are not a Slurm GRES, so --gpus-per-node must not be emitted
        "skip_gpus_directive": True,
        # madengine mounts the MAD checkout into the container and runs it as
        # root. On Ruby that checkout sits on an NFS export with root_squash,
        # where in-container root lands as nobody: `mkdir run_directory` is
        # denied and madengine's own `cp --preserve=all` cannot set ownership.
        # Running as the submitting user is what makes both succeed.
        "run_as_submitting_user": True,
        # Nodes whose sssd/PAM is broken: sudo there answers "PAM account
        # management error" and refuses, so every docker call fails before the
        # model is reached. Probe with probe_sudo.sh before trusting this list;
        # it is a snapshot, not a cluster invariant.
        "exclude_nodes": "cv350-rck-g03-c10-08,cv350-rck-g03-c15-08,cv350-rck-g03-c12-18,cv350-rck-g03-c14-18",
        # Data-plane fabric. Only multi-node (disagg) runs need it; copied from
        # CLUSTER_CONFIGS["ruby"]["nccl_env"] in rccl/ci/scripts/test_madengine.py
        # so the two CI arms describe the same cluster.
        "fabric_env": {
            "NCCL_IB_DISABLE": "0",
            "NCCL_IB_HCA": "bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,"
                           "bnxt_re4:1,bnxt_re5:1,bnxt_re6:1,bnxt_re7:1",
            "IB_DEVICES": "bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,"
                          "bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7",
            "MORI_RDMA_DEVICES": "bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,"
                                 "bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7",
            "NCCL_IB_GID_INDEX": "3",
            "MORI_IB_GID_INDEX": "3",
            "NCCL_IB_TC": "104",
            "NCCL_IB_QPS_PER_CONNECTION": "4",
            "NCCL_SOCKET_IFNAME": "fenic0",
            "GLOO_SOCKET_IFNAME": "fenic0",
            "MORI_SOCKET_IFNAME": "fenic0",
            "RDMAV_DRIVERS": "bnxt_re",
            "IBV_DRIVERS": "bnxt_re",
        },
    },
}

DOCKERFILE = "docker/pyt_infx.ubuntu.amd.Dockerfile"


class Untranslatable(Exception):
    """This scenario cannot be expressed against a MAD runner, with the reason.

    Raised rather than exiting so a sweep over all scenarios reports every gap
    in one pass instead of stopping at the first one.
    """


# --------------------------------------------------------------------------
# Reading the InferenceX side
# --------------------------------------------------------------------------

def load_scenario(infx_root, key):
    path = os.path.join(infx_root, "configs", "amd-master.yaml")
    with open(path) as f:
        master = yaml.safe_load(f)
    if key not in master:
        raise Untranslatable(f"scenario {key!r} not in {path}")
    return master[key]


def infx_revision(infx_root):
    """The InferenceX commit the artifacts were generated from.

    Stamped into every config and manifest: the scenario and its recipe ARE the
    measurement, and upstream moves. Without this a translated row cannot be
    tied back to the definition it claims to reproduce.
    """
    import subprocess
    try:
        out = subprocess.run(["git", "-C", infx_root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_recipe(infx_root, rel_path):
    with open(os.path.join(infx_root, rel_path)) as f:
        return yaml.safe_load(f)


def _agg(block):
    return block.get("roles", {}).get("agg", {})


def _base_tp(recipe):
    for k in TP_KEYS:
        if k in _agg(recipe["base"]).get("args", {}):
            return _agg(recipe["base"])["args"][k]
    return None


def _attr_conflict(point, args, env, base_args):
    """True when the scenario point and a recipe override disagree on an axis.

    A concurrency value is not a unique key: dsv4's agentic recipe has both
    `override_tp8_c128_dp_hicache` (ep=1) and
    `override_tp8_c128_dp_ep8_megamoe_hicache` (ep=8). The scenario's
    search-space point is what says which of the two is meant, through ep,
    dp-attn and kv-offloading. Axes the override does not mention are inherited
    from the recipe base; axes neither states are don't-care.
    """
    def recipe_val(key):
        return args.get(key, base_args.get(key))

    checks = [
        ("ep", recipe_val("expert-parallel-size")),
        ("dp-attn", recipe_val("enable-dp-attention")),
        ("kv-offloading", env.get("KV_OFFLOADING")),
    ]
    for scen_key, rec_val in checks:
        if scen_key not in point or rec_val is None:
            continue
        scen_val = point[scen_key]
        if scen_key == "dp-attn":
            if bool(scen_val) != bool(rec_val):
                return True
        elif str(scen_val) != str(rec_val):
            return True
    return False


def enumerate_points(recipe, tp, point=None):
    """Yield (concurrency, per-point agg args, per-point benchmark env) for `tp`.

    InferenceX writes the sweep two different ways and both reduce to the same
    thing:

      * fixed-seq-len recipes use `zip_override_tp<N>_*` where CONC is a LIST and
        every other list-valued arg is zipped against it element-wise
        (cuda-graph-max-bs is the one that matters);
      * agentic recipes use `override_tp<N>_*c<C>` where CONC is a SCALAR and the
        block carries that point's own args (max-running-requests, the hicache
        group, ...).

    Reading the override blocks rather than the scenario's conc-start/conc-end
    is deliberate: the scenario states a range with no step, so deriving the
    list from it would be a guess.
    """
    base_tp = _base_tp(recipe)
    found = False
    for name, block in recipe.items():
        if name == "base" or not isinstance(block, dict):
            continue
        if "override" not in name:
            continue
        args = dict(_agg(block).get("args", {}))
        block_tp = next((args[k] for k in TP_KEYS if k in args), None)
        if block_tp is None:
            # Some recipes (the atom ones) carry no roles block at all and state
            # no tensor-parallel-size anywhere: the only place the degree
            # appears is the override's NAME, e.g. zip_override_tp4_ep1.
            m = re.search(r"tp(\d+)", name)
            block_tp = m.group(1) if m else base_tp
        # A block that names no tp anywhere (e.g. `zip_override_concurrency`)
        # is the sweep shared by every tp in the scenario's search space, so it
        # must not be skipped for want of a match.
        if block_tp is not None and int(block_tp) != int(tp):
            continue
        env = dict(block.get("benchmark", {}).get("env", {}))
        conc_raw = env.get("CONC")
        if conc_raw is None:
            continue
        base_args = _agg(recipe["base"]).get("args", {})
        if point is not None and _attr_conflict(point, args, env, base_args):
            continue
        allowed = point.get("conc-list") if point else None
        found = True
        point_args = {k: v for k, v in args.items() if k not in TP_KEYS}
        if isinstance(conc_raw, list):
            for i, c in enumerate(conc_raw):
                if allowed is not None and int(c) not in [int(a) for a in allowed]:
                    continue
                per = {k: (v[i] if isinstance(v, list) else v) for k, v in point_args.items()}
                per_env = {k: (v[i] if isinstance(v, list) else v) for k, v in env.items()
                           if k != "CONC"}
                yield int(c), per, per_env
        else:
            if allowed is not None and int(conc_raw) not in [int(a) for a in allowed]:
                continue
            yield int(conc_raw), point_args, {k: v for k, v in env.items() if k != "CONC"}
    if not found:
        raise Untranslatable(
            f"no override block for tp={tp} in recipe {recipe['base']['name']!r}")


def base_extra_args(recipe):
    """Recipe base args -> runner extra_args, minus what the runner emits itself.

    Booleans need care: the runners render any bool as a bare flag, so a `false`
    would be emitted as `--enable-metrics` and turn the option ON. A recipe arg
    set to false means "leave it off", which the engines spell by omitting the
    flag, so drop it here.
    """
    args = _agg(recipe["base"]).get("args", {})
    return {f"--{k}": v for k, v in args.items()
            if k not in RUNNER_OWNED_ARGS and v is not False}


# --------------------------------------------------------------------------
# Workload emitters
# --------------------------------------------------------------------------

def emit_serving(scenario_key, recipe, workloads, runner):
    """fixed-seq-len -> the runners' `benchmark: serving` config blocks."""
    notes = []
    base = recipe["base"]
    bench_env = base.get("benchmark", {}).get("env", {})

    # The recipe jitters prompt length; MAD's runner used to hardcode 1.0 (every
    # prompt exactly `inp` tokens), which is a different workload. It now reads
    # random_range_ratio from the config, so pass the recipe's value through.
    rrr = bench_env.get("RANDOM_RANGE_RATIO")
    if bench_env.get("USE_CHAT_TEMPLATE", "false") != "false":
        notes.append("recipe asks for a chat template; the MAD runner uses dataset-name=random.")

    blocks = []
    for workload in workloads:
      for point in workload["search-space"]:
        tp = point["tp"]
        notes += _ep_note(point, tp)
        for conc, point_args, _ in enumerate_points(recipe, tp, point):
            extra = base_extra_args(recipe)
            extra.update({f"--{k}": v for k, v in point_args.items()
                          if k not in RUNNER_OWNED_ARGS and v is not False})
            block = {
                "benchmark": "serving",
                "model": base["model"]["path"].removeprefix("hf:"),
                "tp": tp,
                "inp": int(workload["isl"]),
                "out": int(workload["osl"]),
                "max_concurrency": conc,
                "env": dict(_agg(base).get("env", {})),
                "extra_args": extra,
            }
            _stamp(block, scenario_key, tp, conc, runner)
            if rrr is not None:
                block["random_range_ratio"] = float(rrr)
            blocks.append(block)
    return blocks, notes


def emit_agentic(scenario_key, recipe, workloads, runner, duration=None):
    """agentic-coding -> the runners' `benchmark: agentic` config blocks.

    The replay itself is MAD's own agentx core (scripts/common/benchmark_agentic.sh
    plus agentic_lib.sh), which is already InferenceX-aware: its default scenario
    is `inferencex-agentx-mvp` and it aggregates through InferenceX's own
    utils.agentic.aggregation. So this emitter does not describe the workload, it
    only fills the env contract that agentic_lib.sh documents.
    """
    notes = []
    base = recipe["base"]
    bench_env = base.get("benchmark", {}).get("env", {})

    blocks = []
    for workload in workloads:
      for point in workload["search-space"]:
        tp = point["tp"]
        notes += _ep_note(point, tp)
        for conc, point_args, point_env in enumerate_points(recipe, tp, point):
            extra = base_extra_args(recipe)
            extra.update({f"--{k}": v for k, v in point_args.items()
                          if k not in RUNNER_OWNED_ARGS and v is not False})
            # agentic_lib.sh is driven by env, not CLI. KV_OFFLOADING is
            # required by its aggregation step; the recipe states it per point.
            agentic_env = {
                "AGENTIC_CONC": str(conc),
                **({"DURATION": str(duration)} if duration else {}),
                "KV_OFFLOADING": str(point_env.get("KV_OFFLOADING",
                                                   point.get("kv-offloading", "none"))),
            }
            for k in ("WEKA_LOADER_OVERRIDE", "AIPERF_APPLY_CHAT_TEMPLATE",
                      "AIPERF_REQUIRED_SERVER_METRIC_PREFIX"):
                if k in bench_env:
                    agentic_env[k] = str(bench_env[k])
            backend = point.get("kv-offload-backend") or {}
            if backend.get("name"):
                agentic_env["KV_OFFLOAD_BACKEND"] = str(backend["name"])

            block = {
                "benchmark": "agentic",
                "model": base["model"]["path"].removeprefix("hf:"),
                "tp": tp,
                # The runners' CSV columns include inp/out; an agentic replay has
                # no fixed sequence length, so 0 marks "not a fixed-seq-len run"
                # rather than inventing a number.
                "inp": 0,
                "out": 0,
                "max_concurrency": conc,
                "env": dict(_agg(base).get("env", {})),
                "extra_args": extra,
                "agentic_env": agentic_env,
            }
            _stamp(block, scenario_key, tp, conc, runner)
            blocks.append(block)

    if duration is None:
        notes.append(
            "no measurement window set; agentic_lib.sh defaults to 120 s while the "
            "InferenceX scenario minimum is 900 s, so the run is marked "
            "submission_valid=false. Pass --agentic-duration 900 for a valid row."
        )
    elif int(duration) < 900:
        notes.append(
            f"measurement window {duration} s is below the scenario minimum of 900 s; "
            "the run is marked submission_valid=false by design."
        )
    return blocks, notes


WORKLOAD_EMITTERS = {"fixed-seq-len": emit_serving, "agentic-coding": emit_agentic}


def _ep_note(point, tp):
    """Expert parallelism needs no note: the recipe override carries
    expert-parallel-size and the emitters pass it straight into extra_args.
    Kept as a hook so a future axis that genuinely is not mapped has a home."""
    return []


def _stamp(block, scenario_key, tp, conc, runner):
    """Add the keys whose presence depends on the runner's config contract."""
    if runner["supports_variant"]:
        block["variant"] = f"{scenario_key}-tp{tp}-c{conc}"
    if runner["dtype_key"]:
        # The checkpoint carries its own quantisation and the InferenceX recipe
        # passes no --dtype at all; the runners always emit one, so 'auto' is
        # the only value that reproduces the recipe instead of overriding the
        # weights.
        block[runner["dtype_key"]] = "auto"


# --------------------------------------------------------------------------
# The madengine side
# --------------------------------------------------------------------------

def resolve_mounts(model_dir, weights_dir, model_host_dir):
    """-> (docker_mounts, in-container model path).

    Either the host dir CONTAINS the checkpoints (a shared model store such as
    /apps/models_blog), or it IS one checkpoint. The second case has a trap: a
    HuggingFace snapshot dir holds only symlinks into ../../blobs, so mounting
    the snapshot alone delivers a tree of dangling links and the engine reports
    an unrecognized model with no config.json — which reads like a bad
    checkpoint rather than a bad mount.
    """
    container_models = "/models"
    if not weights_dir:
        return {container_models: model_host_dir}, f"{container_models}/{model_dir}"
    weights_dir = os.path.realpath(weights_dir)
    head, sep, tail = weights_dir.partition(os.sep + "snapshots" + os.sep)
    if sep:
        return ({f"{container_models}/{model_dir}": head},
                f"{container_models}/{model_dir}/snapshots/{tail}")
    return {f"{container_models}/{model_dir}": weights_dir}, f"{container_models}/{model_dir}"


def translate_manifest(scenario_key, scenario, recipe, prefix, cluster, runner, fw,
                       model_host_dir, timeout, weights_dir=None, slurm_time="04:00:00",
                       base_image_override=None):
    base = recipe["base"]
    model = base["model"]["path"].removeprefix("hf:")
    image = base_image_override or scenario["image"]
    gpu_type = base["resources"]["gpu_type"]
    arch = GPU_ARCH.get(gpu_type)
    if arch is None:
        raise Untranslatable(f"unknown gpu_type {gpu_type!r}")
    if arch != cluster["arch"]:
        raise Untranslatable(
            f"scenario targets {gpu_type} ({arch}) but the cluster is {cluster['arch']}; "
            "running the recipe on a different part would not reproduce it"
        )

    key = f"infx-{scenario_key}"
    model_dir = os.path.basename(model)
    mounts, model_path = resolve_mounts(model_dir, weights_dir, model_host_dir)
    nodes = int(_agg(base).get("nodes", 1))
    gpus = ",".join(str(i) for i in range(cluster["gpus_per_node"]))

    args = f"--model_repo {model} --config configs/{prefix}_infx.yaml"
    if runner["supports_variant"]:
        args += " --variant all"

    return {
        "built_images": {
            key: {
                "model": key,
                "docker_image": f"mad-infx-{prefix}-{fw}:{arch}",
                "dockerfile": DOCKERFILE,
                "base_docker": image,
                "build_duration": 0,
                "local_image": True,
                "registry_image": None,
                "registry": None,
                "gpu_vendor": "AMD",
            }
        },
        "built_models": {
            key: {
                "name": key,
                "url": "",
                "dockerfile": DOCKERFILE,
                "scripts": f"scripts/{runner['dir']}/run_infx.sh",
                "data": "",
                "n_gpus": "-1",
                "owner": "mad.support@amd.com",
                "training_precision": "",
                "multiple_results": f"perf_{model_dir}.csv",
                "tags": ["pyt", fw, "inference", "inferencex", prefix, arch],
                "timeout": timeout,
                "args": args,
                "additional_docker_run_options": (
                    "--privileged --group-add render --shm-size 64G "
                    "--security-opt seccomp=unconfined "
                    "--cap-add IPC_LOCK --ulimit memlock=-1"
                    # appended last so it overrides madengine's own -u root
                    + (f" -u {os.getuid()}:{os.getgid()}"
                       if cluster.get("run_as_submitting_user") else "")
                ),
            }
        },
        "context": {
            "skip_perf_collection": False,
            "docker_env_vars": {
                "MAD_SYSTEM_GPU_ARCHITECTURE": arch,
                # The runners prefer MAD_DATAHOME over an HF pull, which is what
                # keeps a 600+ GB checkpoint off the network every run.
                "MAD_DATAHOME": model_path,
                # NOT the weights mount: sglang's bench_serving downloads the
                # ShareGPT corpus to sample a token-length distribution whenever
                # random-range-ratio != 1.0, and writes it into HF_HUB_CACHE. The
                # weights mount is read-only, so pointing the cache there fails
                # the benchmark minutes after the model has finished loading.
                "HF_HUB_CACHE": "/tmp/hf_cache",
                "HF_HOME": "/tmp/hf_home",
                "NCCL_DEBUG": "WARN",
                "NCCL_DEBUG_SUBSYS": "INIT",
            },
            "docker_mounts": mounts,
            "docker_build_arg": dict(
                {"BASE_DOCKER": image},
                **({"RUN_AS_UID": str(os.getuid()), "RUN_AS_GID": str(os.getgid())}
                   if cluster.get("run_as_submitting_user") else {}),
            ),
            "gpu_vendor": "AMD",
            "guest_os": "UBUNTU",
            "docker_gpus": gpus,
        },
        "credentials_required": [],
        "summary": {
            "successful_builds": [],
            "failed_builds": [],
            "total_build_time": 0,
            "successful_pushes": [],
            "failed_pushes": [],
        },
        "deployment_config": {
            "target": "slurm",
            "slurm": {
                "partition": cluster["partition"],
                "nodes": nodes,
                "gpus_per_node": cluster["gpus_per_node"],
                "time": slurm_time,
                "output_dir": "./slurm_output",
                "exclusive": True,
                "network_interface": cluster["network_interface"],
                "skip_gpus_directive": cluster["skip_gpus_directive"],
                **({"exclude": cluster["exclude_nodes"]}
                   if cluster.get("exclude_nodes") else {}),
            },
            "distributed": {
                "launcher": runner["launcher"],
                "nnodes": nodes,
                "nproc_per_node": cluster["gpus_per_node"],
            },
            "env_vars": {"MAD_DATAHOME": model_path},
            "debug": False,
            "docker_gpus": gpus,
        },
        "_infx_provenance": {
            "scenario": scenario_key,
            "master_config": "configs/amd-master.yaml",
            "srt_recipe": base["name"],
            "framework": fw,
            "mad_runner": f"scripts/{runner['dir']}",
            "precision": scenario["precision"],
            "gpu_type": gpu_type,
        },
    }


def models_json_entry(manifest, key):
    """The same model, in the shape `madengine run --tags` reads from models.json.

    madengine has two entry paths: the manifest drives the slurm deployment the
    RCCL CI uses, while a plain single-node run reads models.json from the MAD
    checkout. Emitting both from one source keeps them from drifting.
    """
    m = dict(manifest["built_models"][key])
    runner_dir = m["scripts"].split("/")[1]
    m["dockerfile"] = "../../docker/pyt_infx"
    m["scripts"] = "run_infx.sh"
    m["data"] = "huggingface"
    m["_runner_dir"] = runner_dir
    return m


# --------------------------------------------------------------------------

def translate_one(args, scenario_key, out_dir):
    try:
        return _translate_one(args, scenario_key, out_dir), None
    except Untranslatable as exc:
        return None, f"{scenario_key}: {exc}"


def _translate_disagg(args, scenario_key, scenario, out_dir):
    """Disagg scenarios emit one manifest per search-space point; see
    disagg_emitter.py for why they cannot share one."""
    import disagg_emitter

    runner = DISAGG_RUNNERS[scenario["framework"]]
    cluster = CLUSTERS[args.cluster]
    gpu_type = scenario["runner"].split(":")[-1].split("-")[0]
    arch = GPU_ARCH.get(gpu_type)
    if arch is None or arch != cluster["arch"]:
        raise Untranslatable(
            f"scenario targets {gpu_type} ({arch}) but the cluster is {cluster['arch']}")

    present = [w for w in ("fixed-seq-len", "agentic-coding") if w in scenario["scenarios"]]
    if len(present) != 1:
        raise Untranslatable(f"expected one known workload, found {present}")
    workload_name = present[0]
    workload = scenario["scenarios"][workload_name][0]

    os.makedirs(out_dir, exist_ok=True)
    made, notes, skipped = [], [], []
    for idx, point in enumerate(workload["search-space"]):
        try:
            man, pnotes = disagg_emitter.emit_disagg(
                scenario_key, scenario, workload_name, workload, point, idx,
                cluster, runner, args.timeout, args.slurm_time,
                args.weights_dir, args.model_host_dir, arch, resolve_mounts)
        except ValueError as exc:
            skipped.append(f"point {idx}: {exc}")
            continue
        path = os.path.join(out_dir, f"{scenario_key}.p{idx}.manifest.json")
        with open(path, "w") as f:
            json.dump(man, f, indent=2)
            f.write("\n")
        made.append(path)
        notes += pnotes
    if not made:
        raise Untranslatable("; ".join(skipped) or "no expressible point")
    notes += [f"point dropped -> {s}" for s in skipped]
    return {"scenario": scenario_key, "framework": scenario["framework"],
            "workload": workload_name, "runner": f"scripts/{runner['dir']}",
            "points": len(made), "config": "(env-driven launcher)",
            "manifest": made[0], "gaps": sorted(set(notes))}


def _translate_one(args, scenario_key, out_dir):
    scenario = load_scenario(args.infx, scenario_key)
    fw = scenario.get("framework")
    if fw in DISAGG_RUNNERS:
        return _translate_disagg(args, scenario_key, scenario, out_dir)
    if fw in NO_RUNNER:
        raise Untranslatable(NO_RUNNER[fw])
    if fw not in RUNNERS:
        raise Untranslatable(f"unknown framework {fw!r}")
    if scenario.get("disagg") or scenario.get("multinode"):
        raise Untranslatable("disagg/multinode not mapped yet")
    runner = RUNNERS[fw]

    workload_name = args.workload
    if workload_name == "auto":
        present = [w for w in WORKLOAD_EMITTERS if w in scenario["scenarios"]]
        if len(present) != 1:
            raise Untranslatable(f"expected one known workload, found {present}")
        workload_name = present[0]
    if workload_name not in scenario["scenarios"]:
        raise Untranslatable(f"no {workload_name} workload")

    entries = scenario["scenarios"][workload_name]
    recipes = {e["search-space"][0]["srt-recipe"] for e in entries}
    if len(recipes) != 1:
        raise Untranslatable(f"{workload_name} entries span {len(recipes)} recipes; "
                             "one config file cannot represent them")
    workload = entries[0]

    recipe = load_recipe(args.infx, workload["search-space"][0]["srt-recipe"])
    emitter = WORKLOAD_EMITTERS[workload_name]
    kwargs = {"duration": args.agentic_duration} if workload_name == "agentic-coding" else {}
    blocks, notes = emitter(scenario_key, recipe, entries, runner, **kwargs)
    prefix = scenario["model-prefix"]
    cluster = CLUSTERS[args.cluster]

    smoke = []
    if args.smoke_first_point:
        blocks = blocks[:1]
        smoke.append(f"only the first point kept: tp={blocks[0]['tp']} "
                     f"conc={blocks[0]['max_concurrency']}")
    if args.smoke_model:
        real = blocks[0]["model"]
        for b in blocks:
            b["model"] = args.smoke_model
        recipe["base"]["model"]["path"] = args.smoke_model
        smoke.append(f"model replaced: {real} -> {args.smoke_model}")
    if smoke:
        notes.append("SMOKE ARTIFACT, not an InferenceX-comparable row: " + "; ".join(smoke))

    os.makedirs(out_dir, exist_ok=True)
    cfg_name = f"{prefix}_{fw}_infx.yaml" if args.per_framework_config else f"{prefix}_infx.yaml"
    cfg_path = os.path.join(out_dir, cfg_name)
    with open(cfg_path, "w") as f:
        f.write(f"# Generated by infx2mad.py from InferenceX scenario {scenario_key}\n")
        f.write(f"# Source recipe: {workload['search-space'][0]['srt-recipe']}\n")
        f.write(f"# Target runner: scripts/{runner['dir']}\n")
        f.write(f"# InferenceX commit: {infx_revision(args.infx)}\n")
        f.write("# Do not hand-edit; regenerate instead.\n")
        for n in notes:
            f.write(f"# FIDELITY GAP: {n}\n")
        yaml.safe_dump(blocks, f, sort_keys=False, default_flow_style=False)

    man = translate_manifest(
        scenario_key, scenario, recipe, prefix, cluster, runner, fw,
        args.model_host_dir, args.timeout, weights_dir=args.weights_dir,
        slurm_time=args.slurm_time, base_image_override=args.base_image,
    )
    man["built_models"][next(iter(man["built_models"]))]["args"] = (
        man["built_models"][next(iter(man["built_models"]))]["args"]
        .replace(f"configs/{prefix}_infx.yaml", f"configs/{cfg_name}")
    )
    if args.base_image:
        notes.append(f"image replaced: {scenario['image']} -> {args.base_image}")
    man["_infx_provenance"]["inferencex_commit"] = infx_revision(args.infx)
    man["_infx_provenance"]["scenario_image"] = scenario["image"]
    man["_infx_provenance"]["workload"] = workload_name
    man["_infx_provenance"]["points"] = len(blocks)
    man["_infx_provenance"]["fidelity_gaps"] = notes
    man_path = os.path.join(out_dir, f"{scenario_key}.manifest.json")
    with open(man_path, "w") as f:
        json.dump(man, f, indent=2)
        f.write("\n")

    key = next(iter(man["built_models"]))
    mj_path = os.path.join(out_dir, f"{scenario_key}_models.json")
    with open(mj_path, "w") as f:
        json.dump([models_json_entry(man, key)], f, indent=2)
        f.write("\n")

    return {"scenario": scenario_key, "framework": fw, "workload": workload_name,
            "runner": f"scripts/{runner['dir']}", "points": len(blocks),
            "config": cfg_path, "manifest": man_path, "gaps": notes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infx", required=True, help="path to an inferencex-e2e checkout")
    ap.add_argument("--scenario", help="one scenario key; omit with --all")
    ap.add_argument("--all", action="store_true",
                    help="translate every AMD scenario that maps to a MAD runner")
    ap.add_argument("--workload", default="auto",
                    choices=["auto", "fixed-seq-len", "agentic-coding"])
    ap.add_argument("--cluster", default="ruby",
                    help="cluster key from the built-in table or --cluster-file")
    ap.add_argument("--cluster-file",
                    help="YAML/JSON file of extra cluster entries, merged over "
                         "the built-ins. Adding a cluster is data, not code: see "
                         "clusters.example.yaml for the fields and why each is "
                         "needed")
    ap.add_argument("--model-host-dir", default="/apps/models_blog",
                    help="host dir holding several checkpoints, mounted at /models")
    ap.add_argument("--weights-dir",
                    help="host dir that IS this scenario's checkpoint (e.g. an HF "
                         "snapshot dir); mounted at /models/<model name>")
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--slurm-time", default="04:00:00",
                    help="SBATCH --time; a full ladder reloads the checkpoint once per "
                         "point, so size it against the model, not the point count")
    ap.add_argument("--per-framework-config", action="store_true",
                    help="name configs <prefix>_<framework>_infx.yaml, so two "
                         "frameworks serving the same model do not collide")
    ap.add_argument("--base-image",
                    help="run on this image instead of the scenario's. The RCCL CI "
                         "uses it to layer the RCCL under test over the scenario's "
                         "image, which is the only way RCCL ends up inside the "
                         "measured number")
    ap.add_argument("--print-scenario-image", action="store_true",
                    help="print the image the scenario pins and exit; the CI needs "
                         "it to know what to build the RCCL overlay on top of")
    ap.add_argument("--agentic-duration", type=int,
                    help="agentic measurement window in seconds; 900 is the "
                         "InferenceX scenario minimum for a valid submission")
    ap.add_argument("--out-dir", default=".")
    # Smoke options. They exist so the plumbing can be proven green before a
    # 600-800 GB checkpoint lands, and they are loud on purpose: anything they
    # change is recorded in the manifest's provenance block, because a smoke
    # artifact must never be mistaken for the InferenceX row it stands in for.
    ap.add_argument("--smoke-model", help="serve this checkpoint instead of the scenario's")
    ap.add_argument("--smoke-first-point", action="store_true",
                    help="keep only the first (tp, concurrency) point")
    args = ap.parse_args()

    if args.cluster_file:
        # YAML, which also parses JSON. The repo's .gitignore excludes *.json,
        # so a cluster file shipped as JSON would be silently dropped from a
        # commit; YAML is both tracked and the convention for configs here.
        with open(args.cluster_file) as f:
            for name, cfg in (yaml.safe_load(f) or {}).items():
                # Keys starting with _ are prose: the example file documents
                # each field inline, and a notes block is not a cluster.
                if name.startswith("_") or not isinstance(cfg, dict):
                    continue
                cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
                CLUSTERS[name] = {**CLUSTERS.get(name, {}), **cfg}
    if args.cluster not in CLUSTERS:
        ap.error(f"unknown cluster {args.cluster!r}; known: {sorted(CLUSTERS)}")

    if not args.all and not args.scenario:
        ap.error("pass --scenario or --all")

    if args.print_scenario_image:
        print(load_scenario(args.infx, args.scenario)["image"])
        return

    if args.scenario:
        keys = [args.scenario]
    else:
        master = yaml.safe_load(open(os.path.join(args.infx, "configs", "amd-master.yaml")))
        keys = sorted(master)

    ok, skipped = [], []
    for k in keys:
        out_dir = args.out_dir if args.scenario else os.path.join(args.out_dir, k)
        result, why = translate_one(args, k, out_dir)
        (ok.append(result) if result else skipped.append(why))

    for r in ok:
        print(f"{r['scenario']:48s} {r['framework']:8s} {r['workload']:14s} "
              f"{r['points']:3d} pts -> {r['runner']}")
        for g in r["gaps"]:
            print(f"{'':48s} FIDELITY GAP: {g}")
    if skipped:
        print(f"\nnot translated ({len(skipped)}):")
        for s in skipped:
            print(f"  {s}")
    print(f"\ntranslated {len(ok)} scenario(s), {sum(r['points'] for r in ok)} points")


if __name__ == "__main__":
    main()
