#!/usr/bin/env python3
"""Translate an InferenceX e2e scenario into the artifacts madengine runs.

Emits, per scenario:

  1. <model-prefix>_infx.yaml   config for the MAD runner that already serves
                                this scenario's framework
  2. <scenario>.manifest.json   the madengine manifest (built_images /
                                built_models / context / deployment_config)
  3. <model-prefix>_models.json the same model in models.json shape, for a
                                plain `madengine run --tags` invocation

Why a translator and not a new harness: 47 of AMD's 52 InferenceX scenarios map
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
    "dtype",
}

# trust-remote-code is runner-owned only where the runner actually emits it.
# run_sglang.py and run_vllm.py always add it; run_atom.py does not -- it only
# forwards it to the benchmark when the config still carries it -- so stripping
# it there silently changes how the model loads.
OWNED_BY_RUNNER = {"sglang": {"trust-remote-code"}, "vllm": {"trust-remote-code"}}

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
        # Where checkpoints already live on this cluster. Searched in order for
        # a directory named after the scenario's model that holds a
        # config.json, so a scenario need not be told its own path. These are
        # five separate stores; four of them are one user's, which is exactly
        # why this belongs in the cluster entry rather than in anyone's memory.
        "model_stores": [
            "/apps/models_blog",
            "/apps/models",
            "/apps/models/htrivedi",
            "/apps/data/models",
            "/apps/huzhao/models",
        ],
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


# Frontend types that ARE the engine's own server rather than something in
# front of it. Observed across the AMD recipes: atomesh, sglang-router and
# vllm-router are routers; sglang and vllm are the engines themselves.
DIRECT_FRONTENDS = {"sglang", "vllm", "atom"}


DEVICE_GROUPS = (("render", "RENDER_GID"), ("video", "VIDEO_GID"))


def device_group_args(cluster):
    """gids for the groups madengine's `docker run` adds by name.

    madengine passes `--group-add render --group-add video`, and docker looks
    those names up in the IMAGE's group file: the vLLM nightly images have
    neither, so the container never starts before the runner is reached. The
    gid must be the COMPUTE NODE's, because /dev/dri/renderD* is bind-mounted
    with that node's numeric ownership and a non-root container cannot read it
    under any other gid.

    So the cluster entry owns these, and the generating host is only a
    fallback for the common case where it shares the cluster's name service.
    A login node that resolves neither, with no cluster entry, is refused
    rather than silently emitting a build that omits the group.
    """
    import grp

    configured = cluster.get("device_gids") or {}
    args, missing = {}, []
    for name, arg in DEVICE_GROUPS:
        if name in configured:
            args[arg] = str(configured[name])
            continue
        try:
            args[arg] = str(grp.getgrnam(name).gr_gid)
        except KeyError:
            missing.append(name)
    if missing:
        raise Untranslatable(
            f"group(s) {', '.join(missing)} resolve on neither this host nor "
            "the cluster entry, and madengine adds them by name at docker run; "
            "set device_gids in the cluster file to the compute node's values")
    return args


def image_tag(prefix, fw, arch, base_image, build_args):
    """Tag that changes whenever the image would.

    madengine reuses a local image, and a saved image tar, by TAG. Two scenarios
    that share a model prefix and framework but pin different engine images --
    or the same scenario built once plainly and once on an RCCL overlay, or for
    a different in-image uid -- would otherwise collide, and a later run would
    silently execute the earlier image.
    """
    import hashlib

    material = base_image + "|" + "|".join(f"{k}={v}" for k, v in sorted(build_args.items()))
    digest = hashlib.sha256(material.encode()).hexdigest()[:10]
    return f"mad-infx-{prefix}-{fw}:{arch}-{digest}"


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


# The only setup_script whose effect we can restate exactly. The others
# (kimik3-atom-dspark-draft.sh, minimaxm3-atom-eagle3-deps.sh,
# lmcache-mp-rocm.sh) patch files inside the container or stand up a connector,
# and guessing at them would be worse than refusing.
REPRODUCIBLE_SETUP_SCRIPTS = {"pip-runtime-deps.sh"}


def resolve_setup_script(recipe, blocks):
    """Packages the recipe's setup_script installs, for the image build.

    A setup_script runs in the container before the engine starts, and a
    recipe declares one because its base image is missing something: DeepSeek-V4's
    vLLM base needs Pillow, fastapi and uvicorn, and the server dies on import
    without them. Passing SETUP_PIP_PACKAGES to the server, as the recipe's env
    does, installs nothing -- so ignoring the script emits a run whose container
    is not the one the recipe describes.

    pip-runtime-deps.sh is exactly `pip install $SETUP_PIP_PACKAGES`, so it
    folds into the image build. Anything else is refused.
    """
    # Base is not the only place: an override may be the ONLY declaration, and
    # reading base alone accepted such a point as reproducible while omitting
    # its packages from the image, so the server started without them.
    declared = {recipe["base"].get("setup_script") or ""}
    for b in blocks:
        declared.update(b.get("_setup") or ())
    declared.discard("")
    for b in blocks:
        b.pop("_setup", None)
    if not declared:
        return "", None
    if len(declared) > 1:
        raise Untranslatable(
            f"points declare different setup scripts ({sorted(declared)}); "
            "one image cannot satisfy both")
    script = next(iter(declared))
    if script not in REPRODUCIBLE_SETUP_SCRIPTS:
        raise Untranslatable(
            f"recipe declares setup_script {script!r}, which prepares the "
            "container in ways this wrapper does not reproduce; running "
            "without it would measure a container the recipe does not describe")
    wanted = {str(b.get("env", {}).get("SETUP_PIP_PACKAGES", "")).strip()
              for b in blocks}
    wanted.discard("")
    if len(wanted) > 1:
        raise Untranslatable(
            f"points disagree on SETUP_PIP_PACKAGES ({sorted(wanted)}); "
            "one image cannot satisfy both")
    packages = next(iter(wanted), "")
    if not packages:
        return "", None
    return packages, (f"setup_script {script} is applied at image build "
                      f"(pip install {packages}) rather than per run")


def _agg(block):
    return block.get("roles", {}).get("agg", {})


def _base_tp(recipe):
    for k in TP_KEYS:
        if k in _agg(recipe["base"]).get("args", {}):
            return _agg(recipe["base"])["args"][k]
    return None


def _effective_tp(args, base_args):
    """TP as the scenario counts it: GPUs the engine spans.

    vLLM expresses DP-attention as tensor-parallel-size 1 with
    data-parallel-size N, while the scenario calls the same point tp=N. SGLang
    keeps tensor-parallel-size at N. Multiplying covers both, since a recipe
    that does not use data parallelism leaves it at 1.
    """
    def val(key, default=None):
        return args.get(key, base_args.get(key, default))

    tp = next((val(k) for k in TP_KEYS if val(k) is not None), None)
    if tp is None:
        return None
    # The two engines spell DP-attention differently and mean different things
    # by the data-parallel degree. SGLang sets enable-dp-attention and shards
    # attention ACROSS the same tensor-parallel group, so the GPU count is tp.
    # vLLM has no such flag and expresses the same topology as
    # tensor-parallel-size 1 with data-parallel-size N, where the GPU count is
    # the product.
    if val("enable-dp-attention"):
        return int(tp)
    return int(tp) * int(val("data-parallel-size", 1) or 1)


def _dp_attention_on(args, base_args):
    """Whether the point runs DP-attention, in either engine's spelling."""
    def val(key, default=None):
        return args.get(key, base_args.get(key, default))

    if val("enable-dp-attention"):
        return True
    return int(val("data-parallel-size", 1) or 1) > 1


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
        ("dp-attn", _dp_attention_on(args, base_args)),
        ("kv-offloading", env.get("KV_OFFLOADING")),
        # Kimi-K3's vLLM recipe has two TP8/conc-1 overrides that differ only in
        # decode-context parallelism; without this both match either point and
        # the sweep doubles.
        ("dcp-size", recipe_val("decode-context-parallel-size")),
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
    """Yield (concurrency, per-point agg args, benchmark env, server env) for `tp`.

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
    base_args_all = _agg(recipe["base"]).get("args", {})
    found = False
    for name, block in recipe.items():
        if name == "base" or not isinstance(block, dict):
            continue
        if "override" not in name:
            continue
        args = dict(_agg(block).get("args", {}))
        block_tp = _effective_tp(args, _agg(recipe["base"]).get("args", {}))
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
        # An override may also set server env of its own (vLLM's
        # VLLM_USE_BREAKABLE_CUDAGRAPH, the offload points' PYTHONHASHSEED);
        # the base env alone is not the point's environment.
        server_env = dict(_agg(block).get("env", {}))
        # Recorded, not applied: some recipes put a router in front of the
        # server with its own routing policy, and the aggregated path replays
        # against the server directly. The override inherits base's frontend
        # when it declares none -- dsr1-fp4-mi355x-atom states `atomesh` in
        # base only, so reading the override alone dropped the gap entirely
        # and the run bypassed Atomesh with nothing saying so.
        server_env["_infx_frontend"] = (
            block.get("frontend") or recipe["base"].get("frontend") or {})
        # Some overrides stand up a side service the point then talks to --
        # DeepSeek-V4's ATOM offload points need an lmcache-server. Neither the
        # wrapper nor the runner starts one, so those points cannot succeed.
        server_env["_infx_setup"] = block.get("setup_script") or ""
        svcs = block.get("services") or []
        server_env["_infx_services"] = sorted(
            str(x.get("type") or x.get("name")) if isinstance(x, dict) else str(x)
            for x in (svcs.values() if isinstance(svcs, dict) else svcs))
        conc_raw = env.get("CONC")
        if conc_raw is None:
            continue
        base_args = _agg(recipe["base"]).get("args", {})
        if point is not None and _attr_conflict(point, args, env, base_args):
            continue
        allowed = point.get("conc-list") if point else None
        found = True
        # The scenario's tp counts GPUs; the engine's tensor-parallel degree can
        # differ -- vLLM's DP-attention points are TP1 x DP8 on eight GPUs. The
        # server must be launched with the engine's degree, so carry it
        # separately instead of dropping it with the other TP keys.
        engine_tp = int(next((args[k] for k in TP_KEYS if k in args),
                             next((base_args_all[k] for k in TP_KEYS if k in base_args_all),
                                  tp)))
        point_args = {k: v for k, v in args.items() if k not in TP_KEYS}
        if isinstance(conc_raw, list):
            for i, c in enumerate(conc_raw):
                if allowed is not None and int(c) not in [int(a) for a in allowed]:
                    continue
                per = {k: (v[i] if isinstance(v, list) else v) for k, v in point_args.items()}
                per_env = {k: (v[i] if isinstance(v, list) else v) for k, v in env.items()
                           if k != "CONC"}
                # A fresh copy per point: _split_marker POPS the markers,
                # so sharing one dict across a zipped override's
                # concurrencies would check only the first and emit the
                # rest without their setup or service dependencies.
                yield int(c), per, per_env, dict(server_env), engine_tp
        else:
            if allowed is not None and int(conc_raw) not in [int(a) for a in allowed]:
                continue
            yield (int(conc_raw), point_args,
                   {k: v for k, v in env.items() if k != "CONC"}, dict(server_env),
                   engine_tp)
    if not found:
        raise Untranslatable(
            f"no override block for tp={tp} in recipe {recipe['base']['name']!r}")


def effective_args(recipe, point_args, owned=frozenset()):
    """Base args overlaid with the point's, then filtered.

    Order matters. Filtering false values out of the base BEFORE the overlay
    would let a base `true` survive a point that sets the same flag to false --
    DeepSeek-V4 enables enforce-shared-experts-fusion in its base and disables
    it for the MegaMoE points. Merge raw, then drop runner-owned args and the
    flags whose effective value is false, which the engines spell by omitting
    them.
    """
    merged = dict(_agg(recipe["base"]).get("args", {}))
    merged.update(point_args)
    skip = RUNNER_OWNED_ARGS | set(owned)
    return {f"--{k}": v for k, v in merged.items()
            if k not in skip and v is not False}


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
    seen_frontends = set()
    rrr = bench_env.get("RANDOM_RANGE_RATIO")
    if bench_env.get("USE_CHAT_TEMPLATE", "false") != "false":
        notes.append("recipe asks for a chat template; the MAD runner uses dataset-name=random.")

    blocks = []
    for workload in workloads:
      for point in workload["search-space"]:
        tp = point["tp"]
        notes += _ep_note(point, tp)
        for conc, point_args, _, point_env_srv, engine_tp in enumerate_points(recipe, tp, point):
            point_services, point_setups = set(), set()
            point_env_srv = _split_marker(point_env_srv, seen_frontends,
                                          point_services, point_setups)
            unusable = point_setups - REPRODUCIBLE_SETUP_SCRIPTS
            point_setup = sorted(point_setups - {""})
            if unusable:
                notes.append(
                    f"tp={tp} conc={conc} dropped: the override declares "
                    f"setup_script {', '.join(sorted(unusable))}, which "
                    "prepares the container in ways this wrapper does not "
                    "reproduce")
                continue
            if point_services:
                notes.append(
                    f"tp={tp} conc={conc} dropped: the override needs the "
                    f"{', '.join(sorted(point_services))} service, which neither "
                    "the wrapper nor the runner starts")
                continue
            extra = effective_args(recipe, point_args,
                                   OWNED_BY_RUNNER.get(runner["dir"], set()))
            block = {
                "benchmark": "serving",
                "model": base["model"]["path"].removeprefix("hf:"),
                "tp": engine_tp,
                "inp": int(workload["isl"]),
                "out": int(workload["osl"]),
                "max_concurrency": conc,
                "env": {**_agg(base).get("env", {}), **point_env_srv},
                "extra_args": extra,
            }
            _stamp(block, scenario_key, engine_tp, conc, runner, _point_suffix(extra))
            # Carried so resolve_setup_script sees declarations that live on
            # the override rather than on base; popped before the YAML write.
            if point_setup:
                block["_setup"] = point_setup
            if rrr is not None:
                block["random_range_ratio"] = float(rrr)
            blocks.append(block)
    notes += _frontend_and_acceptance_notes(seen_frontends, workloads)
    return blocks, notes


def emit_agentic(scenario_key, recipe, workloads, runner, duration=None, model_prefix="",
                 framework="", precision="", image=""):
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
    model_id = base["model"]["path"].removeprefix("hf:")
    seen_frontends = set()
    dram_unstated = set()
    hc = base.get("health_check") or {}
    startup_budget = int(hc.get("interval_seconds", 0) or 0) * \
        int(hc.get("max_attempts", 0) or 0)

    blocks = []
    for workload in workloads:
      for point in workload["search-space"]:
        tp = point["tp"]
        notes += _ep_note(point, tp)
        for conc, point_args, point_env, point_env_srv, engine_tp in enumerate_points(recipe, tp, point):
            point_services, point_setups = set(), set()
            point_env_srv = _split_marker(point_env_srv, seen_frontends,
                                          point_services, point_setups)
            unusable = point_setups - REPRODUCIBLE_SETUP_SCRIPTS
            point_setup = sorted(point_setups - {""})
            if unusable:
                notes.append(
                    f"tp={tp} conc={conc} dropped: the override declares "
                    f"setup_script {', '.join(sorted(unusable))}, which "
                    "prepares the container in ways this wrapper does not "
                    "reproduce")
                continue
            if point_services:
                notes.append(
                    f"tp={tp} conc={conc} dropped: the override needs the "
                    f"{', '.join(sorted(point_services))} service, which neither "
                    "the wrapper nor the runner starts")
                continue
            extra = effective_args(recipe, point_args)
            # agentic_lib.sh is driven by env, not CLI. KV_OFFLOADING is
            # required by its aggregation step; the recipe states it per point.
            kv_off = str(point_env.get("KV_OFFLOADING",
                                       point.get("kv-offloading", "none")))
            ep_deg = int(extra.get("--expert-parallel-size", 0)) or (
                tp if extra.get("--enable-expert-parallel") else 1)
            # Decode-context parallelism is part of the topology the aggregator
            # prints and names the result file with. Kimi-K3's offload points
            # are DCP8; unset, they would be recorded as DCP1.
            dcp_deg = int(extra.get("--decode-context-parallel-size",
                                    point.get("dcp-size", 1)) or 1)
            # The host-DRAM budget is descriptive here, not functional: the
            # recipes that need a concrete figure state it themselves (ATOM),
            # and SGLang sizes its host cache from hicache-ratio instead. Carry
            # whatever the merged server env has so the row is not silently 0.
            # The recipes that state it put it in the point's BENCHMARK env,
            # beside CONC and KV_OFFLOADING, not in the server role's env.
            host_dram = str(point_env.get("TOTAL_CPU_DRAM_GB",
                                          point_env_srv.get("TOTAL_CPU_DRAM_GB",
                                                            bench_env.get("TOTAL_CPU_DRAM_GB",
                                                                          "")))).strip()
            if not host_dram and kv_off == "dram":
                dram_unstated.add(f"tp={tp} conc={conc}")
            agentic_env = {
                "AGENTIC_CONC": str(conc),
                **({"DURATION": str(duration)} if duration else {}),
                "KV_OFFLOADING": kv_off,
                # The aggregator reads the topology and the descriptive fields
                # from the environment. Left unset it defaults TP to 1, which
                # divides nothing and reports per-GPU throughput TP times too
                # high, and writes a row whose concurrency, framework and
                # precision are blank. These are the values the point already
                # states, so there is no reason for the artifact not to carry
                # them.
                "CONC": str(conc),
                "TP": str(tp),
                "EP_SIZE": str(ep_deg),
                "DCP_SIZE": str(dcp_deg),
                **({"TOTAL_CPU_DRAM_GB": host_dram} if host_dram else {}),
                "DP_ATTENTION": "true" if _dp_attention_on(
                    {k.lstrip("-"): v for k, v in extra.items()}, {}) else "false",
                "FRAMEWORK": framework,
                "PRECISION": precision,
                "MODEL": model_id,
                "IMAGE": image,
                "SPEC_DECODING": str(point.get("spec-decoding", "none")),
            }
            backend_name = (point.get("kv-offload-backend") or {}).get("name")
            if kv_off != "none":
                # The aggregator refuses to run unless the backend metadata is
                # present as JSON and its name matches, so an offload point
                # would otherwise die after the replay rather than produce a
                # result.
                if not backend_name:
                    notes.append(
                        f"tp={tp} conc={conc} offloads KV but names no backend; "
                        "the aggregator requires one")
                else:
                    agentic_env["KV_OFFLOAD_BACKEND"] = str(backend_name)
                    agentic_env["KV_OFFLOAD_BACKEND_METADATA"] = json.dumps(
                        {"name": str(backend_name)})
            # Forward every AIPerf knob rather than a whitelist: the recipes
            # also set client timeouts (AIPERF_HTTP_TCP_USER_TIMEOUT) that the
            # shared hook does not restore, and a missed one silently changes
            # the measurement.
            # The point's own benchmark env overrides the base: DeepSeek-V4's
            # overrides set AGENTIC_WARMUP_GRACE_PERIOD and an AIPerf routing
            # key, and the hook would otherwise use its own defaults.
            for k, v in {**bench_env, **point_env}.items():
                if (k.startswith("AIPERF_") or k.startswith("AGENTIC_")
                        or k == "WEKA_LOADER_OVERRIDE"):
                    agentic_env[k] = str(v)
            # The hook picks the trace corpus from MODEL_PREFIX when the recipe
            # states no explicit loader. Without it every model falls back to
            # the 256k-capped corpus, which is a different workload for the
            # models whose scenarios use the uncapped one.
            agentic_env.setdefault("MODEL_PREFIX", model_prefix)
            block = {
                "benchmark": "agentic",
                "model": base["model"]["path"].removeprefix("hf:"),
                "tp": engine_tp,
                # The runners' CSV columns include inp/out; an agentic replay has
                # no fixed sequence length, so 0 marks "not a fixed-seq-len run"
                # rather than inventing a number.
                "inp": 0,
                "out": 0,
                "max_concurrency": conc,
                "env": {**_agg(base).get("env", {}), **point_env_srv},
                "extra_args": extra,
                # How long the recipe says a cold load may take. Kimi-K3's
                # MI355X vLLM recipe allows 7200 s; the shared hook otherwise
                # stops waiting at 1800 and the replay never starts.
                **({"startup_timeout": startup_budget} if startup_budget else {}),
                "agentic_env": agentic_env,
            }
            _stamp(block, scenario_key, engine_tp, conc, runner, _point_suffix(extra))
            # Carried so resolve_setup_script sees declarations that live on
            # the override rather than on base; popped before the YAML write.
            if point_setup:
                block["_setup"] = point_setup
            blocks.append(block)

    notes += _frontend_and_acceptance_notes(seen_frontends, workloads)
    if dram_unstated:
        notes.append(
            "host-DRAM offload budget not reported for "
            f"{len(dram_unstated)} dram-offloading point(s): InferenceX computes "
            "TOTAL_CPU_DRAM_GB at submission from its own runner's memory, and "
            "the recipe states none, so the aggregated row carries no figure. "
            "The engine's own sizing (SGLang's hicache-ratio) is unaffected")
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


def _split_marker(server_env, seen, services=None, setups=None):
    """Remove the internal markers and remember what they said."""
    fe = server_env.pop("_infx_frontend", None) or {}
    # `sglang` and `vllm` name the engine's own HTTP server, which is exactly
    # what the runners start; only a real router in front of it is a gap. Now
    # that base's frontend is inherited, every vLLM recipe would otherwise
    # report one.
    if fe.get("type") and fe["type"] not in DIRECT_FRONTENDS:
        seen.add(fe["type"])
    needed = server_env.pop("_infx_services", None) or []
    if needed and services is not None:
        services.update(needed)
    # An override can carry its own setup_script -- MiniMax-M3's vLLM
    # lmcache point installs LMCache into the worker container -- so the
    # base-level check alone would let an unprepared point through.
    setup = server_env.pop("_infx_setup", None)
    if setup and setups is not None:
        setups.add(setup)
    return {k: v for k, v in server_env.items() if not k.startswith("_infx_")}


def _frontend_and_acceptance_notes(seen, workloads):
    """Deviations that belong to the scenario as a whole, not to one point."""
    notes = []
    if seen:
        notes.append(
            f"recipe puts a {'/'.join(sorted(seen))} frontend in front of the "
            "server with its own routing policy; the aggregated path replays "
            "against the server directly, so that routing is not reproduced")
    if any(str(pt.get("spec-decoding", "none")) != "none"
           for w in workloads for pt in w["search-space"]):
        notes.append(
            "InferenceX applies golden acceptance-length overrides at submission "
            "(synthetic_acceptance.build_overrides); the raw recipe does not "
            "carry them, so these points measure the engine's native acceptance "
            "rather than the pinned one")
    return notes


def _point_suffix(extra):
    """Short discriminator for points that share tp and concurrency."""
    bits = []
    ep = extra.get("--expert-parallel-size")
    if ep and int(ep) > 1:
        bits.append(f"ep{ep}")
    if _dp_attention_on({k.lstrip("-"): v for k, v in extra.items()}, {}):
        bits.append("dp")
    dcp = extra.get("--decode-context-parallel-size")
    if dcp and int(dcp) > 1:
        bits.append(f"dcp{dcp}")
    return ("-" + "-".join(bits)) if bits else ""


def _ep_note(point, tp):
    """Expert parallelism needs no note: the recipe override carries
    expert-parallel-size and the emitters pass it straight into extra_args.
    Kept as a hook so a future axis that genuinely is not mapped has a home."""
    return []


def _stamp(block, scenario_key, tp, conc, runner, suffix=""):
    """Add the keys whose presence depends on the runner's config contract.

    The variant has to identify the POINT: DeepSeek-V4 has an ep=1 and an ep=8
    point at the same tp and concurrency, and a name that cannot tell them
    apart would make `--variant` select both and make their result directories
    collide.
    """
    block["_point_id"] = f"tp{tp}-c{conc}{suffix}"
    if runner["supports_variant"]:
        block["variant"] = f"{scenario_key}-{block['_point_id']}"
    if runner["dtype_key"]:
        # The checkpoint carries its own quantisation and the InferenceX recipe
        # passes no --dtype at all; the runners always emit one, so 'auto' is
        # the only value that reproduces the recipe instead of overriding the
        # weights.
        block[runner["dtype_key"]] = "auto"


# --------------------------------------------------------------------------
# The madengine side
# --------------------------------------------------------------------------

def revision_note(blocks, weights_dir):
    """Say so when a pinned checkpoint revision cannot be enforced.

    A recipe pins `revision` and `tokenizer-revision` because the scenario was
    measured against that exact checkpoint. The engines honour it only when
    they resolve the model through the HuggingFace hub; served from a mounted
    directory, as every run here is, the flag is accepted and ignored. A
    checkpoint that differs then fails in a way that names neither: DeepSeek-V4
    at the wrong revision dies inside the DSpark speculator asking for a
    config field the pinned revision has and an older one does not.

    An HF snapshot path carries its sha, so compare; a plain directory cannot
    be checked at all, which is itself worth saying.
    """
    pinned = {str(b["extra_args"][k]) for b in blocks
              for k in ("--revision", "--tokenizer-revision")
              if k in b.get("extra_args", {})}
    if not pinned:
        return None
    pinned_str = ", ".join(sorted(pinned))
    head, sep, tail = (os.path.realpath(weights_dir or "")
                       .partition(os.sep + "snapshots" + os.sep))
    if sep:
        have = tail.split(os.sep)[0]
        if have in pinned:
            return None
        return (f"recipe pins revision {pinned_str} but the mounted snapshot is "
                f"{have}; the engine ignores the flag for a local path, so this "
                "run measures the mounted checkpoint")
    return (f"recipe pins revision {pinned_str}, which the engine ignores for a "
            "mounted checkpoint; the weights directory carries no revision "
            "marker, so the artifacts cannot confirm it is that checkpoint")


def precision_note(scenario, weights_dir):
    """Say so when the mounted checkpoint is not the precision claimed.

    A scenario states the precision its numbers were measured at, and the
    engine takes the quantization from the checkpoint, not from that label. So
    a mi355x-fp4 scenario served from an fp8 copy of the same model produces a
    row that reads as fp4 and is not: different kernels, different memory
    traffic, different numbers. Both DeepSeek-V4 checkpoints on this cluster
    are fp8 while their scenarios say fp4, so this is not hypothetical.

    Only checkpoints that declare a quantization are checked; an unquantized
    checkpoint says nothing and is left alone.
    """
    want = (scenario.get("precision") or "").lower()
    if not want or not weights_dir:
        return None
    cfg = os.path.join(os.path.realpath(weights_dir), "config.json")
    try:
        with open(cfg) as f:
            have = ((json.load(f).get("quantization_config") or {})
                    .get("quant_method") or "").lower()
    except (OSError, ValueError):
        return (f"scenario states precision {want} and the mounted checkpoint has "
                "no readable config.json, so the artifacts cannot confirm it")
    if not have or have == want:
        return None
    return (f"scenario states precision {want} but the mounted checkpoint is "
            f"quantized {have}; the engine follows the checkpoint, so this run "
            "does not measure what the scenario names")


def resolve_weights(model_path, cluster, explicit):
    """Where this scenario's checkpoint lives on this cluster.

    An explicit --weights-dir always wins. Otherwise the EFFECTIVE model -- the
    stand-in under --smoke-model, the scenario's own otherwise -- is looked up
    by name in the cluster's model stores, because which checkpoints a cluster
    holds is a cluster fact, like its partition or its broken nodes, and
    finding them by hand is how 88 of 280 points were first reported
    unreachable when they were simply in a store nobody had listed.

    Resolving the scenario's model instead of the effective one would mount the
    real checkpoint under the stand-in's name, and every runner prefers those
    weights through MAD_DATAHOME -- so a smoke run advertised as cheap would
    load the full original checkpoint.

    Not found is not an error here: the caller may be generating artifacts to
    run elsewhere. It is recorded, so a run that falls back to MAD's own
    download path does not look like a run against a staged checkpoint.
    """
    if explicit:
        return explicit, None
    name = model_path.split("/")[-1]
    for store in cluster.get("model_stores") or ():
        cand = os.path.join(store, name)
        if os.path.isfile(os.path.join(cand, "config.json")):
            return cand, None
    stores = ", ".join(cluster.get("model_stores") or ()) or "(none configured)"
    return None, (f"no checkpoint named {name!r} in the cluster's model stores "
                  f"[{stores}]; the run will fall back to fetching it")


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
    # Absolute, always: madengine renders the mount verbatim, and docker reads
    # a relative source as a NAMED VOLUME rather than a host directory -- so
    # `--model-host-dir store` would mount an empty volume while MAD_DATAHOME
    # still pointed inside it, and the runner would skip downloading and then
    # fail to load. The explicit weights branch below already does this.
    model_host_dir = os.path.realpath(model_host_dir) if model_host_dir else model_host_dir
    if not weights_dir:
        # Nothing resolved for this scenario: mount the shared store anyway --
        # it may hold the checkpoint under another name -- but report no
        # datahome, because all three runners treat a non-empty MAD_DATAHOME as
        # THE checkpoint path and skip their download branch. Exporting a path
        # that holds no checkpoint turns a missing download into a server that
        # fails to start.
        #
        # os.path.isfile follows symlinks, so a staged check that passes does
        # not mean model_host_dir itself holds the checkpoint: model_host_dir/
        # model_dir can itself be a symlink into an HF cache snapshot elsewhere
        # (a different trap than the one below, one level up). Mounting only
        # model_host_dir then leaves that symlink dangling in the container.
        # realpath it and fall through to the same snapshot-aware resolution
        # used for an explicit --weights-dir, instead of trusting the literal
        # path.
        candidate = os.path.join(model_host_dir, model_dir)
        if os.path.isfile(os.path.join(candidate, "config.json")):
            weights_dir = candidate
        else:
            return {container_models: model_host_dir}, None
    weights_dir = os.path.realpath(weights_dir)
    head, sep, tail = weights_dir.partition(os.sep + "snapshots" + os.sep)
    if sep:
        return ({f"{container_models}/{model_dir}": head},
                f"{container_models}/{model_dir}/snapshots/{tail}")
    return {f"{container_models}/{model_dir}": weights_dir}, f"{container_models}/{model_dir}"


def translate_manifest(scenario_key, scenario, recipe, prefix, cluster, runner, fw,
                       model_host_dir, timeout, weights_dir=None, slurm_time="04:00:00",
                       base_image_override=None, pip_packages=""):
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
    build_args = dict(
        {"BASE_DOCKER": image},
        **({"INFX_PIP_PACKAGES": pip_packages} if pip_packages else {}),
        **({"RUN_AS_UID": str(os.getuid()), "RUN_AS_GID": str(os.getgid())}
           if cluster.get("run_as_submitting_user") else {}),
        **device_group_args(cluster),
    )
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
                "docker_image": image_tag(prefix, fw, arch, image, build_args),
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
                # keeps a 600+ GB checkpoint off the network every run. Absent
                # when nothing is staged, so the runner downloads instead of
                # pointing the engine at an empty directory.
                **({"MAD_DATAHOME": model_path} if model_path else {}),
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
            "docker_build_arg": build_args,
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
            "env_vars": ({"MAD_DATAHOME": model_path} if model_path else {}),
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
    # models.json carries no build arguments and no per-run identity. The
    # Dockerfile has no default BASE_DOCKER on purpose, and the -u the manifest
    # needs on a root_squash cluster is the GENERATOR's uid, which is wrong for
    # anyone else. So this entry is for running an image that already exists,
    # and the manifest stays the self-contained path.
    m["additional_docker_run_options"] = _strip_uid(
        m.get("additional_docker_run_options", ""))
    return m


def _strip_uid(opts):
    """Drop `-u <uid>:<gid>` from docker options; see models_json_entry."""
    out, skip = [], False
    for tok in opts.split():
        if skip:
            skip = False
            continue
        if tok == "-u":
            skip = True
            continue
        out.append(tok)
    return " ".join(out)


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
    # Options the disagg route cannot honour must be refused, not ignored: a
    # smoke run that quietly uses the real checkpoint and every point is worse
    # than no run.
    unsupported = [n for n, v in (("--agentic-duration", args.agentic_duration),
                                  ("--smoke-model", args.smoke_model),
                                  ("--smoke-point", args.smoke_point),
                                  ("--smoke-first-point", args.smoke_first_point)) if v]
    if unsupported:
        raise Untranslatable(
            f"the disagg route does not implement {', '.join(unsupported)}")
    gpu_type = scenario["runner"].split(":")[-1].split("-")[0]
    arch = GPU_ARCH.get(gpu_type)
    if arch is None or arch != cluster["arch"]:
        raise Untranslatable(
            f"scenario targets {gpu_type} ({arch}) but the cluster is {cluster['arch']}")

    # Same selection rule as the aggregated route: auto-detect only when asked
    # to. Ignoring the flag meant `--all --workload agentic-coding` still
    # emitted fixed-seq-len disagg manifests, and asking for a workload a
    # scenario does not have silently produced the other one.
    workload_name = args.workload
    if workload_name == "auto":
        present = [w for w in ("fixed-seq-len", "agentic-coding")
                   if w in scenario["scenarios"]]
        if len(present) != 1:
            raise Untranslatable(f"expected one known workload, found {present}")
        workload_name = present[0]
    if workload_name not in scenario["scenarios"]:
        raise Untranslatable(f"no {workload_name} workload")
    workload = scenario["scenarios"][workload_name][0]

    # Resolved here, not inside the emitter: an unresolvable group has to
    # raise Untranslatable from the module the sweep actually imported.
    device_gids = device_group_args(cluster)
    # The disagg emitter only ever saw --model-host-dir, so a checkpoint in any
    # of the cluster's other stores looked absent to it. Same resolution as the
    # aggregated route.
    disagg_weights, disagg_note = resolve_weights(
        scenario["model"], cluster, args.weights_dir)

    os.makedirs(out_dir, exist_ok=True)
    made, notes, skipped = [], [], []
    for idx, point in enumerate(workload["search-space"]):
        try:
            man, pnotes = disagg_emitter.emit_disagg(
                scenario_key, scenario, workload_name, workload, point, idx,
                cluster, runner, args.timeout, args.slurm_time,
                disagg_weights, args.model_host_dir, arch, resolve_mounts,
                mad_dir=args.mad_dir, infx_commit=infx_revision(args.infx),
                run_logs_host=args.run_logs_dir,
                base_image_override=args.base_image, device_gids=device_gids)
            if args.base_image:
                man["_infx_provenance"]["fidelity_gaps"].append(
                    f"image replaced: {scenario['image']} -> {args.base_image}")
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
    if disagg_note:
        notes.append(disagg_note)
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
    prefix = scenario["model-prefix"]
    emitter = WORKLOAD_EMITTERS[workload_name]
    kwargs = ({"duration": args.agentic_duration, "model_prefix": prefix,
               "framework": fw, "precision": scenario.get("precision", ""),
               "image": args.base_image or scenario["image"]}
              if workload_name == "agentic-coding" else {})
    blocks, notes = emitter(scenario_key, recipe, entries, runner, **kwargs)
    if not blocks:
        # Every matching override can be dropped -- unsupported setup script,
        # a service the wrapper does not start. Writing a zero-point config and
        # manifest leaves artifacts the validator happily reports as OK, and
        # the smoke paths below index blocks[0]. Refuse, keeping the reasons so
        # --all reports them and continues.
        raise Untranslatable("; ".join(notes) or "no expressible point")
    cluster = CLUSTERS[args.cluster]

    smoke = []
    if args.smoke_point:
        # Match on a component boundary, not a substring: `tp8-c1` would
        # otherwise also select tp8-c12 and tp8-c16, quietly running three
        # points where one was asked for. A bare `tp8-c1` still selects
        # `tp8-c1-dp` and friends, which is the discriminator, not a new point.
        kept = [b for b in blocks
                if b["_point_id"] == args.smoke_point
                or b["_point_id"].startswith(args.smoke_point + "-")]
        if not kept:
            raise Untranslatable(
                f"--smoke-point {args.smoke_point!r} matched none of "
                f"{[b['_point_id'] for b in blocks]}")
        blocks = kept
        smoke.append(f"only points matching {args.smoke_point!r} kept "
                     f"({len(blocks)} of them)")
    for b in blocks:
        b.pop("_point_id", None)
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

    # Both of these have to be decided before the config header is written, or
    # the file a human reads would omit a gap the manifest records.
    pip_packages, setup_note = resolve_setup_script(recipe, blocks)
    if setup_note:
        notes.append(setup_note)
    weights_dir, weights_note = resolve_weights(
        recipe["base"]["model"]["path"].removeprefix("hf:"), cluster, args.weights_dir)
    if weights_note:
        notes.append(weights_note)
    rev_note = revision_note(blocks, weights_dir)
    if rev_note:
        notes.append(rev_note)
    prec_note = precision_note(scenario, weights_dir)
    if prec_note:
        notes.append(prec_note)
    if args.base_image:
        notes.append(f"image replaced: {scenario['image']} -> {args.base_image}")

    os.makedirs(out_dir, exist_ok=True)
    # The scenario key, not just the prefix: the FP4, FP8 and MTP SGLang
    # DeepSeek scenarios share a prefix and framework, so a prefix-only name
    # has them overwrite one another's staged config and earlier manifests then
    # run whichever was written last.
    # Fingerprint the blocks the manifest will point at. A full translation and
    # a smoke one -- or two runs differing in --agentic-duration or
    # --base-image -- name the same scenario, so a bare scenario name has the
    # later write replace the file an earlier manifest still references. That
    # manifest then runs a workload it does not describe while keeping its own
    # point count and fidelity claims, which is the one failure these artifacts
    # exist to prevent.
    import hashlib

    digest = hashlib.sha256(
        yaml.safe_dump(blocks, sort_keys=True).encode()).hexdigest()[:8]
    cfg_name = f"{scenario_key}_infx-{digest}.yaml"
    # The runner resolves --config relative to its own directory, so the file
    # has to exist there. Write it straight into the MAD checkout when we know
    # it; otherwise say plainly that it still has to be staged.
    # The manifest first: it is what refuses a scenario the cluster cannot run
    # -- wrong GPU architecture, unresolvable device groups. Writing the config
    # before that check staged a file for every rejected scenario, leaving the
    # runner's configs/ holding translations that no manifest points at.
    man = translate_manifest(
        scenario_key, scenario, recipe, prefix, cluster, runner, fw,
        args.model_host_dir, args.timeout, weights_dir=weights_dir,
        slurm_time=args.slurm_time, base_image_override=args.base_image,
        pip_packages=pip_packages,
    )

    staged = os.path.join(args.mad_dir, "scripts", runner["dir"], "configs")
    cfg_path = os.path.join(staged if os.path.isdir(staged) else out_dir, cfg_name)
    with open(cfg_path, "w") as f:
        f.write(f"# Generated by infx2mad.py from InferenceX scenario {scenario_key}\n")
        f.write(f"# Source recipe: {workload['search-space'][0]['srt-recipe']}\n")
        f.write(f"# Target runner: scripts/{runner['dir']}\n")
        f.write(f"# InferenceX commit: {infx_revision(args.infx)}\n")
        f.write("# Do not hand-edit; regenerate instead.\n")
        for n in notes:
            f.write(f"# FIDELITY GAP: {n}\n")
        yaml.safe_dump(blocks, f, sort_keys=False, default_flow_style=False)
    man["built_models"][next(iter(man["built_models"]))]["args"] = (
        man["built_models"][next(iter(man["built_models"]))]["args"]
        .replace(f"configs/{prefix}_infx.yaml", f"configs/{cfg_name}")
    )
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

    if not os.path.isdir(staged):
        notes.append(
            f"config written to {cfg_path}; copy it to scripts/{runner['dir']}/configs/ "
            "before running, or pass --mad-dir so it is written there directly")
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
                    help="accepted and ignored; config names are keyed by "
                         "scenario, which already separates frameworks and the "
                         "precision/spec-decoding variants that share a prefix")
    ap.add_argument("--mad-dir", default=".",
                    help="MAD checkout, read for scripts/sglang_disagg/models.yaml "
                         "when translating a disaggregated scenario")
    ap.add_argument("--run-logs-dir",
                    help="shared writable dir bound to /run_logs on every node; "
                         "the disagg launcher coordinates readiness through it")
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
    ap.add_argument("--smoke-point",
                    help="keep only points whose id contains this string, e.g. "
                         "'tp1-c64-dp'. The id is tp<N>-c<M> plus the topology "
                         "suffix, and exists for every engine, unlike variant, "
                         "which only SGLang configs carry")
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
