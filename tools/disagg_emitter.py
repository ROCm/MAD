#!/usr/bin/env python3
"""Disaggregated prefill/decode scenarios -> madengine manifests.

Separate from the aggregated emitters because scripts/sglang_disagg has a
different contract: it is driven by args and env (xP prefill nodes, yD decode
nodes, SGLANG_TP_SIZE, KV_TRANSFER_BACKEND, BENCHMARK_CONCURRENCY_LEVELS) rather
than by a config YAML, and it spans several nodes.

One manifest per search-space point, not one per scenario: a point fixes the
topology (how many prefill and decode nodes, which TP, whether DP-attention is
on), and those are launcher arguments, so two points cannot share a manifest.

Two shapes the launcher cannot express, declared rather than approximated:
  * asymmetric TP. scripts/sglang_disagg/run.sh resolves one SGLANG_TP_SIZE into
    IO_EP_TP_SIZE for both roles, so a point with prefill tp=4 and decode tp=8
    has no faithful mapping.
  * a worker smaller than a node. xP and yD are NODE counts; the point states
    workers. They coincide only while tp equals the node's GPU count.
"""

import os
import re
import yaml

from infx2mad import image_tag


def catalog_name(mad_dir, model_dir):
    """The launcher's own key for this checkpoint.

    MODEL_NAME must be a key in scripts/sglang_disagg/models.yaml; the launcher
    rejects anything else before it starts a server, and a checkpoint basename
    is usually not that key (DeepSeek-R1-0528 vs the catalog's DeepSeek-R1).
    """
    cat = os.path.join(mad_dir, "scripts", "sglang_disagg", "models.yaml")
    try:
        with open(cat) as f:
            keys = list(yaml.safe_load(f) or {})
    except OSError:
        raise ValueError(f"cannot read the disagg model catalog at {cat}")
    if model_dir in keys:
        return model_dir
    # Longest key that prefixes the checkpoint name, e.g. DeepSeek-R1 for
    # DeepSeek-R1-0528. Anything else has no supported mapping.
    cands = sorted((k for k in keys if model_dir.startswith(k)), key=len, reverse=True)
    if cands:
        return cands[0]
    raise ValueError(
        f"{model_dir!r} has no entry in scripts/sglang_disagg/models.yaml; "
        "the launcher validates MODEL_NAME against that catalog")


def _worker_nodes(role, gpus_per_node):
    """Nodes a role occupies, or None when a worker does not fill a node."""
    tp = int(role["tp"])
    workers = int(role.get("num-worker", 1))
    if tp > gpus_per_node:
        if tp % gpus_per_node:
            return None
        return workers * (tp // gpus_per_node)
    if tp != gpus_per_node:
        return None
    return workers


def dp_mode1_allowed(mad_dir):
    """The catalog keys the launcher accepts at DP_MODE=1, read from it.

    sglang_disagg_mori_io_ep.sh (used by both transfer routes) refuses any
    other model before a server starts; read rather than copied so the two
    cannot drift. None when the script or the list cannot be found.
    """
    path = os.path.join(mad_dir, "scripts", "sglang_disagg", "sglang_disagg_mori_io_ep.sh")
    try:
        text = open(path).read()
    except OSError:
        return None
    m = re.search(r"MORI_DP_MODE1_ALLOWED_MODELS=\(\s*(.*?)\)", text, re.S)
    if not m:
        return None
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def emit_disagg(scenario_key, scenario, workload_name, workload, point, idx,
                cluster, runner, timeout, slurm_time, weights_dir, model_host_dir,
                gpu_arch, resolve_mounts, mad_dir=".", infx_commit="unknown",
                run_logs_host=None, base_image_override=None, device_gids=None,
                trace_loader=None):
    """-> (manifest dict, notes) for one search-space point."""
    notes = []
    gpus_per_node = cluster["gpus_per_node"]
    prefill, decode = point["prefill"], point["decode"]

    if int(prefill["tp"]) != int(decode["tp"]):
        raise ValueError(
            f"asymmetric TP (prefill tp={prefill['tp']}, decode tp={decode['tp']}); "
            "scripts/sglang_disagg resolves a single SGLANG_TP_SIZE for both roles"
        )
    tp = int(prefill["tp"])
    # DP_MODE=1 makes the launcher derive each role's TP/EP from nodes x GPUs,
    # not from SGLANG_TP_SIZE, so two TP8 prefill workers would silently become
    # one TP16 group. Only accept a layout where that derivation agrees with
    # the point.
    dp_attn = bool(prefill.get("dp-attn")) or bool(decode.get("dp-attn"))
    if bool(prefill.get("dp-attn")) != bool(decode.get("dp-attn")):
        raise ValueError(
            f"asymmetric dp-attn (prefill={prefill.get('dp-attn')}, "
            f"decode={decode.get('dp-attn')}); the launcher takes one DP_MODE")
    xP, yD = _worker_nodes(prefill, gpus_per_node), _worker_nodes(decode, gpus_per_node)
    if xP is None or yD is None:
        raise ValueError(
            f"tp={tp} does not fill a {gpus_per_node}-GPU node, so the point's worker "
            "counts cannot be expressed as the launcher's xP/yD node counts"
        )
    # A worker wider than one node needs the server told its rank in the group.
    # sglang_disagg_mori_io_ep.sh adds --nnodes/--node-rank only under
    # DP_MODE=1; at DP_MODE=0 it starts an independent single-node server per
    # node, so a TP16 worker would never be formed and the manifest would
    # measure something else entirely.
    if not dp_attn and max(int(prefill["tp"]), int(decode["tp"])) > gpus_per_node:
        raise ValueError(
            f"a worker spans more than one {gpus_per_node}-GPU node "
            f"(prefill tp={prefill['tp']}, decode tp={decode['tp']}) while "
            "dp-attn is off; the launcher passes --nnodes/--node-rank only "
            "under DP_MODE=1, so it would start one server per node instead")

    model = scenario["model"]
    model_dir = os.path.basename(model)
    mounts, model_path = resolve_mounts(model_dir, weights_dir, model_host_dir)
    # /run_logs is where the launcher's cross-node rendezvous happens, and
    # run.sh aborts when it is missing or unwritable. Two things have to hold
    # before docker starts, and neither did:
    #   * the path must be ABSOLUTE -- docker reads a relative source as a
    #     named volume, which gives each node its own empty one, so the
    #     readiness files a peer greps for would never appear;
    #   * it must exist and be writable by the submitting user, because docker
    #     otherwise creates it as root and the container cannot write there.
    run_logs = os.path.realpath(run_logs_host or os.path.join(os.getcwd(), "run_logs"))
    try:
        # Only a directory created here is opened up; an existing one keeps
        # its owner's mode.
        if not os.path.isdir(run_logs):
            os.makedirs(run_logs)
            os.chmod(run_logs, 0o1777)
    except OSError as exc:
        raise ValueError(
            f"cannot prepare the shared run-logs directory {run_logs}: {exc}; "
            "the disagg launchers coordinate readiness through it across "
            "nodes, so it has to exist on shared storage and be writable")
    if not os.access(run_logs, os.W_OK):
        raise ValueError(
            f"shared run-logs directory {run_logs} is not writable; "
            "scripts/sglang_disagg/run.sh aborts without it")

    # The aggregated route can leave this unset and let the runner download.
    # This one cannot: the launcher fetches on rank 0 and the peers wait for
    # the files to appear at a SHARED path, so with nothing staged the manifest
    # would carry `MODEL_PATH: null` and `--model-path None`.
    if not model_path:
        raise ValueError(
            f"no checkpoint for {model_dir!r} under {model_host_dir!r}; the "
            "disagg launcher needs a shared path its peers can wait on, so it "
            "cannot fall back to a per-rank download")
    key = f"infx-{scenario_key}-p{idx}"
    arch = gpu_arch

    # The override has to be known here, not patched in afterwards: it feeds the
    # image tag, and a tag computed from the pre-override image would let a
    # plain build and an overlay build share it.
    image = base_image_override or scenario["image"]
    kv = scenario.get("kv-p2p-transfer") or "nixl"
    build_args = dict(
        {"BASE_DOCKER": image},
        **({"RUN_AS_UID": str(os.getuid()), "RUN_AS_GID": str(os.getgid())}
           if cluster.get("run_as_submitting_user") else {}),
        # These manifests pass --group-add render too, so the image needs the
        # group for the same reason the aggregated ones do. Passed in rather
        # than resolved here: run as `python3 tools/infx2mad.py`, the
        # translator is __main__, so importing it by name would load a SECOND
        # copy whose Untranslatable is a different class than the one the
        # sweep catches -- and a missing group would abort the whole run with
        # a traceback instead of reporting one untranslatable scenario.
        **(device_gids or {}),
        # MoRI's kv-transfer path needs RDMA, and this image's distro
        # rdma-core does not speak Ruby's bnxt_re kernel ABI -- see the
        # RDMA_CORE_VERSION stage in pyt_infx.ubuntu.amd.Dockerfile. nixl
        # points do not take this path and the build cost is skipped for them.
        **({"RDMA_CORE_VERSION": cluster.get("rdma_core_version", "")}
           if kv == "mori" and cluster.get("rdma_core_version") else {}),
    )
    dp_mode = 1 if dp_attn else 0
    if dp_mode:
        # With DP_MODE=1 the launcher builds ONE group per role whose degree is
        # nodes x GPUS_PER_NODE, regardless of SGLANG_TP_SIZE. Two TP8 workers
        # on 8-GPU nodes therefore become a single TP16 group while the point
        # asked for two TP8 ones, so compare the derived degree with the
        # requested TP rather than node counts with worker counts.
        for role_name, role, nodes_of_role in (("prefill", prefill, xP), ("decode", decode, yD)):
            derived = nodes_of_role * gpus_per_node
            if derived != int(role["tp"]):
                raise ValueError(
                    f"DP_MODE=1 gives {role_name} one TP{derived}/EP{derived} group "
                    f"({nodes_of_role} node(s) x {gpus_per_node} GPUs), but the point "
                    f"asks for {role.get('num-worker', 1)} x TP{role['tp']}")
            # The launcher sets EP to the same degree; a point asking for any
            # other EP would run EP{derived} while its manifest and agentic
            # metadata still say the requested value.
            if int(role.get("ep", 1)) != derived:
                raise ValueError(
                    f"DP_MODE=1 gives {role_name} EP{derived} "
                    f"({nodes_of_role} node(s) x {gpus_per_node} GPUs), but the point "
                    f"asks for ep={role.get('ep', 1)}")
    if not dp_mode:
        # At DP_MODE=0 the launcher unsets the EP sizes and passes no
        # --ep-size, and the catalog sets none, so every role runs EP1; a
        # point asking for more would run EP1 while reporting its own value.
        for role_name, role in (("prefill", prefill), ("decode", decode)):
            if int(role.get("ep", 1)) != 1:
                raise ValueError(
                    f"{role_name} asks for ep={role.get('ep')} without DP-attention; "
                    "at DP_MODE=0 the launcher passes no --ep-size, so it would run EP1")
    catalog = catalog_name(mad_dir, model_dir)
    if dp_mode:
        allowed = dp_mode1_allowed(mad_dir)
        if allowed is None:
            raise ValueError(
                "cannot read MORI_DP_MODE1_ALLOWED_MODELS from "
                "scripts/sglang_disagg/sglang_disagg_mori_io_ep.sh, so DP_MODE=1 "
                "eligibility cannot be checked")
        if catalog not in allowed:
            raise ValueError(
                f"DP_MODE=1 point for catalog model {catalog!r}, which the "
                f"launcher refuses before starting a server (allowed: "
                f"{', '.join(sorted(allowed))})")
    conc = " ".join(str(c) for c in point["conc-list"])
    # Only fixed-seq-len states a sequence shape; an agentic replay has none.
    isl = workload.get("isl")
    osl = workload.get("osl")

    if int(prefill.get("ep", 1)) != int(decode.get("ep", 1)):
        notes.append(
            f"prefill ep={prefill.get('ep')} but decode ep={decode.get('ep')}; the "
            "launcher derives EP from DP_MODE and TP, so the two roles cannot differ"
        )
    for setting in prefill.get("additional-settings", []) or []:
        notes.append(f"prefill carries an InferenceX-side override not mapped: {setting}")
    # The launcher is driven by its own catalog entry, and this emitter does not
    # read the point's srt-recipe, so per-role serving settings -- speculative
    # decoding and draft configuration among them -- come from MAD's catalog
    # rather than from the scenario.
    if point.get("spec-decoding", "none") != "none":
        notes.append(
            f"point selects spec-decoding={point['spec-decoding']}, which the "
            "catalog-driven launcher does not receive: per-role recipe settings "
            "are not forwarded by this emitter")
    notes.append(
        "per-role serving arguments come from scripts/sglang_disagg/models.yaml, "
        "not from the point's srt-recipe")
    if workload_name == "fixed-seq-len":
        # InferenceX's own client (infx/bench/fixed_seq.py::srt_sweep) sends
        # 10x concurrency prompts; this launcher's benchmark_xPyD.sh computes
        # max(16, 2x concurrency) instead -- a conc=64 point sends 128 prompts
        # here against 640 there. Nothing in this emitter can change that: it
        # is the shared disagg benchmark script's own policy, not a per-point
        # setting, so it is recorded once rather than measured and fixed.
        notes.append(
            "measurement prompt count is max(16, 2x concurrency) "
            "(scripts/sglang_disagg/benchmark_xPyD.sh), not InferenceX's "
            "10x concurrency (infx/bench/fixed_seq.py); a shorter client run "
            "measures a shorter steady-state window")
        # InferenceX's disagg generator (infx/srt_slurm/generate.py) sets
        # RANDOM_RANGE_RATIO=0.8; benchmark_xPyD.sh hardcodes 1.0, so every
        # prompt is exactly isl/osl tokens here -- a different length
        # distribution, independent of the prompt count above.
        notes.append(
            "request lengths use --random-range-ratio 1.0 "
            "(scripts/sglang_disagg/benchmark_xPyD.sh, hardcoded), not "
            "InferenceX's 0.8 (infx/srt_slurm/generate.py); every prompt is "
            "exactly the point's isl/osl instead of jittered below it")
    if scenario.get("router"):
        notes.append(
            f"scenario pins router {scenario['router'].get('name')} "
            f"{scenario['router'].get('version')}; the launcher uses the router its "
            "image ships"
        )

    env_vars = {
        "MODEL_NAME": catalog,
        "MODEL_PATH": model_path,
        # Without this, run.sh picks a repository from the CATALOG KEY when the
        # weights are not already there: catalog key DeepSeek-R1 falls back to
        # deepseek-ai/DeepSeek-R1-0528 even for a scenario naming an AMD FP4
        # build, so a fresh run would fetch a different checkpoint than the
        # provenance claims.
        "MODEL_REPO": scenario["model"],
        "RUN_MORI": "1" if kv == "mori" else "0",
        "DP_MODE": str(dp_mode),
        "xP": str(xP),
        "yD": str(yD),
        "SGLANG_TP_SIZE": str(tp),
        # run.sh turns SGLANG_TP_SIZE into IO_EP_TP_SIZE, but at DP_MODE=0
        # sglang_disagg_mori_io_ep.sh sizes both roles from GENERIC_TP_SIZE
        # (default 8) and never reads it. Without this a TP4 point on a
        # four-GPU node would launch TP8 servers.
        "GENERIC_TP_SIZE": str(tp),
        "GPUS_PER_NODE": str(gpus_per_node),
        "KV_TRANSFER_BACKEND": kv,
        "BENCHMARK_CONCURRENCY_LEVELS": conc,
        "BENCHMARK_FAIL_FAST": "1",
        # benchmark_xPyD.sh names its CSV from MODEL_NAME, the catalog key
        # (DeepSeek-R1), while the manifest collects one named for the
        # checkpoint (DeepSeek-R1-0528). Name it explicitly so producer,
        # collector and provenance agree.
        "MAD_OUTPUT_CSV": f"perf_sglang-disagg-{model_dir}.csv",
        "NCCL_DEBUG": "WARN",
    }
    if isl is not None and osl is not None:
        env_vars["BENCHMARK_COMBINATIONS"] = f"{int(isl)}/{int(osl)}"
    if workload_name == "agentic-coding":
        # --agentic-duration is in the aggregated-only unsupported list above
        # (DURATION is never set here), so this route always falls to
        # agentic_lib.sh's 120 s default against the scenario's 900 s minimum.
        # The aggregated emitter records exactly this when duration is None;
        # match it rather than leave the disagg route the one path silent
        # about a submission_valid=false run.
        notes.append(
            "no measurement window set; agentic_lib.sh defaults to 120 s "
            "while the InferenceX scenario minimum is 900 s, so the run is "
            "marked submission_valid=false. The disagg route does not accept "
            "--agentic-duration.")
        # The disagg launcher already has the agentic hook; selecting it is a
        # single env var, which is why agentic needed new code only on the
        # aggregated side.
        # Both selectors: the launcher keys its agentic setup (metrics,
        # AGENTIC_SERVER_METRICS, dropping --disable-radix-cache) off
        # BENCHMARK_SCRIPT, while BENCHMARK_SCRIPT_FILE only picks the hook.
        # agentic_lib.sh passes AGENTIC_CONC straight after --concurrency in a
        # SINGLE aiperf invocation; it does not loop. A two-value list would
        # become `--concurrency 16 32`, which is one malformed run rather than
        # two replays, so refuse instead of emitting it.
        if len(point["conc-list"]) != 1:
            raise ValueError(
                f"agentic point lists {len(point['conc-list'])} concurrencies "
                f"{point['conc-list']}; the disagg agentic hook runs one per "
                "invocation and does not loop over them")
        env_vars["BENCHMARK_SCRIPT"] = "agentic"
        env_vars["BENCHMARK_SCRIPT_FILE"] = "benchmark_agentic.sh"
        env_vars["AGENTIC_CONC"] = conc
        env_vars.pop("BENCHMARK_COMBINATIONS", None)
        # The pinned InferenceX aggregator (infx/results/agentic/__init__.py,
        # _gpu_shape) reads its OWN names for the replay's topology, separate
        # from what the launcher itself exports and from AGENTIC_CONC above:
        #   * CONC, read directly, not derived from AGENTIC_CONC;
        #   * IS_MULTINODE, which selects the single-node-vs-multinode branch
        #     of _gpu_shape at all -- missing it takes the single-node path
        #     regardless of what else is set, defaulting to TP=1;
        #   * PREFILL_TP/DECODE_TP, not PREFILL_TP_SIZE/DECODE_TP_SIZE (the
        #     launcher's own names for the same values);
        #   * PREFILL_NUM_WORKERS/DECODE_NUM_WORKERS, the role's WORKER count
        #     (recipe num-worker), not xP/yD. xP/yD are NODE counts, and a
        #     worker can span more than one node (TP16 on 8-GPU nodes is one
        #     worker over two nodes, num-worker=1, xP=2) -- using xP/yD here
        #     would count each of that worker's nodes as a separate worker and
        #     double its GPUs.
        # Without these a 1P2D TP8 replay (24 GPUs) is recorded as single-node
        # TP1 conc=0, and per-GPU throughput divides by 1 instead of 24.
        env_vars["CONC"] = conc
        env_vars["IS_MULTINODE"] = "true"
        env_vars["PREFILL_NUM_WORKERS"] = str(prefill.get("num-worker", 1))
        env_vars["PREFILL_TP"] = str(tp)
        env_vars["DECODE_NUM_WORKERS"] = str(decode.get("num-worker", 1))
        env_vars["DECODE_TP"] = str(tp)
        if prefill.get("ep") is not None:
            env_vars["PREFILL_EP"] = str(prefill["ep"])
        if decode.get("ep") is not None:
            env_vars["DECODE_EP"] = str(decode["ep"])
        # Scenario identity the aggregator also records. image is the point's
        # own (base_image_override may differ from scenario["image"]), and
        # MODEL_PREFIX additionally drives agentic_lib.sh's trace-corpus
        # selection -- without it corpus selection falls back to the catalog
        # name, which can pick a different dataset than the scenario's own.
        env_vars["FRAMEWORK"] = scenario["framework"]
        env_vars["PRECISION"] = scenario["precision"]
        env_vars["IMAGE"] = image
        env_vars["MODEL_PREFIX"] = scenario["model-prefix"]
        # MODEL must be present but EMPTY: empty makes the hook discover the
        # served name from /v1/models, and having it in the environment keeps
        # the discovered value exported to the aggregation subprocess (a plain
        # `MODEL=...` assignment in agentic_lib.sh would otherwise not reach
        # it, leaving the aggregate with no model identity). A non-empty value
        # would skip discovery and address an id the server never registered.
        env_vars["MODEL"] = ""
        # This route skips perf collection, so nothing else would notice an
        # aggregate with no measured request or a high error rate: make the
        # hook's own validation decide the run.
        env_vars["AGENTIC_STRICT_RESULT"] = "1"
        # MODEL_PREFIX alone selects the corpus by agentic_lib.sh's own family
        # list, which lags InferenceX's (a Kimi-K2 replay would take the
        # 256k-capped corpus where InferenceX takes the uncapped one). Pin the
        # loader InferenceX itself picks, as the aggregated route does.
        if trace_loader:
            env_vars["WEKA_LOADER_OVERRIDE"] = trace_loader
        else:
            notes.append(
                "the InferenceX trace-loader policy could not be read from the "
                "checkout, so the corpus is agentic_lib.sh's own choice for "
                "MODEL_PREFIX and may differ from InferenceX's")
        env_vars["DISAGG"] = "true"
        env_vars["PREFILL_DP_ATTN"] = "true" if prefill.get("dp-attn") else "false"
        env_vars["DECODE_DP_ATTN"] = "true" if decode.get("dp-attn") else "false"

    nodes = xP + yD
    gpus = ",".join(str(i) for i in range(gpus_per_node))
    args = (
        f"--model-name {catalog} --model-path {model_path} "
        f"--run-mori {env_vars['RUN_MORI']} --dp-mode {dp_mode} "
        f"--xp {xP} --yd {yD} --gpus-per-node {gpus_per_node} "
        f"--kv-transfer-backend {kv}"
    )

    manifest = {
        "built_images": {
            key: {
                "model": key,
                "docker_image": image_tag(scenario["model-prefix"], "disagg", arch,
                                          image, build_args),
                "dockerfile": "docker/pyt_infx.ubuntu.amd.Dockerfile",
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
                "dockerfile": "docker/pyt_infx.ubuntu.amd.Dockerfile",
                "scripts": f"scripts/{runner['dir']}/run.sh",
                "data": "",
                "n_gpus": "-1",
                "owner": "mad.support@amd.com",
                "training_precision": "",
                # The agentic hook writes an aggregate JSON and plots and
                # explicitly does not touch perf.csv (agentic_lib.sh), so
                # naming a results CSV would make every successful replay look
                # like a failed collection. MAD's own disagg agentic entries
                # declare none either; the results are the aggregate under
                # /run_logs. The sweep benchmark does produce the CSV.
                **({"multiple_results": f"perf_sglang-disagg-{model_dir}.csv"}
                   if workload_name != "agentic-coding" else {}),
                "tags": ["pyt", "sglang", "disagg", "inference", "inferencex",
                         scenario["model-prefix"], arch],
                "timeout": timeout,
                "args": args,
                "additional_docker_run_options": (
                    "--privileged --group-add render --shm-size 64G "
                    "--device=/dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 "
                    "--ulimit nofile=1048576:1048576 -v /sys:/sys:ro "
                    "-v /sys/class/infiniband:/sys/class/infiniband:ro "
                    "-v /run/udev:/run/udev:ro"
                    + (f" -u {os.getuid()}:{os.getgid()}"
                       if cluster.get("run_as_submitting_user") else "")
                ),
            }
        },
        "context": {
            # Nothing to collect on the agentic route: its results are the
            # aggregate JSON the hook leaves under /run_logs.
            "skip_perf_collection": workload_name == "agentic-coding",
            "docker_env_vars": dict(env_vars, MAD_SYSTEM_GPU_ARCHITECTURE=arch,
                                    **cluster.get("fabric_env", {})),
            # /run_logs must be shared and writable: the launcher coordinates
            # readiness between prefill, decode and proxy through files there.
            "docker_mounts": dict(mounts, **{
                "/dev/infiniband": "/dev/infiniband",
                "/sys/class/infiniband": "/sys/class/infiniband",
                "/run_logs": run_logs,
            }),
            "docker_build_arg": build_args,
            "gpu_vendor": "AMD",
            "guest_os": "UBUNTU",
            "docker_gpus": gpus,
        },
        "credentials_required": [],
        "summary": {"successful_builds": [], "failed_builds": [], "total_build_time": 0,
                    "successful_pushes": [], "failed_pushes": []},
        "deployment_config": {
            "target": "slurm",
            "slurm": {
                "partition": cluster["partition"],
                "nodes": nodes,
                "gpus_per_node": gpus_per_node,
                "time": slurm_time,
                "output_dir": "./slurm_output",
                "exclusive": True,
                "network_interface": cluster["network_interface"],
                "skip_gpus_directive": cluster["skip_gpus_directive"],
                **({"exclude": cluster["exclude_nodes"]} if cluster.get("exclude_nodes") else {}),
            },
            "distributed": {
                "launcher": "sglang-disagg",
                "backend": "nccl",
                "port": 29500,
                "nnodes": nodes,
                "nproc_per_node": gpus_per_node,
                "sglang_disagg": {"prefill_nodes": xP, "decode_nodes": yD},
            },
            "env_vars": dict(env_vars, **cluster.get("fabric_env", {})),
            "monitor": True,
            "debug": False,
            "docker_gpus": gpus,
            "gpus_per_node": gpus_per_node,
        },
        "_infx_provenance": {
            "scenario": scenario_key,
            "master_config": "configs/amd-master.yaml",
            "framework": scenario["framework"],
            "mad_runner": f"scripts/{runner['dir']}",
            "workload": workload_name,
            "precision": scenario["precision"],
            "point_index": idx,
            "inferencex_commit": infx_commit,
            "model_catalog_name": catalog,
            "topology": {"prefill_nodes": xP, "decode_nodes": yD, "tp": tp,
                         "ep": point["prefill"].get("ep"), "dp_attn": bool(dp_mode),
                         "kv_p2p_transfer": kv},
            "concurrency": point["conc-list"],
            "results": ("aggregate JSON under /run_logs; the agentic hook "
                        "writes no perf CSV"
                        if workload_name == "agentic-coding"
                        else f"perf_sglang-disagg-{model_dir}.csv"),
            "fidelity_gaps": notes,
        },
    }
    return manifest, notes
