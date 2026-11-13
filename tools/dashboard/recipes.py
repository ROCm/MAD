"""
Run recipes for one registered model, per cluster type: the madengine command, and the
plain Docker (or sbatch) commands that reproduce what madengine does without it.

The Docker path follows madengine's ContainerRunner: build with the model's Dockerfile and
build context, run with its AMD security options and MAD_* environment, mount the MAD
checkout at /myworkspace, copy the model's script directory into the run directory and
invoke the script with the card's args. Two differences: madengine passes only the
requested GPUs' render nodes, which are host-specific, so the recipe passes /dev/dri and
limits a smaller run with HIP_VISIBLE_DEVICES; and it leaves out what madengine adds only
for profiling tools, pre-scripts and data providers (SYS_ADMIN, /dev/fuse, data mounts). SLURM multi-node cards
(distributed.launcher = slurm_multi) run their own launcher script, so their plain
recipe builds and pushes the image and submits that script with sbatch, as the
framework READMEs document. Opt-in multi-node configs (hidden from discovery by default)
get madengine's primus launcher on SLURM; registries that document a manual per-node run
(status.yaml k8s: manual) get that run as their Kubernetes recipe.
"""
import json
import os
import re
import shlex

DEFAULT_REPO = "ROCm/MAD"
ISA_OF = {"MI300X": "gfx942", "MI325X": "gfx942", "MI350X": "gfx950", "MI355X": "gfx950",
          "MI455X": "gfx1250"}
GPUS_PER_NODE = 8

# madengine's AMD `docker run` options (ContainerRunner.run_container) plus the GPU devices.
# SYS_ADMIN and /dev/fuse are not here: madengine adds them only for profiling tools,
# pre-scripts and FUSE data mounts, none of which a plain run uses.
DOCKER_RUN_OPTS = [
    "--network host --ipc=host -u root --group-add video",
    "--cap-add=SYS_PTRACE",
    "--security-opt seccomp=unconfined --security-opt apparmor=unconfined",
    "--device=/dev/kfd --device=/dev/dri",
]

HF_TOKEN_NOTE = "# Hugging Face token; required for gated models"

# (key, label, help) for the recipe page's cluster choice, in display order.
CLUSTERS = [
    ("local", "Single node", "Run on one machine with its GPUs."),
    ("slurm", "SLURM", "Submit from a SLURM login node; the job starts one container per node."),
    ("k8s", "Kubernetes", "Start the benchmark on each node yourself (Kubernetes pods or bare metal)."),
]


def is_multi_node(row):
    """A slurm_multi card: its own launcher script starts the containers, under SLURM only."""
    return row["run"]["launcher"] == "slurm_multi"


def clusters(row):
    """Cluster types this card has a recipe for."""
    dep = row["deployment"]
    keys = ["slurm" if dep["scheduler"] == "slurm" else "local"]
    if dep["k8s"] == "manual":
        keys.append("k8s")
    return keys


def unavailable(row):
    """{cluster: why there is no recipe for it}, shown next to the struck-through choice."""
    dep = row["deployment"]
    reasons = {}
    if dep["status"] == "opt-in":
        reasons["local"] = "the config is sized for two or more nodes."
    elif dep["scheduler"] == "slurm":
        reasons["local"] = "the launcher script needs a SLURM allocation, even on one node."
    else:
        reasons["slurm"] = "single-node model card."
    if dep["k8s"] != "manual":
        reasons["k8s"] = ("madengine's slurm_multi launcher is SLURM-only." if is_multi_node(row)
                          else "no Kubernetes recipe in MAD for this model.")
    return reasons


def needs_primus(row):
    return "primus" in row["dockerfile"].lower()


def image_name(row):
    return "mad/" + re.sub(r"[^a-z0-9.]+", "-", row["name"].lower()).strip("-")


def continued(head, parts, indent="  "):
    """'head \\\n  part \\\n  part' — one option group per line, like the vLLM recipes."""
    parts = [p for p in parts if p]
    if not parts:
        return head
    return head + " \\\n" + " \\\n".join(indent + p for p in parts)


def placeholder(value):
    return bool(re.fullmatch(r"<[^<>]+>", value.strip()))


def get_mad(row, ref=None, repo=DEFAULT_REPO):
    """Clone MAD; a frozen release pins the commit it was frozen from."""
    lines = [f"git clone https://github.com/{repo}.git", "cd " + repo.rsplit("/", 1)[-1]]
    if ref:
        lines.append(f"git checkout {ref}")
    if needs_primus(row):
        lines.append("git submodule update --init scripts/Primus   # Primus configs and launcher")
    return "\n".join(lines)


def script_invocation(scripts):
    """(directory to copy, command to run in it) — ContainerRunner's rules for 'scripts'."""
    if scripts.endswith(".sh"):
        return os.path.dirname(scripts), "bash " + os.path.basename(scripts)
    if scripts.endswith(".py"):
        return os.path.dirname(scripts), "python3 " + os.path.basename(scripts)
    return scripts, "bash run.sh"


def gpu_count(run):
    n = run["n_gpus"]
    return GPUS_PER_NODE if n in ("-1", "", "all") else int(n)


def nodes_label(n):
    return f"{n} node{'s' if n != 1 else ''}"


def is_xpyd(row):
    """Disaggregated prefill/decode card: its launcher takes the topology from xP and yD."""
    env = row["run"]["env_vars"]
    return is_multi_node(row) and "xP" in env and "yD" in env


# --------------------------------------------------------------------------- madengine


def madengine_steps(row, sku, ref=None, repo=DEFAULT_REPO):
    tag = row["name"]
    steps = [
        {"title": "Get MAD and madengine", "code": get_mad(row, ref, repo) + "\npip install -r requirements.txt"},
    ]
    if not is_multi_node(row):
        steps.append({
            "title": "Run",
            "code": f"export MAD_SECRETS_HFTOKEN=<your-hf-token>   {HF_TOKEN_NOTE}\n"
                    + continued(f"madengine run --tags {tag}", ["--live-output"]),
            "note": f"madengine detects the {sku} ({ISA_OF[sku]}), builds the image, runs the benchmark and "
                    "appends the result to perf.csv in the MAD directory.",
        })
        return steps

    env = {k: v for k, v in row["run"]["env_vars"].items() if placeholder(v)}
    run_opts = ["--manifest-file build_manifest.json"]
    if env:
        run_opts.append("--additional-context " + shlex.quote(json.dumps({"env_vars": env})))
    # madengine sizes the allocation from slurm.nodes; it hands the card's args (-N 2 -n 2)
    # to the launcher script, not to sbatch, so the node count has to be in the context.
    slurm = {"partition": "<partition>", "nodes": row["nodes"]}
    steps.append({
        "title": "Build on a compute node and push",
        "code": f"export MAD_SECRETS_HFTOKEN=<your-hf-token>   {HF_TOKEN_NOTE}\n" + continued(
            f"madengine build --tags {tag}",
            ["--registry <namespace>/<repo> --build-on-compute",
             "--additional-context " + shlex.quote(json.dumps({"slurm": slurm}))]),
    })
    steps.append({
        "title": f"Run on {nodes_label(row['nodes'])}",
        "code": continued(f"madengine run --tags {tag}", run_opts),
        "note": "Run from a SLURM login node and replace the <placeholders>. The launcher comes from the model "
                "card; slurm.nodes in the build context sets the node count.",
    })
    if is_xpyd(row):
        # A runtime slurm block replaces the manifest's, so it repeats the partition.
        context = {"slurm": {"partition": "<partition>", "nodes": 4}, "env_vars": {**env, "xP": "2", "yD": "2"}}
        steps.append({
            "title": "Other topologies: xP prefill + yD decode nodes",
            "code": continued(f"madengine run --tags {tag}", [
                "--manifest-file build_manifest.json", "--additional-context " + shlex.quote(json.dumps(context))]),
            "note": "slurm.nodes must equal xP + yD; this example runs 2 prefill and 2 decode nodes. "
                    "env_vars here override the card's xP=1 yD=1.",
        })
    return steps


# --------------------------------------------------------------------------- docker


def build_command(row, sku, image):
    run = row["run"]
    build_args = {"MAD_SYSTEM_GPU_ARCHITECTURE": ISA_OF[sku], **run["build_args"]}
    return continued("docker build --network=host --pull", [
        f"-t {shlex.quote(image)}",
        f"-f {shlex.quote(run['dockerfile_path'])}",
        *(f"--build-arg {k}={shlex.quote(str(v))}" for k, v in build_args.items()),
        shlex.quote(run["context"]),
    ])


def container_script(row):
    """What madengine runs inside the container, as a `bash -c` script (one step per line)."""
    run = row["run"]
    script_dir, invoke = script_invocation(run["scripts"])
    steps = []
    if run["url"]:
        model_dir = run["url"].rstrip("/").split("/")[-1]
        steps.append(f"git clone {run['url']}")
    else:
        model_dir = "run_directory"
        steps.append(f"mkdir -p {model_dir}")
    steps.append(f"cp -vLR {script_dir}/. {model_dir}/")
    args = " ".join(shlex.quote(a) for a in shlex.split(run["args"])) if run["args"] else ""
    steps.append(f"cd {model_dir} && {invoke} {args}".rstrip())
    return "\n".join(["set -e"] + steps), model_dir


def docker_steps(row, sku, ref=None, repo=DEFAULT_REPO):
    if is_multi_node(row):
        return slurm_steps(row, sku, ref, repo)
    run = row["run"]
    image = image_name(row)
    n = gpu_count(run)
    env = [
        f"-e MAD_MODEL_NAME={shlex.quote(row['name'])}",
        f"-e MAD_SYSTEM_GPU_ARCHITECTURE={ISA_OF[sku]}",
        f"-e MAD_SYSTEM_GPU_PRODUCT_NAME={shlex.quote('AMD Instinct ' + sku)}",
        f"-e MAD_SYSTEM_NGPUS={GPUS_PER_NODE} -e MAD_RUNTIME_NGPUS={n}",
    ]
    if n < GPUS_PER_NODE:
        env.append("-e HIP_VISIBLE_DEVICES=" + ",".join(str(i) for i in range(n)))
        # As madengine does: Ray in the vLLM images ignores HIP_VISIBLE_DEVICES unless this is cleared.
        env.append("-e RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=")
    if run["multiple_results"]:
        env.append(f"-e MAD_OUTPUT_CSV={shlex.quote(run['multiple_results'])}")
    env.append('-e MAD_SECRETS_HFTOKEN="$HF_TOKEN"')
    env += [f"-e {k}={shlex.quote(v)}" for k, v in run["env_vars"].items()]
    script, model_dir = container_script(row)
    run_cmd = continued("docker run --rm -it", [
        *DOCKER_RUN_OPTS,
        '-v "$PWD":/myworkspace -w /myworkspace',
        *env,
        image,
        # One command per line inside the quotes; the indent is whitespace to the inner shell.
        "bash -c " + shlex.quote(script).replace("\n", "\n    "),
    ])
    if run["multiple_results"]:
        result = f"Results are written to {model_dir}/{run['multiple_results']} in the MAD directory."
    else:
        result = "The script prints the result as a `performance: <value> <unit>` line."
    return [
        {"title": "Get MAD", "code": get_mad(row, ref, repo)},
        {"title": "Build the image", "code": build_command(row, sku, image)},
        {"title": "Run the benchmark", "code": f"export HF_TOKEN=<your-hf-token>   {HF_TOKEN_NOTE}\n{run_cmd}",
         "note": f"Uses {'all ' + str(n) if n == GPUS_PER_NODE else n} GPU{'s' if n > 1 else ''}. {result}"},
    ]


def slurm_steps(row, sku, ref=None, repo=DEFAULT_REPO):
    run = row["run"]
    image = "<registry>/<repo>:" + image_name(row).split("/", 1)[1]
    # Submit from the registry directory (the launchers find their helpers via
    # SLURM_SUBMIT_DIR), naming the script relative to it.
    submit_dir = os.path.join("scripts", row["registry"])
    script = os.path.relpath(run["scripts"], submit_dir)
    # The launchers start one container per node from DOCKER_IMAGE_NAME, except Large-EP's, which
    # reads DOCKER_IMAGE and has the card pin it to a prebuilt image.
    image_vars = ["DOCKER_IMAGE_NAME"] + (["DOCKER_IMAGE"] if "DOCKER_IMAGE" in run["env_vars"] else [])
    exports = [f"export {k}={shlex.quote(image)}" for k in image_vars]
    exports += [f"export {k}={shlex.quote(v)}" for k, v in run["env_vars"].items() if k not in image_vars]
    exports.append(f"export MAD_MODEL_NAME={shlex.quote(row['name'])}")
    if run["multiple_results"]:
        exports.append(f"export MAD_OUTPUT_CSV={shlex.quote(run['multiple_results'])}")
    sbatch_opts = run["args"] or f"-N {row['nodes']} -n {row['nodes']}"
    steps = [
        {"title": "Get MAD", "code": get_mad(row, ref, repo)},
        {"title": "Build and push an image every node can pull",
         "code": build_command(row, sku, image) + f"\ndocker push {shlex.quote(image)}"},
        {"title": f"Submit the {row['nodes']}-node job",
         "code": f"cd {submit_dir}\n" + "\n".join(exports)
                 + f"\nsbatch -p <partition> {sbatch_opts} {script}",
         "note": "Replace the <placeholders>. See the framework guide for node, NIC and shared-storage "
                 "prerequisites."},
    ]
    if is_xpyd(row):
        steps.append({
            "title": "Other topologies: xP prefill + yD decode nodes",
            "code": f"export xP=2 yD=2\nsbatch -p <partition> -N $((xP + yD)) -n $((xP + yD)) {script}",
            "note": "Run in the same shell as the previous step; the launcher reads xP and yD from the environment.",
        })
    return steps


# --------------------------------------------------------------------------- opt-in multi-node


def opt_in_steps(row, sku, ref=None, repo=DEFAULT_REPO):
    """madengine's primus launcher on SLURM for a config discovery hides unless its switch is set."""
    # --config_path x or --config_path=x, as collect.arg_value reads it.
    config = re.search(r"(?:^|\s)--config_path[\s=]+(\S+)", row["run"]["args"]).group(1)
    context = {
        "slurm": {"partition": "<partition>", "nodes": "<nodes>", "gpus_per_node": GPUS_PER_NODE},
        "distributed": {"launcher": "primus", "nnodes": "<nodes>", "nproc_per_node": GPUS_PER_NODE,
                        "primus": {"config_path": config}},
    }
    # <nodes> is a number to fill in, not a string.
    context_json = json.dumps(context, indent=2).replace('"<nodes>"', "<nodes>")
    switch = row["deployment"]["switch"]
    return [
        {"title": "Get MAD and madengine", "code": get_mad(row, ref, repo) + "\npip install -r requirements.txt"},
        {"title": "Describe the multi-node run",
         "code": f"cat > multinode.json <<'EOF'\n{context_json}\nEOF",
         "note": "Replace <partition>, and <nodes> with 2 or more in both places; the config's dcn_* = -1 axes "
                 "scale with the node count."},
        {"title": "Build on a compute node and push",
         "code": f"export {switch}=1   # discover the multi-node-only configs\n"
                 f"export MAD_SECRETS_HFTOKEN=<your-hf-token>   {HF_TOKEN_NOTE}\n"
                 + continued(f"madengine build --tags {row['name']}",
                             ["--registry <namespace>/<repo> --build-on-compute",
                              "--additional-context-file multinode.json"])},
        {"title": "Run",
         "code": "madengine run --manifest-file build_manifest.json",
         "note": "Experimental: MAD does not validate this config. On multi-node MaxText, Primus may apt-install "
                 "InfiniBand packages inside the container; madengine warns when it detects this."},
    ]


# --------------------------------------------------------------------------- manual per-node


def kvcache_manual_steps(row, sku, ref=None, repo=DEFAULT_REPO):
    """run_node.sh on each of the two nodes, the readme's path for Kubernetes and bare metal."""
    env = row["run"]["env_vars"]
    image = "<registry>/<repo>:" + image_name(row).split("/", 1)[1]
    bench = "/workspace/kvcache_transfer_bench"
    run_cmd = continued("docker run --rm", [
        "--device /dev/dri --device /dev/kfd --device /dev/infiniband",
        '--network host --hostname "$(hostname)" --add-host "$(hostname):$(hostname -I | awk \'{print $1}\')"',
        "--ipc host --group-add video --cap-add SYS_PTRACE --privileged=true",
        "--security-opt seccomp=unconfined --ulimit memlock=-1:-1 --shm-size 64G",
        f'-v /sys:/sys -v "$PWD/scripts/{row["registry"]}":{bench}',
        "-e JOB_ID",
        shlex.quote(image),
        f"{bench}/scripts/run_node.sh",
        "--node1 <target-hostname> --node2 <initiator-hostname>",
        f"--kv-cache-test-path {bench} --shared-folder {bench}",
        f"--backends {shlex.quote(env.get('BENCH_BACKENDS', 'all'))}"
        f" --start-size {env.get('START_SIZE', '4096')} --stop-size {env.get('STOP_SIZE', '1073741824')}",
        f"--ibdevice {env.get('IBDEVICE', 'mlx5_0')} --sync-port {env.get('SYNC_PORT', '9999')}",
        f"--model-name {shlex.quote(env.get('BENCH_MODEL_NAME', row['name'].rsplit('/', 1)[-1]))}",
    ])
    return [
        {"title": "Get MAD on storage both nodes mount", "code": get_mad(row, ref, repo)},
        {"title": "Build and push an image both nodes can pull",
         "code": build_command(row, sku, image) + f"\ndocker push {shlex.quote(image)}"},
        {"title": "Run on both nodes",
         "code": "export JOB_ID=<same-id-on-both-nodes>   # names the shared results folder\n" + run_cmd,
         "note": "Run the same command from the MAD directory on each node. The node whose hostname is --node1 "
                 "is the target, the other the initiator; results land in shared/results_<JOB_ID>. On Kubernetes, "
                 "run it in one privileged pod per node with host networking, /dev/infiniband and the shared "
                 "volume (e.g. a PVC)."},
    ]


# Registries with `k8s: manual` in status.yaml, and the recipe for their manual run.
MANUAL_STEPS = {"kvcache_transfer_bench": kvcache_manual_steps}


def recipe(row, sku, ref=None, repo=DEFAULT_REPO):
    """{cluster: {method: steps}} for one model on one GPU product; `ref` pins the MAD commit (frozen releases)."""
    dep = row["deployment"]
    if dep["status"] == "opt-in":
        out = {"slurm": {"madengine": opt_in_steps(row, sku, ref, repo)}}
    else:
        out = {clusters(row)[0]: {
            "madengine": madengine_steps(row, sku, ref, repo),
            "docker": docker_steps(row, sku, ref, repo),
        }}
    if dep["k8s"] == "manual":
        out["k8s"] = {"docker": MANUAL_STEPS[row["registry"]](row, sku, ref, repo)}
    return out
