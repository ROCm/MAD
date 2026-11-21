#!/usr/bin/env python3
"""What every distributed-inference card does on each node, recorded and compared.

For every card in the three multi-node catalogs this runs the card's launcher
(run_xPyD_models.slurm / run_multinode.slurm) offline, against the stand-in
srun / docker / hostname in stubs/, and records per node rank:

  - the `docker run` it issues: flags, mounts, environment (in order), image,
    and the command run in the container;
  - the engine command(s) that rank would run, from the role script
    (vllm_disagg.sh, sglang_disagg_mori_io_ep.sh, serve_colocated.sh) in DRY_RUN
    mode with exactly that rank's container environment.

plus the launcher's exit status and the order of its srun / docker calls. Paths
that differ between runs are replaced with placeholders.

What is recorded is one snapshot per CASE, not per card. A case is a distinct path through
the launcher code: the catalog, the connector and EP backend, wide EP, DP mode, TP-within-EP,
a pool spanning nodes, the router's node, the benchmark, and the engine features the run turns
on (path_of). Every card runs on every check; a card whose path a case already records needs
no snapshot of its own, since what differs between such cards is recipe data (models.yaml),
not code. So:

  - each case's snapshot is compared with cases/<catalog>/<card>.json; any difference fails;
  - every card's path must be a case's: a card on a new path fails until it is recorded
    (--update records it, from that card);
  - the other checks (--spooled, --spec-file, --equivalence) compare every card under that
    dispatch with the same card's own run, live, with nothing stored.

A launcher change must leave every case unchanged, or change it on purpose: rerun with
--update and review the case diff in the commit. Adding a card records nothing unless it
takes a new path.

Each card runs in its own temporary directory, on a copy of scripts/, with HOME,
LOG_PATH, NVME_ROOT, SHARED_MOUNT and the JIT cache pointed inside it, so a run
writes nothing into the repository or the machine. (The launchers still create
/tmp/vllm_cache, which they hardcode.)

Usage:
  python3 scripts/common/tests/snapshot/snapshot.py              # compare
  python3 scripts/common/tests/snapshot/snapshot.py --update     # record the cases
  python3 scripts/common/tests/snapshot/snapshot.py --card qwen3 # cards matching
"""
import argparse
import collections
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[2]          # snapshot -> tests -> common -> scripts
COMPAT = HERE.parents[3] / "compat"   # the capability catalog (a test may point SCRIPTS at a copy)
STUBS = HERE / "stubs"
CASES = HERE / "cases"
CATALOGS = ("vllm_dissag", "sglang_disagg", "vllm_multinode")
JOB_ID = "4242"

# docker run options that take a value as the next argument.
VALUE_OPTS = {
    "-v", "--volume", "-e", "--env", "--env-file", "--device", "--name", "--shm-size",
    "--ulimit", "--entrypoint", "--group-add", "--cap-add", "--security-opt",
    "--network", "--ipc", "-w", "--workdir", "-u", "--user", "--gpus", "--label",
    "--hostname", "--add-host", "--mount", "--memory", "--pids-limit",
}
# Where the launchers mount scripts inside the container, and what that is here.
CONTAINER_SCRIPT_DIRS = {"/opt/nixl-vllm-cookbook": "vllm_dissag", "/opt/mooncake-cookbook": "sglang_disagg",
                         "/mad_scripts": ""}


TOPOLOGY_RECIPE_KEYS = ("EP_TP_SIZE",)   # what a launcher takes from the recipe to size the run


def launch_env(card, catalog, arch=None):
    """The settings a launcher builds its topology from: the card's env_vars over the recipe's
    (the variant of the card's GPU) for the keys the launcher reads from the recipe."""
    import yaml
    sys.path.insert(0, str(SCRIPTS / "common" / "recipe"))
    import models
    env = resolved_env(card, catalog)
    recipes = SCRIPTS / ("vllm_dissag" if catalog in ("vllm_dissag", "vllm_multinode") else catalog) / "models.yaml"
    name = env.get("MODEL_NAME")
    if name and recipes.is_file():
        r = models.effective(yaml.safe_load(recipes.read_text()) or {}, name, arch or gpu_arch(card)) or {}
        for k in TOPOLOGY_RECIPE_KEYS:
            if k in (r.get("env") or {}):
                env.setdefault(k, str(r["env"][k]))
    return env


def _compat():
    """compat/ as one dict (scripts/common/catalog.py)."""
    sys.path.insert(0, str(SCRIPTS / "common"))
    import catalog
    return catalog.load(str(COMPAT))


def gpu_memory_bytes(arch):
    """One GPU's memory for an arch: its first product in the catalog (what the probe would read)."""
    prods = (_compat()["accelerators"].get(arch) or {}).get("products") or {}
    return int(next(iter(prods.values()))["hbm_gb"] * 1e9) if prods else 0


def cards_of(catalog, a):
    """The catalog's cards this run checks."""
    for card in json.loads((SCRIPTS / catalog / "models.json").read_text()):
        if a.card in card["name"]:
            yield card


def submitted_env(card, catalog):
    """The card's env as it is submitted (the launcher applies its workload file itself, MAD_CONFIG,
    as it does without madengine), with the image's builds for a card built from a named spec
    (MAD_IMAGE_COMPONENTS), as madengine's build passes them."""
    env = {k: str(v) for k, v in (card.get("env_vars") or {}).items()}
    sys.path.insert(0, str(SCRIPTS / "common"))
    import catalog as _catalog
    comps = _catalog.components_of_card(card, env.get("MAD_GPU_ARCH") or gpu_arch(card),
                                        _catalog.load(str(COMPAT)))
    if comps:
        env.setdefault("MAD_IMAGE_COMPONENTS", comps)
    return env


def resolved_env(card, catalog):
    """The card's env with its workload file applied under it (MAD's config/workload.py): what the
    launcher works with, for checks made before a launcher runs."""
    env = submitted_env(card, catalog)
    if env.get("MAD_CONFIG"):
        sys.path.insert(0, str(SCRIPTS / "common" / "config"))
        import workload
        data = workload.load(workload.find(env["MAD_CONFIG"], str(SCRIPTS / catalog)))
        env = {**workload.env_of(data, env.get("BENCHMARK_SCRIPT")), **env}
    return env


def default_gpus_per_node():
    """GPUs per node of the catalog's default platform: what cluster.sh gives a run naming none."""
    c = _compat()
    return c["platforms"][c["default_platform"]]["gpus_per_node"]


def gpu_arch(card):
    """The GPU a card runs on: gfx950 when it skips gfx942, else gfx942."""
    skip = card.get("skip_gpu_arch") or []
    if isinstance(skip, str):
        skip = [s.strip() for s in skip.split(",")]
    return "gfx950" if "gfx942" in skip else "gfx942"


def node_count(card):
    for v in ((card.get("slurm") or {}).get("nodes"), (card.get("distributed") or {}).get("nnodes")):
        if v:
            return int(v)
    return 1


def parse_docker_run(argv):
    """`docker run ...` arguments -> flags, mounts, env (ordered), image, command."""
    out = {"flags": [], "mounts": [], "env": [], "env_file": [], "image": None, "command": []}
    args = argv[1:] if argv and argv[0] == "run" else list(argv)
    i = 0
    while i < len(args):
        a = args[i]
        if not a.startswith("-"):
            out["image"] = a
            out["command"] = args[i + 1:]
            break
        name, inline = (a.split("=", 1) + [None])[:2] if a.startswith("--") and "=" in a else (a, None)
        if name in VALUE_OPTS and inline is None:
            value = args[i + 1] if i + 1 < len(args) else ""
            i += 2
        else:
            value = inline
            i += 1
        if name in ("-v", "--volume"):
            out["mounts"].append(value)
        elif name in ("-e", "--env"):
            out["env"].append(value)
        elif name == "--env-file":
            out["env_file"].append(value)
        else:
            out["flags"].append(name if value is None else f"{name} {value}")
    return out


def container_env(docker, client_env=None):
    """The environment the container gets from -e: the last value of a name wins. A bare
    `-e NAME` takes NAME's value from the docker client's environment, and is absent when
    the client has none, as docker does."""
    env = {}
    for e in docker["env"]:
        k, eq, v = e.partition("=")
        if eq:
            env[k] = v
        elif k in (client_env or {}):
            env[k] = client_env[k]
    return env


def engine_dry_run(catalog, docker, work, client_env=None):
    """The engine command(s) this rank's role script would run, from DRY_RUN."""
    scripts = work / "scripts"
    env = {}
    for k, v in container_env(docker, client_env).items():
        for cdir, sub in CONTAINER_SCRIPT_DIRS.items():
            v = v.replace(cdir, str(scripts / sub) if sub else str(scripts))
        v = v.replace("/run_logs/", str(work / "logs") + "/")   # -v $LOG_PATH:/run_logs
        env[k] = v
    env.update({"PATH": f"{STUBS}:{os.environ['PATH']}", "HOME": str(work / "home"), "DRY_RUN": "1"})
    if catalog == "vllm_dissag":
        script = scripts / "vllm_dissag" / "vllm_disagg.sh"
        env["NIXL_COOKBOOK_PATH"] = str(scripts / "vllm_dissag")
    elif catalog == "sglang_disagg":
        script = scripts / "sglang_disagg" / "sglang_disagg_mori_io_ep.sh"
        env["MOONCAKE_COOKBOOK_PATH"] = str(scripts / "sglang_disagg")
        # sglang_compat.sh adapts the recipe to the installed sglang's --help; empty
        # help means "every option exists", i.e. the recipe as written, whether or not
        # sglang is installed where this runs.
        env["_SGL_HELP"] = ""
    else:
        script = scripts / "vllm_multinode" / "serve_colocated.sh"
        env["SHARED_DIR"] = str(scripts / "vllm_dissag")
    p = subprocess.run(["bash", str(script)], cwd=work / "job", env=env,
                       capture_output=True, text=True, timeout=120)
    blocks, cur = [], None
    for line in p.stdout.splitlines():
        if line.startswith("===DRYRUN"):
            cur = {"header": line, "argv": []}
        elif line.startswith("===END"):
            if cur is not None:
                blocks.append(cur)
            cur = None
        elif cur is not None:
            cur["argv"].append(line)
    return {"exit_code": p.returncode, "blocks": blocks}


SHELL_LINE = re.compile(r"(\.(?:sh|slurm): line )\d+:")
# A mktemp name: its random suffix (template .XXXXXX) differs on every run.
MKTEMP = re.compile(r"(/tmp/[^\"/]+)\.[A-Za-z0-9]{6}(?=\")")
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[^\"]*?(?=\"|$)")


def normalize(obj, work):
    """Replace what differs between runs (paths, wall-clock times) with placeholders."""
    text = json.dumps(obj)
    for path, mark in ((str(work), "<WORK>"), (str(STUBS), "<STUBS>")):
        text = text.replace(path, mark)
    text = MKTEMP.sub(r"\1.<TMP>", text)
    # A shell error names the script line it failed on; any edit above it moves it.
    return json.loads(SHELL_LINE.sub(r"\1<N>:", TIMESTAMP.sub("<TIME>", text)))


def run_as_tasks(script, env, nodes, mode, cwd=None):
    """The launcher once per node, all at once, sharing the work directory as the nodes
    share storage: spur-array as the single-node tasks of array job JOB_ID; k8s-indexed
    as the pods of an Indexed Job, with the environment k8s/mad_k8s_node.sh gives them.
    Returns the leader's (rank 0's) exit status and output."""
    procs = []
    for k in range(nodes):
        task_env = dict(env)
        task_env.update({"SLURMD_NODENAME": f"node{k + 1}", "MAD_RENDEZVOUS_TIMEOUT": "60"})
        if mode == "spur-array":
            task_env.update({
                "SLURM_ARRAY_JOB_ID": JOB_ID, "SLURM_ARRAY_TASK_ID": str(k),
                "SLURM_JOB_ID": str(int(JOB_ID) + 1 + k), "SLURM_NNODES": "1",
                "SLURM_JOB_NODELIST": f"node{k + 1}", "MAD_NNODES": str(nodes),
            })
        else:
            task_env.update({
                "MAD_DISPATCH": "k8s-indexed", "JOB_COMPLETION_INDEX": str(k),
                "MAD_K8S_JOB_ID": JOB_ID, "MAD_K8S_NNODES": str(nodes), "MAD_K8S_SELF": f"node{k + 1}",
                "SLURM_JOB_ID": JOB_ID, "SLURM_NNODES": str(nodes), "SLURM_PROCID": str(k),
                "SLURM_JOB_NODELIST": ",".join(f"node{i + 1}" for i in range(nodes)),
            })
        procs.append(subprocess.Popen(["bash", str(script)], cwd=cwd or script.parent, env=task_env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    outs = [p.communicate(timeout=300)[0] for p in procs]
    return procs[0].returncode, outs[0]


def snapshot_card(catalog, card, dispatch="slurm-srun", spooled=False, keep_output=False, submit_env=None):
    work = Path(tempfile.mkdtemp(prefix="madsnap."))
    try:
        shutil.copytree(SCRIPTS, work / "scripts", symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", "cases"))
        # The capability catalog the launchers resolve recipe roles against.
        shutil.copytree(COMPAT, work / "compat")
        for d in ("home", "logs", "stub", "nvme", "shared", "model", "job", "jit"):
            (work / d).mkdir()
        nodes = node_count(card)
        env = {
            "PATH": f"{STUBS}:{os.environ['PATH']}",
            "HOME": str(work / "home"),
            "USER": "t",
            "STUB_DIR": str(work / "stub"),
            "LOG_PATH": str(work / "logs"),
            "SLURM_JOB_ID": JOB_ID,
            "SLURM_NNODES": str(nodes),
            "SLURM_JOB_NODELIST": ",".join(f"node{i + 1}" for i in range(nodes)),
            "MODEL_PATH": str(work / "model"),
            "MAD_GPU_ARCH": gpu_arch(card),
            "CLUSTER_SYSFS_IB": str(work / "none"),     # no RDMA devices: fabric "unknown"
            "NVME_ROOT": str(work / "nvme"),
            "SHARED_MOUNT": str(work / "shared"),
            "JIT_CACHE_HOST": str(work / "jit"),
        }
        env.update(submitted_env(card, catalog) if submit_env is None else submit_env)
        env.setdefault("MAD_GPU_MEMORY_BYTES", str(gpu_memory_bytes(env["MAD_GPU_ARCH"])))   # as the probe would
        env["DOCKER_IMAGE_NAME"] = "img:snapshot"
        script = work / "scripts" / catalog / card["scripts"]
        # From the launcher's own directory, as documented (`cd scripts/<dir> &&
        # sbatch ...`) and as madengine runs it: the SGLang launcher finds its role
        # scripts in the working directory.
        cwd = script.parent
        if spooled:
            # As sbatch runs it: a copy in the scheduler's spool directory, started in
            # the submit directory (the launcher's own, as documented).
            spool = work / "spool"
            spool.mkdir()
            shutil.copy(script, spool / "slurm_script")
            env["SLURM_SUBMIT_DIR"] = str(script.parent)
            script = spool / "slurm_script"
        if dispatch == "slurm-srun":
            p = subprocess.run(["bash", str(script)], cwd=cwd, env=env,
                               capture_output=True, text=True, timeout=300)
            rc, output = p.returncode, p.stdout + p.stderr
        else:
            rc, output = run_as_tasks(script, env, nodes, dispatch, cwd)
        calls_file = work / "stub" / "calls.log"
        calls = calls_file.read_text().splitlines() if calls_file.exists() else []
        ranks = []
        for f in sorted((work / "stub").glob("run_rank*.json"), key=lambda f: f.name):
            docker = parse_docker_run(json.loads(f.read_text()))
            rank = f.stem.replace("run_rank", "")
            cenv = work / "stub" / f"client_env_rank{rank}.json"
            client_env = json.loads(cenv.read_text()) if cenv.exists() else {}
            ranks.append({"rank": rank, "docker": docker,
                          "engine": engine_dry_run(catalog, docker, work, client_env)})
        snap = {
            "card": card["name"],
            "catalog": catalog,
            "inputs": {"nodes": nodes, "gpu_arch": env["MAD_GPU_ARCH"]},
            "exit_code": rc,
            "calls": calls,
            "ranks": ranks,
        }
        if rc != 0:
            snap["stderr_tail"] = output.splitlines()[-5:]
        out = normalize(snap, work)
        if keep_output:              # for tests that read what the launcher said; never in a case
            out["output"] = output
        return out
    finally:
        shutil.rmtree(work)


def show(a):
    """Run each matching card as given and print what every node does."""
    extra = dict(e.split("=", 1) for e in a.env)
    shown = 0
    for catalog in CATALOGS:
        if a.catalog and catalog != a.catalog:
            continue
        for card in cards_of(catalog, a):
            card = json.loads(json.dumps(card))
            card["env_vars"] = dict(card.get("env_vars") or {}, **extra)
            if a.nodes:
                card.setdefault("slurm", {})["nodes"] = a.nodes
                card.setdefault("distributed", {})["nnodes"] = a.nodes
            got = snapshot_card(catalog, card, keep_output=True)
            shown += 1
            print(f"== {catalog}/{card['name']} on {got['inputs']['gpu_arch']}, {got['inputs']['nodes']} node(s): "
                  f"exit {got['exit_code']}")
            for line in got["output"].splitlines():
                if line.startswith(("[recipe]", "[config]", "ERROR", "Error")):
                    print("  " + line)
            for r in got["ranks"]:
                env = dict(e.split("=", 1) for e in r["docker"]["env"] if "=" in e)
                print(f"  node {r['rank']} ({env.get('MAD_TOPO_ROLE', '?')}):")
                for name in filter(None, a.print_env.split(",")):
                    print(f"    {name}={env.get(name, '<not set>')}")
                for b in (r.get("engine") or {}).get("blocks", []):
                    print("    " + " ".join(b["argv"]))
    if not shown:
        print("snapshot: no card matches", file=sys.stderr)
        return 1
    return 0


# What selects a path through the launcher code: container settings of rank 0, and engine
# features the run turns on. Recipe values (sizes, model flags) are data, not paths.
PATH_ENV = ("CONNECTOR", "WIDE_EP", "EP_BACKEND", "RUN_MORI", "RUN_DEEPEP", "DP_MODE", "ENABLE_EP",
            "ALL2ALL_BACKEND", "MAD_TOPO_ROUTER", "BENCHMARK_SCRIPT_FILE", "KV_OFFLOAD")
PATH_FLAGS = ("--enable-expert-parallel", "--headless", "--speculative-config", "--enable-lora",
              "--tensor-parallel-size", "--data-parallel-size", "--pipeline-parallel-size",
              "--enable-dp-attention")


def path_of(catalog, got):
    """The card's path through the launcher code, from its snapshot (a list, as stored)."""
    if not got["ranks"]:
        return [catalog, "refused", got["exit_code"]]
    env = dict(e.split("=", 1) for e in got["ranks"][0]["docker"]["env"] if "=" in e)
    argv = [w for r in got["ranks"] for b in (r.get("engine") or {}).get("blocks", []) for w in b["argv"]]
    backend = sorted({argv[i + 1] for i, w in enumerate(argv[:-1]) if w == "--disaggregation-transfer-backend"})
    connectors = sorted(set(re.findall(r'"kv_connector": ?"(\w+)"', " ".join(argv))))
    multi = "--headless" in argv or any(int(env.get(k, "1") or 1) > 1 for k in ("xP", "yD"))
    return ([catalog] + [f"{k}={env[k]}" for k in PATH_ENV if k in env]
            + ["multi-node pool" if multi else "single-node pools",
               "TP-within-EP" if int(env.get("EP_TP_SIZE", "1") or 1) > 1 else "no TP-within-EP",
               "transfer=" + ",".join(backend), "kv_connector=" + ",".join(connectors),
               "flags=" + ",".join(f for f in PATH_FLAGS if f in argv), got["exit_code"]])


def _cards(a):
    for catalog in CATALOGS:
        if a.catalog and catalog != a.catalog:
            continue
        for card in cards_of(catalog, a):
            yield catalog, card


def _case_files():
    return sorted(CASES.glob("*/*.json"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true", help="record the cases: every path's snapshot, new paths added, stale ones removed")
    ap.add_argument("--card", default="", help="only cards whose name contains this")
    ap.add_argument("--catalog", choices=CATALOGS, help="only this catalog")
    ap.add_argument("--spec-file", action="store_true",
                    help="run every card from a topology spec file (MAD_TOPOLOGY) instead of its "
                         "xP / yD / ... settings and require the same run as from its settings")
    ap.add_argument("--spooled", action="store_true",
                    help="run each launcher as sbatch does: a copy in a spool directory, "
                         "started in the submit directory (with --equivalence: under that "
                         "dispatch); require the same run as the card's own slurm-srun run")
    ap.add_argument("--from-emitted", action="store_true",
                    help="run every card from its generated configuration (config/emit.py: the card's "
                         "MAD_CONFIG pointing at it, nothing else submitted) and require the same run "
                         "as the card's own run, on the card's GPU and on every other GPU its recipe has a "
                         "variant for")
    ap.add_argument("--equivalence", choices=("spur-array", "k8s-indexed"),
                    help="run every card under this dispatch instead and require the same "
                         "per-node docker run and engine command, and exit status, as the "
                         "card's own slurm-srun run; no srun may be called")
    ap.add_argument("--show", action="store_true",
                    help="run the matching card(s) and print what each node does (its role and engine "
                         "command; --env/--nodes change the run), instead of comparing")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                    help="with --show: set NAME for the run, as at submit time (e.g. MAD_CONFIG=workloads/guide/"
                         "tied.yaml, MAD_GPU_ARCH=gfx950, MAD_CLUSTER=<site>); repeatable")
    ap.add_argument("--nodes", type=int, help="with --show: the nodes allocated (default: the card's)")
    ap.add_argument("--print-env", default="", metavar="NAME,...",
                    help="with --show: also print these names from each node's container environment")
    a = ap.parse_args()
    if a.show:
        return show(a)
    if a.spec_file:
        return spec_file_equivalence(a)
    if a.from_emitted:
        return emitted_equivalence(a)
    if a.equivalence or a.spooled:
        return equivalence(a)
    runs, paths = {}, collections.defaultdict(list)
    for catalog, card in _cards(a):
        got = snapshot_card(catalog, card)
        runs[(catalog, card["name"])] = got
        paths[json.dumps(path_of(catalog, got))].append((catalog, card["name"]))
    cases = {}
    for f in _case_files():
        d = json.loads(f.read_text())
        cases[json.dumps(d["path"])] = (f, d)
    if a.update:
        written = set()
        for key, members in paths.items():
            old = cases.get(key)
            pick = (old[1]["catalog"], old[1]["card"]) if old and (old[1]["catalog"], old[1]["card"]) in members \
                else members[0]
            out = CASES / pick[0] / f"{pick[1]}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps({"path": json.loads(key), "catalog": pick[0], "card": pick[1],
                               "snapshot": runs[pick]}, indent=1) + "\n"
            if not out.exists() or out.read_text() != text:
                out.write_text(text)
                print(f"recorded {pick[0]}/{pick[1]}")
            written.add(out)
            if old and old[0] != out:
                old[0].unlink()
        if not (a.card or a.catalog):            # a path no card takes any more is not a case
            for f in _case_files():
                if f not in written:
                    f.unlink()
                    print(f"removed  {f.relative_to(CASES)} (no card takes its path)")
        print(f"snapshot: {sum(map(len, paths.values()))} card(s) on {len(paths)} path(s); cases recorded")
        return 0
    changed, uncovered = [], []
    for key, members in sorted(paths.items()):
        case = cases.get(key)
        if case is None:
            uncovered.append((members[0], json.loads(key)))
            continue
        f, d = case
        mine = (d["catalog"], d["card"])
        if mine not in runs:                         # filtered out of this run
            continue
        got = json.dumps(runs[mine], indent=1)
        want = json.dumps(d["snapshot"], indent=1)
        if got != want:
            changed.append(f"{mine[0]}/{mine[1]}")
            diff = difflib.unified_diff(want.splitlines(), got.splitlines(),
                                        f"cases/{mine[0]}/{mine[1]}.json", "now", lineterm="", n=2)
            print("\n".join(list(diff)[:60]))
    for (catalog, name), path in uncovered:
        print(f"NEW PATH  {catalog}/{name}: {path}\n          no case records it; review its snapshot "
              f"(--show) and record it with --update")
    n = sum(map(len, paths.values()))
    print(f"snapshot: {n} card(s) on {len(paths)} path(s); {len(paths) - len(uncovered)} case(s) "
          f"checked, {len(changed)} differ, {len(uncovered)} new path(s)")
    for c in changed:
        print(f"  DIFF  {c}")
    return 1 if changed or uncovered else 0


# The settings a spec file replaces. With MAD_TOPOLOGY they are all set explicitly,
# so a launcher that forwards one only when set (`${X:+-e X=...}`) passes one more,
# with the value the default gave; the comparison leaves these entries out.
TOPOLOGY_SETTINGS = {"xP", "yD", "ROUTER_NODE", "WIDE_EP", "EP_TP_SIZE", "DP_MODE", "GENERIC_TP_SIZE", "GPUS_PER_NODE",
                     "TP_SIZE", "PP_SIZE", "ENABLE_EP", "ALL2ALL_BACKEND", "MAD_TOPOLOGY", "MAD_TOPOLOGY_NNODES"}


def _without_settings(ranks):
    out = json.loads(json.dumps(ranks))
    for r in out:
        r["docker"]["env"] = [e for e in r["docker"]["env"] if e.split("=", 1)[0] not in TOPOLOGY_SETTINGS]
    return out


def spec_file_equivalence(a):
    """Each card from a spec file (MAD_TOPOLOGY) instead of its topology settings."""
    sys.path.insert(0, str(SCRIPTS / "common" / "topology"))
    import topology
    bad, checked = [], 0
    for catalog in CATALOGS:
        if a.catalog and catalog != a.catalog:
            continue
        for card in cards_of(catalog, a):
            checked += 1
            gold = snapshot_card(catalog, card)        # the card as it runs from its settings
            env = launch_env(card, catalog)
            env["SLURM_NNODES"] = str(node_count(card))
            env.setdefault("GPUS_PER_NODE", str(default_gpus_per_node()))   # as cluster.sh: the platform's
            spec = topology.from_env(env, catalog)
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(spec, f)
            spec_card = dict(card)
            spec_card["env_vars"] = {k: v for k, v in (card.get("env_vars") or {}).items()
                                     if k not in TOPOLOGY_SETTINGS}
            spec_card["env_vars"]["MAD_TOPOLOGY"] = f.name
            try:
                got = snapshot_card(catalog, spec_card)
            finally:
                os.unlink(f.name)
            problems = []
            if got["exit_code"] != gold["exit_code"]:
                problems.append(f"exit {got['exit_code']} (slurm-srun {gold['exit_code']}): {got.get('stderr_tail')}")
            if _without_settings(got["ranks"]) != _without_settings(gold["ranks"]):
                diff = difflib.unified_diff(json.dumps(_without_settings(gold["ranks"]), indent=1).splitlines(),
                                            json.dumps(_without_settings(got["ranks"]), indent=1).splitlines(),
                                            lineterm="", n=1)
                problems.append("ranks differ:\n" + "\n".join(list(diff)[:30]))
            if problems:
                bad.append(f"{catalog}/{card['name']}")
                print(f"DIFF  {catalog}/{card['name']}")
                for pr in problems:
                    print("   " + pr.replace("\n", "\n   "))
    print(f"spec-file equivalence: {checked} card(s) checked, {len(bad)} differ")
    return 1 if bad else 0


# Bookkeeping a run from a generated file adds: which file, and the job's recipe copy it always
# writes (the containers read that instead of the repository's models.yaml -- the same recipe).
EMITTED_BOOKKEEPING = {"MAD_CONFIG", "MAD_RECIPE_FILE"}


def _without_bookkeeping(ranks):
    out = json.loads(json.dumps(ranks))
    for r in out:
        r["docker"]["env"] = [e for e in r["docker"]["env"] if e.split("=", 1)[0] not in EMITTED_BOOKKEEPING]
    return out


def _recipe_archs(catalog, card):
    """The GPUs a card's recipe has a variant for (the card's own first)."""
    import yaml
    archs = [gpu_arch(card)]
    name = resolved_env(card, catalog).get("MODEL_NAME")
    path = SCRIPTS / ("vllm_dissag" if catalog == "vllm_multinode" else catalog) / "models.yaml"
    if name and path.is_file():
        recipes = yaml.safe_load(path.read_text()) or {}
        while isinstance(recipes.get(name), dict) and "alias" in recipes[name]:
            name = recipes[name]["alias"]
        for arch in ((recipes.get(name) or {}).get("variants") or {}):
            if arch.startswith("gfx") and arch not in archs:
                archs.append(arch)
    return archs


def emitted_equivalence(a):
    """Each card run from its generated configuration: the same as running the card itself."""
    sys.path.insert(0, str(SCRIPTS / "common" / "config"))
    import emit
    bad, checked = [], 0
    for catalog in CATALOGS:
        if a.catalog and catalog != a.catalog:
            continue
        for card in cards_of(catalog, a):
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                f.write(emit.dump(emit.compose(catalog, card, os.path.dirname(f.name))))
            try:
                skip = {g.strip() for g in str(card.get("skip_gpu_arch") or "").split(",")}
                for arch in [g for g in _recipe_archs(catalog, card) if g not in skip]:   # where it runs
                    checked += 1
                    on = json.loads(json.dumps(card))
                    on["env_vars"] = dict(on.get("env_vars") or {}, MAD_GPU_ARCH=arch)
                    want = snapshot_card(catalog, on)
                    sub = {"MAD_CONFIG": f.name, "MAD_GPU_ARCH": arch}
                    if (card.get("env_vars") or {}).get("DOCKER_IMAGE_NAME"):   # the card keeps its image
                        sub["DOCKER_IMAGE_NAME"] = card["env_vars"]["DOCKER_IMAGE_NAME"]
                    comps = submitted_env(on, catalog).get("MAD_IMAGE_COMPONENTS")
                    if comps:                      # as the card itself was submitted
                        sub["MAD_IMAGE_COMPONENTS"] = comps
                    got = snapshot_card(catalog, on, submit_env=sub)
                    problems = []
                    if got["exit_code"] != want["exit_code"]:
                        problems.append(f"exit {got['exit_code']} (card {want['exit_code']}): {got.get('stderr_tail')}")
                    g, w = _without_bookkeeping(got["ranks"]), _without_bookkeeping(want["ranks"])
                    if g != w:
                        diff = difflib.unified_diff(json.dumps(w, indent=1).splitlines(),
                                                    json.dumps(g, indent=1).splitlines(), lineterm="", n=1)
                        problems.append("ranks differ:\n" + "\n".join(list(diff)[:30]))
                    if problems:
                        bad.append(f"{catalog}/{card['name']} on {arch}")
                        print(f"DIFF  {catalog}/{card['name']} on {arch}")
                        for pr in problems:
                            print("   " + pr.replace("\n", "\n   "))
            finally:
                os.unlink(f.name)
    print(f"emitted equivalence: {checked} card x GPU run(s) checked, {len(bad)} differ")
    return 1 if bad else 0


def equivalence(a):
    """Same per-node work under another dispatch: compare with the card's own slurm-srun run."""
    bad, checked = [], 0
    for catalog in CATALOGS:
        if a.catalog and catalog != a.catalog:
            continue
        for card in cards_of(catalog, a):
            checked += 1
            gold = snapshot_card(catalog, card)        # the reference: slurm-srun, as submitted
            got = snapshot_card(catalog, card, a.equivalence or "slurm-srun", a.spooled)
            problems = []
            if got["exit_code"] != gold["exit_code"]:
                problems.append(f"exit {got['exit_code']} (slurm-srun {gold['exit_code']}): {got.get('stderr_tail')}")
            if got["ranks"] != gold["ranks"]:
                g = {r["rank"]: r for r in gold["ranks"]}
                for r in got["ranks"]:
                    if r != g.get(r["rank"]):
                        diff = difflib.unified_diff(json.dumps(g.get(r["rank"]), indent=1).splitlines(),
                                                    json.dumps(r, indent=1).splitlines(), lineterm="", n=1)
                        problems.append(f"rank {r['rank']} differs:\n" + "\n".join(list(diff)[:30]))
                if len(got["ranks"]) != len(gold["ranks"]):
                    problems.append(f"{len(got['ranks'])} ranks (slurm-srun {len(gold['ranks'])})")
            srun_calls = [c for c in got["calls"] if c.startswith("srun")]
            if srun_calls and a.equivalence:
                problems.append(f"srun called: {srun_calls}")
            if problems:
                bad.append(f"{catalog}/{card['name']}")
                print(f"DIFF  {catalog}/{card['name']}")
                for pr in problems:
                    print("   " + pr.replace("\n", "\n   "))
    mode = " + ".join(m for m in (a.equivalence, "spooled" if a.spooled else "") if m)
    print(f"equivalence ({mode}): {checked} card(s) checked, {len(bad)} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
