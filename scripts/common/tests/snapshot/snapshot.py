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
that differ between runs are replaced with placeholders. The results are
compared with golden/<catalog>/<card>.json; any difference fails.

A refactor of the launchers must leave every snapshot unchanged, or change it
on purpose: rerun with --update and review the golden diff in the commit.

Each card runs in its own temporary directory, on a copy of scripts/, with HOME,
LOG_PATH, NVME_ROOT, SHARED_MOUNT and the JIT cache pointed inside it, so a run
writes nothing into the repository or the machine. (The launchers still create
/tmp/vllm_cache, which they hardcode.)

Usage:
  python3 scripts/common/tests/snapshot/snapshot.py              # compare
  python3 scripts/common/tests/snapshot/snapshot.py --update     # rewrite goldens
  python3 scripts/common/tests/snapshot/snapshot.py --card qwen3 # cards matching
"""
import argparse
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
STUBS = HERE / "stubs"
GOLDEN = HERE / "golden"
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


def container_env(docker):
    """The environment the container gets from -e: the last value of a name wins."""
    env = {}
    for e in docker["env"]:
        k, _, v = e.partition("=")
        env[k] = v
    return env


def engine_dry_run(catalog, docker, work):
    """The engine command(s) this rank's role script would run, from DRY_RUN."""
    scripts = work / "scripts"
    env = {}
    for k, v in container_env(docker).items():
        for cdir, sub in CONTAINER_SCRIPT_DIRS.items():
            v = v.replace(cdir, str(scripts / sub) if sub else str(scripts))
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
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[^\"]*?(?=\"|$)")


def normalize(obj, work):
    """Replace what differs between runs (paths, wall-clock times) with placeholders."""
    text = json.dumps(obj)
    for path, mark in ((str(work), "<WORK>"), (str(STUBS), "<STUBS>")):
        text = text.replace(path, mark)
    # A shell error names the script line it failed on; any edit above it moves it.
    return json.loads(SHELL_LINE.sub(r"\1<N>:", TIMESTAMP.sub("<TIME>", text)))


def run_as_tasks(script, env, nodes, mode):
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
        procs.append(subprocess.Popen(["bash", str(script)], cwd=script.parent, env=task_env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    outs = [p.communicate(timeout=300)[0] for p in procs]
    return procs[0].returncode, outs[0]


def snapshot_card(catalog, card, dispatch="slurm-srun"):
    work = Path(tempfile.mkdtemp(prefix="madsnap."))
    try:
        shutil.copytree(SCRIPTS, work / "scripts", symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", "golden"))
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
        env.update({k: str(v) for k, v in (card.get("env_vars") or {}).items()})
        env["DOCKER_IMAGE_NAME"] = "img:snapshot"
        script = work / "scripts" / catalog / card["scripts"]
        # From the launcher's own directory, as documented (`cd scripts/<dir> &&
        # sbatch ...`) and as madengine runs it: the SGLang launcher finds its role
        # scripts in the working directory.
        if dispatch == "slurm-srun":
            p = subprocess.run(["bash", str(script)], cwd=script.parent, env=env,
                               capture_output=True, text=True, timeout=300)
            rc, output = p.returncode, p.stdout + p.stderr
        else:
            rc, output = run_as_tasks(script, env, nodes, dispatch)
        calls_file = work / "stub" / "calls.log"
        calls = calls_file.read_text().splitlines() if calls_file.exists() else []
        ranks = []
        for f in sorted((work / "stub").glob("run_rank*.json"), key=lambda f: f.name):
            docker = parse_docker_run(json.loads(f.read_text()))
            ranks.append({"rank": f.stem.replace("run_rank", ""), "docker": docker,
                          "engine": engine_dry_run(catalog, docker, work)})
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
        return normalize(snap, work)
    finally:
        shutil.rmtree(work)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true", help="rewrite the golden files")
    ap.add_argument("--card", default="", help="only cards whose name contains this")
    ap.add_argument("--catalog", choices=CATALOGS, help="only this catalog")
    ap.add_argument("--spec-file", action="store_true",
                    help="run every card from a topology spec file (MAD_TOPOLOGY) instead of its "
                         "xP / yD / ... settings and require the same run as the goldens")
    ap.add_argument("--equivalence", choices=("spur-array", "k8s-indexed"),
                    help="run every card under this dispatch instead and require the same "
                         "per-node docker run and engine command, and exit status, as the "
                         "goldens (recorded under slurm-srun); no srun may be called")
    a = ap.parse_args()
    if a.spec_file:
        return spec_file_equivalence(a)
    if a.equivalence:
        return equivalence(a)
    changed, checked = [], 0
    for catalog in CATALOGS:
        if a.catalog and catalog != a.catalog:
            continue
        for card in json.loads((SCRIPTS / catalog / "models.json").read_text()):
            if a.card not in card["name"]:
                continue
            checked += 1
            got = json.dumps(snapshot_card(catalog, card), indent=1) + "\n"
            gold = GOLDEN / catalog / f"{card['name']}.json"
            want = gold.read_text() if gold.exists() else ""
            if got == want:
                continue
            if a.update:
                gold.parent.mkdir(parents=True, exist_ok=True)
                gold.write_text(got)
                print(f"updated  {catalog}/{card['name']}")
            else:
                changed.append(f"{catalog}/{card['name']}")
                diff = difflib.unified_diff(want.splitlines(), got.splitlines(),
                                            f"golden/{catalog}/{card['name']}.json", "now", lineterm="", n=2)
                print("\n".join(list(diff)[:60]))
    if a.update:
        print(f"snapshot: {checked} card(s) recorded")
        return 0
    print(f"snapshot: {checked} card(s) checked, {len(changed)} differ")
    for c in changed:
        print(f"  DIFF  {c}")
    return 1 if changed else 0


# The settings a spec file replaces. With MAD_TOPOLOGY they are all set explicitly,
# so a launcher that forwards one only when set (`${X:+-e X=...}`) passes one more,
# with the value the default gave; the comparison leaves these entries out.
TOPOLOGY_SETTINGS = {"xP", "yD", "WIDE_EP", "EP_TP_SIZE", "DP_MODE", "GENERIC_TP_SIZE", "GPUS_PER_NODE",
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
        for card in json.loads((SCRIPTS / catalog / "models.json").read_text()):
            if a.card not in card["name"]:
                continue
            checked += 1
            gold = json.loads((GOLDEN / catalog / f"{card['name']}.json").read_text())
            env = {k: str(v) for k, v in (card.get("env_vars") or {}).items()}
            env["SLURM_NNODES"] = str(node_count(card))
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
                problems.append(f"exit {got['exit_code']} (golden {gold['exit_code']}): {got.get('stderr_tail')}")
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


def equivalence(a):
    """Same per-node work under another dispatch: compare with the goldens."""
    bad, checked = [], 0
    for catalog in CATALOGS:
        if a.catalog and catalog != a.catalog:
            continue
        for card in json.loads((SCRIPTS / catalog / "models.json").read_text()):
            if a.card not in card["name"]:
                continue
            checked += 1
            gold = json.loads((GOLDEN / catalog / f"{card['name']}.json").read_text())
            got = snapshot_card(catalog, card, a.equivalence)
            problems = []
            if got["exit_code"] != gold["exit_code"]:
                problems.append(f"exit {got['exit_code']} (golden {gold['exit_code']}): {got.get('stderr_tail')}")
            if got["ranks"] != gold["ranks"]:
                g = {r["rank"]: r for r in gold["ranks"]}
                for r in got["ranks"]:
                    if r != g.get(r["rank"]):
                        diff = difflib.unified_diff(json.dumps(g.get(r["rank"]), indent=1).splitlines(),
                                                    json.dumps(r, indent=1).splitlines(), lineterm="", n=1)
                        problems.append(f"rank {r['rank']} differs:\n" + "\n".join(list(diff)[:30]))
                if len(got["ranks"]) != len(gold["ranks"]):
                    problems.append(f"{len(got['ranks'])} ranks (golden {len(gold['ranks'])})")
            srun_calls = [c for c in got["calls"] if c.startswith("srun")]
            if srun_calls:
                problems.append(f"srun called: {srun_calls}")
            if problems:
                bad.append(f"{catalog}/{card['name']}")
                print(f"DIFF  {catalog}/{card['name']}")
                for pr in problems:
                    print("   " + pr.replace("\n", "\n   "))
    print(f"equivalence ({a.equivalence}): {checked} card(s) checked, {len(bad)} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
