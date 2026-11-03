#!/usr/bin/env python3
"""The topology of a multi-node run, as data.

A topology spec says which roles a run has, how many nodes each takes, how each is
parallelised and where the router runs:

    {"engine": "vllm", "layout": "disagg",
     "roles": [{"name": "prefill", "nodes": 1, "parallel": {"mode": "wide_ep", "ep_tp": 1}},
               {"name": "decode",  "nodes": 1, "parallel": {"mode": "wide_ep", "ep_tp": 1}}],
     "router": {"host": "prefill", "index": 0},
     "gpus_per_node": 8}

    {"engine": "vllm", "layout": "colocated",
     "roles": [{"name": "server", "nodes": 2, "parallel": {"mode": "tp_pp", "tp": 8, "pp": 2}}],
     "gpus_per_node": 8}

With "roles": [..prefill.., ..decode.., {"name": "router", "nodes": 1}] and
"router": {"host": "router", "index": 0}, the router has a node of its own after the
servers (the launchers' ROUTER_NODE=1).

Roles take consecutive node ranks in the order listed. Resolving a spec for a rank
gives that node's role, its index within the role, the role's first rank (its
master), whether the node hosts the router, and for data-parallel roles the DP size
and this node's first DP rank.

Today the launchers take the same decisions from xP / yD / WIDE_EP / EP_TP_SIZE /
TP_SIZE / PP_SIZE, in shell; from_env() states those settings as a spec, and the
snapshot goldens check that the spec resolves to the role every rank's script
actually takes (tests/test_topology.py).

A spec can also be the input: MAD_TOPOLOGY names a spec file, and legacy_env() turns
it into the launcher's own settings (apply.sh, sourced by each launcher before its
defaults), refusing shapes the launcher cannot run and settings that disagree.
Examples: specs/. `snapshot.py --spec-file` checks every card runs the same from one.

CLI:
    topology.py spec       --launcher L           spec from the environment, as JSON
    topology.py node       --launcher L --rank R  this rank's resolution, as shell assignments
    topology.py legacy-env --launcher L           MAD_TOPOLOGY as the launcher's settings
"""
import argparse
import json
import os
import sys

LAUNCHERS = ("vllm_dissag", "sglang_disagg", "vllm_multinode")


class TopologyError(ValueError):
    pass


def _int(env, name, default):
    v = env.get(name, "")
    try:
        return int(v) if v != "" else default
    except ValueError:
        raise TopologyError(f"{name}={v!r} is not an integer")


def from_env(env, launcher):
    """The spec a launcher builds today from its environment."""
    gpus = _int(env, "GPUS_PER_NODE", 8)
    if launcher == "vllm_multinode":
        nnodes = _int(env, "SLURM_NNODES", 1)
        tp = _int(env, "TP_SIZE", gpus)
        pp = _int(env, "PP_SIZE", nnodes)
        parallel = {"mode": "tp_pp", "tp": tp, "pp": pp}
        if env.get("ENABLE_EP", "0") == "1":
            parallel["ep"] = True
            if env.get("ALL2ALL_BACKEND"):
                parallel["all2all"] = env["ALL2ALL_BACKEND"]
        return {"engine": "vllm", "layout": "colocated", "gpus_per_node": gpus,
                "roles": [{"name": "server", "nodes": nnodes, "parallel": parallel}]}
    if launcher not in ("vllm_dissag", "sglang_disagg"):
        raise TopologyError(f"unknown launcher {launcher!r} (one of {', '.join(LAUNCHERS)})")
    xp, yd = _int(env, "xP", 1), _int(env, "yD", 1)
    if launcher == "vllm_dissag":
        # vllm_disagg.sh: the legacy RUN_MORI / RUN_DEEPEP shim sets WIDE_EP=1 unless given.
        wide = env.get("WIDE_EP")
        if not wide:
            legacy = env.get("RUN_MORI") == "1" or env.get("RUN_DEEPEP") == "1"
            wide = "1" if legacy and not env.get("CONNECTOR") else "0"
        if wide == "1":
            parallel = {"mode": "wide_ep", "ep_tp": _int(env, "EP_TP_SIZE", 1)}
        else:
            parallel = {"mode": "tp"}            # degree from the recipe's tp: flags
        engine = "vllm"
    else:
        dp_mode = env.get("DP_MODE", "0") == "1"
        parallel = {"mode": "dp_ep"} if dp_mode else {"mode": "tp", "tp": _int(env, "GENERIC_TP_SIZE", 8)}
        engine = "sglang"
    roles = [{"name": "prefill", "nodes": xp, "parallel": dict(parallel)},
             {"name": "decode", "nodes": yd, "parallel": dict(parallel)}]
    router = {"host": "prefill", "index": 0}
    if env.get("ROUTER_NODE", "0") == "1":      # the router on a node of its own, after decode
        roles.append({"name": "router", "nodes": 1})
        router = {"host": "router", "index": 0}
    return {"engine": engine, "layout": "disagg", "gpus_per_node": gpus, "roles": roles, "router": router}


def validate(spec):
    if spec.get("layout") not in ("disagg", "colocated"):
        raise TopologyError(f"layout {spec.get('layout')!r} (disagg or colocated)")
    roles = spec.get("roles") or []
    if not roles:
        raise TopologyError("a spec needs at least one role")
    names = [r.get("name") for r in roles]
    if len(set(names)) != len(names):
        raise TopologyError(f"role names repeat: {names}")
    for r in roles:
        if not isinstance(r.get("nodes"), int) or r["nodes"] < 1:
            raise TopologyError(f"role {r.get('name')!r}: nodes must be a positive integer")
    gpus = spec.get("gpus_per_node", 8)
    for r in roles:
        p = r.get("parallel") or {}
        if p.get("mode") == "wide_ep" and gpus % p.get("ep_tp", 1):
            raise TopologyError(f"role {r['name']!r}: ep_tp={p.get('ep_tp')} does not divide gpus_per_node={gpus}")
        if p.get("mode") == "tp_pp" and p.get("tp", gpus) * p.get("pp", r["nodes"]) != r["nodes"] * gpus:
            raise TopologyError(f"role {r['name']!r}: tp*pp != nodes*gpus_per_node "
                                f"({p.get('tp')}x{p.get('pp')} vs {r['nodes']}x{gpus})")
    router = spec.get("router")
    if router:
        host = next((r for r in roles if r["name"] == router.get("host")), None)
        if host is None:
            raise TopologyError(f"router host {router.get('host')!r} is not a role")
        if not 0 <= router.get("index", 0) < host["nodes"]:
            raise TopologyError(f"router index {router.get('index')} outside role {host['name']!r}")
    return spec


def nnodes(spec):
    return sum(r["nodes"] for r in spec["roles"])


def resolve(spec, rank):
    """What node `rank` of a run with this spec is."""
    validate(spec)
    if not 0 <= rank < nnodes(spec):
        raise TopologyError(f"rank {rank} outside the run's {nnodes(spec)} nodes")
    first = 0
    for r in spec["roles"]:
        if rank < first + r["nodes"]:
            break
        first += r["nodes"]
    index = rank - first
    out = {"role": r["name"], "role_index": index, "role_nodes": r["nodes"], "role_first_rank": first,
           "nnodes": nnodes(spec)}
    router = spec.get("router")
    out["router"] = bool(router and router.get("host") == r["name"] and router.get("index", 0) == index)
    p = r.get("parallel") or {}
    if p.get("mode") == "wide_ep":
        per_node = spec.get("gpus_per_node", 8) // p.get("ep_tp", 1)
        out.update({"dp_per_node": per_node, "dp_size": r["nodes"] * per_node, "dp_start_rank": index * per_node})
    return out


# The settings each launcher reads, by the spec it can express today. A spec outside
# that shape is refused with what the launcher can run, rather than run differently.
def legacy_env(spec, launcher, env=None):
    """The launcher's own settings for a spec (MAD_TOPOLOGY): what it would otherwise
    read from xP / yD / WIDE_EP / ... A setting the caller also gave must agree."""
    validate(spec)
    roles = spec["roles"]
    gpus = spec.get("gpus_per_node", 8)
    engine = {"vllm_multinode": "vllm", "vllm_dissag": "vllm", "sglang_disagg": "sglang"}.get(launcher)
    if engine and spec.get("engine", engine) != engine:
        raise TopologyError(f"{launcher} runs {engine}, the spec is for {spec['engine']}")
    if launcher == "vllm_multinode":
        if spec["layout"] != "colocated" or len(roles) != 1:
            raise TopologyError("vllm_multinode runs one colocated role: "
                                '{"layout": "colocated", "roles": [{"name": ..., "nodes": N, "parallel": '
                                '{"mode": "tp_pp", "tp": T, "pp": P}}]}')
        p = roles[0].get("parallel") or {}
        if p.get("mode", "tp_pp") != "tp_pp":
            raise TopologyError(f"vllm_multinode: parallel mode {p.get('mode')!r} (tp_pp)")
        out = {"GPUS_PER_NODE": gpus, "TP_SIZE": p.get("tp", gpus), "PP_SIZE": p.get("pp", roles[0]["nodes"]),
               "MAD_TOPOLOGY_NNODES": roles[0]["nodes"]}
        if p.get("ep"):
            out["ENABLE_EP"] = 1
            if p.get("all2all"):
                out["ALL2ALL_BACKEND"] = p["all2all"]
    elif launcher in ("vllm_dissag", "sglang_disagg"):
        names = [r["name"] for r in roles]
        if spec["layout"] != "disagg" or names not in (["prefill", "decode"], ["prefill", "decode", "router"]):
            raise TopologyError(f"{launcher} runs roles prefill then decode, optionally then router (layout disagg)")
        own_node = names[-1] == "router"
        if own_node and roles[2]["nodes"] != 1:
            raise TopologyError(f"{launcher} runs one router node (got {roles[2]['nodes']})")
        router = spec.get("router") or {}
        where = ("router", 0) if own_node else ("prefill", 0)
        if (router.get("host"), router.get("index", 0)) != where:
            raise TopologyError(f"{launcher} runs the router on the first prefill node, or on the router "
                                f'role\'s node when the spec has one (router: {{"host": "{where[0]}", "index": 0}})')
        pp, pd = roles[0].get("parallel") or {}, roles[1].get("parallel") or {}
        if pp != pd:
            raise TopologyError(f"{launcher} runs prefill and decode with the same parallelism (got {pp} and {pd})")
        out = {"GPUS_PER_NODE": gpus, "xP": roles[0]["nodes"], "yD": roles[1]["nodes"],
               "ROUTER_NODE": int(own_node)}
        mode = pp.get("mode")
        if launcher == "vllm_dissag":
            if mode == "wide_ep":
                out.update({"WIDE_EP": 1, "EP_TP_SIZE": pp.get("ep_tp", 1)})
            elif mode == "tp":
                out["WIDE_EP"] = 0
            else:
                raise TopologyError(f"vllm_dissag: parallel mode {mode!r} (wide_ep or tp)")
        else:
            if mode == "dp_ep":
                out["DP_MODE"] = 1
            elif mode == "tp":
                out.update({"DP_MODE": 0, "GENERIC_TP_SIZE": pp.get("tp", 8)})
            else:
                raise TopologyError(f"sglang_disagg: parallel mode {mode!r} (dp_ep or tp)")
    else:
        raise TopologyError(f"unknown launcher {launcher!r}")
    for k, v in out.items():
        given = (env or {}).get(k, "")
        if given != "" and given != str(v):
            raise TopologyError(f"{k}={given} disagrees with MAD_TOPOLOGY ({k}={v}); set one or the other")
    return out


def load(path, scripts_dir=None):
    """A spec file. A relative path is taken from scripts/ (scripts_dir)."""
    p = path if os.path.isabs(path) or not scripts_dir else os.path.join(scripts_dir, path)
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise TopologyError(f"MAD_TOPOLOGY={path}: {e}")


def _shell(d, prefix="MAD_TOPO_", upper=True):
    """KEY=VALUE lines. Every value is a role name, a number or 0/1, so the lines are
    safe both as shell assignments and as `docker run -e` arguments. The launchers'
    own settings keep their names as they are (xP, yD)."""
    return "\n".join(f"{prefix}{k.upper() if upper else k}={int(v) if isinstance(v, bool) else v}"
                     for k, v in d.items())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spec", help="the spec, from MAD_TOPOLOGY (a JSON file) or the environment")
    s.add_argument("--launcher", choices=LAUNCHERS, required=True)
    n = sub.add_parser("node", help="one rank's resolution, as shell assignments (MAD_TOPO_*)")
    n.add_argument("--launcher", choices=LAUNCHERS, required=True)
    n.add_argument("--rank", type=int, required=True)
    e = sub.add_parser("legacy-env", help="the launcher's own settings for MAD_TOPOLOGY, as shell assignments")
    e.add_argument("--launcher", choices=LAUNCHERS, required=True)
    a = ap.parse_args(argv)
    scripts_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        path = os.environ.get("MAD_TOPOLOGY")
        spec = validate(load(path, scripts_dir) if path else from_env(os.environ, a.launcher))
        if a.cmd == "spec":
            print(json.dumps(spec, indent=1))
        elif a.cmd == "legacy-env":
            if not path:
                raise TopologyError("legacy-env needs MAD_TOPOLOGY (a spec file)")
            print(_shell(legacy_env(spec, a.launcher, os.environ), prefix="", upper=False))
        else:
            print(_shell(resolve(spec, a.rank)))
    except TopologyError as e:
        print(f"topology: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
