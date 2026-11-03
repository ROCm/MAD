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

Roles take consecutive node ranks in the order listed. Resolving a spec for a rank
gives that node's role, its index within the role, the role's first rank (its
master), whether the node hosts the router, and for data-parallel roles the DP size
and this node's first DP rank.

Today the launchers take the same decisions from xP / yD / WIDE_EP / EP_TP_SIZE /
TP_SIZE / PP_SIZE, in shell; from_env() states those settings as a spec, and the
snapshot goldens check that the spec resolves to the role every rank's script
actually takes (tests/test_topology.py).

CLI:
    topology.py spec   --launcher L           spec from the environment, as JSON
    topology.py node   --launcher L --rank R  this rank's resolution, as shell assignments
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
    return {"engine": engine, "layout": "disagg", "gpus_per_node": gpus,
            "roles": [{"name": "prefill", "nodes": xp, "parallel": dict(parallel)},
                      {"name": "decode", "nodes": yd, "parallel": dict(parallel)}],
            "router": {"host": "prefill", "index": 0}}


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


def _shell(d, prefix="MAD_TOPO_"):
    """KEY=VALUE lines. Every value is a role name, a number or 0/1, so the lines are
    safe both as shell assignments and as `docker run -e` arguments."""
    return "\n".join(f"{prefix}{k.upper()}={int(v) if isinstance(v, bool) else v}" for k, v in d.items())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spec", help="the spec, from MAD_TOPOLOGY (a JSON file) or the environment")
    s.add_argument("--launcher", choices=LAUNCHERS, required=True)
    n = sub.add_parser("node", help="one rank's resolution, as shell assignments (MAD_TOPO_*)")
    n.add_argument("--launcher", choices=LAUNCHERS, required=True)
    n.add_argument("--rank", type=int, required=True)
    a = ap.parse_args(argv)
    try:
        path = os.environ.get("MAD_TOPOLOGY")
        spec = validate(json.load(open(path)) if path else from_env(os.environ, a.launcher))
        if a.cmd == "spec":
            print(json.dumps(spec, indent=1))
        else:
            print(_shell(resolve(spec, a.rank)))
    except TopologyError as e:
        print(f"topology: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
