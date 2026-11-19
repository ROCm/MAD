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


def gpus_per_node(spec, env=None):
    """The spec's gpus_per_node, else GPUS_PER_NODE (cluster.sh: the run's platform). Never assumed."""
    if spec.get("gpus_per_node") is not None:
        return spec["gpus_per_node"]
    v = _int(env if env is not None else {}, "GPUS_PER_NODE", None)
    if v is None:
        v = _int(os.environ, "GPUS_PER_NODE", None)
    if v is None:
        raise TopologyError("gpus_per_node unknown: give it in the spec, or GPUS_PER_NODE "
                            "(the launchers set it from the platform, compat/rocm.yaml platforms)")
    return v


def from_env(env, launcher):
    """The spec a launcher builds today from its environment."""
    gpus = gpus_per_node({}, env)
    if launcher == "vllm_multinode":
        nnodes = _int(env, "SLURM_NNODES", 1)
        tp = _int(env, "TP_SIZE", gpus)
        pp = _int(env, "PP_SIZE", nnodes)
        parallel = {"mode": "tp_pp", "tp": tp, "pp": pp}
        if env.get("REPLICAS", "0") == "1":          # independent TP_SIZE-GPU servers on every node
            parallel = {"mode": "replicas", "tp": tp}
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


# ---------------------------------------------------------------- topology v2
# A spec may also say what it runs on and how it is placed (compat/rocm.yaml platforms):
#   "platform": "mi455x-helios-tray"      GPUs per node and the scale-up domain from the catalog
#   "partition": "CPX"                    logical GPUs: the platform's GPUs x the mode's size
#   role "gpus": 72                       instead of nodes (must fill whole nodes)
#   role "platform": "..."                a role on another node type (heterogeneous pools)
#   role "kind": "draft"|"encoder"|...    what the role is, when its name does not say
#   role "kv_format": "fp8"               the KV cache a role hands over or takes (disagg)
#   "placement": [{"same_domain": [roles]}, {"spread_domains": [roles]}]
#   "strict": true                        a parallel group that leaves its scale-up domain is refused
#   parallel {"mode": "replicas", "tp": 8} independent TP-8 copies over the role's GPUs (multi-tenant)
# A spec using none of these resolves exactly as before.
KINDS = ("prefill", "decode", "router", "server", "draft", "encoder", "embedding", "kv_cache")
CATALOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "compat", "rocm.yaml")
_CATALOG = None


def catalog():
    global _CATALOG
    if _CATALOG is None:
        import yaml
        path = os.environ.get("MAD_CATALOG") or CATALOG
        try:
            with open(path) as f:
                _CATALOG = yaml.safe_load(f) or {}
        except OSError as e:
            raise TopologyError(f"the capability catalog is needed for platforms and domains: {e}")
    return _CATALOG


def _platform(name, where):
    p = (catalog().get("platforms") or {}).get(name)
    if p is None:
        raise TopologyError(f"{where}: unknown platform {name!r} (compat/rocm.yaml platforms: "
                            f"{', '.join(sorted(catalog().get('platforms') or {}))})")
    return p


def kind(role):
    return role.get("kind") or (role["name"] if role.get("name") in KINDS else "server")


def normalize(spec):
    """Fill the v1 fields (gpus_per_node, role nodes) from the v2 ones; idempotent."""
    if spec.get("platform"):
        plat = _platform(spec["platform"], "spec")
        per = plat["gpus_per_node"]
        if spec.get("partition"):
            sizes = (catalog()["accelerators"][plat["accelerator"]] or {}).get("partitions") or {}
            factor = sizes.get(spec["partition"])
            if not factor:
                raise TopologyError(f"partition {spec['partition']!r}: its size on {plat['accelerator']} is not "
                                    f"known (compat/rocm.yaml accelerators.{plat['accelerator']}.partitions)")
            per = per * factor
        if spec.get("gpus_per_node") not in (None, per):
            raise TopologyError(f"gpus_per_node={spec['gpus_per_node']} disagrees with platform "
                                f"{spec['platform']}{' in ' + spec['partition'] if spec.get('partition') else ''} ({per})")
        spec["gpus_per_node"] = per
    elif spec.get("partition"):
        raise TopologyError("partition needs a platform (its accelerator says how many GPUs each mode makes)")
    for r in spec.get("roles") or []:
        if r.get("platform"):
            r["gpus_per_node"] = _platform(r["platform"], f"role {r.get('name')!r}")["gpus_per_node"]
        if "gpus" in r:
            per = r.get("gpus_per_node") or gpus_per_node(spec)
            if not isinstance(r["gpus"], int) or r["gpus"] < 1 or r["gpus"] % per:
                raise TopologyError(f"role {r.get('name')!r}: gpus={r['gpus']} is not a whole number of "
                                    f"{per}-GPU nodes")
            if r.get("nodes") not in (None, r["gpus"] // per):
                raise TopologyError(f"role {r.get('name')!r}: gpus={r['gpus']} and nodes={r['nodes']} disagree")
            r["nodes"] = r["gpus"] // per
        if kind(r) not in KINDS:
            raise TopologyError(f"role {r.get('name')!r}: kind {r.get('kind')!r} ({', '.join(KINDS)})")
    return spec


def domain_size(spec, role=None):
    """(GPUs per scale-up domain, nodes per domain) for a role (its own platform, else the spec's),
    in the GPUs the run counts (logical, under a partition mode); None without a platform."""
    name = (role or {}).get("platform") or spec.get("platform")
    if not name:
        return None
    plat = _platform(name, "spec")
    gpd = (plat.get("scale_up") or {}).get("gpus_per_domain") or plat["gpus_per_node"]
    nodes = max(1, gpd // plat["gpus_per_node"])
    if spec.get("partition") and not (role or {}).get("platform"):
        gpd *= gpus_per_node(spec) // plat["gpus_per_node"]
    return gpd, nodes


def _role_ranks(spec):
    first, out = 0, {}
    for r in spec["roles"]:
        out[r["name"]] = range(first, first + r["nodes"])
        first += r["nodes"]
    return out


def check(spec):
    """Placement and fabric tiers: ([errors], [notes]). Errors: placement constraints the rank
    order cannot meet; under strict, a parallel group that leaves its scale-up domain. Notes: what
    crosses scale-out (RDMA), and KV boundary formats."""
    errors, notes = [], []
    ds = domain_size(spec)
    ranks = _role_ranks(spec)
    for c in spec.get("placement") or []:
        if ds is None:
            errors.append(f"placement {c}: needs a platform (its scale-up domain)")
            continue
        npd = ds[1]
        dom = {name: {i // npd for i in ranks[name]} for name in ranks}
        for name in [n for v in c.values() for n in v]:
            if name not in ranks:
                errors.append(f"placement {c}: {name!r} is not a role")
        if errors:
            continue
        if "same_domain" in c:
            together = set().union(*(dom[n] for n in c["same_domain"]))
            if len(together) != 1:
                errors.append(f"same_domain {c['same_domain']}: in rank order they span domains {sorted(together)} "
                              f"({npd} nodes per domain)")
        if "spread_domains" in c:
            seen = {}
            for n in c["spread_domains"]:
                for d in dom[n]:
                    if d in seen and seen[d] != n:
                        errors.append(f"spread_domains {c['spread_domains']}: {seen[d]!r} and {n!r} share domain {d}")
                    seen[d] = n
    for r in spec["roles"]:
        rds = domain_size(spec, r)
        gpd = rds[0] if rds else None
        # A role's own ranks, in its own domains (a role on another platform starts its own).
        base = ranks[r["name"]].start if r.get("platform") else 0
        span = len({(i - base) // rds[1] for i in ranks[r["name"]]}) if rds else 1
        p = r.get("parallel") or {}
        per = r.get("gpus_per_node") or gpus_per_node(spec)
        tp = (p.get("tp") if p.get("mode") in ("tp", "tp_pp", "replicas")
              else p.get("ep_tp") if p.get("mode") == "wide_ep" else None)
        if p.get("mode") == "replicas" and (r["nodes"] * per) % p.get("tp", 1):
            errors.append(f"role {r['name']!r}: {r['nodes'] * per} GPUs are not whole TP-{p.get('tp')} replicas")
        ep = r["nodes"] * per if p.get("mode") in ("wide_ep", "dp_ep") or p.get("ep") else None
        if gpd and tp and tp > gpd:
            (errors if spec.get("strict") else notes).append(
                f"role {r['name']!r}: a TP group of {tp} GPUs exceeds the {gpd}-GPU scale-up domain")
        if gpd and ep and ep > gpd:
            notes.append(f"role {r['name']!r}: its EP group of {ep} GPUs is larger than the {gpd}-GPU scale-up "
                         "domain: the all-to-all crosses scale-out (RDMA)")
        elif gpd and ep and span > 1:
            (errors if spec.get("strict") else notes).append(
                f"role {r['name']!r}: its EP group of {ep} GPUs would fit one scale-up domain, but in rank "
                "order its nodes span two: the all-to-all crosses scale-out (RDMA)")
    kv = {r["name"]: r.get("kv_format") for r in spec["roles"]}
    if kv.get("prefill") and kv.get("decode") and kv["prefill"] != kv["decode"]:
        errors.append(f"KV cache handed from prefill as {kv['prefill']} to decode as {kv['decode']}: "
                      "no conversion between roles is supported")
    return errors, notes


def scheduler_hints(spec, scheduler, domain_label=None):
    """What to ask the scheduler for so the spec's placement holds: {"options": [...], "notes": [...]}.
    SLURM: --segment=<nodes per scale-up domain> (the topology/block plugin keeps each segment in
    one block) when a same_domain constraint covers more than one node. Kubernetes: required pod
    affinity on the node label that names a scale-up domain (the site profile's domain_label)."""
    validate(spec)
    ds = domain_size(spec)
    together = [c["same_domain"] for c in spec.get("placement") or [] if "same_domain" in c]
    ranks = _role_ranks(spec)
    out = {"options": [], "notes": []}
    if not ds or ds[1] <= 1 or not together:
        out["notes"].append("no placement beyond single nodes: nothing to ask the scheduler for")
        return out
    width = max(len(set().union(*(set(ranks[n]) for n in group))) for group in together)
    if scheduler == "slurm":
        out["options"].append(f"--segment={min(width, ds[1])}")
        out["notes"].append("needs SLURM's topology/block plugin with one block per scale-up domain")
    elif scheduler == "kubernetes":
        # One Indexed Job has one pod template: it can hold the whole run inside one scale-up
        # domain (pod affinity on the domain's node label), not put roles in different domains.
        spread = [c for c in spec.get("placement") or [] if "spread_domains" in c]
        if spread:
            raise TopologyError("Kubernetes: one Job cannot put roles in different scale-up domains "
                                f"({spread}); that needs a Job per role, not supported yet")
        if nnodes(spec) > ds[1]:
            raise TopologyError(f"Kubernetes: same_domain {together} on a run of {nnodes(spec)} nodes, larger "
                                f"than a {ds[1]}-node domain: one Job can only hold the whole run in one domain")
        if not domain_label:
            out["notes"].append("no domain label (k8s.domain_label / MAD_K8S_DOMAIN_LABEL): placement is not requested")
        else:
            out["options"].append({"pod_affinity_topology_key": domain_label})
    else:
        raise TopologyError(f"scheduler {scheduler!r} (slurm or kubernetes)")
    return out


def validate(spec):
    normalize(spec)
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
    gpus = gpus_per_node(spec)
    for r in roles:
        p = r.get("parallel") or {}
        if p.get("mode") == "wide_ep" and gpus % p.get("ep_tp", 1):
            raise TopologyError(f"role {r['name']!r}: ep_tp={p.get('ep_tp')} does not divide gpus_per_node={gpus}")
        if p.get("mode") == "tp_pp" and p.get("tp", gpus) * p.get("pp", r["nodes"]) != r["nodes"] * (r.get("gpus_per_node") or gpus):
            raise TopologyError(f"role {r['name']!r}: tp*pp != nodes*gpus_per_node "
                                f"({p.get('tp')}x{p.get('pp')} vs {r['nodes']}x{gpus})")
    router = spec.get("router")
    if router:
        host = next((r for r in roles if r["name"] == router.get("host")), None)
        if host is None:
            raise TopologyError(f"router host {router.get('host')!r} is not a role")
        if not 0 <= router.get("index", 0) < host["nodes"]:
            raise TopologyError(f"router index {router.get('index')} outside role {host['name']!r}")
    errors, _ = check(spec)
    if errors:
        raise TopologyError("; ".join(errors))
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
    # v2 only (a spec without these resolves exactly as before):
    if "kind" in r or kind(r) not in ("prefill", "decode", "router", "server"):
        out["kind"] = kind(r)
    if r.get("gpus_per_node") is not None:
        out["gpus_per_node"] = r["gpus_per_node"]
    ds = domain_size(spec)
    if ds and ds[1] > 1:
        out["domain"] = rank // ds[1]
    router = spec.get("router")
    out["router"] = bool(router and router.get("host") == r["name"] and router.get("index", 0) == index)
    p = r.get("parallel") or {}
    if p.get("mode") == "wide_ep":
        per_node = (r.get("gpus_per_node") or gpus_per_node(spec)) // p.get("ep_tp", 1)
        out.update({"dp_per_node": per_node, "dp_size": r["nodes"] * per_node, "dp_start_rank": index * per_node})
    return out


# The settings each launcher reads, by the spec it can express today. A spec outside
# that shape is refused with what the launcher can run, rather than run differently.
def legacy_env(spec, launcher, env=None):
    """The launcher's own settings for a spec (MAD_TOPOLOGY): what it would otherwise
    read from xP / yD / WIDE_EP / ... A setting the caller also gave must agree."""
    validate(spec)
    roles = spec["roles"]
    gpus = gpus_per_node(spec, env)
    per_role = {r.get("gpus_per_node") for r in roles if r.get("gpus_per_node") is not None}
    if per_role - {gpus}:
        raise TopologyError(f"{launcher} runs one node type; roles on other platforms ({sorted(per_role)} GPUs "
                            f"per node, the run {gpus}) need a launcher for heterogeneous pools")
    runs = ("prefill", "decode", "router", "server") + (("embedding",) if launcher == "vllm_multinode" else ())
    other = [r["name"] for r in roles if kind(r) not in runs]
    if other:
        raise TopologyError(f"{launcher} runs prefill/decode/router/server roles; {other} need their serving type")
    engine = {"vllm_multinode": "vllm", "vllm_dissag": "vllm", "sglang_disagg": "sglang"}.get(launcher)
    if engine and spec.get("engine", engine) != engine:
        raise TopologyError(f"{launcher} runs {engine}, the spec is for {spec['engine']}")
    if launcher == "vllm_multinode":
        if spec["layout"] != "colocated" or len(roles) != 1:
            raise TopologyError("vllm_multinode runs one colocated role: "
                                '{"layout": "colocated", "roles": [{"name": ..., "nodes": N, "parallel": '
                                '{"mode": "tp_pp", "tp": T, "pp": P}}]}')
        p = roles[0].get("parallel") or {}
        if p.get("mode", "tp_pp") not in ("tp_pp", "replicas"):
            raise TopologyError(f"vllm_multinode: parallel mode {p.get('mode')!r} (tp_pp or replicas)")
        if p.get("mode") == "replicas":
            return _agree({"GPUS_PER_NODE": gpus, "TP_SIZE": p.get("tp", 1), "PP_SIZE": 1, "REPLICAS": 1,
                           "MAD_TOPOLOGY_NNODES": roles[0]["nodes"]}, env)
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
    return _agree(out, env)


def _agree(out, env):
    """The launcher settings, refusing any the caller also gave differently."""
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
    c = sub.add_parser("check", help="placement and fabric tiers of MAD_TOPOLOGY: errors and notes")
    c.add_argument("--launcher", choices=LAUNCHERS, required=True)
    h = sub.add_parser("scheduler-hints", help="what to ask the scheduler for, for MAD_TOPOLOGY's placement")
    h.add_argument("--launcher", choices=LAUNCHERS, required=True)
    h.add_argument("--scheduler", choices=("slurm", "kubernetes"), required=True)
    h.add_argument("--domain-label", default=os.environ.get("MAD_K8S_DOMAIN_LABEL"))
    a = ap.parse_args(argv)
    scripts_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        path = os.environ.get("MAD_TOPOLOGY")
        spec = validate(load(path, scripts_dir) if path else from_env(os.environ, a.launcher))
        if a.cmd == "spec":
            print(json.dumps(spec, indent=1))
        elif a.cmd == "scheduler-hints":
            print(json.dumps(scheduler_hints(spec, a.scheduler, a.domain_label), indent=1))
        elif a.cmd == "check":
            for note in check(spec)[1]:
                print(f"note: {note}")
            print(f"{nnodes(spec)} nodes, {gpus_per_node(spec)} GPUs per node: ok")
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
