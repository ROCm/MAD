#!/usr/bin/env python3
"""SGLang disaggregated P/D server and router commands, from models.yaml.

Run settings come from the environment, as run_xPyD_models.slurm and node.sh pass
them into the container (defaults in brackets):

  MODEL_NAME            models.yaml entry (required)
  xP, yD                prefill and decode nodes [1, 1]
  DP_MODE               0: tensor parallel (mode tp), 1: DP attention + EP (mode dp_ep) [0]
  ROUTER_NODE           1: the router has a node of its own, after decode. Unset (as in
                        the container): 1 when MAD_TOPO_NNODES is xP + yD + 1, or when
                        ROUTER_ADDR differs from MASTER_ADDR; else 0
  KV_TRANSFER_BACKEND   models.yaml backend [mori]
  GPUS_PER_NODE [8], GENERIC_TP_SIZE [GPUS_PER_NODE]: one tp-mode server's TP size
  IPADDRS               node addresses in rank order (required except for plan / validate)
  MODEL_PATH, IB_DEVICES (--disaggregation-ib-device), HOST_IP (this node's --host,
  else its IPADDRS entry), DIST_INIT_PORT [5757]
  BENCHMARK_SCRIPT, ENABLE_RADIX_CACHE, ENABLE_SERVER_METRICS  select overlays
  EXTRA_SERVER_ARGS, EXTRA_PREFILL_ARGS, EXTRA_DECODE_ARGS, EXTRA_ROUTER_ARGS
  MAD_TOPO_ROLE / _ROLE_INDEX / _ROUTER   the launcher's topology for this node: when
                        given, it decides the node's role (as in the other role scripts)
  _SGL_HELP             SGLang's --help text; empty means unknown (flags as written).
                        Unset: read from `python3 -m sglang.launch_server --help`.

Commands:
  models            catalog models and the modes each supports
  validate          model, mode, backend and topology; non-zero exit on error
  plan [--json]     every node's role and commands (a dry run; needs no GPUs)
  shell --rank R    bash assignments for node R: ROLE, ROLE_INDEX, RUNS_SERVER,
                    RUNS_ROUTER, TP_SIZE, SERVER_ARGV=(...), ROUTER_ARGV=(...),
                    IP_FIRST_PREFILL, IP_FIRST_DECODE
  check             every flag the catalog can render exists in the installed SGLang
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Tuple

import yaml

HERE = Path(__file__).resolve().parent
MODES = ("tp", "dp_ep")
ROLES = ("prefill", "decode")
SERVER_PORT = 3000
ROUTER_PORT = 2322

# Set per node from the topology; a catalog entry setting one is a mistake.
LAUNCHER_FLAGS = frozenset({
    "--model-path", "--disaggregation-mode", "--host", "--port", "--tp-size",
    "--dp-size", "--ep-size", "--nnodes", "--node-rank", "--dist-init-addr",
    "--disaggregation-ib-device",
})

_RANGE = re.compile(r"\s*(\d+)\s*\.\.\s*(\d+)\s*")
_OPTION = re.compile(r"(?<![\w-])--[A-Za-z0-9][A-Za-z0-9-]*")

Flags = Dict[str, object]


class RenderError(Exception):
    pass


# --------------------------------------------------------------------------- catalog

def _flag_map(block, where: str) -> Flags:
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise RenderError(f"{where}: expected a map of --flag: value, got {type(block).__name__}")
    for key in block:
        if not str(key).startswith("--"):
            raise RenderError(f"{where}: {key!r} is not a --flag")
        if key in LAUNCHER_FLAGS:
            raise RenderError(f"{where}: {key} is set by the launcher, not the catalog")
    return dict(block)


def _resolve_layer(cfg: Mapping, mode: str, role: str, where: str) -> Flags:
    """base -> modes[mode] -> roles[role] -> roles[role][mode], later keys winning."""
    flags = _flag_map(cfg.get("base"), f"{where}.base")
    modes = cfg.get("modes") or {}
    for name in modes:
        if name not in MODES:
            raise RenderError(f"{where}.modes: unknown mode {name!r} (expected one of {MODES})")
    flags.update(_flag_map(modes.get(mode), f"{where}.modes.{mode}"))
    roles = cfg.get("roles") or {}
    for name in roles:
        if name not in ROLES:
            raise RenderError(f"{where}.roles: unknown role {name!r} (expected one of {ROLES})")
    block = dict(roles.get(role) or {})
    by_mode = {m: block.pop(m) for m in MODES if m in block}
    flags.update(_flag_map(block, f"{where}.roles.{role}"))
    flags.update(_flag_map(by_mode.get(mode), f"{where}.roles.{role}.{mode}"))
    return flags


def _apply(acc: Flags, layer: Mapping) -> None:
    for key, value in layer.items():
        if value is None or value is False:
            acc.pop(key, None)
        else:
            acc[key] = value


def _inherit(parent: Mapping, child: Mapping) -> Dict:
    """Deep-merge `child` over `parent`; a null in the child drops the inherited key."""
    out = dict(parent)
    for key, value in child.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _inherit(out[key], value)
        else:
            out[key] = value
    return out


class Catalog:
    def __init__(self, path: Path):
        data = yaml.safe_load(path.read_text()) or {}
        self.defaults = data.get("defaults") or {}
        self.backends = data.get("backends") or {}
        self.overlays = data.get("overlays") or {}
        compat = data.get("compat") or {}
        self.renamed: Dict[str, str] = dict(compat.get("renamed") or {})
        self.removed: List[str] = list(compat.get("removed") or [])
        self._models = data.get("models") or {}

    def names(self) -> List[str]:
        return list(self._models)

    def model(self, name: str, _chain: Tuple[str, ...] = ()) -> Dict:
        if name not in self._models:
            raise RenderError(f"unknown MODEL_NAME {name!r}; models.yaml has: {', '.join(self._models)}")
        if name in _chain:
            raise RenderError(f"models.yaml: extends cycle {' -> '.join(_chain + (name,))}")
        cfg = dict(self._models[name] or {})
        parent = cfg.pop("extends", None)
        if parent:
            cfg = _inherit(self.model(parent, _chain + (name,)), cfg)
        return cfg

    def modes_for(self, name: str) -> List[str]:
        declared = self.model(name).get("modes") or {}
        return [m for m in MODES if m == "tp" or m in declared]

    def backend(self, name: str) -> Mapping:
        if name not in self.backends:
            raise RenderError(f"unknown KV_TRANSFER_BACKEND {name!r}; models.yaml backends: {', '.join(self.backends)}")
        return self.backends[name] or {}

    def server_flags(self, name: str, mode: str, role: str, backend: str, overlays=()) -> Flags:
        acc: Flags = {}
        _apply(acc, _resolve_layer(self.defaults, mode, role, "defaults"))
        _apply(acc, _resolve_layer(self.model(name), mode, role, f"models.{name}"))
        _apply(acc, _flag_map(self.backend(backend).get("args"), f"backends.{backend}.args"))
        for overlay in overlays:
            if overlay not in self.overlays:
                raise RenderError(f"models.yaml has no overlay {overlay!r}")
            _apply(acc, _flag_map(self.overlays[overlay], f"overlays.{overlay}"))
        return acc

    def compat(self, flags: Flags, options: Optional[Set[str]]) -> Tuple[Flags, List[str]]:
        """`flags` in the installed SGLang's names. Unknown options: unchanged."""
        if options is None:
            return flags, []
        out: Flags = {}
        notes = []
        for key, value in flags.items():
            new = self.renamed.get(key)
            if new and key not in options and new in options:
                notes.append(f"this SGLang has {new} instead of {key}")
                key = new
            elif key in self.removed and key not in options:
                notes.append(f"this SGLang has no {key}; launching without it")
                continue
            out[key] = value
        return out, notes


def _values(value) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [v for item in value for v in _values(item)]
    if isinstance(value, str):
        match = _RANGE.fullmatch(value)
        if match:
            return [str(i) for i in range(int(match.group(1)), int(match.group(2)) + 1)]
        return [value]
    return [str(value)]


def flags_to_argv(flags: Mapping[str, object]) -> List[str]:
    argv: List[str] = []
    for key, value in flags.items():
        argv.append(key)
        if value is not True:
            argv.extend(_values(value))
    return argv


def sglang_options(env: Mapping[str, str]) -> Optional[Set[str]]:
    """The installed SGLang's server options, or None when they cannot be read."""
    if "_SGL_HELP" in env:
        text = env["_SGL_HELP"]
    else:
        # Not from the image's /sgl-workspace: its sglang/ checkout shadows the package.
        try:
            text = subprocess.run([sys.executable, "-m", "sglang.launch_server", "--help"],
                                  capture_output=True, text=True, timeout=300, cwd="/").stdout
        except (OSError, subprocess.TimeoutExpired):
            text = ""
    options = set(_OPTION.findall(text))
    return options or None


# --------------------------------------------------------------------------- settings

def _int(env: Mapping[str, str], key: str, default: Optional[int]) -> int:
    raw = env.get(key, "")
    if raw == "":
        if default is None:
            raise RenderError(f"{key} is not set")
        return default
    try:
        return int(raw)
    except ValueError:
        raise RenderError(f"{key}={raw!r} is not an integer")


@dataclass
class Settings:
    model: str
    xp: int
    yd: int
    mode: str
    backend: str
    router_node: bool
    ips: List[str]
    gpus_per_node: int
    tp_size: int
    model_path: str
    ib_devices: str
    host_ip: str
    dist_init_port: int
    overlays: List[str]
    extra_all: List[str]
    extra_role: Dict[str, List[str]]
    extra_router: List[str]

    @classmethod
    def from_env(cls, env: Mapping[str, str], require_ips: bool = True) -> "Settings":
        model = env.get("MODEL_NAME", "")
        if not model:
            raise RenderError("MODEL_NAME is not set")
        xp, yd = _int(env, "xP", 1), _int(env, "yD", 1)
        dp_mode = env.get("DP_MODE", "0") or "0"
        if dp_mode not in ("0", "1"):
            raise RenderError(f"DP_MODE={dp_mode!r} (0 or 1)")
        router_node = env.get("ROUTER_NODE", "")
        if router_node == "":
            # node.sh does not pass ROUTER_NODE into the container.
            if env.get("MAD_TOPO_NNODES"):
                router_node = "1" if _int(env, "MAD_TOPO_NNODES", 0) == xp + yd + 1 else "0"
            elif env.get("ROUTER_ADDR") and env.get("ROUTER_ADDR") != env.get("MASTER_ADDR"):
                router_node = "1"
            else:
                router_node = "0"
        if router_node not in ("0", "1"):
            raise RenderError(f"ROUTER_NODE={router_node!r} (0 or 1)")
        nodes = xp + yd + int(router_node)
        ips = [ip.strip() for ip in env.get("IPADDRS", "").split(",") if ip.strip()]
        if not ips:
            if require_ips:
                raise RenderError("IPADDRS is not set")
            ips = [f"node{i}" for i in range(nodes)]
        gpus = _int(env, "GPUS_PER_NODE", 8)
        agentic = env.get("BENCHMARK_SCRIPT") == "agentic"
        overlays = [name for name, on in (
            ("radix_cache", agentic or env.get("ENABLE_RADIX_CACHE") == "1"),
            ("server_metrics", agentic or env.get("ENABLE_SERVER_METRICS") == "1")) if on]
        return cls(
            model=model, xp=xp, yd=yd, mode="dp_ep" if dp_mode == "1" else "tp",
            backend=env.get("KV_TRANSFER_BACKEND") or "mori", router_node=router_node == "1",
            ips=ips, gpus_per_node=gpus, tp_size=_int(env, "GENERIC_TP_SIZE", gpus),
            model_path=env.get("MODEL_PATH") or f"<models>/{model}",
            ib_devices=env.get("IB_DEVICES", ""), host_ip=env.get("HOST_IP", ""),
            dist_init_port=_int(env, "DIST_INIT_PORT", 5757), overlays=overlays,
            extra_all=shlex.split(env.get("EXTRA_SERVER_ARGS", "")),
            extra_role={"prefill": shlex.split(env.get("EXTRA_PREFILL_ARGS", "")),
                        "decode": shlex.split(env.get("EXTRA_DECODE_ARGS", ""))},
            extra_router=shlex.split(env.get("EXTRA_ROUTER_ARGS", "")),
        )


# --------------------------------------------------------------------------- topology

@dataclass
class Node:
    rank: int
    ip: str
    role: str                  # prefill | decode | router
    role_index: int = 0        # position within the role (MAD_TOPO_ROLE_INDEX)
    runs_router: bool = False
    tp: int = 0
    dp: int = 1
    ep: int = 1
    nnodes: int = 1            # nodes in this node's server
    node_rank: int = 0         # rank within that server
    leader_ip: str = ""


def build_topology(s: Settings) -> Tuple[Dict[int, Node], Dict[str, List[str]]]:
    """Nodes by rank (prefill, then decode, then the router node if any) and each
    role's server leaders. A tp server spans GENERIC_TP_SIZE / GPUS_PER_NODE nodes;
    a dp_ep server spans all of its role's nodes."""
    if s.xp < 1 or s.yd < 1:
        raise RenderError(f"need at least one prefill and one decode node (xP={s.xp}, yD={s.yd})")
    needed = s.xp + s.yd + int(s.router_node)
    if len(s.ips) != needed:
        raise RenderError(f"{len(s.ips)} address(es) in IPADDRS for xP={s.xp} + yD={s.yd}"
                          f"{' + a router node' if s.router_node else ''} = {needed} nodes")
    nodes: Dict[int, Node] = {}
    leaders: Dict[str, List[str]] = {}
    for role, first, count in (("prefill", 0, s.xp), ("decode", s.xp, s.yd)):
        if s.mode == "dp_ep":
            tp = dp = ep = count * s.gpus_per_node
            per = count
        else:
            tp, dp, ep = s.tp_size, 1, 1
            if tp > s.gpus_per_node and tp % s.gpus_per_node:
                raise RenderError(f"GENERIC_TP_SIZE={tp} is not a multiple of GPUS_PER_NODE={s.gpus_per_node}")
            per = max(1, tp // s.gpus_per_node)
        if count % per:
            raise RenderError(f"{count} {role} node(s) cannot be split into servers of {per} node(s)")
        leaders[role] = []
        for start in range(first, first + count, per):
            leader = s.ips[start]
            leaders[role].append(leader)
            for i, rank in enumerate(range(start, start + per)):
                nodes[rank] = Node(rank=rank, ip=s.ips[rank], role=role, role_index=rank - first,
                                   tp=tp, dp=dp, ep=ep, nnodes=per, node_rank=i, leader_ip=leader)
    if s.router_node:
        rank = needed - 1
        nodes[rank] = Node(rank=rank, ip=s.ips[rank], role="router", runs_router=True)
    else:
        nodes[0].runs_router = True
    return nodes, leaders


def this_node(nodes: Dict[int, Node], s: Settings, rank: int,
              env: Mapping[str, str]) -> Tuple[Node, Optional[str]]:
    """Node `rank`, and a note when the launcher's topology (MAD_TOPO_*,
    common/topology/topology.py) places it differently from xP / yD. The topology
    decides the role; the layout gives that role's sizes and leader."""
    if rank not in nodes:
        raise RenderError(f"NODE_RANK={rank} is outside the {len(nodes)}-node topology")
    node = nodes[rank]
    role = env.get("MAD_TOPO_ROLE", "")
    if not role:
        return node, None
    index = _int(env, "MAD_TOPO_ROLE_INDEX", 0)
    template = next((n for n in nodes.values() if n.role == role and n.role_index == index), None)
    if template is None:
        raise RenderError(f"node {rank}: the topology spec makes it {role} {index}, which "
                          f"xP={s.xp} yD={s.yd} has no place for")
    router = env.get("MAD_TOPO_ROUTER", "")
    placed = replace(template, rank=rank, ip=s.ips[rank],
                     runs_router=template.runs_router if router == "" else router == "1")
    if (placed.role, placed.role_index, placed.runs_router) == (node.role, node.role_index, node.runs_router):
        return placed, None
    return placed, (f"node {rank}: following the topology spec ({role} {index}"
                    f"{', router' if placed.runs_router else ''}) over its place in xP={s.xp} yD={s.yd} "
                    f"({node.role} {node.role_index})")


def validate(cat: Catalog, s: Settings):
    modes = cat.modes_for(s.model)
    if s.mode not in modes:
        raise RenderError(f"{s.model} does not support DP_MODE=1 (mode {s.mode}); "
                          f"add a modes.{s.mode} block to its models.yaml entry to enable it")
    backend_modes = (cat.backend(s.backend).get("modes") or list(MODES))
    if s.mode not in backend_modes:
        raise RenderError(f"KV_TRANSFER_BACKEND={s.backend} is not validated with mode {s.mode} "
                          f"(models.yaml backends.{s.backend}.modes: {', '.join(backend_modes)})")
    for role in ROLES:
        cat.server_flags(s.model, s.mode, role, s.backend, s.overlays)
    return build_topology(s)


# --------------------------------------------------------------------------- commands

def server_argv(cat: Catalog, s: Settings, node: Node, options: Optional[Set[str]],
                notes: Optional[List[str]] = None) -> List[str]:
    argv = ["--model-path", s.model_path, "--disaggregation-mode", node.role,
            "--host", s.host_ip or node.ip, "--port", str(SERVER_PORT), "--tp-size", str(node.tp)]
    if s.mode == "dp_ep":
        argv += ["--dp-size", str(node.dp), "--ep-size", str(node.ep)]
    if node.nnodes > 1 or s.mode == "dp_ep":
        argv += ["--dist-init-addr", f"{node.leader_ip}:{s.dist_init_port}",
                 "--nnodes", str(node.nnodes), "--node-rank", str(node.node_rank)]
    if s.ib_devices:
        argv += ["--disaggregation-ib-device", s.ib_devices]
    flags, compat_notes = cat.compat(cat.server_flags(s.model, s.mode, node.role, s.backend, s.overlays), options)
    if notes is not None:
        notes.extend(compat_notes)
    return argv + flags_to_argv(flags) + s.extra_all + s.extra_role[node.role]


def router_argv(s: Settings, leaders: Mapping[str, List[str]]) -> List[str]:
    argv = ["--pd-disaggregation"]
    for ip in leaders["prefill"]:
        argv += ["--prefill", f"http://{ip}:{SERVER_PORT}"]
    for ip in leaders["decode"]:
        argv += ["--decode", f"http://{ip}:{SERVER_PORT}"]
    return argv + ["--host", "0.0.0.0", "--port", str(ROUTER_PORT)] + s.extra_router


def _bash_array(argv: List[str]) -> str:
    return "(" + " ".join(shlex.quote(a) for a in argv) + ")"


def shell_vars(cat: Catalog, s: Settings, node: Node, leaders,
               options: Optional[Set[str]], notes: List[str]) -> List[str]:
    lines = [f"ROLE={node.role}", f"ROLE_INDEX={node.role_index}",
             f"RUNS_SERVER={0 if node.role == 'router' else 1}",
             f"RUNS_ROUTER={1 if node.runs_router else 0}", f"TP_SIZE={node.tp}",
             f"IP_FIRST_PREFILL={leaders['prefill'][0]}", f"IP_FIRST_DECODE={leaders['decode'][0]}"]
    if node.role != "router":
        lines.append(f"SERVER_ARGV={_bash_array(server_argv(cat, s, node, options, notes))}")
    if node.runs_router:
        lines.append(f"ROUTER_ARGV={_bash_array(router_argv(s, leaders))}")
    return lines


def print_plan(cat: Catalog, s: Settings, nodes, leaders, options, as_json: bool) -> None:
    rows = []
    for rank in sorted(nodes):
        node = nodes[rank]
        row = asdict(node)
        if node.role != "router":
            row["server_cmd"] = ["python3", "-m", "sglang.launch_server"] + server_argv(cat, s, node, options)
        if node.runs_router:
            row["router_cmd"] = ["python3", "-m", "sglang_router.launch_router"] + router_argv(s, leaders)
        rows.append(row)
    if as_json:
        print(json.dumps(rows, indent=2))
        return
    print(f"model={s.model} mode={s.mode} backend={s.backend} xP={s.xp} yD={s.yd} "
          f"router={'own node' if s.router_node else 'node 0'} overlays={','.join(s.overlays) or '-'}")
    for row in rows:
        print(f"\n[node {row['rank']}] {row['ip']}  {row['role']} {row['role_index']}"
              f"{' + router' if row['runs_router'] and row['role'] != 'router' else ''}")
        for key in ("server_cmd", "router_cmd"):
            if key in row:
                print("  " + shlex.join(row[key]))


def cmd_check(cat: Catalog, options: Optional[Set[str]]) -> int:
    if options is None:
        print("check: cannot read the installed SGLang's options "
              "(python3 -m sglang.launch_server --help)", file=sys.stderr)
        return 1
    used: Dict[str, Set[str]] = {}
    for name in cat.names():
        for mode in cat.modes_for(name):
            for backend, spec in cat.backends.items():
                if mode not in ((spec or {}).get("modes") or MODES):
                    continue
                for role in ROLES:
                    flags, _ = cat.compat(cat.server_flags(name, mode, role, backend, list(cat.overlays)), options)
                    for flag in flags:
                        used.setdefault(flag, set()).add(f"{name}/{mode}/{role}/{backend}")
    for flag in LAUNCHER_FLAGS:
        used.setdefault(flag, set()).add("launcher")
    unknown = sorted(flag for flag in used if flag not in options)
    for flag in unknown:
        wheres = sorted(used[flag])
        print(f"  flag this SGLang does not accept: {flag}  (used by {len(wheres)}: "
              f"{', '.join(wheres[:4])}{', ...' if len(wheres) > 4 else ''})")
    if unknown:
        return 1
    print(f"OK: every flag the catalog renders exists in this SGLang ({len(cat.names())} models)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=HERE / "models.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a run setting from the environment")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("models", "validate", "check"):
        sub.add_parser(name)
    sub.add_parser("plan").add_argument("--json", action="store_true")
    sub.add_parser("shell").add_argument("--rank", type=int, required=True)
    args = ap.parse_args(argv)

    env = dict(os.environ)
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            ap.error(f"--set expects KEY=VALUE, got {item!r}")
        env[key] = value

    try:
        cat = Catalog(args.models)
        if args.cmd == "models":
            for name in cat.names():
                print(f"{name}\t{','.join(cat.modes_for(name))}")
            return 0
        if args.cmd == "check":
            return cmd_check(cat, sglang_options(env))
        s = Settings.from_env(env, require_ips=args.cmd == "shell")
        nodes, leaders = validate(cat, s)
        if args.cmd == "validate":
            return 0
        options = sglang_options(env)
        if args.cmd == "plan":
            print_plan(cat, s, nodes, leaders, options, args.json)
            return 0
        node, placement = this_node(nodes, s, args.rank, env)
        if placement:
            print(f"[topology] {placement}", file=sys.stderr)
        notes: List[str] = []
        lines = shell_vars(cat, s, node, leaders, options, notes)
        for note in notes:
            print(f"[sglang] {note}", file=sys.stderr)
        print("\n".join(lines))
        return 0
    except RenderError as exc:
        print(f"render.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
