#!/usr/bin/env python3
r"""A card's workload file (MAD_CONFIG): MAD's own reader, so MAD runs without madengine.

The format is madengine's layered config (docs: madengine distributed-config.md); madengine reads
it at submit time, MAD's launchers read it here, and a test holds the two to the same result.

    version: 2
    defaults: [../../../common/workloads/family/kimi-k3.yaml, ../../../common/workloads/bench/niah.yaml]
    model:
      env: {MODEL_NAME: Kimi-K3-MXFP4, xP: "2", TAG: "tp${model.env.TP_SIZE}"}
    benchmark:
      niah: {env: {NIAH_WORDS: "50000"}}

- `defaults`: base files, relative to the file, merged in order (recursively; a cycle or a missing
  base is refused), then the file itself on top. Mappings merge; lists and scalars are replaced.
- `${a.b.c}` refers to another value of the merged file, `${oc.env:NAME,default}` to the
  environment; resolved after the merge.
- `benchmark`: a list of entries (`kind`: which benchmark; none: every one) or, in version 2, a
  map by kind (`default`: every one).
- Version 1: no inheritance, no interpolation.

A file may also carry the layers a launcher otherwise reads from its directory, so one file
holds a card's whole configuration (config/emit.py writes such files from existing cards):

    recipes:                   # replaces the launcher's models.yaml: the card's recipe (MODEL_NAME)
      Kimi-K3-MXFP4: {...}     # and any recipe it is an alias of, written as in models.yaml
    connectors:                # replaces connectors/<name>.env: KEY: VALUE per connector
      moriio: {PYTORCH_ALLOC_CONF: "expandable_segments:False", ...}
    topology: common/topology/specs/v2/x.json    # as MAD_TOPOLOGY (a path; relative to scripts/)
    card: {...}                # what the file was generated from (informational)
    image: {...}               # the card's image (informational to the launcher; for the build)

`\${` is a literal `${` (as in OmegaConf).

The environment it gives, lowest first: model.env, the benchmark entries without a kind, the
entries of the selected kind (BENCHMARK_SCRIPT: the run's, else the file's). The launcher applies
it under everything set at submit time (config/layers.sh config_apply_workload).

    workload.py env FILE [--kind KIND]     NAME=VALUE\0 ...
    workload.py show FILE                  the merged, resolved file as JSON
    workload.py find MAD_CONFIG DIR        the file a card with MAD_CONFIG (may be empty) in DIR reads
    workload.py sections FILE --out DIR    write its recipes (DIR/models.yaml) and connectors
                                           (DIR/connectors/NAME.env); print recipes=, connectors=,
                                           topology= for those present (NUL-separated)
Standard library and PyYAML only.
"""
import argparse
import json
import os
import re
import sys

import yaml

VERSIONS = (1, 2)
REF = re.compile(r"(?<!\\)\$\{([^${}]+)\}")
ESCAPED = "\\${"


class WorkloadError(ValueError):
    pass


def _read(path):
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise WorkloadError(f"{path}: {e}")
    data = data or {}
    if not isinstance(data, dict):
        raise WorkloadError(f"{path}: top level must be a map")
    if data.get("version") not in VERSIONS:
        raise WorkloadError(f"{path}: version {data.get('version')!r} is not supported ({list(VERSIONS)})")
    return data


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def _chain(path, data, seen):
    path = os.path.realpath(path)
    if path in seen:
        raise WorkloadError("defaults cycle: " + " -> ".join(seen + (path,)))
    merged = {}
    for ref in data.get("defaults") or []:
        if not isinstance(ref, str):
            raise WorkloadError(f"{path}: defaults entries are file paths, got {ref!r}")
        bpath = os.path.normpath(os.path.join(os.path.dirname(path), ref))
        if not os.path.exists(bpath):
            raise WorkloadError(f"{path}: default {ref!r} not found ({bpath})")
        bdata = _read(bpath)
        if bdata.get("version") != 2:
            raise WorkloadError(f"{bpath}: a base file must be version 2")
        merged = _merge(merged, _chain(bpath, bdata, seen + (path,)))
    return _merge(merged, {k: v for k, v in data.items() if k != "defaults"})


def _lookup(root, dotted, where):
    node = root
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise WorkloadError(f"{where}: ${{{dotted}}} refers to nothing")
        node = node[part]
    return node


def _resolve(root, value, where, depth=0):
    if isinstance(value, dict):
        return {k: _resolve(root, v, f"{where}.{k}", depth) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(root, v, where, depth) for v in value]
    if not isinstance(value, str) or "${" not in value:
        return value
    if not REF.search(value):
        return value.replace(ESCAPED, "${")
    if depth > 20:
        raise WorkloadError(f"{where}: interpolation loop")

    def one(m):
        expr = m.group(1).strip()
        if expr.startswith("oc.env:"):
            name, _, default = expr[len("oc.env:"):].partition(",")
            got = os.environ.get(name.strip())
            if got is None:
                if not _:
                    raise WorkloadError(f"{where}: ${{{expr}}}: {name.strip()} is not set and has no default")
                got = default.strip()
            return got
        return str(_resolve(root, _lookup(root, expr, where), where, depth + 1))

    whole = REF.fullmatch(value)
    if whole and not whole.group(1).startswith("oc.env:"):
        return _resolve(root, _lookup(root, whole.group(1).strip(), where), where, depth + 1)
    return REF.sub(one, value).replace(ESCAPED, "${")


def load(path):
    """The workload file, merged over its defaults and resolved."""
    data = _read(path)
    if data.get("version") == 2:
        data = _chain(path, data, ())
        data.pop("defaults", None)
        data = _resolve(data, data, os.path.basename(path))
        bench = data.get("benchmark")
        if isinstance(bench, dict):
            data["benchmark"] = [dict(v or {}, **({} if k == "default" else {"kind": k})) for k, v in bench.items()]
    if data.get("site"):
        raise WorkloadError(f"{path}: 'site:' is not read -- site facts belong to the run (site profile, context)")
    if data.get("recipes") is not None and not isinstance(data["recipes"], dict):
        raise WorkloadError(f"{path}: recipes: is a map of recipe name to recipe (as models.yaml)")
    conns = data.get("connectors")
    if conns is not None and not (isinstance(conns, dict) and all(isinstance(v, dict) for v in conns.values())):
        raise WorkloadError(f"{path}: connectors: is a map of connector name to KEY: VALUE")
    if data.get("topology") is not None and not isinstance(data["topology"], str):
        raise WorkloadError(f"{path}: topology: is a spec file path (as MAD_TOPOLOGY), relative to scripts/")
    if (data.get("model") or {}).get("serve"):
        raise WorkloadError(f"{path}: model.serve is not applied by any launcher; put flags in models.yaml")
    return data


def env_of(data, kind=None):
    """The environment the file gives (as strings): model.env, then the kind-less benchmark entries,
    then those of `kind` (default: the file's own BENCHMARK_SCRIPT, from model.env or a kind-less entry)."""
    entries = data.get("benchmark") or []
    if isinstance(entries, dict):
        entries = [entries]
    common = [e for e in entries if isinstance(e, dict) and not e.get("kind")]
    if kind is None:
        kind = ((data.get("model") or {}).get("env") or {}).get("BENCHMARK_SCRIPT")
        for e in common:
            kind = (e.get("env") or {}).get("BENCHMARK_SCRIPT", kind)
    out = dict((data.get("model") or {}).get("env") or {})
    for e in common + [e for e in entries if isinstance(e, dict) and kind and e.get("kind") == kind]:
        out.update(e.get("env") or {})
    return {str(k): str(v) for k, v in out.items()}


def write_sections(data, out):
    """Write the file's recipes and connectors where a launcher reads them; {name: value} of
    those present: recipes (a models.yaml path), connectors (a directory), topology."""
    got = {}
    if data.get("recipes"):
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "models.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(data["recipes"], f, sort_keys=False, width=1 << 16)
        got["recipes"] = path
    if data.get("connectors"):
        d = os.path.join(out, "connectors")
        os.makedirs(d, exist_ok=True)
        for name, env in data["connectors"].items():
            with open(os.path.join(d, f"{name}.env"), "w") as f:
                for k, v in env.items():
                    f.write(f"{k}={'' if v is None else v}\n")
        got["connectors"] = d
    if data.get("topology"):
        got["topology"] = data["topology"]
    return got


DEFAULT_FILENAME = "mad-config.yaml"


def find(mad_config, script_dir):
    """The card's workload file, as madengine locates it: MAD_CONFIG (relative to the card's script
    directory), else a DEFAULT_FILENAME beside the script, else None."""
    if mad_config:
        return mad_config if os.path.isabs(mad_config) else os.path.join(script_dir, mad_config)
    sibling = os.path.join(script_dir, DEFAULT_FILENAME)
    return sibling if os.path.isfile(sibling) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["env", "show", "find", "sections"])
    ap.add_argument("file", help="the workload file (find: the card's MAD_CONFIG, may be empty)")
    ap.add_argument("dir", nargs="?", default=".", help="find: the card's script directory")
    ap.add_argument("--kind", default=None, help="the benchmark selected (BENCHMARK_SCRIPT) when the run set one")
    ap.add_argument("--out", help="sections: where to write them")
    a = ap.parse_args(argv)
    if a.cmd == "find":
        path = find(a.file, a.dir)
        if path:
            print(path)
        return 0
    try:
        data = load(a.file)
    except WorkloadError as e:
        print(f"workload: {e}", file=sys.stderr)
        return 1
    if a.cmd == "show":
        print(json.dumps(data, indent=1))
    elif a.cmd == "sections":
        if not a.out:
            print("workload: sections needs --out", file=sys.stderr)
            return 2
        for k, v in write_sections(data, a.out).items():
            sys.stdout.write(f"{k}={v}\0")
    else:
        for k, v in env_of(data, a.kind or None).items():
            sys.stdout.write(f"{k}={v}\0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
