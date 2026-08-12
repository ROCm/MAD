#!/usr/bin/env python3
"""AgentX suite config loader.

Parses the suite config (agentic.yaml) and the per-workload profile files, then
yields fully-resolved, per-workload parameter sets to the bash suite driver.

The config carries a `serving:` block (one served endpoint for the whole run), a
`run:` block (default concurrency/duration), and a `workloads:` LIST. Each
workload entry is either:
  - source: profile  -> carries the distribution params inline, or `preset: conformance_256k`
                        to inherit scripts/common/agentx/profiles/conformance_256k.yaml
  - source: hf       -> carries a `loader` name (an aiperf --public-dataset id)

Design notes:
  * Runs in the aiperf venv. Uses PyYAML if importable; otherwise falls back to a
    small pure-stdlib parser for the restricted YAML subset the shipped files use
    (block maps/seqs, inline [..]/{..} flows, scalars, comments). No hard third-
    party dependency. Set AGENTX_YAML_FALLBACK=1 to force the fallback (tests).
  * Environment variables OVERRIDE file values:
        MODEL -> serving.model            MAX_MODEL_LEN -> serving.max_model_len
        AGENTIC_PORT -> serving.port       AGENTIC_SERVER_METRICS -> serving.server_metrics
        AGENTIC_CONC -> run.concurrency    DURATION -> run.duration
  * Single-workload shorthand: AGENTIC_WORKLOAD=<name> restricts the run to that
    one entry (a 1-entry list). With no --config, conformance_256k/conformance_512k/inferencex are
    synthesized from the shipped presets so the shorthand works standalone.

CLI:
  agentx_config.py --profile <file> --emit-json
       Resolve a single profile file (yaml/json) and print it as JSON.
  agentx_config.py --config <file> --emit-config-shell
       Print SUITE_* globals + SUITE_WORKLOAD_NAMES (eval-able in bash).
  agentx_config.py --config <file> --workload <name> [--profile-out P] --emit-workload-shell
       Resolve one workload; print WL_* (eval-able). For source=profile, write the
       resolved profile JSON to P (for gen/verify).
  agentx_config.py --config <file> --dump-json
       Print the fully-resolved config (serving/run/workloads) as JSON.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(HERE, "profiles")

_RUN_KEYS = ("concurrency", "duration")
# Keys that steer resolution / run knobs but are NOT part of a generator profile
# dict (so they are stripped when building the source=profile profile JSON).
_CONTROL_KEYS = ("source", "preset", "loader", "filter",
                 "num_dataset_entries", "trajectory") + _RUN_KEYS


# --------------------------------------------------------------------------
# YAML loading: PyYAML if available, else a small restricted-subset fallback.
# --------------------------------------------------------------------------
def _yaml_load(text):
    if os.environ.get("AGENTX_YAML_FALLBACK", "") != "1":
        try:
            import yaml  # type: ignore
            return yaml.safe_load(text)
        except ImportError:
            pass
    return _fallback_load(text)


def _scalar(s):
    s = s.strip()
    if s == "":
        return None
    if (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"):
        return s[1:-1]
    low = s.lower()
    if low in ("null", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _split_top(s):
    """Split a flow-collection body on top-level commas (respects nesting/quotes)."""
    parts = []
    depth = 0
    inq = None
    cur = ""
    for ch in s:
        if inq:
            cur += ch
            if ch == inq:
                inq = None
        elif ch in "\"'":
            inq = ch
            cur += ch
        elif ch in "[{":
            depth += 1
            cur += ch
        elif ch in "]}":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip() != "" or parts:
        parts.append(cur)
    return parts


def _parse_node(s):
    s = s.strip()
    if s.startswith("["):
        inner = s[1:-1].strip()
        return [] if inner == "" else [_parse_node(p) for p in _split_top(inner)]
    if s.startswith("{"):
        inner = s[1:-1].strip()
        d = {}
        if inner:
            for p in _split_top(inner):
                k, _, v = p.partition(":")
                d[str(_scalar(k))] = _parse_node(v)
        return d
    return _scalar(s)


def _strip_comment(line):
    inq = None
    out = ""
    for i, ch in enumerate(line):
        if inq:
            out += ch
            if ch == inq:
                inq = None
        elif ch in "\"'":
            inq = ch
            out += ch
        elif ch == "#" and (i == 0 or line[i - 1] == " "):
            break
        else:
            out += ch
    return out


def _parse_block(lines, i, indent):
    _, content = lines[i]
    if content.startswith("-"):
        seq = []
        while i < len(lines):
            ind, c = lines[i]
            if ind != indent or not c.startswith("-"):
                break
            rest = c[1:].lstrip()
            item_indent = indent + (len(c) - len(c[1:].lstrip()))
            if rest == "":
                i += 1
                if i < len(lines) and lines[i][0] > indent:
                    val, i = _parse_block(lines, i, lines[i][0])
                else:
                    val = None
                seq.append(val)
            else:
                sub = [(item_indent, rest)]
                i += 1
                while i < len(lines) and lines[i][0] > indent:
                    sub.append(lines[i])
                    i += 1
                val, _ = _parse_block(sub, 0, item_indent)
                seq.append(val)
        return seq, i
    d = {}
    while i < len(lines):
        ind, c = lines[i]
        if ind != indent:
            break
        key, _, rest = c.partition(":")
        key = str(_scalar(key.strip()))
        rest = rest.strip()
        if rest == "":
            i += 1
            if i < len(lines) and lines[i][0] > indent:
                val, i = _parse_block(lines, i, lines[i][0])
            else:
                val = None
            d[key] = val
        else:
            d[key] = _parse_node(rest)
            i += 1
    return d, i


def _fallback_load(text):
    lines = []
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if line.strip() == "":
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((indent, line.strip()))
    if not lines:
        return None
    val, _ = _parse_block(lines, 0, lines[0][0])
    return val


def _load_file(path):
    with open(path) as f:
        text = f.read()
    if path.endswith(".json"):
        return json.loads(text)
    return _yaml_load(text)


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------
def load_profile_file(path):
    return _load_file(path)


def _merge_preset(entry, _visited):
    """Return `entry` merged over its `preset:` chain (entry keys win).

    Works for any source: a profiles/<preset>.yaml may declare distribution
    params (source=profile), an hf loader + Tier 1/Tier 2 knobs (source=hf),
    and/or run knobs (concurrency/duration). Circular references raise.
    """
    name = entry.get("preset")
    if not name:
        return dict(entry)
    if name in _visited:
        raise ValueError(f"circular preset: {name}")
    _visited.add(name)
    base = _merge_preset(
        load_profile_file(os.path.join(PROFILES_DIR, f"{name}.yaml")) or {}, _visited)
    for k, v in entry.items():
        if k == "preset":
            continue
        base[k] = v
    return base


def _profile_from_merged(merged, name):
    """Build a generator profile dict from a merged workload dict."""
    profile = {k: v for k, v in merged.items() if k not in _CONTROL_KEYS}
    profile.setdefault("name", name)
    return profile


def _validate_tier1(name, nde, tmin, tmax):
    if nde is not None and int(nde) < 1:
        raise ValueError(f"workload '{name}': num_dataset_entries must be >= 1")
    if tmin is not None or tmax is not None:
        lo = 0.0 if tmin is None else float(tmin)
        hi = 1.0 if tmax is None else float(tmax)
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(
                f"workload '{name}': trajectory requires 0.0 <= min <= max <= 1.0 "
                f"(got min={tmin}, max={tmax})")


def _hf_isl_tail(loader):
    """ISL tail (max input tokens) for an hf loader, for context gating.

    Option A explicit matching: the `_256k` suffix is definitional and checked
    FIRST (wins over the date substring); the full-corpus loaders use a
    conservative HIGH default. The gate caps --max-context-length at the served
    window, so over-estimation only over-WARNs. Override with AGENTIC_HF_ISL_TAIL.
    """
    env = os.environ.get("AGENTIC_HF_ISL_TAIL")
    if env:
        return int(env)
    if loader.endswith("_256k"):
        return 262144            # definitional: 256k cap
    if "062126" in loader or "061526" in loader:
        # full corpus: conservative ~1M. Measured max per-turn ISL is 989824 for
        # both 062126 and 061526 (in-container, all sessions); 1048576 (2^20) is a
        # safe over-estimate and rounds to the same power-of-two window as 989824.
        return 1048576
    return 1048576               # unknown loader -> conservative default (errs to WARN)


def _resolve_workload_entry(entry, _visited=None):
    """Merge a workload entry with its preset (any source) and resolve it."""
    if _visited is None:
        _visited = set()
    merged = _merge_preset(entry, _visited)
    src = merged.get("source", "profile")
    name = entry.get("name") or entry.get("preset") or "workload"
    nde = merged.get("num_dataset_entries")
    traj = merged.get("trajectory") or {}
    tmin = traj.get("min")
    tmax = traj.get("max")
    _validate_tier1(name, nde, tmin, tmax)
    wl = {
        "name": name,
        "source": src,
        "concurrency": _norm_concurrency(merged.get("concurrency")),
        "duration": merged.get("duration"),
        "num_dataset_entries": nde,
        "traj_min": tmin,
        "traj_max": tmax,
    }
    if src == "profile":
        prof = _profile_from_merged(merged, name)
        wl["profile"] = prof
        wl["isl_tail"] = _isl_tail(prof)
    elif src == "hf":
        wl["loader"] = merged.get("loader", "")
        wl["isl_tail"] = _hf_isl_tail(wl["loader"])
        wl["filter"] = merged.get("filter") or {}
    else:
        raise ValueError(f"workload '{name}': unknown source '{src}'")
    return wl


def _isl_tail(profile):
    clamps = profile.get("clamps", {}) or {}
    if "isl" in clamps:
        return int(clamps["isl"][1])
    return int(profile["isl_p"][2])


def _norm_concurrency(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return " ".join(str(int(x)) for x in v)
    return " ".join(str(int(x)) for x in str(v).replace(",", " ").split())


def resolve_config(config):
    serving = dict(config.get("serving", {}) or {})
    run = dict(config.get("run", {}) or {})

    env = os.environ
    if env.get("MODEL"):
        serving["model"] = env["MODEL"]
    if env.get("MAX_MODEL_LEN"):
        serving["max_model_len"] = int(env["MAX_MODEL_LEN"])
    if env.get("AGENTIC_PORT"):
        serving["port"] = env["AGENTIC_PORT"]
    if env.get("AGENTIC_SERVER_METRICS"):
        serving["server_metrics"] = env["AGENTIC_SERVER_METRICS"]
    if env.get("AGENTIC_CONC"):
        run["concurrency"] = env["AGENTIC_CONC"]
    if env.get("DURATION"):
        run["duration"] = int(env["DURATION"])

    serving.setdefault("model", "auto")
    serving.setdefault("max_model_len", 0)
    serving.setdefault("port", "auto")
    serving.setdefault("server_metrics", "auto")
    run.setdefault("concurrency", 16)
    run.setdefault("duration", 900)

    workloads = []
    for entry in config.get("workloads", []) or []:
        workloads.append(_resolve_workload_entry(entry))

    want = env.get("AGENTIC_WORKLOAD")
    if want:
        filtered = [w for w in workloads if w["name"] == want]
        if not filtered:
            raise ValueError(f"AGENTIC_WORKLOAD='{want}' not found in workloads list")
        workloads = filtered

    return {"serving": serving, "run": run, "workloads": workloads}


# Presets used for the config-less single-workload shorthand.
_HF_PRESETS = {"inferencex": "semianalysis_cc_traces_weka_062126_256k"}


def _synth_config_from_env():
    want = os.environ.get("AGENTIC_WORKLOAD")
    if not want:
        raise SystemExit("no --config and no AGENTIC_WORKLOAD set")
    if want in _HF_PRESETS:
        wl = {"name": want, "source": "hf", "loader": _HF_PRESETS[want]}
    else:
        wl = {"name": want, "source": "profile", "preset": want}
    return {"serving": {}, "run": {}, "workloads": [wl]}


def _load_config_arg(path):
    if path:
        return resolve_config(_load_file(path))
    return resolve_config(_synth_config_from_env())


# --------------------------------------------------------------------------
# Shell emitters
# --------------------------------------------------------------------------
def _sh(v):
    return "'" + str(v).replace("'", "'\\''") + "'"


def emit_config_shell(resolved):
    s = resolved["serving"]
    r = resolved["run"]
    names = [w["name"] for w in resolved["workloads"]]
    out = []
    out.append(f"SUITE_SERVING_MODEL={_sh(s['model'])}")
    out.append(f"SUITE_MAX_MODEL_LEN={_sh(s['max_model_len'])}")
    out.append(f"SUITE_PORT={_sh(s['port'])}")
    out.append(f"SUITE_SERVER_METRICS={_sh(s['server_metrics'])}")
    out.append(f"SUITE_CONCURRENCY={_sh(_norm_concurrency(r['concurrency']))}")
    out.append(f"SUITE_DURATION={_sh(r['duration'])}")
    out.append(f"SUITE_WORKLOAD_NAMES={_sh(' '.join(names))}")
    return "\n".join(out)


def emit_workload_shell(resolved, name, profile_out):
    wl = next((w for w in resolved["workloads"] if w["name"] == name), None)
    if wl is None:
        raise SystemExit(f"workload '{name}' not in resolved config")
    r = resolved["run"]
    conc = wl["concurrency"] or _norm_concurrency(r["concurrency"])
    dur = wl["duration"] if wl["duration"] is not None else r["duration"]
    def _opt(v):
        return "" if v is None else v

    out = [
        f"WL_NAME={_sh(wl['name'])}",
        f"WL_SOURCE={_sh(wl['source'])}",
        f"WL_CONCURRENCY={_sh(conc)}",
        f"WL_DURATION={_sh(dur)}",
        f"WL_ISL_TAIL={_sh(wl.get('isl_tail', 0))}",
        f"WL_NUM_DATASET_ENTRIES={_sh(_opt(wl.get('num_dataset_entries')))}",
        f"WL_TRAJ_MIN={_sh(_opt(wl.get('traj_min')))}",
        f"WL_TRAJ_MAX={_sh(_opt(wl.get('traj_max')))}",
    ]
    if wl["source"] == "hf":
        out.append(f"WL_LOADER={_sh(wl.get('loader', ''))}")
        f = wl.get("filter") or {}
        out.append(f"WL_FILTER_MAX_ISL={_sh(_opt(f.get('max_isl')))}")
        out.append(f"WL_FILTER_MAX_TURNS={_sh(_opt(f.get('max_turns')))}")
        out.append(f"WL_FILTER_SAMPLE={_sh(_opt(f.get('sample')))}")
        out.append("WL_PROFILE_FILE=''")
    else:
        out.append("WL_LOADER=''")
        out.append("WL_FILTER_MAX_ISL=''")
        out.append("WL_FILTER_MAX_TURNS=''")
        out.append("WL_FILTER_SAMPLE=''")
        if profile_out:
            with open(profile_out, "w") as f:
                json.dump(wl["profile"], f)
            out.append(f"WL_PROFILE_FILE={_sh(profile_out)}")
        else:
            out.append("WL_PROFILE_FILE=''")
        out.append(f"WL_MODEL_TAG={_sh(wl['profile'].get('model_tag', ''))}")
    return "\n".join(out)


def main(argv):
    config_path = None
    profile_path = None
    workload = None
    profile_out = None
    mode = None
    it = iter(argv)
    for a in it:
        if a == "--config":
            config_path = next(it)
        elif a == "--profile":
            profile_path = next(it)
        elif a == "--workload":
            workload = next(it)
        elif a == "--profile-out":
            profile_out = next(it)
        elif a in ("--emit-json", "--emit-config-shell", "--emit-workload-shell", "--dump-json"):
            mode = a
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            sys.stderr.write(f"[agentx_config] unknown arg: {a}\n")
            return 2

    if mode == "--emit-json":
        if not profile_path:
            sys.stderr.write("--emit-json requires --profile\n")
            return 2
        prof = _load_file(profile_path)
        json.dump(prof, sys.stdout)
        sys.stdout.write("\n")
        return 0

    resolved = _load_config_arg(config_path)

    if mode == "--dump-json":
        json.dump(resolved, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if mode == "--emit-config-shell":
        print(emit_config_shell(resolved))
        return 0
    if mode == "--emit-workload-shell":
        if not workload:
            sys.stderr.write("--emit-workload-shell requires --workload\n")
            return 2
        print(emit_workload_shell(resolved, workload, profile_out))
        return 0

    sys.stderr.write("no mode selected (see --help)\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
