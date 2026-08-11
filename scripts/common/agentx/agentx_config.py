#!/usr/bin/env python3
"""AgentX suite config loader.

Parses the suite config (agentic.yaml) and the per-workload profile files, then
yields fully-resolved, per-workload parameter sets to the bash suite driver.

The config carries a `serving:` block (one served endpoint for the whole run), a
`run:` block (default concurrency/duration), and a `workloads:` LIST. Each
workload entry is either:
  - source: profile  -> carries the distribution params inline, or `preset: caseA`
                        to inherit scripts/common/agentx/profiles/caseA.yaml
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
    one entry (a 1-entry list). With no --config, caseA/caseB/inferencex are
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


def _resolve_profile_entry(entry):
    """Return a full profile dict for a source=profile workload entry."""
    profile = {}
    preset = entry.get("preset")
    if preset:
        profile.update(load_profile_file(os.path.join(PROFILES_DIR, f"{preset}.yaml")))
    for k, v in entry.items():
        if k in ("source", "preset") or k in _RUN_KEYS:
            continue
        profile[k] = v
    profile.setdefault("name", entry.get("name", preset or "workload"))
    return profile


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
        src = entry.get("source", "profile")
        wl = {
            "name": entry.get("name", "workload"),
            "source": src,
            "concurrency": _norm_concurrency(entry.get("concurrency")),
            "duration": entry.get("duration"),
        }
        if src == "profile":
            prof = _resolve_profile_entry(entry)
            wl["profile"] = prof
            wl["isl_tail"] = _isl_tail(prof)
        elif src == "hf":
            wl["loader"] = entry.get("loader", "")
            wl["isl_tail"] = 0
        else:
            raise ValueError(f"workload '{wl['name']}': unknown source '{src}'")
        workloads.append(wl)

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
    out = [
        f"WL_NAME={_sh(wl['name'])}",
        f"WL_SOURCE={_sh(wl['source'])}",
        f"WL_CONCURRENCY={_sh(conc)}",
        f"WL_DURATION={_sh(dur)}",
        f"WL_ISL_TAIL={_sh(wl.get('isl_tail', 0))}",
    ]
    if wl["source"] == "hf":
        out.append(f"WL_LOADER={_sh(wl.get('loader', ''))}")
        out.append("WL_PROFILE_FILE=''")
    else:
        out.append("WL_LOADER=''")
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
