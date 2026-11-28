#!/usr/bin/env python3
"""Is the model this run script runs a PyTorch program?  (debug-numeric step 1, first check)

    framework_check.py <run script> [--source <model.py> ...]

Every rung of debug-numeric hooks PyTorch: the capture is a __torch_dispatch__ mode, a
capsule is torch.save'd ATen ops, the replay re-dispatches them.  A JAX, TensorFlow or
other non-PyTorch model gives no capsule and no replay -- only hours of empty waves --
so the skill must refuse it before anything is created.

Static: reads files, imports nothing, runs no GPU work (works on a login node).

Evidence, strongest first; the verdict comes from the first level with any hit:

  1. entry     the Python files the run script executes (python X.py, torchrun X.py,
               python -m pkg.mod resolved to a file, a .py run script itself, --source):
               their own imports.
  2. launcher  the run script and the shell scripts it calls: torchrun /
               torch.distributed.run, `python -m <known module>`, KERAS_BACKEND, the
               container image tag (rocm/pytorch, rocm/jax, rocm/tensorflow), inline
               `python -c 'import ...'`.
  3. tree      imports anywhere under the entry files' directories (no venvs, .git,
               artifacts) -- a fallback for launchers that hide their entry point.

Last line, always:

  FRAMEWORK=pytorch        VERDICT=RUN   exit 0
  FRAMEWORK=<jax|tensorflow|...> VERDICT=SKIP  exit 3   not PyTorch: do not apply the skill
  FRAMEWORK=mixed|unknown  VERDICT=ASK   exit 2   ask the user which framework runs the
                                                  model's compute (it may be hidden in an
                                                  image, or torch only loads the data)
"""
import argparse
import os
import re
import subprocess
import sys

# Import roots -> framework.  Keras 3 runs on any backend: KERAS_BACKEND decides.
IMPORTS = {
    "pytorch": ("torch", "torchvision", "torchaudio", "torchtitan", "torchrec", "lightning",
                "pytorch_lightning", "fairseq", "megatron", "deepspeed", "fbgemm_gpu",
                "torchtune", "accelerate"),
    "jax": ("jax", "jaxlib", "flax", "optax", "haiku", "equinox", "maxtext", "MaxText",
            "paxml", "praxis", "t5x", "orbax"),
    "tensorflow": ("tensorflow", "tf_keras"),
    "paddle": ("paddle",),
    "mxnet": ("mxnet",),
    "mindspore": ("mindspore",),
    "oneflow": ("oneflow",),
}
IMAGE_HINTS = (("pytorch", r"pytorch|torch"), ("jax", r"\bjax|maxtext"),
               ("tensorflow", r"tensorflow"))
SKIP_DIRS = {".git", "__pycache__", "artifacts", "node_modules", "site-packages"}

_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+(?:\s*,\s*[\w.]+)*))",
                        re.M)
_INLINE_RE = re.compile(r"\b(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))")


def framework_of(module):
    root = module.split(".")[0]
    for fw, roots in IMPORTS.items():
        if root in roots:
            return fw
    if root == "keras":
        return "keras"
    return None


def imports_in(text, inline=False):
    """{framework: [module, ...]} for the imports in `text`."""
    found = {}
    rx = _INLINE_RE if inline else _IMPORT_RE
    for m in rx.finditer(text):
        mods = [m.group(1)] if m.group(1) else [x.strip() for x in m.group(2).split(",")]
        for mod in mods:
            fw = framework_of(mod)
            if fw:
                found.setdefault(fw, [])
                if mod not in found[fw]:
                    found[fw].append(mod)
    return found


def read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def git_root(path):
    try:
        return subprocess.run(["git", "-C", os.path.dirname(path), "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:                                   # noqa: BLE001
        return ""


def find_file(name, roots):
    """A file named like `name` (a relative path or a basename) under `roots`."""
    for r in roots:
        p = os.path.normpath(os.path.join(r, name))
        if os.path.isfile(p):
            return p
    # By basename -- but a name with directories (torchtitan/train.py) must match its
    # literal tail: an unrelated train.py elsewhere in the tree is not that file.
    base = os.path.basename(name)
    tail = [c for c in os.path.normpath(name).split(os.sep)[:-1] if c and c not in (".", "..")]
    tail = tail[len(tail) - next((i for i, c in enumerate(reversed(tail)) if "$" in c), len(tail)):]
    for r in roots:
        for d, dirs, files in os.walk(r):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith(".venv")]
            if d[len(r):].count(os.sep) > 3:
                dirs[:] = []
            if base in files and (not tail or d.split(os.sep)[-len(tail):] == tail):
                return os.path.join(d, base)
    return None


def shell_closure(run_script, roots, depth=2):
    """The run script plus the shell scripts it calls (bash x.sh, ./x.sh, source x.sh)."""
    seen, todo = [], [run_script]
    for _ in range(depth + 1):
        nxt = []
        for s in todo:
            if s in seen:
                continue
            seen.append(s)
            for name in re.findall(r"(?:\bbash|\bsh|\bsource|^\s*\.|\s\./)\s*[\"']?([\w./${}-]+\.sh)",
                                   read(s), re.M):
                p = find_file(clean_path(name), roots)
                if p and p not in seen:
                    nxt.append(p)
        todo = nxt
    return seen


def clean_path(token):
    """Drop shell variables and quotes: "$HERE/train.py" -> train.py."""
    token = token.strip("\"'")
    token = re.sub(r"\$\{?\w+\}?/?|\$\([^)]*\)/?", "", token)
    return token.lstrip("/") if token.startswith("/") and not os.path.isfile(token) else token


def entry_points(shell_texts, roots):
    """(python files, -m modules) the shell scripts execute."""
    files, modules = [], []
    for t in shell_texts:
        # python X.py / torchrun ... X.py / accelerate launch X.py / deepspeed X.py
        for tok in re.findall(r"([\w./${}()\"'-]*\.py)\b", t):
            p = find_file(clean_path(tok), roots)
            if p and p not in files:
                files.append(p)
        for m in re.findall(r"-m\s+([\w][\w.]*)", t):
            if m not in modules:
                modules.append(m)
    return files, modules


def module_file(mod, roots):
    rel = mod.replace(".", os.sep)
    for r in roots:
        for cand in (rel + ".py", os.path.join(rel, "__main__.py")):
            p = os.path.join(r, cand)
            if os.path.isfile(p):
                return p
    return None


def tree_imports(dirs):
    found = {}
    for top in dirs:
        for d, sub, files in os.walk(top):
            sub[:] = [x for x in sub if x not in SKIP_DIRS and not x.startswith(".venv")]
            for f in files:
                if f.endswith(".py"):
                    for fw, mods in imports_in(read(os.path.join(d, f))).items():
                        found.setdefault(fw, set()).update(mods)
    return {k: sorted(v) for k, v in found.items()}


def resolve_keras(found, env_texts):
    """keras imports count for the backend KERAS_BACKEND names (default: tensorflow)."""
    if "keras" not in found:
        return found
    m = None
    for t in env_texts:
        m = re.search(r"KERAS_BACKEND\s*=\s*[\"']?(\w+)", t) or m
    name = m.group(1) if m else "tensorflow"
    backend = {"torch": "pytorch", "jax": "jax"}.get(name, "tensorflow")
    found = dict(found)
    found.setdefault(backend, []).append("keras(%s backend)" % name)
    del found["keras"]
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_script")
    ap.add_argument("--source", action="append", default=[], help="model file(s), if known")
    a = ap.parse_args()

    rs = os.path.realpath(a.run_script)
    if not os.path.isfile(rs):
        print("FRAMEWORK=unknown VERDICT=ASK  (no such file: %s)" % a.run_script)
        return 2
    roots = [os.path.dirname(rs)]
    gr = git_root(rs)
    if gr and gr not in roots:
        roots.append(gr)

    first = read(rs).split("\n")[0]
    is_py = rs.endswith(".py") or (first.startswith("#!") and "python" in first)
    shells = [] if is_py else shell_closure(rs, roots)
    shell_texts = [read(s) for s in shells]
    files, modules = ([rs], []) if is_py else entry_points(shell_texts, roots)
    for s in a.source:
        p = os.path.realpath(s)
        if p not in files:
            files.append(p)
    unresolved = []
    for m in modules:
        p = module_file(m, roots)
        if p:
            files.append(p)
        else:
            unresolved.append(m)

    print("framework_check: run=%s" % rs)
    print("  shell : %s" % (", ".join(os.path.relpath(s, roots[-1]) for s in shells) or "-"))
    print("  entry : %s" % (", ".join(os.path.relpath(f, roots[-1]) for f in files) or "-"))

    # 1. entry files' own imports
    entry = {}
    for f in files:
        for fw, mods in imports_in(read(f)).items():
            entry.setdefault(fw, [])
            entry[fw] += [x for x in mods if x not in entry[fw]]
    entry = resolve_keras(entry, shell_texts + [read(f) for f in files])

    # 2. launcher evidence
    launch = {}

    def add(fw, why):
        launch.setdefault(fw, [])
        if why not in launch[fw]:
            launch[fw].append(why)
    for t in shell_texts:
        if re.search(r"\btorchrun\b|torch\.distributed\.(run|launch)", t):
            add("pytorch", "torchrun")
        for fw, mods in imports_in(t, inline=True).items():
            for x in mods:
                add(fw, "inline import %s" % x)
        for img in re.findall(r"(?:IMAGE\w*=\$\{\w+:-|IMAGE\w*=|docker://|docker run[^\n]*?\s)"
                              r"([\w./-]+:[\w.-]+)", t):
            for fw, rx in IMAGE_HINTS:
                if re.search(rx, img, re.I):
                    add(fw, "image %s" % img)
                    break
    for m in unresolved:
        fw = framework_of(m)
        if fw:
            add(fw, "python -m %s" % m)
    launch = resolve_keras(launch, shell_texts)

    # 3. the tree around the entry files
    tree = {}
    if not entry and not launch:
        tree = resolve_keras(tree_imports(sorted({os.path.dirname(f) for f in files}
                                                  or {os.path.dirname(rs)})), shell_texts)

    for name, ev in (("entry", entry), ("launcher", launch), ("tree", tree)):
        if ev:
            print("  %-8s: %s" % (name, "; ".join("%s (%s)" % (fw, ", ".join(v[:4]))
                                                  for fw, v in sorted(ev.items()))))
    level = entry or launch or tree
    fws = sorted(level)
    if not fws:
        print("  no framework import found -- the model may run inside an image or a "
              "package this script cannot read")
        print("FRAMEWORK=unknown VERDICT=ASK")
        return 2
    if fws == ["pytorch"]:
        print("FRAMEWORK=pytorch VERDICT=RUN")
        return 0
    if "pytorch" in fws:
        print("FRAMEWORK=mixed(%s) VERDICT=ASK" % "+".join(fws))
        return 2
    print("FRAMEWORK=%s VERDICT=SKIP" % "+".join(fws))
    return 3


if __name__ == "__main__":
    sys.exit(main())
