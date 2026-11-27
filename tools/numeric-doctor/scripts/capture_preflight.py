#!/usr/bin/env python3
"""Can this run script see a NaN, and can a capture be wired into its model?
One report instead of a dozen greps (debug-numeric steps 1 and 4.1).

    capture_preflight.py <run script> [--source <model.py>] [--signal REGEX]
                         [--root DIR] [--venv DIR]

Checks, each one line plus where it looked:

  IMAGE    the image the run script uses, per the profile's RUNTIME (ND_ENV):
           apptainer -> its SIF exists (MISSING = get it now); docker/podman ->
           nd pulls/loads it; a run script that BUILDS its image (Dockerfile,
           .def) -> build it once first (env.md *Images*).
  SIGNAL   every alternative of the run script's NaN signal (from --signal, else
           the patterns its grep/zgrep -E calls use) is searched for in the model
           repo and its venv.  NOT RAISED ANYWHERE = the script can never report
           that NaN: it will score NaN runs as clean.  A regex alternative is
           searched as a regex (not found = unverifiable, not an alarm); a run
           script whose greps mention no NaN gets the source's own not-finite
           messages as --signal candidates instead.  `grep -o` field extractions
           (`grep -o 'nan_request=[^ ]*'`) are not tests and are skipped; a
           pattern that also matches the field's clean value (`nan_request=-`)
           is flagged MATCHES CLEAN OUTPUT.
  MASK     nan_to_num / masked loss in the repo: a NaN that never reaches a check.
  HOOK     is a numeric-doctor capture already wired (capture_from_env /
           OpNanCapture / cap.step) -- and where.
  ENV      how the run script's env reaches the trainer (set -a + source, docker
           -e / --env-file, env -i), and subprocess env dicts in the repo that do
           not start from os.environ (they would drop NAN_* settings).
  LAZY     torch.compile / compile: true in specs -> NAN_CAPTURE_ARM_AFTER_STEPS=1.
  VLLM     a vLLM engine in the source: is it in-process (the capture sees the
           model only then), tensor-parallel, a server; its compiled default.
  SGLANG   an sglang.Engine in the source: the model runs in its scheduler
           subprocess -- the capture goes there (scripts/sglang_capture).
  LOOP     find_capture_points.py's top loop for --source (where cap.step() goes).

Static: reads files, imports nothing, runs no GPU work.
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=300).stdout
    except Exception:                                   # noqa: BLE001
        return ""


def grep(pattern, paths, fixed=False, include=("*.py", "*.sh", "*.yaml", "*.yml", "*.env"),
         venv_ok=False):
    """[(file, line, text)] for `pattern` under `paths` (grep -rn, skipping run dirs)."""
    # -H: with a single file argument grep omits the file name, and every hit was dropped below
    args = ["grep", "-rnIH", "-F" if fixed else "-E", "--exclude-dir=.git",
            "--exclude-dir=__pycache__", "--exclude-dir=qualification-runs*",
            "--exclude-dir=artifacts",
            # a framework's unit tests raise and compile on purpose: torchtitan's tests/ were the
            # SIGNAL / LAZY candidates of a qwen3 MoE preflight (2026-11-20)
            "--exclude-dir=tests", "--exclude-dir=test", "--exclude-dir=unit_tests"]
    if not venv_ok:
        args.append("--exclude-dir=.venv*")      # repo-scoped: libraries are noise here
    args += ["--include=%s" % i for i in include]
    out = []
    for line in sh(args + ["--", pattern] + [p for p in paths if p]).splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3:
            out.append((parts[0], parts[1], parts[2].strip()))
    return out


def short(hits, n=3, root=""):
    return "; ".join("%s:%s" % (os.path.relpath(f, root) if root else f, ln)
                     for f, ln, _ in hits[:n]) + (" (+%d)" % (len(hits) - n) if len(hits) > n else "")


# What a signal pattern mentions when it tests for a bad value: a NaN / Inf, or a wrong finite
# value (`RESULT ... wrong=yes` is a valid signal, step4-explode.md; 2026-11-24: preflight called
# deepseek_mla_decode_wrong's `^RESULT` + `wrong=` verdict "greps no NaN line").
_SIGNAL_WORDS = r"nan|finite|\binf\b|wrong|mismatch|differ"
# a verdict field's bad value, when the run script compares the field to literals
_BAD_VALUES = ("yes", "true", "1", "nan", "wrong", "bad")


def _shell_mask(src):
    """`src` with quoted text and comments blanked (newlines kept): shell keywords (`if`, `fi`)
    found in it are code, not the words of an echo or a comment ("if a stored login...")."""
    out, i, n, q = list(src), 0, len(src), None
    while i < n:
        c = src[i]
        if q:
            if c == "\\" and q == '"' and i + 1 < n:
                out[i] = " "
                if src[i + 1] != "\n":
                    out[i + 1] = " "
                i += 2
                continue
            if c == q:
                q = None
            elif c != "\n":
                out[i] = " "
        elif c in "'\"":
            q = c
        elif c == "\\":
            i += 1
        elif c == "#" and (i == 0 or src[i - 1] in " \t\n;"):
            while i < n and src[i] != "\n":
                out[i] = " "
                i += 1
            continue
        i += 1
    return "".join(out)


class Arms:
    """The run script's arms (`./run.sh <arm>`: an `ARMS="..."` table, `want="..."` groups), which
    arms each line of the script belongs to (`if [ "$name" = op_issue ]; then ... else ... fi`,
    `$OUT_DIR/op_issue.log`), which arms run the model source, and which `./run.sh` runs bare.
    2026-11-24 (deepseek_sparse_mla_decode_nan): the op_issue arm's `grep -q 'nan_count=0/'`
    over the issue repro's output was taken as the signal of the serve script, whose arms print
    `Logits are not finite`.  No arm table -> every line belongs to every run (names empty)."""

    def __init__(self, src, sources):
        self.src = src
        groups = {}
        for var, body in re.findall(r"""^\s*(\w+)=["']([^"'$]*)["']""", src, re.M):
            groups.setdefault(var, body)
        names = []
        m = re.search(r"""^\s*ARMS=["']([^"']*)["']""", src, re.M)
        if m:
            names += [l.split()[0] for l in m.group(1).splitlines() if l.split()]
        for body in re.findall(r"""\bwant=["']([^"'$]*)["']""", src):
            names += body.split()
        self.names = [x for x in dict.fromkeys(names) if re.match(r"[A-Za-z_][\w-]*$", x)]
        self.default = None
        m = re.search(r"""case\s+"?\$\{1:-([\w-]+)\}"?\s+in""", src)
        if m and self.names:
            d = re.search(r"^\s*%s\)\s*want=(?:[\"']([^\"'$]*)[\"']|\$\{?(\w+)\}?)" % re.escape(m.group(1)),
                          src, re.M)
            if d:
                self.default = set((d.group(1) if d.group(1) is not None else groups.get(d.group(2), "")).split())
            else:
                self.default = None
        self.lines = src.splitlines()
        self._ctx = self._contexts() if self.names else None
        self.source_arms = set(self.names)
        runs = []
        for s in sources:
            b = re.escape(os.path.basename(s))
            runs += [i for i, l in enumerate(self.lines)
                     if re.search(r"""(python[\d.]*|torchrun)\b[^|;&]*?["'/\s]%s\b""" % b, l)]
        if self.names and runs:
            self.source_arms = set().union(*(self.of_line(i) for i in runs))

    def _contexts(self):
        """Per line: [(arm names in an enclosing if/elif condition, negated: in its else)]."""
        masked = _shell_mask(self.src).splitlines()
        stack, ctx = [], []
        for i, ml in enumerate(masked):
            line_ctx = [tuple(x) for x in stack]
            for m in re.finditer(r"(?<![\w$-])(if|elif|else|fi)(?![\w-])", ml):
                kw = m.group(1)
                if kw == "if":
                    cond = self.lines[i][m.end():]
                    cond = re.split(r"\bthen\b", cond)[0]
                    stack.append([self._named(cond), False, set()])
                elif kw == "elif" and stack:
                    seen = stack[-1][2] | (stack[-1][0] if not stack[-1][1] else set())
                    stack[-1] = [self._named(re.split(r"\bthen\b", self.lines[i][m.end():])[0]), False, seen]
                elif kw == "else" and stack:
                    seen = stack[-1][2] | (stack[-1][0] if not stack[-1][1] else set())
                    stack[-1] = [seen, True, seen]
                elif kw == "fi" and stack:
                    stack.pop()
                # the code after a keyword on the same line runs in the new context
                line_ctx = [tuple(x) for x in stack] if kw != "fi" else line_ctx
            ctx.append(line_ctx)
        return ctx

    def _named(self, text):
        return {a for a in self.names if re.search(r"(?<![\w-])%s(?![\w-])" % re.escape(a), text)}

    def of_line(self, i):
        """The arms whose run executes line i (0-based)."""
        if not self.names:
            return set()
        cur = set(self.names)
        for arms, neg, _ in self._ctx[i]:
            if arms:
                cur = cur - arms if neg else cur & arms
        on_line = self._named(self.lines[i])      # `"$OUT_DIR/op_issue.log"`
        if on_line:
            cur &= on_line
        return cur

    def label(self, arms):
        if not self.names or arms == set(self.names):
            return "all arms"
        txt = "arm%s %s" % ("s" if len(arms) > 1 else "", ",".join(a for a in self.names if a in arms))
        if self.default is not None:
            txt += " (default run)" if arms & self.default else " (not in the default run)"
        return txt


def signal_alternatives(run_script, given, sources=()):
    """The run script's signal candidates: [(alternative, is_regex, arms label, kind)], kind
    'grep' (a literal / regex alternative of a grep) or 'field' (a verdict field of a result line,
    `res=$(grep '^RESULT' ...); wrong=$(grep -o 'wrong=[^ ]*' <<< "$res"); [ "$wrong" = yes ]`
    -> `RESULT .*wrong=yes`), and [notes] on candidates dropped because they test another arm.

    `grep -E 'a|b'` is split on `|`; any other grep (`grep -q "..."`, `grep -F`, plain)
    is a single literal -- splitting it would invent alternatives that were never there.
    """
    if given:
        return [(given, True, "", "grep")], []
    src = open(run_script).read()
    arms = Arms(src, sources)
    line_of = lambda pos: src.count("\n", 0, pos)          # noqa: E731
    pats, notes, fields = [], [], []
    for m in re.finditer(r"""(?<![\w-])z?grep\b((?:\s+-[a-zA-Z0-9]+)*)\s+['"]([^'"]+)['"]""", src):
        flags, pat = m.group(1), m.group(2)
        # (?<![\w-]): `pgrep -af 'worker|torchrun'` (an idle-GPU check) is not a log test.
        # `grep -o 'nan_request=[^ ]*'` extracts a field, NaN or not: taken as the signal it
        # matched every clean RESULT line too (qwen_vllm_serve_nan, 2026-11-19).
        if "o" in re.sub(r"[^a-zA-Z]", "", flags):
            continue
        ln = line_of(m.start())
        on = arms.of_line(ln)
        if arms.names and not on & arms.source_arms:
            if re.search(_SIGNAL_WORDS, pat, re.I):
                notes.append("'%s' (run.sh:%d) tests %s, which does not run the --source -- skipped"
                             % (pat[:32], ln + 1, arms.label(on)))
            continue
        pats.append((pat, "E" in flags and "F" not in flags, arms.label(on) if arms.names else "", on))
    # verdict fields of a result line: VAR=$(grep -o 'KEY=[^ ]*' <<< "$RES") compared to a literal
    for m in re.finditer(r"""(\w+)=\$\(\s*grep\s+-o\s+['"]\s*(\w+)=[^'"]*['"]\s*<<<\s*"?\$\{?(\w+)""", src):
        var, key, res = m.groups()
        if not re.search(_SIGNAL_WORDS, key, re.I):
            continue
        ln = line_of(m.start())
        on = arms.of_line(ln)
        if arms.names and not on & arms.source_arms:
            continue
        vals = re.findall(r"""["']?\$\{?%s(?::-[^}]*)?\}?["']?\s*!?==?\s*["']?(\w+)""" % var, src)
        bad = next((v for v in _BAD_VALUES if v in vals), None)
        if not bad:
            continue
        r = re.search(r"""\b%s=\$\(\s*grep\b[^'"\n]*['"]\^?(\w+)[^'"]*['"]""" % res, src)
        prefix = (r.group(1) + " .*") if r else ""
        fields.append(("%s%s=%s" % (prefix, key, bad), True, arms.label(on) if arms.names else "", on))
    # A run script greps for more than its NaN line (exit codes, step numbers): keep the
    # patterns that mention a NaN or a wrong value, unless none does.
    pats = [x for x in pats if re.search(_SIGNAL_WORDS, x[0], re.I)] or ([] if fields else pats)
    # the default run's candidates first (`./run.sh` bare), then the other arms'
    rank = lambda x: 0 if arms.default is None or not arms.names or x[3] & arms.default else 1   # noqa: E731
    alts = []
    for p, regex, lab, on in sorted(pats, key=rank):
        for a in (p.split("|") if regex else [p]):
            a = a.strip("()^$ ") if regex else a
            if len(a) >= 6 and a not in [x[0] for x in alts]:
                alts.append((a, regex, lab, "grep"))
    for a, regex, lab, on in sorted(fields, key=rank):
        if a not in [x[0] for x in alts]:
            alts.append((a, regex, lab, "field"))
    return alts, notes


def raised_messages(scan):
    """[(file, line, message, code line)] for the source's own not-finite messages."""
    out = []
    for f, ln, text in grep(r"""["'][^"']*(not finite|non-finite|[Ii]s ?[Nn]a[Nn]|NaN detected)""", scan,
                            include=("*.py",)):
        m = re.search(r"""["']([^"']*(?:not finite|non-finite|[Ii]s ?[Nn]a[Nn]|NaN detected)[^"'{]*)""", text)
        out.append((f, ln, (m.group(1) if m else text).strip(), text))
    return out


# Values a key=value field takes when nothing went wrong.
# Not "0": a step or request index can be 0 (`nan_step=[0-9]` is a valid signal).
_CLEAN_VALUES = ("-", "None", "none", "False", "false")


def matches_clean(alt):
    """A regex alternative that also matches its own field's clean value (`nan_request=[^ ]*`
    matches `nan_request=-`): the literal prefix up to the first regex metacharacter, followed
    by each clean value.  Only for key=value / key: value shaped patterns."""
    m = re.match(r"([^\\\[\](){}*+?.|^$]*[=:]\s*)", alt)
    if not m:
        return None
    for v in _CLEAN_VALUES:
        try:
            if re.search(alt, m.group(1) + v):
                return m.group(1) + v
        except re.error:
            return None
    return None


def _dedup(hits):
    seen, out = set(), []
    for h in hits:
        k = (os.path.realpath(h[0]), h[1])
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out


def vllm_check(files, rs, root):
    """VLLM: a vLLM engine in the model source, and what the capture needs from it.
    vLLM V1 runs the engine core (and with it the model) in a subprocess unless
    VLLM_ENABLE_V1_MULTIPROCESSING=0: a capture wired into the serving loop then sees no model op.
    `files`: the model source(s) -- never the whole repo (a vendored torchtitan has RL actors that
    import vLLM without being on the run's path)."""
    serve = re.search(r"\bvllm\s+serve\b|vllm\.entrypoints", open(rs, errors="replace").read())
    uses = _dedup(grep(r"^\s*(from vllm\b|import vllm\b)", files, include=("*.py",))) if files else []
    if not uses and not serve:
        return
    if serve and not uses:
        print("VLLM     the run script starts a vLLM server (`vllm serve` / api_server): run it as is through\n"
              "           scripts/serve_run.sh --server '<the serve command, --port {port}>' --requests <the user's\n"
              "           requests.jsonl> -- the capture shims act in the server's workers (step1-ask.md *vLLM*)")
        return
    both = files + [rs]
    inproc = [h for h in _dedup(grep(r"VLLM_ENABLE_V1_MULTIPROCESSING", both, include=("*.py", "*.sh")))
              if re.search(r"""^\W*["']?0\b""", h[2].split("MULTIPROCESSING", 1)[1])]
    prof = "VLLM_ENABLE_V1_MULTIPROCESSING=0" in (_profile_var("EXTRA_ENV") or "")
    if inproc or prof:
        print("VLLM     %s: in-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0 at %s) -- the capture sees the model"
              % (short(uses, 1, root=root), short(inproc, 2, root=root) if inproc else "the profile's EXTRA_ENV"))
    else:
        print("VLLM     %s: vLLM V1 runs the model in an engine-core SUBPROCESS by default -- a capture in\n"
              "           this process sees no model op: add VLLM_ENABLE_V1_MULTIPROCESSING=0 to EXTRA_ENV (step1-ask.md *vLLM*)"
              % short(uses, 1, root=root))
    tp = [h for h in _dedup(grep(r"tensor_parallel_size\s*=|pipeline_parallel_size\s*=|--tensor-parallel-size", both,
                                 include=("*.py", "*.sh"))) if not re.search(r"=\s*1\b", h[2])]
    if tp:
        print("VLLM     parallel size set at %s: > 1 puts the model in one worker process per rank -- capture lines\n"
              "           get PYTHONPATH=$ND_REPO/scripts/vllm_capture (the capture is built inside every TP worker)\n"
              "           and a GPU-list field (0,1 for TP 2: one trial over those GPUs); pipeline parallel > 1 and\n"
              "           several nodes (vLLM's nnodes / node_rank, not Ray) too -- the manifest tags each rank's stage\n"
              "           (step1-ask.md *vLLM*)" % short(tp, 2, root=root))
    forced = _dedup(grep(r"enforce_eager\s*=\s*True", both, include=("*.py", "*.sh")))
    switch = _dedup(grep(r"enforce.eager", both, include=("*.py", "*.sh")))
    if forced:
        print("VLLM     enforce_eager=True at %s: eager ops, visible to the capture" % short(forced, 2, root=root))
    else:
        print("VLLM     default mode: torch.compile + HIP graphs -- a NaN born in compiled code reaches the capture\n"
              "           only PROPAGATED (step5n-compiled.md *vLLM*)%s"
              % ("; an enforce_eager switch at %s is C0's compile-off flag" % short(switch, 2, root=root)
                 if switch else "; C0's compile-off flag: enforce_eager=True"))


def sglang_check(files, rs, root):
    """SGLANG: an sglang.Engine in the model source. The engine always runs the model in a spawned
    scheduler subprocess, so the capture is installed there by scripts/sglang_capture (a sitecustomize
    patching Scheduler.run_batch), not wired into the user's request loop. `files`: the model source(s)."""
    server = re.search(r"sglang\.launch_server|sglang\s+serve\b", open(rs, errors="replace").read())
    uses = _dedup(grep(r"^\s*(from sglang\b|import sglang\b)", files, include=("*.py",))) if files else []
    if not uses and not server:
        return
    if server and not uses:
        print("SGLANG   the run script starts an SGLang server (sglang.launch_server): run it as is through\n"
              "           scripts/serve_run.sh --api sglang --server '<the command, --port {port}>' --requests <the\n"
              "           user's requests.jsonl>; capture lines as below (step1-ask.md *SGLang*)")
        return
    print("SGLANG   %s: sglang.Engine runs the model in a scheduler SUBPROCESS -- don't wire the request loop;\n"
          "           capture lines get PYTHONPATH=$ND_REPO/scripts/sglang_capture (step1-ask.md *SGLang*)"
          % short(uses, 1, root=root))
    both = files + [rs]
    tp = [h for h in _dedup(grep(r"tp_size\s*=|--tp(-size)?\b|dp_size\s*=|--dp(-size)?\b", both,
                                 include=("*.py", "*.sh"))) if not re.search(r"=\s*1\b|\s1\b", h[2])]
    if tp:
        print("SGLANG   parallel size set at %s: > 1 -- one capture per scheduler rank (a replica's ranks stop\n"
              "           together), one capsule dir per data-parallel replica; a GPU-list field per trial\n"
              "           (step1-ask.md *SGLang*)" % short(tp, 2, root=root))
    if (_profile_var("RUNTIME") or "docker") in ("docker", "podman") and _profile_var("DOCKER_AS_USER") != "0":
        print("SGLANG   ROCm SGLang images: aiter writes root-owned dirs (its JIT dir, /tmp/aiter_configs) -- as the\n"
              "           host uid every process fails at import: DOCKER_AS_USER=0 in the profile")


def _profile_var(name):
    """A variable of the investigation's profile (the caller's environment first, then ND_ENV's
    file, the last assignment winning as in bash), or None."""
    if os.environ.get(name):
        return os.environ[name]
    f, val = os.environ.get("ND_ENV"), None
    if f and os.path.isfile(f):
        for line in open(f, errors="replace"):
            m = re.match(r"\s*%s=[\"']?([^\"'\s#]*)" % name, line)
            val = m.group(1) if m else val
    return val or None


def _profile_runtime():
    """RUNTIME of the investigation's profile (ND_ENV, else the caller's RUNTIME), or None."""
    return _profile_var("RUNTIME")


# Dockerfile instructions that only configure: a build made of these and COPY/ADD of files
# nd mounts anyway needs no build (env-runtime.md *Images*: use its base image).
_THIN = {"FROM", "ARG", "ENV", "LABEL", "WORKDIR", "COPY", "ADD", "USER", "ENTRYPOINT", "CMD"}


def thin_dockerfile(path):
    """The base image of a Dockerfile that only sets ENV / copies files, else None."""
    try:
        lines = open(path, errors="replace").read().replace("\\\n", " ").splitlines()
    except OSError:
        return None
    instr = [l.split(None, 1) for l in lines if l.strip() and not l.lstrip().startswith("#")]
    if not instr or any(i[0].upper() not in _THIN for i in instr):
        return None
    args = dict(re.findall(r"^ARG\s+(\w+)=(\S+)", "\n".join(lines), re.M))
    base = next((i[1].split()[0] for i in instr if i[0].upper() == "FROM" and len(i) > 1), None)
    if base:
        base = re.sub(r"\$\{?(\w+)\}?", lambda m: args.get(m.group(1), m.group(0)), base)
    return base


def image_check(rs):
    """IMAGE: the image the run script needs, and what to do before any job is spent.
    Runtime-aware (env.md *Images*): a SIF is a file to check only under apptainer; a docker/
    podman image is pulled or loaded per node by nd; an image the run script BUILDS must be
    built once by hand -- that one used to surface only after a failed job."""
    text = open(rs, errors="replace").read()
    here = os.path.dirname(rs)
    rt = _profile_runtime()
    refs = re.findall(r"docker://(\S+?)['\"\s]", text)
    # IMAGE=${IMAGE:-x}, BUGGY_BASE=${BUGGY_BASE:-x}, ...: defaults that look like image refs
    refs += re.findall(r"\$\{\w*(?:IMAGE|BASE)\w*:-([^}\s]+)\}", text)
    refs = sorted(set(r for r in refs if not r.startswith("$") and ("/" in r or ":" in r)
                      and not re.search(r"\.(tar|gz|zst|sif)$", r)))
    # an image of its own: docker/podman build, or apptainer build from a .def (not a SIF
    # converted from a pulled image)
    builds_cmd = re.search(r"\b(docker|podman)\s+build\b|\b(apptainer|singularity)\s+build\b.*\.def\b", text)
    sifs = []
    for p in sorted(set(re.findall(r"""([^\s'"=(]*\.sif)\b""", text))):
        p = re.sub(r"\$\{?HERE\}?", here, p)
        if not re.search(r"[$<()]", p) and not os.path.basename(p).startswith("."):
            sifs.append(p if os.path.isabs(p) else os.path.join(here, p))
    # a build file in the folder is THIS script's image only if the script names it or has no
    # other image (a sibling run.sh / Dockerfile often builds the SIF this script just uses:
    # reported "BUILDS" for a saved-SIF script, 2026-11-11)
    files = [f for f in sorted(os.listdir(here))
             if f in ("Dockerfile", "Containerfile") or f.endswith((".def", ".Dockerfile"))]
    builds = [f for f in files if f in text or not (sifs or refs or builds_cmd)]
    for f in files:
        if f not in builds:
            print("IMAGE    (%s beside the run script is not its image -- it uses %s; ignored)"
                  % (f, os.path.basename(sifs[0]) if sifs else refs[0] if refs else "its own build"))
    uses_apptainer = re.search(r"\b(apptainer|singularity)\s+(exec|run|build|pull)", text)
    uses_oci = re.search(r"\b(docker|podman)\s+(run|build|pull|load)", text)

    if rt is None:
        print("IMAGE    (no ND_ENV / RUNTIME: runtime unknown -- run this after `nd profile`, step 1)")
    for r in refs:
        print("IMAGE    source %s" % r)
    tar = _profile_var("IMAGE_TAR")
    thin = [b for b in builds if "Dockerfile" in b or "Containerfile" in b]
    # the run script's own `docker build` of that Dockerfile is still thin; an apptainer .def build is not
    thin_base = (thin_dockerfile(os.path.join(here, thin[0])) if len(thin) == 1 and not (
        builds_cmd and re.match(r"(apptainer|singularity)", builds_cmd.group(0))) else None)
    if (builds or builds_cmd) and tar and os.path.isfile(tar):
        print("IMAGE    the run script builds its image (%s), and the profile already has it: IMAGE=%s\n"
              "           IMAGE_TAR=%s (%.1f GB) -- nothing to do now"
              % (", ".join(builds) or builds_cmd.group(0), _profile_var("IMAGE") or "?", tar,
                 os.path.getsize(tar) / 1e9))
    elif thin_base:
        print("IMAGE    the run script's %s only sets ENV / copies files onto %s: no build needed --\n"
              "           IMAGE=%s, its ENV lines as EXTRA_ENV (nd mounts the repo) -- env-runtime.md *Images*"
              % (thin[0], thin_base, thin_base))
    elif builds or builds_cmd:
        what = ", ".join(builds) or builds_cmd.group(0)
        print("IMAGE    the run script BUILDS its image (%s): build it once for RUNTIME=%s and set IMAGE=\n"
              "           (+ IMAGE_TAR= so other nodes load it) in the profile -- env-runtime.md *Images*"
              % (what, rt or "?"))
    if rt == "apptainer":
        nan = os.environ.get("NAN", os.path.join(os.environ.get("PROJ", os.getcwd()), "nan"))
        sifs = list(dict.fromkeys(sifs))
        if not sifs and not (builds or builds_cmd):
            # IMAGE tag -> $SIF_DIR/<tag with / : @ -> _>.sif (runtime_apptainer.sh); SIF= overrides it
            sifs = [os.path.join(os.environ.get("SIF_DIR", os.path.join(nan, "docker_imgs", "sif")),
                                 re.sub(r"[/:@]", "_", t) + ".sif") for t in refs if not t.startswith("$")]
        for p in sifs:
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                print("IMAGE    %s  OK (%.1f GB)" % (p, os.path.getsize(p) / 1e9))
            else:
                print("IMAGE    %s  MISSING -- get it before step 4 (env.md *Images*: IMAGE_TAR=, or\n"
                      "           `apptainer pull` in a job, ~35 min for ~25 GB)" % p)
    elif rt in ("docker", "podman"):
        if uses_apptainer and not uses_oci and not (builds or builds_cmd):
            print("IMAGE    the run script is apptainer-only%s; on RUNTIME=%s use the docker image it starts\n"
                  "           from, or build one from its .def -- env.md *Images*"
                  % (" (SIF %s)" % sifs[0] if sifs else "", rt or "docker"))
        elif refs and not (builds or builds_cmd):
            print("IMAGE    nd pulls (DOCKER_PULL=1) or loads (IMAGE_TAR=) it on each node: nothing to do now")
    elif rt == "native":
        print("IMAGE    RUNTIME=native: no image; VENV=/CONDA_ENV= in the profile")
    if not refs and not sifs and not builds and not builds_cmd:
        print("IMAGE    ?  no image reference, SIF or build file found -- find the image by hand")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_script")
    ap.add_argument("--source", action="append", default=[], help="model file(s) to wire")
    ap.add_argument("--signal", default=None, help="NaN signal regex (default: from the run script)")
    ap.add_argument("--root", default=None, help="model repo (default: run script's git root)")
    ap.add_argument("--venv", default=None, help="model venv (default: .venv* under root)")
    a = ap.parse_args()

    rs = os.path.realpath(a.run_script)          # realpath: root (git) is resolved too
    root = a.root or sh(["git", "-C", os.path.dirname(rs), "rev-parse", "--show-toplevel"]).strip() \
        or os.path.dirname(rs)
    pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if not a.root and rs.startswith(pkg + os.sep):   # the package lives inside a larger repo (MAD)
        root = pkg
    venv = a.venv
    if venv is None:
        cands = sorted(d for d in os.listdir(root) if d.startswith(".venv")) if os.path.isdir(root) else []
        venv = os.path.join(root, cands[0]) if cands else None
    site = None
    if venv:
        libs = sh(["bash", "-c", "ls -d %s/lib/python*/site-packages 2>/dev/null" % venv]).split()
        site = libs[0] if libs else None
    # A model that lives INSIDE the numeric-doctor checkout (examples/...): the git root is
    # the tool repo, whose own scripts and tests are full of nan_to_num, capture hooks
    # and torch.compile fixtures. Scan only the run script's and the sources' dirs.
    scan = [root]
    if not a.root and os.path.isfile(os.path.join(root, "scripts", "replay_nan_op.py")):
        scan = sorted({os.path.dirname(rs)} | {os.path.dirname(os.path.realpath(s)) for s in a.source})
    print("preflight: run=%s root=%s venv=%s" % (rs, root, venv or "(none)"))
    if scan != [root]:
        print("scope:    numeric-doctor checkout -> scanning only %s"
              % ", ".join(os.path.relpath(d, root) for d in scan))

    image_check(rs)

    # the model sources: --source, else the .py files the run script names
    named = [os.path.join(os.path.dirname(rs), f) for f in re.findall(r"([\w./-]+\.py)\b", open(rs, errors="replace").read())]
    srcs = [os.path.realpath(f) for f in a.source] or [f for f in named if os.path.isfile(f)]

    # SIGNAL: is each alternative raised anywhere?
    alts, notes = signal_alternatives(rs, a.signal, srcs)
    for n in notes:
        print("SIGNAL   (%s)" % n)
    if not alts:
        print("SIGNAL   ?  no quoted grep pattern in the run script; pass --signal")
    elif not a.signal and not any(re.search(_SIGNAL_WORDS, x[0], re.I) for x in alts):
        # 2026-11-09 (qwen3 MoE run.sh): its only grep built a per-step table; the NaN is
        # the framework's own raise. Searching that table regex reported a false alarm.
        print("SIGNAL   ?  the run script greps no NaN line (only %s): the NaN surfaces in the "
              "model/framework itself -- pass its error line as --signal"
              % ", ".join("'%s'" % x[0][:32] for x in alts))
        alts = []
    real_srcs = {os.path.realpath(f) for f in srcs}
    # the --source's own lines first (else the folder's: a sibling op repro's `isnan` is noise), and a
    # print/raise before a docstring sentence that only mentions non-finite values
    raises = raised_messages(scan)
    raises = [h for h in raises if os.path.realpath(h[0]) in real_srcs] or raises
    raises.sort(key=lambda h: (os.path.realpath(h[0]) not in real_srcs,
                               not re.search(r"\b(print|raise|log\w*|warn\w*)\b\W", h[3]),
                               # "X is/are not finite" states the failure; a PROBE line only reports it
                               not re.search(r"\b(is|are) (not finite|non-finite|nan)\b|NaN detected", h[2], re.I)))
    msgs = []
    for f, ln, text, _ in raises:
        if text not in [m[0] for m in msgs]:
            msgs.append((text, "%s:%s" % (os.path.relpath(f, root), ln)))
    msgs_txt = "; ".join("'%s' (%s)" % m for m in msgs[:3]) + (" (+%d)" % (len(msgs) - 3) if len(msgs) > 3 else "")
    if not alts:
        if raises:
            print("SIGNAL   the source raises/logs: %s -- candidates for <NAN_SIGNAL>" % msgs_txt)
        # a finite signal: the model's own wrong-value / parity / determinism line, no NaN
        # (2026-11-21: a `RESULT ... wrong=yes` line went unsuggested twice)
        wrong = grep(r"""["'][^"']*(wrong=|RESULT |mismatch|differ|parity)""", scan, include=("*.py",))
        if wrong:
            print("SIGNAL   the source prints: %s -- finite-signal candidates (wrong values, no NaN: "
                  "step4-explode.md)" % short(wrong, root=root))
    for alt, _, lab, kind in alts:
        tag = "  [%s]" % lab if lab else ""
        clean = matches_clean(alt)
        if clean:
            print("SIGNAL   %-34s MATCHES CLEAN OUTPUT ('%s') -- not a NaN signal; use the line printed on the NaN path%s"
                  % ("'%s'" % alt[:32], clean, tag))
            continue
        # a pattern with regex syntax (grep -o 'step: *[0-9]*', grep -E) is searched as one;
        # read literally it can never match the source
        rx = bool(re.search(r"[\\\[\]*+?{}]", alt))
        def look(pat):
            if rx:   # ^...$ anchor a log line; the source prints it from inside a print(f"...")
                pat = pat.lstrip("^").rstrip("$") or pat
            hits = [h for h in grep(pat, scan, fixed=not rx) if os.path.abspath(h[0]) != rs
                    and not h[0].endswith("~")]
            # the --source, then the model's raise before sibling scripts
            hits.sort(key=lambda h: (os.path.realpath(h[0]) not in real_srcs, h[0].endswith(".sh")))
            return hits or (grep(pat, [site], fixed=not rx, include=("*.py",), venv_ok=True) if site else [])
        where, verdict = look(alt), "found"
        # `RESULT .*wrong=yes`: the source prints `wrong={wrong}` -- the field is static, its value is not
        field = re.match(r"(.*\w=)\w+$", alt) if rx and (kind == "field" or not where) else None
        if field and look(field.group(1)):
            where = look(field.group(1))
            verdict = "found as a field (its value is set at runtime)"
        if kind == "field":
            verdict += " -- the run script's verdict field: a wrong-value signal (step4-explode.md)"
        if not where:
            verdict = ("? not in the source as a regex -- a pattern over formatted output cannot be "
                       "checked statically: confirm it on one model log" if rx else
                       "NOT RAISED ANYWHERE -- the run script cannot see this NaN")
        print("SIGNAL   %-34s %s%s%s" % ("'%s'" % alt[:32], verdict,
                                         (": " + short(where, root=root)) if where else "", tag))
    if alts and all(k == "field" for *_, k in alts) and msgs:
        # 2026-11-24 (deepseek_sparse_mla_decode_nan): the run script only reads a RESULT verdict,
        # the source also prints the NaN itself, mid-run -- the line a capture can stop on
        print("SIGNAL   the source also raises/logs: %s -- a NaN-specific line: prefer it as <NAN_SIGNAL>;\n"
              "           the verdict field above is the wrong-value fallback" % msgs_txt)

    # MASK: values that would hide a NaN before any check
    mask = grep(r"nan_to_num|masked_fill\(.*isnan|where\(.*isfinite", scan, include=("*.py",))
    loss = [h for h in mask if re.search(r"loss|total", h[2])]
    print("MASK     %d nan_to_num/mask site(s)%s" % (len(mask), (", on a loss: " + short(loss, root=root))
                                                    if loss else ""))

    # HOOK: already wired?
    hook = grep(r"capture_from_env|OpNanCapture|op_nan_capture|cap\.step\(\)", scan, include=("*.py",))
    titan = re.search(r"torchtitan", open(rs, errors="replace").read())
    print("HOOK     %s" % ("wired: " + short(hook, root=root) if hook else
                           "not wired in the source -- a torchtitan run is wired by "
                           "scripts/torchtitan_capture_train.py instead (step4-distributed.md): "
                           "nothing to edit" if titan else
                           "not wired (wire_capture.py apply <source>)"))

    # ENV: does the run script's env reach the trainer?
    src = open(rs).read()
    called = re.findall(r"(?:bash|sh)\s+(\S+\.sh)", src)
    texts = [src] + [open(os.path.join(root, c)).read() for c in called
                     if os.path.isfile(os.path.join(root, c))]
    mech = []
    for t in texts:
        if re.search(r"set -a", t) and re.search(r"^\s*(source|\.)\s", t, re.M):
            mech.append("set -a + source")
        if re.search(r"docker run[^\n]*(-e |--env-file)", t):
            mech.append("docker -e/--env-file")
        if re.search(r"\benv -i\b", t):
            mech.append("env -i (CLEARS the environment)")
    sanit = [h for h in grep(r"env\s*=\s*\{", scan, include=("*.py",))]
    sanit = [h for h in sanit if "os.environ" not in h[2]]
    print("ENV      %s%s" % (", ".join(sorted(set(mech))) or "no mechanism seen -- check by hand",
                             ("; subprocess env dict(s) to read: " + short(sanit, root=root))
                             if sanit else ""))

    # LAZY
    lazy = grep(r"torch\.compile\(|^\s*compile:\s*true", scan, include=("*.py", "*.yaml", "*.yml"))
    print("LAZY     %s" % (("torch.compile/compile:true at " + short(lazy, root=root)
                            + " -> NAN_CAPTURE_ARM_AFTER_STEPS=1") if lazy else "none seen"))

    vllm_check(srcs, rs, root)
    sglang_check(srcs, rs, root)

    # LOOP
    for s in a.source:
        out = sh([sys.executable, os.path.join(HERE, "find_capture_points.py"), s, "--windows", "0"])
        top = next((l for l in out.splitlines() if l.strip().startswith("LOOP")), None)
        print("LOOP     %s: %s" % (s, " ".join(top.split()[1:]) if top else
                                    "no LOOP candidate -- read the entry point"))


if __name__ == "__main__":
    main()
