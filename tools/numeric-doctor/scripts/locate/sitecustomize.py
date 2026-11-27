"""Locate the training loop that actually runs (debug-numeric step 4.2 smoke test).

Put this directory first on PYTHONPATH for the smoke test only:

    PYTHONPATH=<REPO_ROOT>/scripts/locate${PYTHONPATH:+:$PYTHONPATH}

Every Python process the launcher starts (torchrun workers, subprocess env dicts
built from os.environ) then imports it at startup.  Once torch is imported it
hooks Optimizer.step (the global pre-hook) and torch.autograd.backward -- and, for
inference, every top-level nn.Module forward run with grad disabled (no_grad /
inference_mode; e.g. each forward of a Hugging Face generate()) -- and for each
distinct call site prints the innermost loop around it, in the user's code, on
stderr:

    [op-nan-locate] LOOP /repo/launch.py:925 in run_distributed() via optimizer.step wired=no ...
    [op-nan-locate] summary pid 123: /repo/launch.py:925 run_distributed() x200 optimizer.step wired=no; ...
    [op-nan-locate] NEXT: wire_capture.py apply /repo/launch.py --loop-line 925

LOOP = a for/while in the model's own code (where wire_capture.py puts
cap.step()).  HOOK = the nearest loop is in site-packages: a framework owns the
loop, use its callback (step4-capture.md).  wired= says whether that loop
already starts with the capture's cap.step(); NAN_CAPTURE_DIR= whether the
capture env reached this process.  A static guess that names a loop which never
runs (dispatch through a registry) or runs but is not training (a memory probe)
shows up here as the summary naming another loop.

Exists because find_capture_points.py ranks loops one file at a time and cannot
see which one the run reaches; a wrong guess cost a whole smoke test plus the
diagnosis, or worse, passed a smoke test while capturing a probe loop.

NAN_LOCATE_OUT=<dir>: also append the lines to <dir>/locate-<pid>.txt (for a
harness that drops the workers' stderr).  Ubuntu's own sitecustomize, shadowed
by this file, is still run.
"""
import ast
import atexit
import importlib.abc
import importlib.util
import os
import sys
import sysconfig

TAG = "[op-nan-locate]"
MARK = "# numeric-doctor capture (wire_capture.py)"
_HERE = os.path.dirname(os.path.abspath(__file__))
_SYS = tuple(os.path.realpath(p) + os.sep for p in
             {sysconfig.get_paths()["stdlib"], sysconfig.get_paths()["platstdlib"]})
_trees, _sites, _counts, _order = {}, {}, {}, []


def _say(line):
    line = "%s %s" % (TAG, line)
    print(line, file=sys.stderr, flush=True)
    out = os.environ.get("NAN_LOCATE_OUT")
    if out:
        try:
            with open(os.path.join(out, "locate-%d.txt" % os.getpid()), "a") as f:
                f.write(line + "\n")
        except OSError:
            pass


def _framework(path):
    return "site-packages" in path or "dist-packages" in path


def _user_frames(frame):
    """(path, line, func) from innermost outward, skipping torch's, the stdlib's and ours."""
    out = []
    while frame is not None:
        raw = frame.f_code.co_filename
        path = os.path.realpath(raw)
        if not (raw.startswith("<") or path.startswith(_SYS) or path.startswith(_HERE + os.sep)
                or (os.sep + "torch" + os.sep) in path):
            out.append((path, frame.f_lineno, frame.f_code.co_name))
        frame = frame.f_back
    return out


def _tree(path):
    if path not in _trees:
        try:
            _trees[path] = ast.parse(open(path).read())
        except (OSError, SyntaxError, ValueError):
            _trees[path] = None
    return _trees[path]


def _loop_at(path, line):
    """Innermost for/while around `line` in the same function, or None."""
    tree = _tree(path)
    if tree is None:
        return None
    best = None

    def walk(node):
        nonlocal best
        for child in ast.iter_child_nodes(node):
            if not (getattr(child, "lineno", 0) <= line <= (getattr(child, "end_lineno", 0) or 0)):
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                best = None                       # a loop outside this def is not this frame's loop
                walk(child)
                return
            if isinstance(child, (ast.For, ast.AsyncFor, ast.While)) and child.body[0].lineno <= line:
                best = child
            walk(child)
            return

    walk(tree)
    return best


def _wired(path, loop):
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return "?"
    return "yes" if MARK in lines[loop.body[0].lineno - 1] else "no"


def _site(frames, user_first=False):
    """user_first: the nearest loop in the user's own code wins over a framework's inner
    loop (generate()'s decode loop) -- the request loop around it is where cap.step() goes."""
    order = ([f for f in frames if not _framework(f[0])] + frames) if user_first else frames
    for path, line, func in order:
        loop = _loop_at(path, line)
        if loop is not None:
            return ("HOOK" if _framework(path) else "LOOP", path, loop.lineno, func, _wired(path, loop))
    return None


def _event(kind):
    try:
        frames = _user_frames(sys._getframe(1))
        key = (kind, tuple(frames))
        site = _sites.get(key)
        if site is None:
            site = _sites[key] = _site(frames, user_first=(kind == "inference forward")) or ("NONE", frames[-1][0] if frames else "?", 0, "?", "?")
            if (kind, site) not in _counts:
                stack = " > ".join("%s:%d %s()" % (os.path.basename(p), l, f) for p, l, f in reversed(frames[:6]))
                _say("%s %s:%d in %s() via %s wired=%s NAN_CAPTURE_DIR=%s pid=%d  stack: %s"
                     % (site[0], site[1], site[2], site[3], kind, site[4],
                        "set" if os.environ.get("NAN_CAPTURE_DIR") else "unset", os.getpid(), stack))
                _order.append((kind, site))
        _counts[(kind, site)] = _counts.get((kind, site), 0) + 1
    except Exception as e:                        # never break the model
        _say("locator error (ignored): %r" % (e,))


def _summary():
    if not _counts:
        return
    parts = ["%s:%d %s() x%d %s wired=%s" % (s[1], s[2], s[3], _counts[(k, s)], k, s[4]) for k, s in _order]
    _say("summary pid %d: %s" % (os.getpid(), "; ".join(parts)))
    for kind in ("optimizer.step", "backward", "inference forward"):
        ranked = sorted(((n, s) for (k, s), n in _counts.items() if k == kind), key=lambda x: -x[0])
        if not ranked:
            continue
        s = ranked[0][1]
        if s[0] == "LOOP" and s[4] != "yes":
            _say("NEXT: wire_capture.py apply %s --loop-line %d   (the loop with the most %s calls)"
                 % (s[1], s[2], kind))
        elif s[0] == "LOOP":
            _say("OK: the wired loop %s:%d is the one with the most %s calls" % (s[1], s[2], kind))
        elif s[0] == "HOOK":
            _say("NEXT: the loop is a framework's (%s:%d) -- use its capture callback (step4-capture.md)"
                 % (s[1], s[2]))
        else:
            _say("NEXT: no loop around %s -- read the entry point" % kind)
        return


def _patch(torch):
    try:
        from torch.optim.optimizer import register_optimizer_step_pre_hook
        register_optimizer_step_pre_hook(lambda opt, args, kwargs: _event("optimizer.step"))
    except Exception as e:
        _say("no optimizer pre-hook (%r); backward only" % (e,))
    orig = torch.autograd.backward

    def backward(*args, **kwargs):
        _event("backward")
        return orig(*args, **kwargs)

    torch.autograd.backward = backward
    try:                                      # inference: top-level forwards with grad disabled
        from torch.nn.modules.module import (register_module_forward_pre_hook,
                                             register_module_forward_hook)
        depth = [0]

        def pre(mod, args):
            if depth[0] == 0 and not torch.is_grad_enabled():
                _event("inference forward")
            depth[0] += 1

        def post(mod, args, out):
            depth[0] = max(0, depth[0] - 1)

        register_module_forward_pre_hook(pre)
        try:
            register_module_forward_hook(post, always_call=True)
        except TypeError:                     # torch < 2.1: no always_call
            register_module_forward_hook(post)
    except Exception as e:
        _say("no module forward hooks (%r); training events only" % (e,))
    atexit.register(_summary)


class _TorchFinder(importlib.abc.MetaPathFinder):
    """Patch torch right after its own import finishes (torch.optim and torch.autograd included)."""

    def find_spec(self, name, path, target=None):
        if name != "torch":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if "torch" in sys.modules:
    _patch(sys.modules["torch"])
else:
    sys.meta_path.insert(0, _TorchFinder())


def _chain():
    """Run the sitecustomize this file shadows (e.g. Ubuntu's apport hook)."""
    for d in sys.path:
        if not d or os.path.realpath(d) == _HERE:
            continue
        cand = os.path.join(d, "sitecustomize.py")
        if os.path.isfile(cand):
            spec = importlib.util.spec_from_file_location("_nd_shadowed_sitecustomize", cand)
            try:
                spec.loader.exec_module(importlib.util.module_from_spec(spec))
            except Exception:
                pass
            return


_chain()
