"""The twin check inside the capture (T2b) and its replay (T6-lite), on test ops with a planted bug.

    python3 tests/twins/test_capture.py <work dir>      (any PyTorch image, one GPU)

Each case runs a short "model" loop under capture_from_env in a fresh subprocess (the capture is a process-wide
dispatch mode) and reads what it wrote:
  hit             TWIN_EVERY=1, buggy op: captured at the 37-row call, origin wrong_value, verify kind twin
  default off     TWIN_EVERY unset, same bug: no capsule (the opt-in holds)
  filtered        TWIN_OPS naming another op: no capsule
  in-place        the in-place buggy op: captured, its twin fed the pre-call snapshot
  nan             an op writing NaN, with a twin registered: still captured by the NaN path
  late            an op 1% off its twin every call, wrong (x1.2) on call 21: caught past the median rule
  sampled         TWIN_EVERY=4 and the bad call is the 2nd of its site: not checked, no capsule
  clean           the correct op over many shapes: no capsule, the census counts its checks (no false positive)
  inductor        a torch.compile'd module under the capture: its kernels twin-checked (fp64), no recursion into the
                  capture, no false positive
  replay          replay_nan_op on the hit's capsule counts every run (the twin fails again), 20/20
Last line: TWIN_CAPTURE=PASS|FAIL <n> <failed>.
"""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "..", "scripts")
FAILS, N = [], [0]

LOOP = r'''
import os, sys, torch
sys.path[:0] = [%(scripts)r, %(here)r]
import ndtest_ops  # noqa: F401
from op_nan_capture import capture_from_env
op = getattr(torch.ops.ndtest, %(op)r).default
cap = capture_from_env()
w = torch.randn(256, device="cuda", dtype=torch.bfloat16) * 0.1 + 1
try:
    for step, rows in enumerate(%(shapes)r):
        cap.step()
        x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        y = op(x, w, 1e-6)
        if y is None:
            y = x
        torch.cuda.synchronize()
        print("step", step, "rows", rows, "max", float(y.float().abs().max()), flush=True)
    cap.close()
    print("LOOP_DONE", flush=True)
except RuntimeError as e:
    print("CAPTURE_RAISED", str(e)[:200], flush=True)
'''


def check(name, cond, detail=""):
    N[0] += 1
    print("%s  %s%s" % ("PASS" if cond else "FAIL", name, ("  -- " + str(detail)) if detail else ""), flush=True)
    if not cond:
        FAILS.append(name)


def run(work, label, op, shapes, env):
    d = os.path.join(work, label)
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(d, "**", "*.pt*"), recursive=True):
        os.remove(f)
    e = dict(os.environ, NAN_CAPTURE_DIR=d, NAN_CAPTURE_DATA="1", NAN_CAPTURE_VERBOSE="1", **env)
    for k in ("NAN_CAPTURE_TWIN_EVERY", "NAN_CAPTURE_TWIN_OPS"):
        if k not in env:
            e.pop(k, None)
    src = LOOP % {"scripts": SCRIPTS, "here": HERE, "op": op, "shapes": shapes}
    p = subprocess.run([sys.executable, "-c", src], env=e, capture_output=True, text=True, timeout=600)
    log = p.stdout + p.stderr
    open(os.path.join(d, "log.txt"), "w").write(log)
    caps = [f for f in glob.glob(os.path.join(d, "**", "*.pt"), recursive=True)]
    return log, caps


COMPILED = r'''
import os, sys, torch
sys.path[:0] = [%(scripts)r]
from op_nan_capture import capture_from_env
class M(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.w = torch.nn.Parameter(torch.randn(256, device="cuda") * 0.1 + 1)
    def forward(s, x):
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6) * s.w
        return (torch.softmax(torch.tanh(y) * 3, -1) * y.sigmoid()).to(x.dtype)
m = torch.compile(M())
cap = capture_from_env()
for step in range(6):
    cap.step()
    out = m(torch.randn(64, 256, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
cap.close()
print("COMPILED_DONE", flush=True)
'''


def run_compiled(work):
    d = os.path.join(work, "compiled")
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(d, "**", "*.pt*"), recursive=True):
        os.remove(f)
    # armed after step 1, as the skill runs a lazily compiled model: armed at step 0 the first call compiles under
    # the capture, and torch.compile falls back to eager (no Inductor kernel to check)
    e = dict(os.environ, NAN_CAPTURE_DIR=d, NAN_CAPTURE_DATA="1", NAN_CAPTURE_TWIN_EVERY="1",
             NAN_CAPTURE_ARM_AFTER_STEPS="1",
             NAN_CAPTURE_SUSPEND="torch._dynamo.eval_frame:OptimizedModule.__call__")
    p = subprocess.run([sys.executable, "-c", COMPILED % {"scripts": SCRIPTS}], env=e, capture_output=True,
                       text=True, timeout=900)
    log = p.stdout + p.stderr
    open(os.path.join(d, "log.txt"), "w").write(log)
    return log


def capsule(path):
    import torch
    sys.path[:0] = [SCRIPTS, HERE]
    return torch.load(path, map_location="cpu", weights_only=False)


def main(work):
    good_then_bad = [16, 16, 37, 16]
    log, caps = run(work, "hit", "rmsnorm_buggy", good_then_bad, {"NAN_CAPTURE_TWIN_EVERY": "1"})
    check("hit: a capsule is written", len(caps) == 1, "%d capsule(s); log tail: %s" % (len(caps), log[-300:]))
    hit_cap = caps[0] if caps else None
    if hit_cap:
        c = capsule(hit_cap)
        v = c.get("verify") or {}
        check("hit: origin wrong_value", (c.get("nan_origin") or {}).get("class") == "wrong_value", c.get("nan_origin"))
        check("hit: verify kind twin, source named", v.get("kind") == "twin" and v.get("twin_source") == "test native norm",
              {k: v.get(k) for k in ("kind", "twin_source", "twin_kind", "rel_err", "threshold")})
        check("hit: at the 37-row call", tuple(c.get("out_size") or ()) == (37, 256), c.get("out_size"))
        check("hit: rel err ~0.2 over a 3e-2 floor", 0.1 < (v.get("rel_err") or 0) < 0.3 and v.get("threshold") == 3e-2,
              v.get("rel_err"))
    log, caps = run(work, "off", "rmsnorm_buggy", good_then_bad, {})
    check("default off: no capsule", not caps, caps)
    log, caps = run(work, "filtered", "rmsnorm_buggy", good_then_bad,
                    {"NAN_CAPTURE_TWIN_EVERY": "1", "NAN_CAPTURE_TWIN_OPS": "ndtest.rmsnorm_ok"})
    check("filtered: no capsule", not caps, caps)
    log, caps = run(work, "inplace", "rmsnorm_inplace_buggy", good_then_bad, {"NAN_CAPTURE_TWIN_EVERY": "1"})
    ok = len(caps) == 1 and (capsule(caps[0]).get("verify") or {}).get("kind") == "twin"
    check("in-place: captured through the pre-call snapshot", ok, "%d capsule(s); %s" % (len(caps), log[-300:]))
    log, caps = run(work, "nan", "rmsnorm_nan", good_then_bad, {"NAN_CAPTURE_TWIN_EVERY": "1"})
    c = capsule(caps[0]) if len(caps) == 1 else {}
    check("NaN with a twin registered: still the NaN path's capsule",
          len(caps) == 1 and (c.get("nan_origin") or {}).get("class") != "wrong_value" and c.get("out_nan", 0) > 0,
          "%d capsule(s) origin %s nan %s" % (len(caps), (c.get("nan_origin") or {}).get("class"), c.get("out_nan")))
    log, caps = run(work, "late", "rmsnorm_noisy_buggy", [16] * 20 + [37, 16], {"NAN_CAPTURE_TWIN_EVERY": "1"})
    check("late: a bug after the median is set (1% steady gap) is still caught",
          len(caps) == 1 and tuple(capsule(caps[0]).get("out_size") or ()) == (37, 256),
          "%d capsule(s); %s" % (len(caps), log[-300:]))
    log, caps = run(work, "sampled", "rmsnorm_buggy", [37 if i == 1 else 16 for i in range(6)],
                    {"NAN_CAPTURE_TWIN_EVERY": "4"})
    check("sampled: the unsampled bad call is not checked", not caps, caps)
    log, caps = run(work, "clean", "rmsnorm_ok", [1, 7, 16, 37, 64, 128, 333, 1024] * 3,
                    {"NAN_CAPTURE_TWIN_EVERY": "1"})
    check("clean: no capsule on a correct op", not caps and "LOOP_DONE" in log, "%d capsule(s); %s" % (len(caps), log[-300:]))
    census = [l for l in log.splitlines() if "ndtest.rmsnorm_ok" in l and "checked" in l]
    check("clean: the census shows the twin checked every call", any("24 checked" in l for l in census), census)
    log = run_compiled(work)
    sites = [l for l in log.splitlines() if " twin N=-:" in l and "inductor." in l]
    checked = [int(l.split(" calls, ")[1].split(" checked")[0]) for l in sites]
    check("inductor: compiled kernels twin-checked under the capture (no recursion into the capture)",
          sites and all(c > 0 for c in checked) and "RecursionError" not in log and "COMPILED_DONE" in log,
          "%d site(s), checked %s; %s" % (len(sites), checked, log[-300:]))
    check("inductor: no false positive on correct compiled kernels",
          not glob.glob(os.path.join(work, "compiled", "**", "*.pt"), recursive=True))
    if hit_cap:
        src = ("import sys, runpy; sys.path[:0] = [%r, %r]; import ndtest_ops; "
               "sys.argv = ['replay_nan_op.py', %r, '--iters', '20']; runpy.run_path(%r, run_name='__main__')"
               % (SCRIPTS, HERE, hit_cap, os.path.join(SCRIPTS, "replay_nan_op.py")))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE, SCRIPTS, os.environ.get("PYTHONPATH", "")]))
        p = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=600, env=env)
        out = p.stdout + p.stderr
        open(os.path.join(work, "replay.txt"), "w").write(out)
        line = [l for l in out.splitlines() if "non-finite on" in l or "[verdict]" in l or "wrong-value capsule" in l]
        check("replay: 20/20 runs fail the twin again", any("20/20" in l for l in line), line or out[-400:])
    print("TWIN_CAPTURE=%s %d %d" % ("FAIL" if FAILS else "PASS", N[0], len(FAILS)))


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]))
