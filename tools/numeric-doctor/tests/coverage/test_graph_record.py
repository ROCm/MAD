"""G5: a framework that records its own CUDA/HIP graphs during the model's steps, under the capture.

    python3 tests/coverage/test_graph_record.py <work dir>      (any PyTorch image, one GPU)

The capture lets every op through while a graph is being recorded (`graph_capture_skipped` in the progress
lines): a recorded op's outputs are never computed, and a host read during recording is illegal. Only vLLM's
graphs had exercised that (2026-11-23). Each case runs a short loop under capture_from_env in a fresh process:

  rot default      torch.compile(mode="reduce-overhead") on an nn.Module, the capture as it comes: the compiled
                   module's ops run EAGERLY under the capture's dispatch mode (no graph recorded) and a NaN made
                   inside is caught at the op that made it
  rot suspended    NAN_CAPTURE_SUSPEND=OptimizedModule.__call__ (compiled code kept as the user runs it): CUDA-
                   graph trees record and replay. Clean: no crash, no capsule. An eager op after the module makes a
                   NaN: captured there, inputs finite. A NaN made INSIDE (finite input): captured at the first eager
                   op after it with non-finite inputs -- PROPAGATED, as for any compiled region
  manual_clean     a hand-written torch.cuda.graph recorded at step 2 while the capture is armed, replayed after:
                   no crash, no capsule, graph_capture_skipped > 0
  manual_inside    the same graph, a NaN made inside it at step 6: PROPAGATED at the first eager op after
  manual_eager     the same graph, an eager op after it makes the NaN: captured there, inputs finite
Last line: GRAPH_RECORD=PASS|FAIL <n checks> <n failed>.
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "..", "scripts")
FAILS, N = [], [0]

PRE = r'''
import os, sys, torch
sys.path[:0] = [%(scripts)r]
from op_nan_capture import capture_from_env
torch.manual_seed(0)
BAD = %(bad)d
W = torch.randn(256, 256, device="cuda") * 0.05
def body(x):
    h = torch.tanh(x @ W)
    return torch.sqrt(h + 1.0 - 0.5 * (x[:, :1] > 50).float() * 4)   # sqrt(negative) only on a planted row
def make_x(step, inside):
    x = torch.randn(64, 256, device="cuda")
    if inside and step == BAD:
        x[3, 0] = 100.0                         # finite input; the body turns row 3 into NaN
    return x
def tail(y, step, eager):
    z = y * 2.0                                  # the first eager op after the graphed part
    if eager and step == BAD:
        z = torch.log(z - 1e4)                   # an eager op that makes the NaN itself
    return z.sum()
'''

ROT = PRE + r'''
class Body(torch.nn.Module):
    def forward(self, x):
        return body(x)
m = torch.compile(Body(), mode="reduce-overhead")
cap = capture_from_env()
try:
    for step in range(8):
        cap.step()
        x = make_x(step, %(inside)r)
        y = m(x).clone()                         # cudagraph-tree outputs are overwritten on the next replay
        s = tail(y, step, %(eager)r)
        torch.cuda.synchronize()
        print("step", step, "sum", float(s), flush=True)
    cap.close()
    print("LOOP_DONE", flush=True)
except RuntimeError as e:
    print("CAPTURE_RAISED", str(e)[:200], flush=True)
'''

MANUAL = PRE + r'''
cap = capture_from_env()
static_x = torch.zeros(64, 256, device="cuda")
g, static_y = None, None
try:
    for step in range(9):
        cap.step()
        static_x.copy_(make_x(step, %(inside)r))
        if step < 2:
            y = body(static_x)                   # eager warm-up steps
        elif step == 2:
            s_ = torch.cuda.Stream()
            s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                body(static_x)                   # warm-up on the side stream, as the docs require
            torch.cuda.current_stream().wait_stream(s_)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):            # recorded while the capture is armed
                static_y = body(static_x)
            g.replay()
            y = static_y
        else:
            g.replay()
            y = static_y
        s = tail(y, step, %(eager)r)
        torch.cuda.synchronize()
        print("step", step, "sum", float(s), flush=True)
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


SUSPEND = {"NAN_CAPTURE_SUSPEND": "torch._dynamo.eval_frame:OptimizedModule.__call__"}


def run(work, label, tmpl, inside, eager, bad=5, arm_after=1, env=None):
    d = os.path.join(work, label)
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(d, "**", "*.pt*"), recursive=True):
        os.remove(f)
    e = dict(os.environ, NAN_CAPTURE_DIR=d, NAN_CAPTURE_DATA="1", NAN_PROGRESS_EVERY="1",
             NAN_CAPTURE_ARM_AFTER_STEPS=str(arm_after), **(env or {}))
    src = tmpl % {"scripts": SCRIPTS, "bad": bad, "inside": inside, "eager": eager}
    p = subprocess.run([sys.executable, "-c", src], env=e, capture_output=True, text=True, timeout=900)
    log = p.stdout + p.stderr
    open(os.path.join(d, "log.txt"), "w").write(log)
    caps = sorted(glob.glob(os.path.join(d, "**", "*.pt"), recursive=True))
    return log, caps, p.returncode


def capsule(path):
    import torch
    sys.path[:0] = [SCRIPTS]
    c = torch.load(path, map_location="cpu", weights_only=False)
    return c.get("op"), c.get("inputs_finite"), c.get("step")


def skipped(log):
    vals = [int(x.split("=")[1]) for x in log.split() if x.startswith("graph_capture_skipped=")]
    return max(vals) if vals else 0


def tail_err(log):
    lines = [x for x in log.splitlines() if "Error" in x or "Traceback" in x]
    return lines[-1][:200] if lines else log.strip().splitlines()[-1][:200] if log.strip() else "-"


def main(work):
    # default capture: torch.compile under the capture's dispatch mode runs the compiled module's ops eagerly --
    # no graph is recorded, and a NaN made inside is caught at the op that made it (2026-11-24)
    log, caps, rc = run(work, "rot_default_inside", ROT, True, False)
    ok = bool(caps)
    op, fin, st = capsule(caps[0]) if ok else (None, None, None)
    check("rot default: compiled module runs eagerly under the capture -> NaN caught at the op inside (sqrt)",
          ok and "sqrt" in str(op) and fin is True, (op, fin, st, tail_err(log)))

    # suspended (the skill's way to keep compiled code as the user runs it): CUDA-graph trees record and replay
    log, caps, rc = run(work, "rot_clean", ROT, False, False, env=SUSPEND)
    check("rot suspended, clean: reduce-overhead graphs run to the end under the capture", "LOOP_DONE" in log,
          tail_err(log))
    check("rot suspended, clean: no capsule", not caps, caps)

    log, caps, rc = run(work, "rot_eager_nan", ROT, False, True, env=SUSPEND)
    ok = bool(caps)
    op, fin, st = capsule(caps[0]) if ok else (None, None, None)
    check("rot suspended, eager NaN after the graphs: captured at aten.log, inputs finite",
          ok and "log" in str(op) and fin is True, (op, fin, st, tail_err(log)))

    log, caps, rc = run(work, "rot_inside_nan", ROT, True, False, env=SUSPEND)
    ok = bool(caps)
    op, fin, st = capsule(caps[0]) if ok else (None, None, None)
    check("rot suspended, NaN inside the graphed module: PROPAGATED at the first eager op after it (the clone)",
          ok and fin is False and "clone" in str(op), (op, fin, st, tail_err(log)))

    log, caps, rc = run(work, "manual_clean", MANUAL, False, False)
    check("manual_clean: hand-recorded graph inside the armed loop runs to the end", "LOOP_DONE" in log,
          tail_err(log))
    check("manual_clean: no capsule (recorded ops' never-computed outputs not judged)", not caps, caps)
    check("manual_clean: recording counted (graph_capture_skipped > 0)", skipped(log) > 0, skipped(log))

    log, caps, rc = run(work, "manual_inside", MANUAL, True, False, bad=6)
    ok = bool(caps)
    op, fin, st = capsule(caps[0]) if ok else (None, None, None)
    check("manual_inside: NaN made inside the replayed graph -> PROPAGATED at the first eager op",
          ok and fin is False and "mul" in str(op), (op, fin, st, tail_err(log)))

    log, caps, rc = run(work, "manual_eager", MANUAL, False, True, bad=6)
    ok = bool(caps)
    op, fin, st = capsule(caps[0]) if ok else (None, None, None)
    check("manual_eager: eager op after the replayed graph made it -> captured there, inputs finite",
          ok and "log" in str(op) and fin is True, (op, fin, st, tail_err(log)))

    print("GRAPH_RECORD=%s %d %d" % ("PASS" if not FAILS else "FAIL", N[0], len(FAILS)), flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
