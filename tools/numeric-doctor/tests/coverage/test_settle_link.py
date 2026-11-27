"""Settle linking: a NaN made in a step-local temporary must be written at the op that MADE it, not at a
downstream op where it was seen surviving.

    python3 tests/coverage/test_settle_link.py <work dir>      (any PyTorch image with Triton, one GPU)

#4 G3/G1 (2026-11-24) left two vLLM capsules at the wrong op, with the creator (`aten.sqrt`, origin math) still
pending in the census.  Each case is a short loop under capture_from_env in a fresh process; the NaN is made by
sqrt(x - 1e6) on finite x at step BAD:

  cat_persist   bad = sqrt(h[:1].float() - 1e6).to(bf16); h = cat([bad, h[1:]]); the step's result is then
                written into a buffer that outlives the step (a persistent write)       -> capsule at aten.sqrt
  cat_host      the same, the NaN leaves through .item() (an escape, no op in hand)     -> capsule at aten.sqrt
  inplace_tri   h[:1].sub_(1e6).sqrt_() in place, then logits from h are copied into a fresh buffer and a
                Triton kernel adds a bias IN PLACE (every pointer taken as written: it becomes a candidate of
                its own, inputs_finite=None), then .item()                               -> capsule at aten.sqrt_
  lost_tri      the taint is lost: sqrt makes the NaN, a function the capture is suspended in copies it into a
                fresh buffer (unseen), the Triton kernel runs in place on it, then .item() -> capsule at aten.sqrt
                (an earlier candidate that made a NaN from finite inputs beats one handed NaN operands)
  tri_own       control: the Triton kernel itself makes the NaN (finite in)              -> capsule at the kernel
  clean         no NaN                                                                  -> no capsule
Last line: SETTLE_LINK=PASS|FAIL <n checks> <n failed>.
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.environ.get("ND_TEST_SCRIPTS") or os.path.join(HERE, "..", "..", "scripts")   # another copy: a baseline
FAILS, N = [], [0]

SRC = r'''
import os, sys, torch, triton, triton.language as tl
sys.path[:0] = [%(scripts)r]
from op_nan_capture import capture_from_env
CASE, BAD = %(case)r, 4
torch.manual_seed(0)

@triton.jit
def _bias_kernel(logits_ptr, bias_ptr, n, MAKE_NAN: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(logits_ptr + offs, mask=m)
    b = tl.load(bias_ptr + offs, mask=m)
    y = tl.where(offs < 2, float("-inf"), x + b)              # masks 2 tokens, as vLLM's does
    if MAKE_NAN:
        y = tl.sqrt(y - 1e6)
    tl.store(logits_ptr + offs, y, mask=m)

def unseen_copy(dst, src):                                     # NAN_CAPTURE_SUSPEND=__main__:unseen_copy
    dst.copy_(src)

W = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) * 0.05
Wout = torch.randn(256, 512, device="cuda", dtype=torch.float32) * 0.05
bias = torch.randn(512, device="cuda")
state = torch.zeros(64, 256, device="cuda", dtype=torch.bfloat16)      # outlives every step
cap = capture_from_env()
ctx = torch.inference_mode() if %(infer)r else torch.enable_grad()   # inference mode: aten.to.dtype reaches
try:                                                                   # the capture undecomposed, as in vLLM
  with ctx:
    for step in range(8):
        cap.step()
        x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
        h = torch.tanh(x @ W)
        if step == BAD and CASE in ("cat_persist", "cat_host"):
            bad = torch.sqrt(h[:1].float() - 1e6).to(h.dtype)          # finite in, NaN out
            h = torch.cat([bad, h[1:]], 0)
        if step == BAD and CASE == "inplace_tri":
            h[:1].sub_(1e6).sqrt_()
        h = h * 0.5 + 0.1
        if CASE == "cat_persist":
            state.copy_(h)                                             # persistent write
            print("step", step, flush=True)
            continue
        z = h[:1].float() @ Wout                                       # the "logits" row (row 0)
        logits = torch.empty_like(z)
        if CASE == "lost_tri":
            unseen_copy(logits, torch.sqrt(z - 1e6) if step == BAD else z)
        else:
            logits.copy_(z)
        _bias_kernel[(1,)](logits, bias, 512, MAKE_NAN=(step == BAD and CASE == "tri_own"), BLOCK=512)
        s = logits.max().item()                                        # escape: host read
        print("step", step, "max", s, flush=True)
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


def run(work, case, env=None, infer=False):
    d = os.path.join(work, case)
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(d, "**", "*.pt*"), recursive=True):
        os.remove(f)
    script = os.path.join(d, "loop.py")              # a file: Triton reads its kernels' source
    open(script, "w").write(SRC % {"scripts": SCRIPTS, "case": case, "infer": infer})
    e = dict(os.environ, NAN_CAPTURE_DIR=d, NAN_CAPTURE_DATA="1", NAN_PROGRESS_EVERY="1",
             NAN_CAPTURE_ARM_AFTER_STEPS="1", **(env or {}))
    p = subprocess.run([sys.executable, script], env=e, capture_output=True, text=True, timeout=900)
    log = p.stdout + p.stderr
    open(os.path.join(d, "log.txt"), "w").write(log)
    caps = sorted(glob.glob(os.path.join(d, "**", "*.pt"), recursive=True))
    return log, caps


def capsule(path):
    import torch
    sys.path[:0] = [SCRIPTS]
    c = torch.load(path, map_location="cpu", weights_only=False)
    return str(c.get("op")), (c.get("nan_origin") or {}).get("class"), (c.get("settle") or {}).get("reason")


def tail_err(log):
    lines = [x for x in log.splitlines() if "Error" in x or "Traceback" in x]
    return lines[-1][:200] if lines else log.strip().splitlines()[-1][:200] if log.strip() else "-"


def expect(work, case, want, env=None, infer=False):
    log, caps = run(work, case, env, infer)
    got = capsule(caps[0]) if caps else (None, None, None)
    check("%s%s: one capsule, at %s" % (os.path.basename(work), case, want), len(caps) == 1 and got[0] is not None and want in got[0],
          (len(caps),) + got + (tail_err(log),))


def main(work):
    for batch in ("0", "64"):                        # immediate checks and deferred (batched) checks
      for infer in (False, True):
        env = {"NAN_CAPTURE_CHECK_BATCH": batch}
        w = os.path.join(work, "batch%s%s" % (batch, "-infer" if infer else ""))
        expect(w, "cat_persist", "aten.sqrt.default", env, infer)
        expect(w, "cat_host", "aten.sqrt.default", env, infer)
        expect(w, "inplace_tri", "aten.sqrt_.default", env, infer)
        expect(w, "lost_tri", "aten.sqrt.default", dict(env, NAN_CAPTURE_SUSPEND="__main__:unseen_copy"), infer)
        expect(w, "tri_own", "_bias_kernel", env, infer)
        log, caps = run(w, "clean", env, infer)
        check("%s clean: no capsule, loop done" % os.path.basename(w), not caps and "LOOP_DONE" in log,
              (caps, tail_err(log)))
    print("SETTLE_LINK=%s %d %d" % ("PASS" if not FAILS else "FAIL", N[0], len(FAILS)), flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
