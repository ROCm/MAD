"""G2 (graphs): SGLang's graph runner under debug-numeric -- graphs off from the environment, and the graph-replay
check (scripts/sglang_graphs), on the SGLang serving example in a clean setting with a planted decode NaN.

    python3 tests/coverage/test_sglang_graphs.py <work dir>      (the SGLang image, one GPU)

  graph_only + check     a NaN written only while a graph is recorded: GRAPH_CHECK verdict=GRAPH_ONLY
  decode_nan + check     a NaN in every decode step, graph or not: verdict=NOT_GRAPH_ONLY
  graph_only, graphs off ND_SGLANG_DISABLE_CUDA_GRAPH=1: the switch line, and the model's logprobs stay finite
  decode_nan, capture, graphs off   the decode NaN is captured (the decode steps run eagerly, the capture sees them)
  decode_nan, capture, graphs on    decode steps are graph replays the capture cannot see: the NaN is caught only
                                    downstream, PROPAGATED (non-finite inputs) -- the contrast that makes the
                                    switch necessary
Last line: SGLANG_GRAPHS=PASS|FAIL <n checks> <n failed>.
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EX = os.path.join(REPO, "examples", "inference", "qwen_sglang_serve_nan", "qwen_sglang_serve.py")
PLANT = os.path.join(REPO, "tests", "diff", "plant")
FAILS, N = [], [0]


def check(name, cond, detail=""):
    N[0] += 1
    print("%s  %s%s" % ("PASS" if cond else "FAIL", name, ("  -- " + str(detail)) if detail else ""), flush=True)
    if not cond:
        FAILS.append(name)


def run(work, label, plant, shims, env):
    d = os.path.join(work, label)
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(d, "**", "*.pt"), recursive=True):
        os.remove(f)
    for v in env.values():                       # the capture asserts its capsule dir exists
        if isinstance(v, str) and v.endswith("/caps"):
            os.makedirs(v, exist_ok=True)
    pp = ":".join([PLANT] + [os.path.join(REPO, "scripts", s) for s in shims] + [os.environ.get("PYTHONPATH", "")])
    e = dict(os.environ, PYTHONPATH=pp, ND_PLANT=plant, ND_PLANT_LAYER="3", ND_GRAPH_CHECK_DIR=d, **env)
    cmd = [sys.executable, EX, "--requests", "6", "--mix-every", "0", "--config", '{"rms_norm_eps": 1e-6}',
           "--prompt-len", "64", "--new-tokens", "8", "--seed", "0"]
    p = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=1200)
    log = p.stdout + p.stderr
    open(os.path.join(d, "log.txt"), "w").write(log)
    caps = sorted(glob.glob(os.path.join(d, "caps", "**", "*.pt"), recursive=True))
    return log, caps


def capsule(path):
    import torch
    c = torch.load(path, map_location="cpu", weights_only=False)
    return c.get("op"), c.get("inputs_finite")


def line(log, key):
    ls = [x for x in log.splitlines() if key in x]
    return ls[0][:220] if ls else "-"


def main(work):
    log, _ = run(work, "graph_only_check", "sglang_graph_only", ["sglang_graphs"], {"ND_GRAPH_CHECK": "1"})
    check("graph-only NaN + graph check -> GRAPH_ONLY", "GRAPH_CHECK verdict=GRAPH_ONLY" in log,
          line(log, "GRAPH_CHECK") if "GRAPH_CHECK" in log else line(log, "Error"))
    check("graph-only NaN reaches the model's output with graphs on", "not finite" in log, line(log, "RESULT"))

    log, _ = run(work, "decode_nan_check", "sglang_decode_nan", ["sglang_graphs"], {"ND_GRAPH_CHECK": "1"})
    check("decode NaN + graph check -> NOT_GRAPH_ONLY", "GRAPH_CHECK verdict=NOT_GRAPH_ONLY" in log,
          line(log, "GRAPH_CHECK") if "GRAPH_CHECK" in log else line(log, "Error"))

    log, _ = run(work, "graph_only_off", "sglang_graph_only", ["sglang_graphs"], {"ND_SGLANG_DISABLE_CUDA_GRAPH": "1"})
    check("graphs off: the switch is applied, and the scheduler's decode graphs are off",
          "disable_cuda_graph=True" in log and "decode backend" in log and "capturing=True" not in log,
          line(log, "decode backend"))
    check("graphs off: the graph-only NaN is gone (logprobs finite)", "not finite" not in log and "RESULT" in log,
          line(log, "RESULT"))

    cenv = {"NAN_CAPTURE_DATA": "1", "NAN_CAPTURE_ARM_AFTER_STEPS": "1", "ND_SGLANG_DISABLE_CUDA_GRAPH": "1"}
    d = os.path.join(work, "decode_nan_capture_off")
    log, caps = run(work, "decode_nan_capture_off", "sglang_decode_nan", ["sglang_graphs", "sglang_capture"],
                    dict(cenv, NAN_CAPTURE_DIR=os.path.join(d, "caps")))
    op, fin = capsule(caps[0]) if caps else (None, None)
    check("graphs off + capture: the decode-step NaN is captured where it is made (inputs finite)",
          bool(caps) and fin is True, (op, fin) if caps else line(log, "op-nan"))

    cenv.pop("ND_SGLANG_DISABLE_CUDA_GRAPH")
    d = os.path.join(work, "decode_nan_capture_on")
    log, caps = run(work, "decode_nan_capture_on", "sglang_decode_nan", ["sglang_capture"],
                    dict(cenv, NAN_CAPTURE_DIR=os.path.join(d, "caps")))
    op, fin = capsule(caps[0]) if caps else (None, None)
    check("graphs on + capture: decode replays unseen -> at most a PROPAGATED capsule downstream",
          not caps or fin is False, (op, fin) if caps else line(log, "RESULT"))

    print("SGLANG_GRAPHS=%s %d %d" % ("PASS" if not FAILS else "FAIL", N[0], len(FAILS)), flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
