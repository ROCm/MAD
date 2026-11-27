"""Differential mode without a GPU (U): output_diff.py's verdicts on synthetic tap files, the module comparison,
configs.py's arms and framework detection.

    python3 tests/diff/test_unit.py         (no torch needed: the login node)

Last line: DIFF_UNIT=PASS|FAIL <n checks> <n failed>.
"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import configs  # noqa: E402
import output_diff as od  # noqa: E402

FAILS, N = [], [0]
TMP = tempfile.mkdtemp(prefix="diff-unit-")


def check(name, cond, detail=""):
    N[0] += 1
    print("%s  %s%s" % ("PASS" if cond else "FAIL", name, ("  -- " + str(detail)) if detail else ""))
    if not cond:
        FAILS.append(name)


def write(name, recs, dtype="bfloat16"):
    p = os.path.join(TMP, name + ".jsonl")
    with open(p, "w") as f:
        f.write(json.dumps({"kind": "meta", "framework": "vllm", "dtype": dtype}) + "\n")
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return p


def gen(i, tokens, lp, margin=None, inp="h%d"):
    return {"kind": "gen", "i": i, "inp": inp % i if "%" in inp else inp, "tokens": tokens, "lp": lp,
            "margin": margin or [5.0] * len(tokens)}


def run(*argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        od.main(list(argv))
    out = buf.getvalue()
    line = [x for x in out.splitlines() if x.startswith("ND_DIFF verdict=")]
    return (line[0] if line else ""), out


def verdict(line):
    return line.split()[1].split("=")[1] if line else None


BASE = [gen(i, [10 + i, 20, 30, 40], [-0.5, -1.0, -0.2, -2.0]) for i in range(20)]


def jitter(recs, eps):
    out = []
    for r in recs:
        r = json.loads(json.dumps(r))
        r["lp"] = [x + eps * (1 if k % 2 else -1) for k, x in enumerate(r["lp"])]
        out.append(r)
    return out


# --- judge_gen --------------------------------------------------------------------------------------------
v, d = od.judge_gen(BASE[0], BASE[0], 0.1)
check("gen identical -> SAME", v == "SAME", v)
v, d = od.judge_gen(jitter([BASE[0]], 0.05)[0], BASE[0], 0.1)
check("gen small logprob jitter -> SAME", v == "SAME", (v, d))
v, d = od.judge_gen(jitter([BASE[0]], 0.5)[0], BASE[0], 0.1)
check("gen logprob gap above threshold -> DIVERGE", v == "DIVERGE" and d["why"] == "logprob gap", (v, d))
a = gen(0, [10, 21, 31, 41], [-0.5, -0.7, -0.2, -2.0], margin=[5, 0.02, 5, 5])
r = gen(0, [10, 20, 30, 40], [-0.5, -0.69, -0.2, -2.0], margin=[5, 0.03, 5, 5])
v, d = od.judge_gen(a, r, 0.1)
check("gen token flip at a near-tie -> TIE_FLIP", v == "TIE_FLIP" and d["pos"] == 1, (v, d))
a2 = dict(a, margin=[5, 3.0, 5, 5])
v, d = od.judge_gen(a2, r, 0.1)
check("gen token flip away from a tie -> DIVERGE", v == "DIVERGE" and d["pos"] == 1, (v, d))
a3 = dict(a, lp=[-0.5, -0.7, -9.0, -9.0])
v, d = od.judge_gen(a3, r, 0.1)
check("gen logprobs after the first differing token are not compared", v == "TIE_FLIP", (v, d))
v, d = od.judge_gen(dict(BASE[0], inp="other"), BASE[0], 0.1)
check("gen different prompt -> INPUTS_DIFFER", v == "INPUTS_DIFFER", v)
v, d = od.judge_gen(dict(BASE[0], lp=[-0.5, float("nan"), -0.2, -2.0]), BASE[0], 0.1)
check("gen non-finite logprob in one run -> DIVERGE", v == "DIVERGE" and d["gap"] == float("inf"), (v, d))
v, d = od.judge_gen(dict(BASE[0], tokens=[10, 20]), BASE[0], 0.1)
check("gen shorter output -> differing at the end (no margin: DIVERGE)", v == "DIVERGE" and d["pos"] == 2, (v, d))

u1 = {"kind": "gen", "i": 0, "inp": "call0/row0", "tokens": [1, 2, 3], "lp": [], "margin": []}
u2 = dict(u1, tokens=[1, 2, 4])
v, d = od.judge_gen(u1, u2, 0.1)
check("gen token-only records (func tap), token differs -> UNMARGINED", v == "UNMARGINED", (v, d))
cmp_ = {"verdicts": {("gen", 0): ("UNMARGINED", {})}, "missing": 0}
check("UNMARGINED counts against the same configuration (A vs A2)",
      od.summarize(cmp_, {"gen"}, same_config=True)["bad"] == [("gen", 0)])
check("UNMARGINED does not count against the reference (reported only)",
      od.summarize(cmp_, {"gen"})["bad"] == [] and od.summarize(cmp_, {"gen"})["unmargined"] == 1)

# --- judge_step / step_diverged ---------------------------------------------------------------------------
v, d = od.judge_step({"loss": 2.0, "gnorm": 1.0}, {"loss": 2.01, "gnorm": 1.0}, 0.02)
check("step 0.5% loss gap under a 2% threshold -> SAME", v == "SAME", (v, d))
v, d = od.judge_step({"loss": 2.0, "gnorm": 1.5}, {"loss": 2.0, "gnorm": 1.0}, 0.02)
check("step two-way: a grad-norm gap alone does not decide (too chaotic)", v == "SAME", (v, d))
v, d = od.judge_step({"loss": 2.0, "gnorm": float("inf")}, {"loss": 2.0, "gnorm": 1.0}, 0.02)
check("step two-way: a non-finite grad norm in one run -> DIVERGE", v == "DIVERGE" and d["what"] == "gnorm", (v, d))
check("relgap: log-ratio for positive values (755 vs 15.6 stays large)", abs(od.relgap(755, 15.6) - 3.879) < 0.01)

# --- the three-way training rule --------------------------------------------------------------------------
FL = {"loss": 0.02, "gnorm": 1.39}
mk = lambda vals: {("step", i): {"kind": "step", "i": i, "loss": l, "gnorm": g} for i, (l, g) in enumerate(vals)}  # noqa
chaos_a = mk([(5.0, 8.5), (5.1, 11.7), (5.0, 68.8)])
chaos_b = mk([(5.01, 12.8), (5.09, 32.3), (5.02, 20.2)])
chaos_r = mk([(4.99, 27.7), (5.08, 4.86), (4.98, 60.4)])
o = od.odd_one_out(chaos_a, chaos_b, chaos_r, FL)
check("three-way: grad-norm chaos below 4x and close losses -> no odd run", all(c == "SAME" for c, _ in o.values()), o)
spike_a = mk([(5.43, 15.0), (7.28, 755.0)])
spike_b = mk([(5.43, 15.2), (5.43, 15.6)])
spike_r = mk([(5.42, 15.1), (5.42, 6.7)])
o = od.odd_one_out(spike_a, spike_b, spike_r, FL)
check("three-way: A's loss spike while A2 and REF agree -> SELF, odd A", o[1][0] == "SELF" and o[1][1]["odd"] == "A", o[1])
ref_a = mk([(5.0, 3.0), (4.0, 3.0)])
ref_r = mk([(5.0, 3.0), (4.4, 3.0)])
o = od.odd_one_out(ref_a, ref_a, ref_r, FL)
check("three-way: A and A2 agree, REF 10% off -> REF_ODD (a jump from 0)", o[1][0] == "REF_ODD", o[1])
drift_a = mk([(5.0, 3.0)] * 8)
drift_r = mk([(5.0 * (1 + 0.003 * 2.5 ** i), 3.0) for i in range(8)])
o = od.odd_one_out(drift_a, drift_a, drift_r, FL)
check("three-way: smooth drift of the reference (x2.5 per step) -> never REF_ODD, only REF_FAR",
      all(c in ("SAME", "REF_FAR") for c, _ in o.values()) and any(c == "REF_FAR" for c, _ in o.values()), o)
nan_a = mk([(5.0, 3.0), (float("nan"), float("nan"))])
o = od.odd_one_out(nan_a, ref_a, ref_a, FL)
check("three-way: a NaN in A only -> NONFINITE_SELF", o[1][0] == "NONFINITE_SELF", o[1])
v, d = od.judge_step({"loss": float("nan")}, {"loss": 2.0}, 0.02)
check("step non-finite loss in one run -> DIVERGE", v == "DIVERGE", (v, d))
vs = {("step", i): ("DIVERGE" if i in (3, 7, 8, 9, 10) else "SAME", {"gap": 0.1}) for i in range(12)}
check("step_diverged: an isolated noisy step is not a divergence, a run of 3 is (starts at 7)",
      od.step_diverged(vs) == 7, od.step_diverged(vs))
vs2 = {("step", i): ("SAME", {"gap": 0.0}) for i in range(5)}
vs2[("step", 2)] = ("DIVERGE", {"gap": float("inf")})
steps_a1 = [{"kind": "step", "i": i, "loss": 3.0} for i in range(10)]
steps_r1 = [{"kind": "step", "i": i, "loss": 3.0 * (1.3 if i == 5 else 1.0)} for i in range(10)]
line, out = run("--a", write("SA1", steps_a1), "--a2", write("SA12", steps_a1), "--ref", write("SR1", steps_r1))
check("training: a one-step jump of the reference alone -> AGREE (the reference hiccuped)", verdict(line) == "AGREE", line)
check("step_diverged: a non-finite step counts at once", od.step_diverged(vs2) == 2, od.step_diverged(vs2))

# --- judge_fwd --------------------------------------------------------------------------------------------
f0 = {"kind": "fwd", "i": 0, "inp": "x", "argmax": [1, 2, 3], "l2": 10.0, "absmax": 2.0, "finite": True}
check("fwd identical -> SAME", od.judge_fwd(f0, f0, 0.03)[0] == "SAME")
check("fwd 10% l2 gap -> DIVERGE", od.judge_fwd(dict(f0, l2=11.0), f0, 0.03)[0] == "DIVERGE")
check("fwd argmax flip within the floor -> TIE_FLIP", od.judge_fwd(dict(f0, argmax=[1, 2, 4]), f0, 0.03)[0] == "TIE_FLIP")
check("fwd non-finite in one run -> DIVERGE", od.judge_fwd(dict(f0, finite=False), f0, 0.03)[0] == "DIVERGE")

# --- whole runs: AGREE / DIVERGE / NONDETERMINISTIC / INPUTS_DIFFER, the noise floor, the bisection -----
A = write("A", BASE)
A2 = write("A2", BASE)
R = write("R", jitter(BASE, 0.05))
line, out = run("--a", A, "--a2", A2, "--ref", R)
check("run: ref within the bf16 floor -> AGREE", verdict(line) == "AGREE", line)

bad = json.loads(json.dumps(BASE))
for i in (4, 11):
    bad[i]["tokens"][2] = 99
    bad[i]["lp"][1] -= 0.6
Ab = write("Ab", bad)
Ab2 = write("Ab2", bad)
line, out = run("--a", Ab, "--a2", Ab2, "--ref", R, "--json", os.path.join(TMP, "d.json"))
check("run: two requests 0.6 off -> DIVERGE, 2 of 20, first gen:4", verdict(line) == "DIVERGE" and
      " diverge=2 " in line and "first=gen:4" in line, line)
res = json.load(open(os.path.join(TMP, "d.json")))
check("run: --json holds thresholds and the diverging records", res["thresholds"]["gen"] >= 0.25 and
      res["a_vs_ref"]["diverge"] == [["gen", 4], ["gen", 11]], res.get("a_vs_ref"))

line, out = run("--a", Ab, "--a2", A2, "--ref", R)
check("run: A and A2 disagree -> NONDETERMINISTIC (before DIVERGE)", verdict(line) == "NONDETERMINISTIC", line)

noisy = jitter(BASE, 0.2)
An = write("An", noisy)
line, out = run("--a", A, "--a2", An, "--ref", R)
check("run: a 0.2 self-gap raises the threshold to 5 x 0.2 (k rule)", "gen:1" in line, line)

line, out = run("--a", A, "--a2", A2, "--ref", write("Rin", [dict(r, inp="z%d" % r["i"]) for r in BASE]))
check("run: other prompts in the reference -> INPUTS_DIFFER", verdict(line) == "INPUTS_DIFFER", line)

line, out = run("--a", Ab, "--a2", Ab2, "--ref", R, "--arm", "flip_aiter=" + A, "--arm", "flip_graphs=" + Ab)
check("bisection: the arm that agrees names the switch", "ND_DIFF_SWITCH=flip_aiter" in out, out[-300:])
check("bisection: the arm that still diverges is reported DIVERGE",
      "ND_DIFF_ARM name=flip_graphs verdict=DIVERGE" in out, out[-300:])
line, out = run("--a", Ab, "--a2", Ab2, "--ref", R, "--arm", "flip_x=" + Ab)
check("bisection: no single switch -> NONE_SINGLE", "ND_DIFF_SWITCH=NONE_SINGLE" in out, out[-200:])

steps_a = [{"kind": "step", "i": i, "loss": 3.0 - 0.1 * i} for i in range(20)]
steps_r = [{"kind": "step", "i": i, "loss": (3.0 - 0.1 * i) * (1.1 if i >= 12 else 1.00001)} for i in range(20)]
line, out = run("--a", write("SA", steps_a, "float32"), "--a2", write("SA2", steps_a, "float32"),
                "--ref", write("SR", steps_r, "float32"))
check("training: loss 10% off from step 12 -> DIVERGE first=step:12", verdict(line) == "DIVERGE" and
      "first=step:12" in line, line)
gn = os.path.join(TMP, "gn.jsonl")
with open(gn, "w") as f:
    f.write(json.dumps({"kind": "step", "i": 0, "loss": 1.0}) + "\n")
    f.write(json.dumps({"kind": "step", "i": 0, "gnorm": 2.0}) + "\n")
_, recs = od.load(gn)
check("load: a later record of the same key adds its fields (gnorm)", recs[("step", 0)] == {
      "kind": "step", "i": 0, "loss": 1.0, "gnorm": 2.0}, recs)

# --- modules ----------------------------------------------------------------------------------------------


def mods(name, corrupt=None):
    p = os.path.join(TMP, name + ".modules.jsonl")
    with open(p, "w") as f:
        seq = 0
        for layer in range(3):
            for m in ("RMSNorm#%d" % layer, "Attention#%d" % layer, "MLP#%d" % layer, "DecoderLayer#%d" % layer):
                l2 = 10.0 + layer
                s = 1.0 if m != corrupt and not (corrupt and m.startswith("DecoderLayer#") and
                                                 int(m[-1]) >= int(corrupt[-1])) else 1.2
                f.write(json.dumps({"req": 3, "seq": seq, "mod": m, "n": 1000, "l2": l2 * s, "absmax": 1.0,
                                    "proj": [0.5 * s, -0.3 * s, 0.2 * s], "finite": True}) + "\n")
                seq += 1
    return p


buf = io.StringIO()
with redirect_stdout(buf):
    od.main(["--modules", mods("ma", corrupt="MLP#1"), "--modules-ref", mods("mr")])
o = buf.getvalue()
check("modules: the innermost corrupted module is named (not its parent layer)",
      "ND_DIFF_LOCATE verdict=FOUND module=MLP#1" in o, o.strip().splitlines()[-1])
buf = io.StringIO()
with redirect_stdout(buf):
    od.main(["--modules", mods("ma2"), "--modules-ref", mods("mr2")])
check("modules: identical runs -> NONE_ABOVE", "verdict=NONE_ABOVE" in buf.getvalue(), buf.getvalue()[-200:])

def drift_mods(name, base, grow):
    p = os.path.join(TMP, name + ".modules.jsonl")
    with open(p, "w") as f:
        for i in range(12):
            s = 1 + base * grow ** i
            f.write(json.dumps({"req": 0, "seq": i, "mod": "Layer#%d" % i, "n": 100, "l2": 10.0 * s, "absmax": 1.0,
                                "proj": [1.0 * s, 2.0, 3.0], "finite": True}) + "\n")
    return p


buf = io.StringIO()
with redirect_stdout(buf):
    od.main(["--modules", drift_mods("dr_a", 0.01, 1.6), "--modules-ref", drift_mods("dr_r", 0.0, 1.0)])
check("modules: gaps growing ~1.6x per module (drift) -> NO_JUMP, no module blamed",
      "verdict=NO_JUMP" in buf.getvalue(), buf.getvalue().strip().splitlines()[-1])

# --- configs ----------------------------------------------------------------------------------------------
arms = dict(configs.arms("vllm"))
check("configs: vLLM arms are A, A2, REF + one flip per switch",
      list(arms)[:3] == ["A", "A2", "REF"] and "flip_aiter" in arms and "flip_fp32" not in arms, list(arms))
check("configs: REF holds every default switch's conservative side",
      arms["REF"].get("VLLM_ROCM_USE_AITER") == "0" and arms["REF"].get("ND_VLLM_CUDAGRAPH_MODE") == "NONE", arms["REF"])
check("configs: fp32 only with --extra", "flip_fp32" in dict(configs.arms("vllm", extra=("fp32",))))
ov = dict(configs.arms("vllm", overrides={"ND_VLLM_ATTENTION_BACKEND": "TRITON_MLA"}))
check("configs: --set overrides a switch's value", ov["flip_attention"]["ND_VLLM_ATTENTION_BACKEND"] == "TRITON_MLA")
one = configs.arms("torch", only=("tf32",))
check("configs: one switch -> no separate flip arm (it is the reference)", [a for a, _ in one] == ["A", "A2", "REF"], one)
root = os.path.join(HERE, "..", "..")
check("configs: detect vLLM example", configs.detect(
    ["python3", "$ND_REPO/examples/inference/qwen_vllm_serve_nan/qwen_vllm_serve.py"], root) == "vllm")
check("configs: detect SGLang example", configs.detect(
    ["python3", "$ND_REPO/examples/inference/qwen_sglang_serve_nan/qwen_sglang_serve.py"], root) == "sglang")
check("configs: detect a training script -> torch", configs.detect(
    ["python3", "$ND_REPO/examples/training/softmoe_train_nan/train.py"], root) == "torch")

print("DIFF_UNIT=%s %d %d" % ("PASS" if not FAILS else "FAIL", N[0], len(FAILS)))
sys.exit(1 if FAILS else 0)
