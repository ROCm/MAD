# debug-numeric differential mode: record what the model returned, and apply the configuration switches -- from
# the environment, no model edit (`step2-diff.md`).
#
#   PYTHONPATH=<REPO_ROOT>/scripts/diff_tap:<REPO_ROOT>/scripts  ND_DIFF_OUT=<dir>  python3 serve.py ...
#
# THE TAP (ND_DIFF_OUT=<dir>: one <dir>/outputs.jsonl per process that returns outputs; the format is
# output_diff.py's):
#   vllm.LLM.generate         per request: prompt-id hash, output token ids, the chosen token's logprob and the
#                             top-1 minus top-2 margin per position. SamplingParams.logprobs is raised to >= 2 on a
#                             copy (the engine computes two more logprobs; the model's math is unchanged); the
#                             caller still gets RequestOutputs, with up to 2 more logprob entries per position.
#   sglang.Engine.generate    the same, through return_logprob / top_logprobs_num >= 2.
#   transformers generate     the same, through an extra LogitsProcessor that reads the scores (unchanged).
#   training                  per Tensor.backward() on a scalar: the loss; clip_grad_norm_'s result: the grad norm
#                             (rank 0 only under torchrun). Off for a process that never calls backward.
#   ND_DIFF_TAP=fwd           the fallback: every call of the outermost nn.Module -- last-dim argmax of its output
#                             (<= 4096 rows), l2, absmax, finite -- for a hand-written loop with none of the above.
#   ND_DIFF_FUNC=[module:]name  the last resort, for a script with no generate API and no nn.Module: the return value
#                             of every call of that Python function (module default __main__), through
#                             sys.setprofile -- an integer tensor's rows as token records (no logprobs: a differing
#                             token is a divergence; inputs unverified), a float tensor as an fwd record.
#   ND_DIFF_TAP=<list>        restrict the taps: gen,step,fwd (default gen,step).
#   One record = one host read of the outputs the caller is about to read anyway (gen), or of one scalar (step).
#
# CHECK MODE (ND_DIFF_REF=<reference outputs.jsonl>, ND_DIFF_THR=<diff.json from output_diff.py --json>):
#   each record is judged against the reference's record of the same index as it is written; a divergence prints
#   `ND_DIFF_HIT kind=<k> i=<n> pos=<p> gap=<g> threshold=<t>` -- the <NAN_SIGNAL> of steps 2-5 for a model that
#   prints nothing itself. At exit: `ND_DIFF_CHECK records=<n> hits=<h> tie_flips=<t> inputs_differ=<d>`.
#   ND_DIFF_SAME_CONFIG=1: the reference is a run of the SAME configuration (A2 of a NONDETERMINISTIC wave):
#   a differing token counts even without logprobs (the ND_DIFF_FUNC tap).
#
# PER-MODULE RECORDING (ND_DIFF_MODULES=<request / step indices, comma list>, with ND_DIFF_OUT): while those
#   requests run, every nn.Module call's output is summarised -- class#instance (first-seen order), call order,
#   l2, absmax, finite, the sum and two fixed random-sign projections -- into <dir>/modules.jsonl; output_diff.py --modules compares two runs
#   and names the first module that disagrees. Needs module forwards in Python: eager configurations (a compiled
#   module is seen at its boundary only) and an in-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0).
#
# THE SWITCHES (configs.py): engine arguments and torch settings with no variable of their own --
#   ND_VLLM_CUDAGRAPH_MODE=<mode>  ND_VLLM_COMPILE=0  ND_VLLM_CUSTOM_OPS=none  ND_VLLM_ATTENTION_BACKEND=<name>
#   ND_VLLM_DTYPE=<dtype>          (vllm.engine.arg_utils.EngineArgs, after __post_init__)
#   ND_SGLANG_DISABLE_CUDA_GRAPH=1  ND_SGLANG_ATTENTION_BACKEND=<name>  ND_SGLANG_DTYPE=<dtype>
#                                  (sglang Engine(**kwargs), before ServerArgs is built)
#   ND_DIFF_TF32=0                 TF32 off however the model asks for it (allow_tf32, fp32_precision,
#                                  set_float32_matmul_precision)
#   ND_DIFF_SDPA=math              flash / mem-efficient / cuDNN SDPA off: the math backend
# Each applied switch prints `[diff-tap] switch <what>`. Validated with vLLM 0.33.0, SGLang 0.5.23, transformers
# 5.20.2, torch 2.10 (2026-11-24); re-check the patch points after an upgrade. A sitecustomize later on sys.path
# (the image's own, or another of ours) is still run.
import atexit
import hashlib
import importlib.abc
import importlib.util
import json
import os
import sys
import threading

_OUT = os.environ.get("ND_DIFF_OUT", "")
_TAPS = set((os.environ.get("ND_DIFF_TAP") or "gen,step").split(","))
_REF = os.environ.get("ND_DIFF_REF", "")
_MOD_REQS = {int(x) for x in os.environ.get("ND_DIFF_MODULES", "").split(",") if x.strip().isdigit()}
_FUNC = os.environ.get("ND_DIFF_FUNC", "")
_LOCK = threading.Lock()
_S = {"gen": 0, "step": 0, "fwd": 0, "hits": 0, "ties": 0, "inp": 0, "n": 0, "streak": 0, "files": {},
      "ref": None, "thr": None, "mod_on": False, "mod_seq": 0, "mod_req": None, "mod_ids": {}}


def _say(msg):
    print("[diff-tap] " + msg, flush=True)


def _rank0():
    return os.environ.get("RANK", "0") == "0" and os.environ.get("LOCAL_RANK", "0") == "0"


def _file(name):
    f = _S["files"].get(name)
    if f is None:
        os.makedirs(_OUT, exist_ok=True)
        f = _S["files"][name] = open(os.path.join(_OUT, name), "a")
    return f


def _hash(ids):
    return hashlib.sha1(json.dumps([int(x) for x in ids]).encode()).hexdigest()[:16]


def _load_ref():
    if _S["ref"] is not None or not _REF:
        return
    import output_diff
    _, _S["ref"] = output_diff.load(_REF)
    thr = os.environ.get("ND_DIFF_THR", "")
    if thr and os.path.isfile(thr):
        _S["thr"] = json.load(open(thr)).get("thresholds")
    if not _S["thr"]:
        _S["thr"] = output_diff.thresholds(os.environ.get("ND_DIFF_DTYPE", "bfloat16"), {}, 0.0)
    _say("check mode: %d reference records from %s, thresholds %s" % (len(_S["ref"]), _REF, _S["thr"]))


def _check(rec):
    import output_diff
    _load_ref()
    r = _S["ref"].get((rec["kind"], rec["i"]))
    if r is None:
        return
    _S["n"] += 1
    v, d = output_diff.JUDGE[rec["kind"]](rec, r, _S["thr"][rec["kind"]])
    if v == "INPUTS_DIFFER":
        _S["inp"] += 1
        if _S["inp"] == 1:
            _say("ND_DIFF_INPUTS_DIFFER kind=%s i=%d: this run's inputs are not the reference's -- pin the inputs "
                 "(seed, data order) or the check is meaningless" % (rec["kind"], rec["i"]))
        return
    if v == "TIE_FLIP":
        _S["ties"] += 1
    if v == "UNMARGINED":                # a differing token with no logprobs: a hit only against the same config
        if os.environ.get("ND_DIFF_SAME_CONFIG") != "1":
            return
        v = "DIVERGE"
    if rec["kind"] == "step":           # a training gap counts once it persists (output_diff.STEP_RUN)
        _S["streak"] = _S["streak"] + 1 if v == "DIVERGE" else 0
        finite = output_diff._finite(d.get("gap"))
        if not (v == "DIVERGE" and (not finite or _S["streak"] == output_diff.STEP_RUN)):
            return
    elif v != "DIVERGE":
        return
    _S["hits"] += 1
    print("ND_DIFF_HIT kind=%s i=%d pos=%s gap=%.4g threshold=%.4g" % (
        rec["kind"], rec["i"], d.get("pos", "-"), d.get("gap", float("nan")), _S["thr"][rec["kind"]]), flush=True)


def _emit(rec):
    with _LOCK:
        if _OUT:
            f = _file("outputs.jsonl")
            f.write(json.dumps(rec) + "\n")
            f.flush()
        if _REF:
            _check(rec)


def _meta(**kw):
    if _OUT and not _S.get("meta"):
        _S["meta"] = True
        rec = {"kind": "meta", "pid": os.getpid(), "host": os.uname().nodename,
               "switches": {k: v for k, v in os.environ.items()
                            if k.startswith(("ND_VLLM_", "ND_SGLANG_", "ND_DIFF_TF32", "ND_DIFF_SDPA", "VLLM_ROCM_USE",
                                             "SGLANG_FORCE", "TORCHDYNAMO_DISABLE", "HIPBLASLT_ALLOW_TF32",
                                             "TORCH_BLAS_PREFER"))}}
        rec.update(kw)
        with _LOCK:
            _file("outputs.jsonl").write(json.dumps(rec) + "\n")


def _dtype_name(dt):
    s = str(dt).replace("torch.", "")
    return s if s in ("bfloat16", "float16", "float32") else None


def _f(x):
    x = float(x)
    return x if x == x and abs(x) != float("inf") else (None if x != x else x)


def _gen_record(prompt_ids, tokens, per_pos):
    """per_pos: one {token_id: logprob} dict per position (top-k + the chosen token)."""
    i = _S["gen"]
    _S["gen"] += 1
    lp, margin = [], []
    for t, d in zip(tokens, per_pos):
        if not d:
            lp.append(None)
            margin.append(None)
            continue
        lp.append(float(d[t]) if t in d else None)
        top = sorted(d.values(), reverse=True)
        margin.append(float(top[0] - top[1]) if len(top) > 1 else None)
    rec = {"kind": "gen", "i": i, "inp": _hash(prompt_ids), "plen": len(prompt_ids), "tokens": [int(t) for t in tokens],
           "lp": lp, "margin": margin}
    _emit(rec)
    return i


# ---- module recording (localisation) -------------------------------------------------------------------------


def _mod_arm(i):
    _S["mod_on"] = i in _MOD_REQS
    _S["mod_req"] = i if _S["mod_on"] else None
    _S["mod_seq"] = 0


_SIGNS = {}


def _signs(n, device):
    """Two fixed pseudo-random +-1 vectors of length n (seeded: the same in every run on this device type) --
    a projection on them changes under any rearrangement of the values (a head swap leaves the l2 and the plain
    sum unchanged; 2026-11-24)."""
    import torch
    key = (n, str(device))
    v = _SIGNS.get(key)
    if v is None:
        g = torch.Generator(device=device).manual_seed(1234567)
        v = _SIGNS[key] = torch.randint(0, 2, (2, n), generator=g, device=device).float().mul_(2).sub_(1)
    return v


def _summ(t):
    import torch
    x = t.detach()
    if x.is_complex() or not x.is_floating_point():
        return None
    x = x.float().flatten()
    n = x.numel()
    if n == 0:
        return None
    fin = bool(torch.isfinite(x).all())
    sg = _signs(n, x.device)
    v = torch.cat([torch.stack([x.norm(), x.abs().max(), x.sum()]), sg @ x]).tolist()
    return {"n": n, "l2": v[0], "absmax": v[1], "proj": v[2:], "finite": fin}


def _first_tensor(out):
    import torch
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            t = _first_tensor(o)
            if t is not None:
                return t
    if isinstance(out, dict):
        for k in ("logits", "last_hidden_state"):
            if isinstance(out.get(k), torch.Tensor):
                return out[k]
        for o in out.values():
            t = _first_tensor(o)
            if t is not None:
                return t
    if hasattr(out, "logits"):
        return getattr(out, "logits")
    return None


def _install_module_hooks():
    import torch
    from torch.nn.modules import module as M
    depth = [0]

    def pre(mod, args):
        depth[0] += 1

    def post(mod, args, out):
        depth[0] -= 1
        if _S["mod_on"]:
            t = _first_tensor(out)
            s = _summ(t) if t is not None else None
            if s is not None:
                key = id(mod)
                name = _S["mod_ids"].get(key)
                if name is None:
                    k = type(mod).__name__
                    n = sum(1 for v in _S["mod_ids"].values() if v.split("#")[0] == k)
                    name = _S["mod_ids"][key] = "%s#%d" % (k, n)
                s.update(req=_S["mod_req"], seq=_S["mod_seq"], mod=name, depth=depth[0])
                _S["mod_seq"] += 1
                with _LOCK:
                    _file("modules.jsonl").write(json.dumps(s) + "\n")
        if depth[0] == 0 and "fwd" in _TAPS and torch.is_tensor(_first_tensor(out)):
            _fwd_record(args, out)

    M.register_module_forward_pre_hook(pre)
    M.register_module_forward_hook(post)


def _fwd_record(args, out):
    import torch
    t = _first_tensor(out)
    x = t.detach().float()
    inp = _first_tensor(args)
    if torch.is_tensor(inp):
        a = inp.detach()
        h = _hash(a.flatten()[:4096].tolist()) if not a.is_floating_point() else \
            "f%.6e" % float(a.float().sum()) + "/%.6e" % float(a.float().abs().sum())
    else:
        h = "-"
    am = x.reshape(-1, x.shape[-1]).argmax(-1)[:4096].tolist() if x.dim() >= 1 and x.numel() else []
    i = _S["fwd"]
    _S["fwd"] += 1
    _emit({"kind": "fwd", "i": i, "inp": h, "argmax": am, "l2": _f(x.norm()), "absmax": _f(x.abs().max()),
           "finite": bool(torch.isfinite(x).all())})
    if _MOD_REQS:
        _mod_arm(_S["fwd"])


def _install_func_tap():
    mod, _, name = _FUNC.rpartition(":")
    mod = mod or "__main__"
    calls = [0]

    def prof(frame, event, arg):
        if event != "return" or frame.f_code.co_name != name or frame.f_globals.get("__name__") != mod:
            return
        try:
            _func_record(calls[0], arg)
        except Exception as e:
            _say("tap failed on %s's return value: %r" % (_FUNC, e))
        calls[0] += 1

    sys.setprofile(prof)
    threading.setprofile(prof)
    _say("tap on the return value of %s:%s (pid %d)" % (mod, name, os.getpid()))


def _func_record(call, ret):
    import torch
    t = _first_tensor(ret)
    if t is None:
        if call == 0:
            _say("%s returned no tensor: nothing recorded" % _FUNC)
        return
    if not _S.get("meta"):
        _meta(framework="func", dtype=_dtype_name(t.dtype) if t.is_floating_point() else None, func=_FUNC)
    if t.is_floating_point():
        x = t.detach().float()
        am = x.reshape(-1, x.shape[-1]).argmax(-1)[:4096].tolist() if x.dim() >= 1 and x.numel() else []
        i = _S["fwd"]
        _S["fwd"] += 1
        _emit({"kind": "fwd", "i": i, "inp": "call%d" % call, "argmax": am, "l2": _f(x.norm()),
               "absmax": _f(x.abs().max()), "finite": bool(torch.isfinite(x).all())})
        return
    rows = t.detach().reshape(-1, t.shape[-1]).cpu().tolist() if t.dim() >= 1 else [[int(t)]]
    for r, toks in enumerate(rows):
        _emit({"kind": "gen", "i": _S["gen"], "inp": "call%d/row%d" % (call, r), "tokens": toks, "lp": [],
               "margin": []})
        _S["gen"] += 1


# ---- vLLM ----------------------------------------------------------------------------------------------------


def _patch_vllm_args(mod):
    ea = mod.EngineArgs
    orig = ea.__post_init__
    gmode = os.environ.get("ND_VLLM_CUDAGRAPH_MODE", "").strip().upper()
    comp = os.environ.get("ND_VLLM_COMPILE", "")
    cops = os.environ.get("ND_VLLM_CUSTOM_OPS", "")
    attn = os.environ.get("ND_VLLM_ATTENTION_BACKEND", "")
    dt = os.environ.get("ND_VLLM_DTYPE", "")

    def __post_init__(self, *a, **k):
        orig(self, *a, **k)
        cc = self.compilation_config
        done = []

        def setcc(key, val):
            if isinstance(cc, dict):
                cc[key] = val
            else:
                setattr(cc, key, val)
        if gmode:
            from vllm.config.compilation import CUDAGraphMode
            setcc("cudagraph_mode", gmode if isinstance(cc, dict) else CUDAGraphMode[gmode])
            done.append("cudagraph_mode=" + gmode)
        if comp == "0":
            from vllm.config.compilation import CompilationMode
            setcc("mode", 0 if isinstance(cc, dict) else CompilationMode.NONE)
            done.append("compilation mode NONE")
        if cops:
            setcc("custom_ops", [c for c in cops.split(",") if c])
            done.append("custom_ops=" + cops)
        if attn:
            self.attention_backend = attn
            done.append("attention_backend=" + attn)
        if dt:
            self.dtype = dt
            done.append("dtype=" + dt)
        if done:
            _say("switch vLLM " + ", ".join(done))

    ea.__post_init__ = __post_init__


def _patch_vllm_llm(mod):
    LLM = mod.LLM
    orig = LLM.generate

    def generate(self, *a, **kw):
        import copy
        sp_pos = 1 if len(a) > 1 else None
        sp = a[1] if sp_pos else kw.get("sampling_params")
        if sp is None:
            sp = self.get_default_sampling_params()

        def raise_lp(s):
            s = copy.copy(s)
            s.logprobs = max(s.logprobs or 0, 2)
            return s
        sp2 = [raise_lp(s) for s in sp] if isinstance(sp, (list, tuple)) else raise_lp(sp)
        if sp_pos:
            a = (a[0], sp2) + tuple(a[2:])
        else:
            kw["sampling_params"] = sp2
        if not _S.get("meta"):
            try:
                dt = _dtype_name(self.llm_engine.model_config.dtype)
            except Exception:
                dt = None
            _meta(framework="vllm", dtype=dt)
        if _MOD_REQS:
            _mod_arm(_S["gen"])
        outs = orig(self, *a, **kw)
        try:
            for ro in outs:
                co = ro.outputs[0]
                per = []
                for p, t in enumerate(co.token_ids):
                    d = co.logprobs[p] if co.logprobs is not None and p < len(co.logprobs) else None
                    per.append({int(k): v.logprob for k, v in d.items()} if d else None)
                _gen_record(ro.prompt_token_ids or [], list(co.token_ids), per)
                if _MOD_REQS:
                    _mod_arm(_S["gen"])
        except Exception as e:  # never break the model's own loop
            _say("tap failed on a vLLM output: %r" % (e,))
        return outs

    LLM.generate = generate
    _say("tap on vllm.LLM.generate (pid %d)" % os.getpid())


# ---- SGLang --------------------------------------------------------------------------------------------------


def _patch_sglang_engine(mod):
    Engine = mod.Engine
    oinit, ogen = Engine.__init__, Engine.generate
    ncg = os.environ.get("ND_SGLANG_DISABLE_CUDA_GRAPH", "")
    attn = os.environ.get("ND_SGLANG_ATTENTION_BACKEND", "")
    dt = os.environ.get("ND_SGLANG_DTYPE", "")

    def __init__(self, *a, **kw):
        done = []
        if ncg == "1":
            kw["disable_cuda_graph"] = True
            done.append("disable_cuda_graph")
        if attn:
            kw["attention_backend"] = attn
            done.append("attention_backend=" + attn)
        if dt:
            kw["dtype"] = dt
            done.append("dtype=" + dt)
        if done:
            _say("switch SGLang " + ", ".join(done))
        oinit(self, *a, **kw)
        d = kw.get("dtype")
        _meta(framework="sglang", dtype=_dtype_name(d) if d not in (None, "auto") else None)

    def generate(self, *a, **kw):
        if not _OUT and not _REF:
            return ogen(self, *a, **kw)
        if kw.get("stream"):
            return ogen(self, *a, **kw)
        kw["return_logprob"] = True
        kw["top_logprobs_num"] = max(kw.get("top_logprobs_num") or 0, 2)
        ids = kw.get("input_ids")
        outs = ogen(self, *a, **kw)
        try:
            lst = outs if isinstance(outs, list) else [outs]
            if ids is not None and ids and not isinstance(ids[0], (list, tuple)):
                ids = [ids]
            for j, o in enumerate(lst):
                mi = o.get("meta_info", {})
                toks = o.get("output_ids") or [t[1] for t in mi.get("output_token_logprobs", [])]
                chosen = mi.get("output_token_logprobs") or []
                tops = mi.get("output_top_logprobs") or []
                per = []
                for p, t in enumerate(toks):
                    d = {}
                    if p < len(tops) and tops[p]:
                        d.update({int(e[1]): e[0] for e in tops[p]})
                    if p < len(chosen) and chosen[p]:
                        d[int(chosen[p][1])] = chosen[p][0]
                    per.append(d or None)
                prompt = ids[j] if ids is not None and j < len(ids) else \
                    [ord(c) for c in str((kw.get("prompt") or [""] * (j + 1))[j] if isinstance(kw.get("prompt"), list)
                                         else kw.get("prompt") or "")]
                _gen_record(prompt, toks, per)
        except Exception as e:
            _say("tap failed on an SGLang output: %r" % (e,))
        return outs

    Engine.__init__ = __init__
    Engine.generate = generate
    _say("tap on sglang Engine.generate (pid %d)" % os.getpid())


# ---- transformers --------------------------------------------------------------------------------------------


def _patch_hf(mod):
    GM = mod.GenerationMixin
    orig = GM.generate

    def generate(self, *a, **kw):
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList
        steps = []

        class _Rec(LogitsProcessor):
            def __call__(self, input_ids, scores):
                v, ix = torch.log_softmax(scores.float(), -1).topk(2, dim=-1)
                steps.append((v.detach(), ix.detach()))
                return scores

        lp = kw.get("logits_processor")
        kw["logits_processor"] = LogitsProcessorList(list(lp or []) + [_Rec()])
        inp = kw.get("input_ids", a[0] if a and torch.is_tensor(a[0]) else kw.get("inputs"))
        if not _S.get("meta"):
            _meta(framework="hf", dtype=_dtype_name(getattr(self, "dtype", None)))
        if _MOD_REQS:
            _mod_arm(_S["gen"])
        out = orig(self, *a, **kw)
        try:
            seq = out if torch.is_tensor(out) else out.sequences
            n0 = inp.shape[-1] if torch.is_tensor(inp) else 0
            if steps:
                V = torch.stack([s[0] for s in steps], 1).cpu().tolist()   # (batch, steps, 2)
                I = torch.stack([s[1] for s in steps], 1).cpu().tolist()
            for b in range(seq.shape[0]):
                toks = seq[b, n0:].tolist()
                per = [dict(zip(I[b][p], V[b][p])) if steps and p < len(I[b]) else None for p in range(len(toks))]
                _gen_record(inp[b].tolist() if torch.is_tensor(inp) else [], toks, per)
        except Exception as e:
            _say("tap failed on a transformers output: %r" % (e,))
        return out

    GM.generate = generate
    _say("tap on transformers GenerationMixin.generate (pid %d)" % os.getpid())


# ---- torch: training taps and the torch-level switches -------------------------------------------------------


def _patch_torch(torch):
    if os.environ.get("ND_DIFF_TF32") == "0":
        m = torch.backends.cuda.matmul
        cls = type(m)
        oset = cls.__setattr__

        def __setattr__(self, name, value):
            if name == "allow_tf32":
                value = False
            elif name == "fp32_precision" and value == "tf32":
                value = "ieee"
            return oset(self, name, value)
        cls.__setattr__ = __setattr__
        m.allow_tf32 = False
        osp = torch.set_float32_matmul_precision
        torch.set_float32_matmul_precision = lambda p: osp("highest")
        osp("highest")
        try:
            ccls = type(torch.backends.cudnn)
            ocset = ccls.__setattr__

            def cset(self, name, value):
                return ocset(self, name, False if name == "allow_tf32" else value)
            ccls.__setattr__ = cset
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        _say("switch TF32 off (allow_tf32 / fp32_precision / set_float32_matmul_precision pinned)")
    if os.environ.get("ND_DIFF_SDPA") == "math":
        for f in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp"):
            if hasattr(torch.backends.cuda, f):
                getattr(torch.backends.cuda, f)(False)
                setattr(torch.backends.cuda, f, lambda enabled, _f=f: None)
        _say("switch SDPA -> math only")
    if not (_OUT or _REF):
        return
    if "step" in _TAPS and _rank0():
        T = torch.Tensor
        obw = T.backward

        def backward(self, *a, **kw):
            if self.dim() == 0 or self.numel() == 1:
                if not _S.get("meta"):
                    _meta(framework="torch", dtype=None)   # autocast or not: unknown here, output_diff defaults to bf16 floors
                i = _S["step"]
                _S["step"] += 1
                if _MOD_REQS:
                    _mod_arm(_S["step"])
                rec = {"kind": "step", "i": i, "loss": _f(self.detach().float().reshape(()).item())}
                _S["last_step"] = rec
                _emit(rec)
            return obw(self, *a, **kw)
        T.backward = backward
        import torch.nn.utils as U
        import torch.nn.utils.clip_grad as CG
        ocl = CG.clip_grad_norm_

        def clip_grad_norm_(*a, **kw):
            n = ocl(*a, **kw)
            last = _S.get("last_step")
            if last is not None and _OUT:
                try:
                    with _LOCK:
                        _file("outputs.jsonl").write(json.dumps(
                            {"kind": "step", "i": last["i"], "gnorm": _f(n.float().item())}) + "\n")
                except Exception:
                    pass
            return n
        CG.clip_grad_norm_ = clip_grad_norm_
        U.clip_grad_norm_ = clip_grad_norm_
        _say("tap on Tensor.backward / clip_grad_norm_ (pid %d)" % os.getpid())
    if _FUNC:
        _install_func_tap()
    if "fwd" in _TAPS or _MOD_REQS:
        if not _S.get("meta") and "fwd" in _TAPS:
            _meta(framework="torch", dtype=None)
        _install_module_hooks()
        if _MOD_REQS:
            _mod_arm(0)
        _say("module hooks installed (fwd tap: %s, per-module recording for %s)"
             % ("fwd" in _TAPS, sorted(_MOD_REQS) or "-"))


def _close():
    if _REF:
        print("ND_DIFF_CHECK records=%d hits=%d tie_flips=%d inputs_differ=%d"
              % (_S["n"], _S["hits"], _S["ties"], _S["inp"]), flush=True)
    for f in _S["files"].values():
        try:
            f.close()
        except Exception:
            pass


_TARGETS = {}
if any(os.environ.get(k) for k in ("ND_VLLM_CUDAGRAPH_MODE", "ND_VLLM_COMPILE", "ND_VLLM_CUSTOM_OPS",
                                   "ND_VLLM_ATTENTION_BACKEND", "ND_VLLM_DTYPE")):
    _TARGETS["vllm.engine.arg_utils"] = _patch_vllm_args
if _OUT or _REF:
    if "gen" in _TAPS:
        _TARGETS["vllm.entrypoints.llm"] = _patch_vllm_llm
        _TARGETS["transformers.generation.utils"] = _patch_hf
if _OUT or _REF or any(os.environ.get(k) for k in ("ND_SGLANG_DISABLE_CUDA_GRAPH", "ND_SGLANG_ATTENTION_BACKEND",
                                                   "ND_SGLANG_DTYPE")):
    _TARGETS["sglang.srt.entrypoints.engine"] = _patch_sglang_engine
if _OUT or _REF or _FUNC or os.environ.get("ND_DIFF_TF32") == "0" or os.environ.get("ND_DIFF_SDPA"):
    _TARGETS["torch"] = _patch_torch


class _Finder(importlib.abc.MetaPathFinder):
    """Patches each target module right after it executes, wherever it is first imported."""

    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module, patch = spec.loader.exec_module, _TARGETS[name]

        def exec_and_patch(module):
            exec_module(module)
            try:
                patch(module)
            except Exception as e:
                _say("patching %s failed: %r -- running unpatched" % (name, e))

        spec.loader.exec_module = exec_and_patch
        return spec


def _chain():
    """Run the next sitecustomize on sys.path -- one AFTER this directory: this one shadows it, and an
    earlier one (another of ours on the same PYTHONPATH) already ran and chained here."""
    here = os.path.dirname(os.path.abspath(__file__))
    seen = False
    for d in sys.path:
        if os.path.abspath(d or ".") == here:
            seen = True
            continue
        f = os.path.join(d or ".", "sitecustomize.py")
        if seen and os.path.isfile(f):
            name = "_sitecustomize_next_%x" % (hash(os.path.abspath(f)) & 0xffffffff)
            spec = importlib.util.spec_from_file_location(name, f)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return


if _TARGETS:
    sys.meta_path.insert(0, _Finder())
    atexit.register(_close)
_chain()
