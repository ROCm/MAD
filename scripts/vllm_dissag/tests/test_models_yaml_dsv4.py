#!/usr/bin/env python3
"""models.yaml DSV4 entries: the values DSV4 cannot boot or score without.

Each of these was learned from a failed or non-comparable run, and a yaml edit
breaks none of them loudly: the job boots on the wrong KV dtype, or the 35k
rung is silently truncated by max-model-len. No cluster.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

SRC = Path(__file__).resolve().parent.parent
FAILED = []
DSV4 = ("DeepSeek-V4-Flash-FP8", "DeepSeek-V4-Pro-FP8")
HEAP = {"DeepSeek-V4-Flash-FP8": "34359738368", "DeepSeek-V4-Pro-FP8": "17179869184"}
# Measured on Flash's tokenizer: 1.0799-1.0812 tok/word on the NIAH prompt.
TOK_PER_WORD = 1.09


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def flag(s, name):
    m = re.search(r"--%s[ =](\S+)" % re.escape(name), s or "")
    return m.group(1) if m else None


def yaml_env(model, **pre):
    """Run vllm_disagg.sh's own models.yaml env loader and return what it exports."""
    src = (SRC / "vllm_disagg.sh").read_text()
    m = re.search(r"_yaml_env=\"\$\(python3 - <<'PY'\n(.*?)\nPY\n", src, re.S)
    if not m:
        return None
    e = {"PATH": os.environ["PATH"], "MODELS_YAML": str(SRC / "models.yaml"), "MODEL_NAME": model}
    e.update(pre)
    r = subprocess.run([sys.executable, "-c", m.group(1)], env=e, capture_output=True, text=True)
    return dict(re.findall(r"^export (\w+)=(.*)$", r.stdout, re.M))


def main():
    y = yaml.safe_load((SRC / "models.yaml").read_text())
    slurm = (SRC / "run_xPyD_models.slurm").read_text()
    wrapper = (SRC / "run_wideep_bench.sh").read_text()
    ladder = re.search(r'NIAH_WORDS="\$\{NIAH_WORDS:-([\d,]+)\}"', wrapper).group(1)
    top_rung = max(int(w) for w in ladder.split(","))

    for model in DSV4:
        print(model)
        e = y.get(model) or {}
        env = e.get("env") or {}
        check("entry exists", bool(e))
        check("is WideEP-only in the slurm", f'"{model}"' in slurm.split("WIDE_EP_ONLY_MODELS=", 1)[1].split("\n", 1)[0])
        check("KV dtype fp8_e4m3 (fp8 remaps to fp8_ds_mla, which ROCm MLA rejects)",
              env.get("KV_CACHE_DTYPE") == "fp8_e4m3", env.get("KV_CACHE_DTYPE"))
        check("AITER MLA on (V4 sparse attention needs it)", env.get("VLLM_ROCM_USE_AITER_MLA") == "1")
        bs = int(env.get("KV_BLOCK_SIZE", "0"))
        check("block size is a multiple of lcm(4,128)=128", bs > 0 and bs % 128 == 0, str(bs))
        check("HMA off", env.get("DSV4_ENABLE_HMA") == "0")
        check("decode graphs FULL_DECODE_ONLY (FULL is downgraded on DSV4)",
              env.get("DECODE_CUDAGRAPH_MODE") == "FULL_DECODE_ONLY")
        check("heap matches the wrapper", env.get("MORI_SHMEM_HEAP_SIZE") == HEAP[model])
        check("values are strings", all(isinstance(v, str) for v in env.values()))
        flags = " ".join(str(v) for v in [e.get("dp_flags"), *(e.get("prefill") or {}).values(),
                                           *(e.get("decode") or {}).values()])
        check("never --enforce-eager", "enforce-eager" not in flags)
        check("deepseek_v4 tokenizer mode", flag(e.get("dp_flags"), "tokenizer-mode") == "deepseek_v4")
        for role in ("prefill", "decode"):
            dp = (e.get(role) or {}).get("dp", "")
            check(f"{role}: no TP flags (WideEP is -tp 1 + DP)", not (e.get(role) or {}).get("tp"))
            mml = int(flag(dp, "max-model-len") or 0)
            need = int(top_rung * TOK_PER_WORD) + 512
            check(f"{role}: max-model-len fits the {top_rung}-word rung + 512 answer",
                  mml >= need, f"{mml} < {need}")
            check(f"{role}: mnbt 1024", flag(dp, "max-num-batched-tokens") == "1024")

    print("other models")
    leaks = [m for m, e in y.items() if m not in DSV4 and isinstance(e, dict)
             and any(k.startswith("DSV4_") for k in (e.get("env") or {}))]
    check("no DSV4_ knob outside the DSV4 entries", not leaks, str(leaks))

    print("loader precedence")
    got = yaml_env("DeepSeek-V4-Flash-FP8")
    check("loader found in vllm_disagg.sh", got is not None)
    if got is not None:
        check("yaml value is exported", got.get("KV_CACHE_DTYPE") == "fp8_e4m3", str(got.get("KV_CACHE_DTYPE")))
        got = yaml_env("DeepSeek-V4-Flash-FP8", KV_CACHE_DTYPE="auto")
        check("submit-time env wins over yaml", "KV_CACHE_DTYPE" not in got)
        check("unknown model exports nothing", yaml_env("no-such-model") == {})

    if FAILED:
        print(f"\n{len(FAILED)} models.yaml check(s) failed")
        return 1
    print("\nall models.yaml DSV4 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
