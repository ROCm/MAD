#!/usr/bin/env python3
"""run_wideep.sh --dry-run: the knobs it derives from BENCH + MODEL + TOPO.

Every DSV4 cell goes through this wrapper, and a wrong derived value does not
fail at submit -- it runs for hours and produces a row that is not comparable
(an unprimed Flash NIAH, a Pro con=1 sweep that overruns the wall, a --time
that slurm holds forever). Each case runs in a clean env so a knob exported in
the caller's shell cannot mask a wrong default. No slurm needed.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

WRAPPER = Path(__file__).resolve().parent.parent / "run_wideep.sh"
FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def plan(*args, **env):
    e = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp")}
    e.update({k: str(v) for k, v in env.items()})
    r = subprocess.run(["bash", str(WRAPPER), *args, "--dry-run"],
                       env=e, capture_output=True, text=True, timeout=30)
    return r


def field(out, line_prefix, key):
    """Value of KEY=... on the plan line starting with line_prefix."""
    for line in out.splitlines():
        if line.startswith(line_prefix):
            m = re.search(r"(?:^|\s)%s=(.*?)(?=\s+[A-Z_]+=|$)" % re.escape(key), line)
            return m.group(1).strip() if m else None
    return None


def extra(out):
    for line in out.splitlines():
        if line.startswith("EXTRA "):
            return line[len("EXTRA "):].split()
    return []


def sbatch(out):
    for line in out.splitlines():
        if line.startswith("SBATCH: "):
            return line[len("SBATCH: "):]
    return ""


def main():
    ladder = "2000,8000,16000,20000,28000,35000"

    print("validate dsv4fls ep32")
    r = plan("validate", "dsv4fls", "ep32")
    o = r.stdout
    check("exits 0", r.returncode == 0, r.stderr)
    check("8 nodes", "-N 8 -n 8 " in sbatch(o), sbatch(o))
    check("con 1 8 16 32", field(o, "SMOKE", "CON") == "1 8 16 32", field(o, "SMOKE", "CON"))
    check("isl/osl 1024", field(o, "SMOKE", "COMBOS") == "1024/1024")
    check("full NIAH ladder", field(o, "NIAH_METHOD", "NIAH_WORDS") == ladder)
    check("hybrid / 512", field(o, "NIAH_METHOD", "NIAH_METHOD") == "hybrid"
          and field(o, "NIAH_METHOD", "MAXTOK") == "512")
    check("validate does not halt NIAH", field(o, "NIAH_METHOD", "HALT") == "0")
    check("Flash is primed 1.", field(o, "NIAH_METHOD", "PRIME") == "1.")
    x = extra(o)
    for kv in ("KV_CACHE_DTYPE=fp8_e4m3", "VLLM_ROCM_USE_AITER_MLA=1",
               "MORI_SHMEM_HEAP_SIZE=34359738368", "DSV4_ENABLE_HMA=0",
               "NIAH_DS_WRAP=0", "NIAH_LIST_PRIME=1.", "CURL_SUITE=short"):
        check(f"EXTRA has {kv}", kv in x, " ".join(x))
    check("no --enforce-eager anywhere", "enforce-eager" not in o)
    check("default 12h clamped to 8h", field(o, "TIME", "TIME") == "08:00:00", field(o, "TIME", "TIME"))

    print("validate dsv4pro ep32")
    o = plan("validate", "dsv4pro", "ep32").stdout
    check("Pro 4P/4D skips con=1", field(o, "SMOKE", "CON") == "8 16 32", field(o, "SMOKE", "CON"))
    check("Pro 4P/4D step budget 3 s/tok", field(o, "SMOKE", "STEP_SEC_PER_TOK") == "3")
    check("Pro is unprimed", field(o, "NIAH_METHOD", "PRIME") == "")
    check("Pro heap 16 GiB", "MORI_SHMEM_HEAP_SIZE=17179869184" in extra(o))

    print("prime overrides")
    o = plan("validate", "dsv4pro", "ep16", NIAH_LIST_PRIME="1.").stdout
    check("Pro ignores a leaked Flash prime", field(o, "NIAH_METHOD", "PRIME") == "")
    o = plan("niah", "dsv4fls", "2p2d", NIAH_LIST_PRIME="off").stdout
    check("Flash prime=off is unprimed", field(o, "NIAH_METHOD", "PRIME") == "")
    o = plan("validate", "dsv4pro", "ep16").stdout
    check("Pro 2P/2D keeps con=1", field(o, "SMOKE", "CON") == "1 8 16 32")
    o = plan("niah", "dsv4fls", "2p2d", NIAH_METHOD="product").stdout
    check("product method caps at 64", field(o, "NIAH_METHOD", "MAXTOK") == "64")

    print("bench shapes")
    o = plan("niah", "dsv4fls", "2p2d").stdout
    check("niah runs the full ladder", field(o, "NIAH_METHOD", "HALT") == "0")
    o = plan("niah", "dsv4pro", "2p2d", NIAH_HALT_ON_FAIL="1").stdout
    check("NIAH_HALT_ON_FAIL=1 is honoured", field(o, "NIAH_METHOD", "HALT") == "1")
    check("niah has no sweep line", not any(l.startswith("SMOKE") for l in o.splitlines()))
    o = plan("accuracy", "dsv4fls", "2p2d").stdout
    check("accuracy has no NIAH line", not any(l.startswith("NIAH_METHOD") for l in o.splitlines()))
    check("accuracy defaults to chat", "ACC_THINKING=chat" in extra(o))
    o = plan("smoke", "dsv4fls", "1p1d", DSV4_EAGER="1").stdout
    check("DSV4_EAGER=1 sets decode graphs NONE", "DECODE_CUDAGRAPH_MODE=NONE" in extra(o))

    print("walltime")
    o = plan("validate", "dsv4fls", "ep32", QOS="low").stdout
    check("QOS=low clamps to 4h", field(o, "TIME", "TIME") == "04:00:00")
    check("QOS=low reaches sbatch", "--qos=low" in sbatch(o))
    r = plan("validate", "dsv4fls", "ep32", "--time", "10:00:00")
    check("explicit --time over the cap is kept", field(r.stdout, "TIME", "TIME") == "10:00:00")
    check("explicit --time over the cap warns", "exceeds assoc MaxWall" in r.stderr, r.stderr)
    r = plan("smoke", "dsv4fls", "2p3d")
    check("asymmetric 2p3d is 5 nodes", "-N 5 -n 5 " in sbatch(r.stdout))
    check("asymmetric gets a walltime", field(r.stdout, "TIME", "TIME") == "04:00:00",
          field(r.stdout, "TIME", "TIME"))

    print("sbatch flags")
    o = plan("validate", "dsv4fls", "ep16", "--after", "12345", "--exclude", "node-[1-2]").stdout
    s = sbatch(o)
    check("--after chains afterany", "--dependency=afterany:12345" in s, s)
    check("--exclude reaches sbatch", "--exclude=node-[1-2]" in s, s)
    o = plan("validate", "dsv4fls", "ep16", "--nodelist", "node-3", "--exclude", "node-1").stdout
    s = sbatch(o)
    check("--nodelist replaces --exclude", "--nodelist=node-3" in s and "--exclude" not in s, s)
    o = plan("validate", "dsv4fls", "ep16", PARTITION="p1", ACCOUNT="a1").stdout
    check("partition and account reach sbatch", "-A a1 -p p1 " in sbatch(o), sbatch(o))
    check("router port below the ephemeral range",
          "MORI_PROXY_PING_PORT=31367" in extra(plan("validate", "dsv4fls", "ep16").stdout))

    print("bad input")
    for args in (("validate", "nosuchmodel", "ep16"), ("validate", "dsv4fls", "9p9d"),
                 ("nosuchbench", "dsv4fls", "ep16")):
        r = plan(*args)
        check(f"rejects {' '.join(args)}", r.returncode == 2, f"rc={r.returncode}")

    if FAILED:
        print(f"\n{len(FAILED)} wrapper check(s) failed")
        return 1
    print("\nall wrapper dry-run checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
