#!/usr/bin/env python3
"""Smoke test of a di_vllm image against the pin set it was built from.

Run inside the image build (docker/di_vllm.ubuntu.amd.Dockerfile):
  check_vllm_stack.py aiter        AITER (and flydsl) are what the pin set says
  check_vllm_stack.py aiter mori   ...and MoRI imports (after the vLLM install)
  check_vllm_stack.py all          ...and vLLM, vllm-router, the transport stack,
                                   and /app/versions.txt has every pinned commit
"""
import os
import shlex
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, files, version

ENV_FILE = "/opt/mad/pins/build.env"
VERSIONS = "/app/versions.txt"


def build_env():
    env = {}
    with open(ENV_FILE) as f:
        for tok in shlex.split(f.read(), posix=True):
            if "=" in tok:
                k, v = tok.split("=", 1)
                env[k] = v
    return env


def dist_version(names):
    for n in names:
        try:
            return version(n)
        except PackageNotFoundError:
            pass
    return None


def check_aiter(env):
    v = dist_version(("amd-aiter", "amd_aiter", "aiter"))
    assert v, "AITER is not installed"
    if env["AITER_KIND"] == "wheel":
        assert v.split("+", 1)[0] == env["AITER_VERSION"], f"AITER {v!r}, pin set {env['AITER_VERSION']}"
    else:
        assert env["AITER_COMMIT"][:8] in v, f"AITER {v!r} is not commit {env['AITER_COMMIT'][:8]}"
    if env.get("AITER_CHECK_CONV") == "1":
        assert any(str(p).startswith("aiter/ops/triton/conv/") for p in files("amd-aiter") or []), \
            "aiter.ops.triton.conv missing"
    if env.get("FLYDSL_VERSION"):
        fv = dist_version(("flydsl",))
        assert fv == env["FLYDSL_VERSION"], f"flydsl {fv!r}, pin set {env['FLYDSL_VERSION']}"
    print("AITER OK:", v)


def check_mori(env):
    import mori, mori.io, mori.ops  # noqa: E401,F401
    print("MoRI OK at", mori.__path__[0])


def check_all(env):
    check_aiter(env)
    check_mori(env)
    import vllm
    print("vLLM OK:", vllm.__version__)
    assert shutil.which("vllm-router"), "vllm-router not on PATH"
    out = subprocess.run(["vllm-router", "--help"], capture_output=True, text=True).stdout
    assert "moriio" in out, "vllm-router has no moriio support"
    lines = []
    with open(VERSIONS) as f:
        for line in f:
            if "=" in line:
                lines.append(tuple(line.strip().split("=", 1)))
    recorded = dict(lines)
    want = {"VLLM_COMMIT": env["VLLM_COMMIT"], "VLLM_ROUTER_COMMIT": env["ROUTER_COMMIT"]}
    if env["MORI_BUILD"] == "1":
        want["MORI_COMMIT"] = env["MORI_COMMIT"]
    if env["AITER_KIND"] == "source":
        want["AITER_COMMIT"] = env["AITER_COMMIT"]
    if env["TRANSPORTS_BUILD"] == "1":
        for k in ("UCX", "RIXL", "ROCSHMEM", "DEEPEP"):
            want[f"{k}_COMMIT"] = env[f"{k}_COMMIT"]
        assert os.path.exists("/usr/local/ucx/bin/ucx_info"), "UCX is not installed"
    for comp in ("AITER", "VLLM", "ROUTER"):
        names = [os.path.basename(f) for f in env.get(f"{comp}_PATCHES", "").split()]
        got = [v.split()[0] for k, v in lines if k == f"{comp}_PATCH"]
        if names != got:
            want[f"{comp}_PATCH"] = " ".join(names)
            recorded[f"{comp}_PATCH"] = " ".join(got)
    bad = {k: (recorded.get(k), v) for k, v in want.items() if recorded.get(k) != v}
    assert not bad, f"versions.txt disagrees with the pin set: {bad}"
    print(f"pin set {env['PIN_SET']}: {len(want)} commits match")


def main(argv):
    env = build_env()
    checks = {"aiter": check_aiter, "mori": check_mori, "all": check_all}
    for what in argv or ["all"]:
        checks[what](env)


if __name__ == "__main__":
    main(sys.argv[1:])
