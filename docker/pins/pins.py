#!/usr/bin/env python3
"""Pin sets: what a distributed-inference image is built from, as data.

A pin set (docker/pins/<name>.json) names, for one image:

  base_image   the image to build on, per GPU architecture
  gpu_archs    the architectures it may be built for
  components   each source as a repository and a full 40-hex commit (or, for a
               prebuilt wheel, its URL and version), with why it is pinned there
  pip          extra pip packages, each with an exact version
  build        compile parallelism, NIC target, pip constraints

docker/di_vllm.ubuntu.amd.Dockerfile and docker/di_sglang.ubuntu.amd.Dockerfile
build any pin set: `--build-arg PINS=<name>`, plus the base image and the
architecture, which `build-args` prints. The Dockerfile runs `env` to read the
set; nothing about a model is in the Dockerfile or the image name.

Rules (validate):
  * a component commit is a full 40-hex SHA; a symbolic ref may be kept beside it
    in "ref" as a note, never used to build;
  * a base image is a digest or a tag in the set's "immutable_tags";
  * a pip package has an exact version, unless it is listed with "version": null
    and a "pending" reason -- reported by `validate`, refused by `validate --strict`.

CLI:
  pins.py validate [--strict] [NAME ...]       check pin sets (all by default)
  pins.py build-args NAME --arch ARCH          the docker build --build-arg list
  pins.py env NAME --arch ARCH --base IMAGE    shell assignments for the Dockerfile
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# docker/patches next to docker/pins, in the repo and in the image (/opt/mad/{pins,patches}).
PATCHES = os.path.join(os.path.dirname(HERE), "patches")
SHA = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")


class PinError(ValueError):
    pass


def load(name):
    path = name if name.endswith(".json") else os.path.join(HERE, f"{name}.json")
    try:
        with open(path) as f:
            spec = json.load(f)
    except (OSError, ValueError) as e:
        raise PinError(f"pin set {name}: {e}")
    spec.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return spec


def names():
    return sorted(f[:-5] for f in os.listdir(HERE) if f.endswith(".json"))


def problems(spec, strict=False):
    """(errors, pending) for a pin set."""
    err, pending = [], []
    n = spec.get("name")
    if spec.get("engine") not in ("vllm", "sglang"):
        err.append(f"{n}: engine {spec.get('engine')!r} (vllm or sglang)")
    archs = spec.get("gpu_archs") or []
    if not archs:
        err.append(f"{n}: gpu_archs is empty")
    bases = spec.get("base_image") or {}
    immutable = set(spec.get("immutable_tags") or [])
    for arch in archs:
        b = bases.get(arch)
        if not b:
            err.append(f"{n}: no base_image for {arch}")
        elif not DIGEST.search(b) and b not in immutable:
            err.append(f"{n}: base_image {b} is neither a digest nor in immutable_tags")
    for comp, c in (spec.get("components") or {}).items():
        if not isinstance(c, dict) or c.get("enabled") is False:
            continue
        if "commit" in c and not SHA.match(str(c["commit"])):
            err.append(f"{n}: {comp}.commit {c['commit']!r} is not a full 40-hex SHA")
        if "wheel" in c and not (c["wheel"].get("url") and c["wheel"].get("version")):
            err.append(f"{n}: {comp}.wheel needs url and version")
        for p in c.get("patches") or []:
            f = os.path.join(PATCHES, p.get("file", ""))
            if not os.path.isfile(f):
                err.append(f"{n}: {comp} patch {p.get('file')} not found in docker/patches")
            elif hashlib.sha256(open(f, "rb").read()).hexdigest() != p.get("sha256"):
                err.append(f"{n}: {comp} patch {p['file']} does not match its sha256")
        if "fork_commit" in c and not SHA.match(str(c["fork_commit"])):
            err.append(f"{n}: {comp}.fork_commit {c['fork_commit']!r} is not a full 40-hex SHA")
        if c.get("archs") and not set(c["archs"]) <= set(archs):
            err.append(f"{n}: {comp}.archs {c['archs']} not all in gpu_archs")
        if "commit" not in c and "wheel" not in c and "subcomponents" not in c:
            err.append(f"{n}: {comp} has neither a commit nor a wheel")
        for sub, s in (c.get("subcomponents") or {}).items():
            if "commit" in s and not SHA.match(str(s["commit"])):
                err.append(f"{n}: {comp}.{sub}.commit {s['commit']!r} is not a full 40-hex SHA")
            if "commit" not in s and "tarball" not in s:
                err.append(f"{n}: {comp}.{sub} has neither a commit nor a tarball")
    for p in spec.get("pip") or []:
        if p.get("version") is None:
            if not p.get("pending"):
                err.append(f"{n}: pip {p.get('name')} has no version and no pending reason")
            else:
                pending.append(f"{n}: pip {p['name']} unpinned ({p['pending']})")
        elif not re.match(r"^[0-9][0-9A-Za-z.+!-]*$", str(p["version"])):
            err.append(f"{n}: pip {p.get('name')} version {p['version']!r} is not exact")
    if strict:
        err += pending
    return err, pending


def validate(spec, strict=False):
    err, _ = problems(spec, strict)
    if err:
        raise PinError("; ".join(err))
    return spec


def build_args(spec, arch):
    validate(spec)
    if arch not in spec["gpu_archs"]:
        raise PinError(f"{spec['name']} is built for {', '.join(spec['gpu_archs'])}, not {arch}")
    out = {"PINS": spec["name"], "BASE_IMAGE": spec["base_image"][arch], "MAD_SYSTEM_GPU_ARCHITECTURE": arch}
    if spec["engine"] == "vllm":
        out["MAX_JOBS"] = str((spec.get("build") or {}).get("max_jobs", 32))
    return out


def _pip_line(packages):
    return " ".join(f"{p['name']}=={p['version']}" if p.get("version") else p["name"] for p in packages)


def build_env(spec, arch, base):
    """Everything the Dockerfile needs, as NAME=value (shell-quoted)."""
    want = build_args(spec, arch)
    if base != want["BASE_IMAGE"]:
        raise PinError(f"BASE_IMAGE={base}, but pin set {spec['name']} builds {arch} on {want['BASE_IMAGE']}")
    c = spec.get("components") or {}
    b = spec.get("build") or {}
    out = {"PIN_SET": spec["name"], "GPU_ARCH": arch, "MAX_JOBS": b.get("max_jobs", 32),
           "NVCC_THREADS": b.get("nvcc_threads", ""), "NIC_ARCH": b.get("nic_arch", "cx7"),
           "PIP_CONSTRAINTS": "\n".join(b.get("pip_constraints") or []),
           "PIP_AFTER_ENGINE": _pip_line([p for p in spec.get("pip") or [] if p.get("stage") == "after_engine"]),
           "PIP_EARLY": _pip_line([p for p in spec.get("pip") or []
                                   if p.get("stage", "early") == "early" and not p.get("upgrade")]),
           "PIP_EARLY_UPGRADE": _pip_line([p for p in spec.get("pip") or []
                                           if p.get("stage", "early") == "early" and p.get("upgrade")])}

    mori = c.get("mori") or {}
    mori_on = bool(mori) and mori.get("enabled", True) and arch in (mori.get("archs") or [arch])
    out.update(MORI_BUILD=int(mori_on), MORI_REPO=mori.get("repo", ""),
               MORI_COMMIT=mori.get("commit", ""), MORI_MODE=mori.get("mode", "pip"))

    aiter = c.get("aiter") or {}
    flydsl = aiter.get("flydsl") or {}
    out.update(AITER_BUILD=int(bool(aiter) and aiter.get("enabled", True)),
               AITER_KIND="wheel" if "wheel" in aiter else ("source" if "commit" in aiter else ""),
               AITER_WHEEL_URL=(aiter.get("wheel") or {}).get("url", ""),
               AITER_VERSION=(aiter.get("wheel") or {}).get("version", ""),
               AITER_REPO=aiter.get("repo", ""), AITER_COMMIT=aiter.get("commit", ""),
               AITER_PATCHES=_patches(aiter),
               AITER_ARCH_ENV=int(bool(aiter.get("gpu_archs_env"))),
               AITER_CHECK_CONV=int(bool(aiter.get("check_triton_conv"))),
               FLYDSL_VERSION=flydsl.get("version", ""), FLYDSL_PIP_FLAGS=" ".join(flydsl.get("pip_flags") or []))

    for comp, prefix in (("vllm", "VLLM"), ("router", "ROUTER")):
        x = c.get(comp) or {}
        out[f"{prefix}_BUILD"] = int(bool(x) and x.get("enabled", True))
        out[f"{prefix}_REPO"] = x.get("repo", "")
        out[f"{prefix}_COMMIT"] = x.get("commit", "")
        out[f"{prefix}_PATCHES"] = _patches(x)
    out["RUST_TOOLCHAIN"] = (c.get("router") or {}).get("rust_toolchain", "")
    out["SGLANG_ROUTER"] = (c.get("sglang_router") or {}).get("pip", "")

    t = c.get("transports") or {}
    subs = t.get("subcomponents") or {}
    out["TRANSPORTS_BUILD"] = int(bool(t) and t.get("enabled", False))
    for sub, prefix in (("ucx", "UCX"), ("rixl", "RIXL"), ("rocshmem", "ROCSHMEM"), ("deepep", "DEEPEP")):
        s = subs.get(sub) or {}
        out[f"{prefix}_REPO"] = s.get("repo", "")
        out[f"{prefix}_COMMIT"] = s.get("commit", "")
    out["GTEST_TARBALL"] = (subs.get("googletest") or {}).get("tarball", "")
    return out


def _patches(component):
    """The component's patch files, as paths in the image, space-separated."""
    return " ".join(f"/opt/mad/patches/{p['file']}" for p in component.get("patches") or [])


def _shell(env):
    return "\n".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--strict", action="store_true")
    v.add_argument("names", nargs="*")
    for cmd in ("build-args", "env"):
        p = sub.add_parser(cmd)
        p.add_argument("name")
        p.add_argument("--arch", required=True)
        if cmd == "env":
            p.add_argument("--base", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "validate":
            bad = 0
            for n in a.names or names():
                err, pending = problems(load(n), a.strict)
                for line in err:
                    print("ERROR   " + line)
                for line in [] if a.strict else pending:
                    print("PENDING " + line)
                bad += bool(err)
                if not err:
                    print(f"ok      {n}")
            return 1 if bad else 0
        spec = load(a.name)
        if a.cmd == "build-args":
            print(" ".join(f"--build-arg {k}={shlex.quote(v)}" for k, v in build_args(spec, a.arch).items()))
        else:
            print(_shell(build_env(spec, a.arch, a.base)))
    except PinError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
