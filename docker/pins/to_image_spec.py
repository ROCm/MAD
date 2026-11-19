#!/usr/bin/env python3
"""A pin set (docker/pins/<name>.json) as an image spec over the compatibility matrix.

The pin sets and their hand-written Dockerfiles (di_vllm, di_sglang) become what #295's
image specs describe: compat/images/<name>.yaml selects component builds authorized in
compat/rocm.yaml, and madengine generates the Dockerfile from it. Each build here does what
the pin set's Dockerfile step did, with the values pins.py resolves for it, and the image keeps
the same records and self-check: /opt/mad/pins/build.env, /app/versions.txt (every commit and
patch), check_vllm_stack.py at the end.

    to_image_spec.py show NAME           the spec and the matrix builds, as YAML
    to_image_spec.py write [NAME ...]    write compat/images/NAME.yaml and add the builds to
                                         compat/rocm.yaml (default: every pin set)
    to_image_spec.py check [NAME ...]    the written spec and builds give, per GPU, the commits,
                                         patches and base image the pin set does (exit 1 if not)

The pin sets stay until images built from the specs are shown to match (same component commits
and patch digests in /app/versions.txt, per pin set and GPU); then they and the hand-written
Dockerfiles are removed.
"""
import argparse
import hashlib
import os
import shlex
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MATRIX = os.path.join(ROOT, "compat", "rocm.yaml")
IMAGES = os.path.join(ROOT, "compat", "images")
PATCHES = os.path.join(ROOT, "docker", "patches")
sys.path.insert(0, HERE)
import pins  # noqa: E402



def _sha(file):
    with open(os.path.join(PATCHES, file), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _patches(component, prefix):
    """Matrix patch entries, and the versions.txt lines the Dockerfile wrote for them."""
    out, lines = [], []
    for p in component.get("patches") or []:
        sha = p.get("sha256") or _sha(p["file"])
        out.append({"file": p["file"], "sha256": sha, "upstream": p.get("upstream", "")})
        record = f"{prefix}_PATCH={os.path.basename(p['file'])} sha256:{sha}"
        lines.append(f"echo {shlex.quote(record)} >> /app/versions.txt")
    return out, lines


def _short(commit):
    return str(commit)[:8]


def _key(commit, patches, origin):
    """<commit>[+<patch digest>]@<origin>: one build per commit and patch set."""
    tag = _short(commit)
    if patches:
        tag += "+" + hashlib.sha256("".join(p["sha256"] for p in patches).encode()).hexdigest()[:8]
    return f"{tag}@{origin}"


def vllm_stack(spec):
    """(matrix builds {component: {key: entry}}, image spec) for a vLLM pin set."""
    c = spec["components"]
    b = spec.get("build") or {}
    builds, sel, order = {}, {}, []

    def add(name, key, entry):
        builds.setdefault(name, {})[key] = entry
        sel[name] = key
        order.append(name)

    mori = c.get("mori") or {}
    if not mori or not mori.get("enabled", True) or mori.get("archs"):
        raise SystemExit(f"{spec['name']}: a vLLM pin set without MoRI on every GPU is not converted")
    add("mori", _key(mori["commit"], [], "di"), {
        "source": {"repo": mori["repo"], "commit": mori["commit"], "path": "/tmp/mori-src", "steps": [
            "pip uninstall -y amd_mori amd-mori amd-mori-nightly mori 2>/dev/null || true",
            "BUILD_UMBP=OFF pip install .",
            "python3 -c 'import mori, mori.io, mori.ops; print(\"MoRI OK at\", mori.__path__[0])'",
            "echo MORI_COMMIT=$(git -C /tmp/mori-src rev-parse HEAD) >> /app/versions.txt",
            "cd / && rm -rf /tmp/mori-src"]},
        "note": mori.get("why", "MoRI at the pinned commit, replacing the base image's"),
    })

    aiter = c["aiter"]
    fly = aiter.get("flydsl") or {}
    fly_steps = ([f"pip install {' '.join(fly.get('pip_flags') or [])} flydsl=={fly['version']}".replace("  ", " ")]
                 if fly.get("version") else [])
    after = [f"python3 /opt/mad/pins/check_vllm_stack.py aiter",
             "rm -rf /opt/vllm_cache/aiter_jit /root/.aiter && echo 'cleared stale AITER JIT cache'"]
    if "wheel" in aiter:
        w = aiter["wheel"]
        fname = "/tmp/" + os.path.basename(w["url"]).replace("%2B", "+")
        add("amd-aiter", f"{w['version']}{'+flydsl' + fly['version'] if fly.get('version') else ''}@di", {
            "run": ["pip uninstall -y amd_aiter amd-aiter aiter 2>/dev/null || true",
                    f"curl -fL --retry 3 --retry-delay 2 -o {shlex.quote(fname)} {shlex.quote(w['url'])}",
                    f"pip install --no-deps {shlex.quote(fname)} && rm -f {shlex.quote(fname)}",
                    f"echo AITER_VERSION={w['version']} >> /app/versions.txt"] + fly_steps + after,
            "note": aiter.get("why", "the AITER release wheel"),
        })
    else:
        patches, plines = _patches(aiter, "AITER")
        arch_env = ["export GPU_ARCHS={gpu_arch}"] if aiter.get("gpu_archs_env") else []
        key = _key(aiter["commit"], patches, "di")
        if fly.get("version"):
            key = key.replace("@", f"+flydsl{fly['version']}@")
        entry = {"source": {"repo": aiter["repo"], "commit": aiter["commit"], "path": "/tmp/aiter-src",
                            "steps": ["pip uninstall -y amd_aiter amd-aiter aiter 2>/dev/null || true"] + plines
                            + arch_env + ["pip install --no-build-isolation --no-deps -v .",
                                          "echo AITER_COMMIT=$(git -C /tmp/aiter-src rev-parse HEAD) >> /app/versions.txt",
                                          "cd / && rm -rf /tmp/aiter-src"] + fly_steps + after},
                 "note": aiter.get("why", "AITER at the pinned commit, from source")}
        if patches:
            entry["source"]["patches"] = patches
        add("amd-aiter", key, entry)

    vllm = c["vllm"]
    vp, vlines = _patches(vllm, "VLLM")
    after_engine = pins._pip_line([p for p in spec.get("pip") or [] if p.get("stage") == "after_engine"])
    entry = {"source": {"repo": vllm["repo"], "commit": vllm["commit"], "path": "/tmp/vllm-src", "steps": [
        "echo VLLM_COMMIT=$(git -C /tmp/vllm-src rev-parse HEAD) >> /app/versions.txt"] + vlines + [
        "pip uninstall -y vllm 2>/dev/null || true",
        "pip install --no-deps --no-build-isolation -v .",
        "python3 -c 'import vllm; print(\"vLLM\", vllm.__version__, \"from\", vllm.__file__)'",
        "cd / && rm -rf /tmp/vllm-src"]
        + ([f"pip install {after_engine}"] if after_engine else [])
        + ["python3 /opt/mad/pins/check_vllm_stack.py aiter mori"]},
        "note": vllm.get("why", "vLLM compiled at the pinned commit")}
    if vp:
        entry["source"]["patches"] = vp
    if vllm.get("fork_commit"):
        entry["fork_commit"] = vllm["fork_commit"]
    add("vllm", _key(vllm["commit"], vp, spec["name"]), entry)   # after-engine packages differ per set

    r = c["router"]
    rp, rlines = _patches(r, "ROUTER")
    entry = {"source": {"repo": r["repo"], "commit": r["commit"], "path": "/tmp/vllm-router-src", "steps": [
        "command -v cargo >/dev/null 2>&1 || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs"
        f" | sh -s -- -y --default-toolchain {r.get('rust_toolchain', 'stable')}",
        "export PATH=/root/.cargo/bin:$PATH"] + rlines + [
        "cargo build --release",
        "install -m 755 target/release/vllm-router /usr/local/bin/vllm-router",
        "vllm-router --help 2>&1 | grep -q moriio",
        "echo VLLM_ROUTER_COMMIT=$(git -C /tmp/vllm-router-src rev-parse HEAD) >> /app/versions.txt",
        "cd / && rm -rf /tmp/vllm-router-src"]},
        "note": r.get("why", "vllm-router at the pinned commit")}
    if rp:
        entry["source"]["patches"] = rp
    if r.get("fork_commit"):
        entry["fork_commit"] = r["fork_commit"]
    add("vllm-router", _key(r["commit"], rp, "di"), entry)

    t = c.get("transports") or {}
    packages = ["git", "build-essential", "cmake", "ninja-build", "ccache", "libssl-dev", "pkg-config"]
    if t.get("enabled"):
        s = t["subcomponents"]
        nic = b.get("nic_arch", "cx7")
        tpk = ["autoconf", "automake", "libtool", "autogen", "m4", "gcc", "make", "librdmacm-dev",
               "rdmacm-utils", "infiniband-diags", "ibverbs-utils", "perftest", "ethtool", "libibverbs-dev",
               "rdma-core", "strace", "libgflags-dev", "libaio-dev", "liburing-dev", "libcpprest-dev",
               "libgrpc-dev", "libgrpc++-dev", "libprotobuf-dev", "protobuf-compiler-grpc"]
        add("ucx", _key(s["ucx"]["commit"], [], "di"), {
            "source": {"repo": s["ucx"]["repo"], "commit": s["ucx"]["commit"], "path": "/tmp/ucx", "steps": [
                "pip install meson==0.64.0 'pybind11[global]' pyyaml",
                "./autogen.sh", "mkdir -p build && cd build",
                "../configure --prefix=/usr/local/ucx/ --with-rocm=/opt/rocm --disable-go --disable-java"
                " --disable-assertions --enable-mt",
                "make -j && make install",
                "echo UCX_COMMIT=$(git -C /tmp/ucx rev-parse HEAD) >> /app/versions.txt",
                "cd / && rm -rf /tmp/ucx"]},
            "system_packages": tpk})
        tar = s["googletest"]["tarball"]
        add("googletest", os.path.basename(tar).replace(".tar.gz", "") + "@di", {
            "run": [f"cd /tmp && wget -q -O googletest.tar.gz {shlex.quote(tar)}",
                    "mkdir -p googletest && tar -xzf googletest.tar.gz -C googletest --strip-components=1",
                    "cd googletest && mkdir -p build && cd build && cmake -DBUILD_SHARED_LIBS=on .. && make -j && make install",
                    "rm -rf /tmp/googletest /tmp/googletest.tar.gz"]})
        add("rixl", _key(s["rixl"]["commit"], [], "di"), {
            "source": {"repo": s["rixl"]["repo"], "commit": s["rixl"]["commit"], "path": "/tmp/RIXL", "steps": [
                "meson setup build/ --prefix=/usr/local/RIXL/install -Ducx_path=/usr/local/ucx/"
                " -Ddisable_gds_backend=true -Dcudapath_inc=/opt/rocm/include -Dcudapath_lib=/opt/rocm/lib",
                "cd build && ninja && ninja install && cd /tmp/RIXL",
                "pip install --config-settings=setup-args=-Dcudapath_inc=/opt/rocm/include"
                " --config-settings=setup-args=-Dcudapath_lib=/opt/rocm/lib"
                " --config-settings=setup-args=-Ducx_path=/usr/local/ucx/"
                " --config-settings=setup-args=-Ddisable_gds_backend=true .",
                "echo RIXL_COMMIT=$(git -C /tmp/RIXL rev-parse HEAD) >> /app/versions.txt"]},
            "requires": ["ucx"]})
        add("rocshmem", _key(s["rocshmem"]["commit"], [], "di"), {
            "source": {"repo": s["rocshmem"]["repo"], "commit": s["rocshmem"]["commit"], "path": "/tmp/rocm-systems",
                       "sparse": ["projects/rocshmem"], "steps": [
                           "mkdir -p /tmp/rocshmem-build && cd /tmp/rocshmem-build",
                           "/tmp/rocm-systems/projects/rocshmem/scripts/build_configs/all_backends"
                           " -DUSE_EXTERNAL_MPI=OFF -DGPU_TARGETS={gpu_arch}",
                           "echo ROCSHMEM_COMMIT=$(git -C /tmp/rocm-systems rev-parse HEAD) >> /app/versions.txt",
                           "rm -rf /tmp/rocm-systems /tmp/rocshmem-build"]}})
        add("deepep", _key(s["deepep"]["commit"], [], f"di-{nic}"), {
            "source": {"repo": s["deepep"]["repo"], "commit": s["deepep"]["commit"], "path": "/tmp/DeepEP", "steps": [
                "PYTORCH_ROCM_ARCH={gpu_arch} CFLAGS='-O3 -fPIC' CXXFLAGS='-O3 -fPIC --offload-arch={gpu_arch}'"
                f" HIP_CXX_FLAGS='-O3 -fPIC' python3 setup.py --variant rocm --nic {nic} build develop",
                "echo DEEPEP_COMMIT=$(git -C /tmp/DeepEP rev-parse HEAD) >> /app/versions.txt"]},
            "requires": ["rocshmem"]})

    env = {"PIP_CONSTRAINT": "/etc/pip-constraints.txt", "MORI_GPU_ARCHS": "{gpu_arch}",
           "BUILD_UMBP": "OFF", "BUILD_UMBP_SPDK": "OFF", "VLLM_TARGET_DEVICE": "rocm",
           "PYTORCH_ROCM_ARCH": "{gpu_arch}", "MAX_JOBS": str(b.get("max_jobs", 32))}
    if b.get("nvcc_threads"):
        env["NVCC_THREADS"] = str(b["nvcc_threads"])
    if t.get("enabled"):
        env.update({"LD_LIBRARY_PATH": "/usr/local/ucx/lib:/usr/local/lib:/usr/local/RIXL/install/lib:$LD_LIBRARY_PATH",
                    "PATH": "/usr/local/ucx/bin:$PATH"})
    env.update({"AITER_JIT_DIR": "/opt/vllm_cache/aiter_jit", "VLLM_CACHE_ROOT": "/opt/vllm_cache/vllm",
                "TRITON_CACHE_DIR": "/opt/vllm_cache/triton", "COMGR_CACHE_DIR": "/opt/vllm_cache/comgr"})
    image = {
        "about": spec.get("about", ""),
        "base": dict(spec["base_image"]),
        "copy": [{"from": "pins/check_vllm_stack.py", "to": "/opt/mad/pins/check_vllm_stack.py"}],
        "system_packages": packages,
        "env": env,
        "variants": {arch: {"setup": _setup(spec, arch)} for arch in spec["gpu_archs"]},
        "components": sel,
        "order": order,
        "finish": ["python3 /opt/mad/pins/check_vllm_stack.py all",
                   "rm -rf /root/.mori /tmp/mori_jit_* && mkdir -p /root/.mori",
                   "echo JIT_SCRUBBED=1 >> /app/versions.txt",
                   "cat /app/versions.txt"],
    }
    return builds, image


def _setup(spec, arch):
    """What the Dockerfile's first step did: the records and pip constraints, before apt."""
    env = pins.build_env(spec, arch, spec["base_image"][arch])
    lines = [f"{k}={shlex.quote(str(v))}" for k, v in env.items()]
    constraints = (spec.get("build") or {}).get("pip_constraints") or []
    return [
        "sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true",
        "sed -i 's|http://|https://|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true",
        "mkdir -p /app /opt/mad/pins",
        "printf '%s\\n' " + " ".join(shlex.quote(x) for x in lines) + " > /opt/mad/pins/build.env",
        f"printf 'PIN_SET=%s\\nGPU_ARCH=%s\\nBASE_IMAGE=%s\\n' {spec['name']} {arch} "
        f"{shlex.quote(spec['base_image'][arch])} > /app/versions.txt",
        "printf '%s\\n' " + (" ".join(shlex.quote(x) for x in constraints) if constraints else "''")
        + " > /etc/pip-constraints.txt",
        "pip install meson==0.64.0 'pybind11[global]' tqdm prettytable",
    ]


def sglang_stack(spec):
    """(matrix builds, image spec) for the SGLang pin set."""
    c = spec.get("components") or {}
    builds, variants = {}, {}
    early_up = pins._pip_line([p for p in spec.get("pip") or [] if p.get("stage", "early") == "early" and p.get("upgrade")])
    early = pins._pip_line([p for p in spec.get("pip") or [] if p.get("stage", "early") == "early" and not p.get("upgrade")])
    pkey = hashlib.sha256((early_up + "|" + early).encode()).hexdigest()[:8] + "@" + spec["name"]
    builds["sglang-extras"] = {pkey: {"run": ([f"pip install --upgrade {early_up}"] if early_up else [])
                                      + ([f"pip install {early}"] if early else []),
                                      "note": "the pip packages the validated image adds"}}
    mori = c.get("mori") or {}
    mkey = None
    if mori:
        mkey = _key(mori["commit"], [], "sglang")
        builds["mori"] = {mkey: {"source": {
            "repo": mori["repo"], "commit": mori["commit"], "path": "/sgl-workspace/mori", "steps": [
                "pip install -r requirements-build.txt", "pip install -e .",
                "echo MORI_COMMIT=$(git -C /sgl-workspace/mori rev-parse HEAD) >> /app/versions.txt"]},
            "note": mori.get("why", "")}}
    for arch in spec["gpu_archs"]:
        v = {"setup": [
            "sed -i 's|http://|https://|g' /etc/apt/sources.list",
            "mkdir -p /app /opt/mad/pins",
            "printf '%s\\n' " + " ".join(shlex.quote(f"{k}={shlex.quote(str(x))}")
                                         for k, x in pins.build_env(spec, arch, spec["base_image"][arch]).items())
            + " > /opt/mad/pins/build.env",
            f"printf 'PIN_SET=%s\\nGPU_ARCH=%s\\nBASE_IMAGE=%s\\n' {spec['name']} {arch} "
            f"{shlex.quote(spec['base_image'][arch])} > /app/versions.txt"]}
        if mori and arch in (mori.get("archs") or spec["gpu_archs"]):
            v["components"] = {"mori": mkey}
            v["order"] = ["sglang-extras", "mori"]
        else:
            v["finish"] = ["echo MORI_COMMIT=base >> /app/versions.txt"] + _sglang_finish()
        variants[arch] = v
    image = {
        "about": spec.get("about", ""),
        "base": dict(spec["base_image"]),
        "system_packages": ["git", "ibverbs-utils", "libibverbs-dev", "openmpi-bin", "libopenmpi-dev", "libpci-dev",
                            "libdw1", "locales", "libgrpc-dev", "libgrpc++-dev", "libprotobuf-dev",
                            "protobuf-compiler-grpc", "cmake"],
        "env": {"PYTHONPATH": "$PYTHONPATH:/sgl-workspace/mori:/sgl-workspace/aiter:"},
        "variants": variants,
        "components": {"sglang-extras": pkey},
        "order": ["sglang-extras"],
        "finish": _sglang_finish(),
    }
    return builds, image


def _sglang_finish():
    return ["pip list --format=freeze | grep -iE '^(sglang|sglang-router|amd-aiter|mori|torch)==' >> /app/versions.txt || true",
            "cat /app/versions.txt"]


def convert(name):
    spec = pins.load(name)
    if spec["engine"] == "vllm":
        return vllm_stack(spec)
    if spec["engine"] == "sglang":
        return sglang_stack(spec)
    raise SystemExit(f"{name}: engine {spec['engine']} is not converted")


def _dump(data):
    return yaml.safe_dump(data, sort_keys=False, width=1 << 16)


def _insert_builds(text, builds):
    """The matrix text with each build added under its component (comments kept): under an
    existing component's block, or as a new component at the end of `components:` (the last
    section of the file)."""
    lines = text.rstrip("\n").split("\n")
    start = next(i for i, l in enumerate(lines) if l.rstrip() == "components:")

    def section_end():
        """The line after components:'s last entry (before the next top-level key or its comments)."""
        nxt = next((i for i in range(start + 1, len(lines)) if lines[i] and not lines[i][0].isspace()
                    and not lines[i].startswith("#")), len(lines))
        while nxt > start + 1 and (not lines[nxt - 1].strip() or lines[nxt - 1].startswith("#")):
            nxt -= 1
        return nxt

    for comp, entries in builds.items():
        block = yaml.safe_dump(entries, sort_keys=False, width=1 << 16).rstrip("\n").split("\n")
        head = next((i for i in range(start + 1, section_end()) if lines[i].rstrip() == f"  {comp}:"), None)
        if head is None:
            end = section_end()
            lines[end:end] = ["", f"  {comp}:"] + ["    " + l for l in block]
            continue
        end = head + 1
        while end < len(lines) and (not lines[end].strip() or lines[end].startswith("    ")
                                    or lines[end].lstrip().startswith("#") and lines[end].startswith("   ")):
            end += 1
        while end > head + 1 and not lines[end - 1].strip():   # keep the blank line before the next one
            end -= 1
        lines[end:end] = ["    " + l for l in block]
    return "\n".join(lines) + "\n"


def write(names):
    with open(MATRIX) as f:
        text = f.read()
    matrix = yaml.safe_load(text)
    all_builds = {}
    os.makedirs(IMAGES, exist_ok=True)
    for name in names:
        builds, image = convert(name)
        for comp, entries in builds.items():
            all_builds.setdefault(comp, {}).update(entries)
        with open(os.path.join(IMAGES, f"{name}.yaml"), "w") as f:
            f.write(f"# The {name} stack (was docker/pins/{name}.json): generated by docker/pins/to_image_spec.py.\n"
                    f"# A card selects it with \"image_spec\": \"${{images.{name}}}\" (madengine image_spec).\n")
            f.write(_dump(image))
    have = matrix.get("components") or {}
    new = {c: {k: e for k, e in entries.items() if k not in (have.get(c) or {})} for c, entries in all_builds.items()}
    new = {c: e for c, e in new.items() if e}
    changed = [f"{c} {k}" for c, entries in all_builds.items() for k, e in entries.items()
               if k in (have.get(c) or {}) and have[c][k] != e]
    if changed:
        raise SystemExit("builds already in the matrix differ (edit them there, or remove them to "
                         "regenerate): " + ", ".join(changed))
    out = _insert_builds(text, new)
    got = yaml.safe_load(out)["components"]
    assert all(got[c][k] == e for c, entries in all_builds.items() for k, e in entries.items())
    with open(MATRIX, "w") as f:
        f.write(out)
    print(f"wrote {len(names)} spec(s); {sum(len(v) for v in new.values())} build(s) added to the matrix")


def check(names):
    """The written spec composes, per GPU, to the pin set's commits, patches and base."""
    with open(MATRIX) as f:
        matrix = yaml.safe_load(f)
    bad = 0
    for name in names:
        spec = pins.load(name)
        with open(os.path.join(IMAGES, f"{name}.yaml")) as f:
            image = yaml.safe_load(f)
        for arch in spec["gpu_archs"]:
            env = pins.build_env(spec, arch, spec["base_image"][arch])
            sel = dict(image["components"], **((image.get("variants") or {}).get(arch, {}).get("components") or {}))
            got = {}
            for comp, key in sel.items():
                e = matrix["components"][comp][key]
                src = e.get("source") or {}
                got[comp] = (src.get("commit"), [p["file"] for p in src.get("patches") or []])
            want = {}
            if spec["engine"] == "vllm":
                want["mori"] = (env["MORI_COMMIT"], [])
                if env["AITER_KIND"] == "source":
                    want["amd-aiter"] = (env["AITER_COMMIT"], [os.path.relpath(p, "/opt/mad/patches")
                                                               for p in env["AITER_PATCHES"].split()])
                want["vllm"] = (env["VLLM_COMMIT"], [os.path.relpath(p, "/opt/mad/patches") for p in env["VLLM_PATCHES"].split()])
                want["vllm-router"] = (env["ROUTER_COMMIT"], [os.path.relpath(p, "/opt/mad/patches")
                                                              for p in env["ROUTER_PATCHES"].split()])
                if env["TRANSPORTS_BUILD"] == 1:
                    for k in ("ucx", "rixl", "rocshmem", "deepep"):
                        want[k] = (env[f"{k.upper()}_COMMIT"], [])
            elif env["MORI_BUILD"] == 1:
                want["mori"] = (env["MORI_COMMIT"], [])
            problems = [f"{k}: spec {got.get(k)} != pin set {v}" for k, v in want.items() if got.get(k) != v]
            base = image["base"][arch] if isinstance(image["base"], dict) else image["base"]
            if base != spec["base_image"][arch]:
                problems.append(f"base {base} != {spec['base_image'][arch]}")
            for comp, key in sel.items():
                for p in ((matrix["components"][comp][key].get("source") or {}).get("patches") or []):
                    if _sha(p["file"]) != p["sha256"]:
                        problems.append(f"{comp}: {p['file']} digest {p['sha256'][:12]} is not the file's")
            print(f"{'ok  ' if not problems else 'DIFF'} {name} {arch}")
            for pr in problems:
                print("     " + pr)
            bad += bool(problems)
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "write", "check"])
    ap.add_argument("names", nargs="*")
    a = ap.parse_args(argv)
    names = a.names or pins.names()
    if a.cmd == "show":
        for n in names:
            builds, image = convert(n)
            print(f"# {n}\n" + _dump({"image": image, "builds": builds}))
        return 0
    if a.cmd == "write":
        write(names)
        return 0
    return check(names)


if __name__ == "__main__":
    sys.exit(main())
