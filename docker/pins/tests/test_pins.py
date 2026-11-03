"""Pin sets (docker/pins/*.json) and the two Dockerfiles that build them.

LegacyEquivalence proves each pin set builds what the per-model Dockerfile it
replaces builds: same base, architectures, MoRI / AITER / flydsl / vLLM / router /
transport sources and switches, compile settings and pip constraints. Symbolic and
short refs in the legacy files are compared through RESOLVED, the commits they named
when the pin sets were written (git ls-remote / rev-parse, 2026-11-03). The only
differences allowed are in DEVIATIONS, each with its reason.

Run from the MAD repo root:  python3 -m unittest discover -s docker/pins/tests -v
"""
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

PINS_DIR = Path(__file__).resolve().parents[1]
DOCKER = PINS_DIR.parent
sys.path.insert(0, str(PINS_DIR))
import pins  # noqa: E402

LEGACY = {
    "vllm-mori121": "vllm_disagg_inference.ubuntu.amd.Dockerfile",
    "vllm-glm51": "vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile",
    "vllm-kimi-k3": "vllm_kimi_k3.ubuntu.amd.Dockerfile",
    "sglang-disagg": "sglang_disagg_inference.ubuntu.amd.Dockerfile",
}

# (repository, ref as the legacy Dockerfile wrote it) -> the commit it named.
RESOLVED = {
    ("https://github.com/ROCm/mori.git", "v1.2.1"): "e31d426a13e96e1cbff96a1c904d291aefe8c46a",
    ("https://github.com/ROCm/mori.git", "624002c897a3"): "624002c897a30928ec04c14fc167bf2b8a93f6ce",
    ("https://github.com/raviguptaamd/aiter.git", "624e43586b"): "624e43586bbdb33617812ff1de5abce83d42db65",
    ("https://github.com/shikamd123/vllm.git", "vllm_2p2d_wide-ep_write_shikpate_test_06_29_customer"):
        "1ba8a2a905a4dac96f74a6f0fba8f39ef8e94a31",
    ("https://github.com/raviguptaamd/router.git", "ravgupta/discovery-dp-rank-roundrobin"):
        "82dc9811af17412e6e24b5942a5486bc502df23a",
    ("https://github.com/ROCm/ucx.git", "da3fac2a"): "da3fac2afcb55df7605f619ee19b37f4505bd4e4",
    ("https://github.com/ROCm/RIXL.git", "f33a5599"): "f33a5599400eb8da47ecdad20111ccb35b41a709",
}

# Differences from the legacy file that are intended, with the reason.
DEVIATIONS = {
    ("vllm-glm51", "rocshmem"): "legacy cloned rocm-systems develop, which no longer builds with this "
                                "ROCm 7.2; pinned to the known-good parent of 16dc5673 (transports are "
                                "off by default in this image)",
    ("*", "deepep"): "legacy cloned DeepEP's default branch with no ref; pinned to its tip on 2026-11-03",
}


def source(c):
    """(repository, commit) a component reproduces: its fork commit when carried as patches."""
    if c.get("patches"):
        return ("fork", c["fork_commit"])
    return (c["repo"], c["commit"])


def legacy_source(repo, ref):
    sha = resolve(repo, ref)
    return ("fork", sha) if any(u in repo for u in FORK_OWNERS) else (repo, sha)


# Legacy repositories that are personal forks; their pins are upstream + patches now.
FORK_OWNERS = ("github.com/raviguptaamd/", "github.com/shikamd123/")


def resolve(repo, ref):
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    return RESOLVED[(repo, ref)]


# The commit the per-model Dockerfiles were last in, before they were retired.
LEGACY_COMMIT = "13cfcac"


def legacy_text(name):
    """A legacy Dockerfile's text: from the tree while it exists, else from git."""
    f = DOCKER / LEGACY[name]
    if f.exists():
        return f.read_text()
    p = subprocess.run(["git", "-C", str(DOCKER), "show", f"{LEGACY_COMMIT}:docker/{LEGACY[name]}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise unittest.SkipTest(f"legacy {LEGACY[name]} not in the tree and not in git: {p.stderr.strip()}")
    return p.stdout


def legacy_args(text):
    """ARG / ENV defaults of a Dockerfile (last definition wins)."""
    out = {}
    text = re.sub(r"\\\n\s*", " ", text)          # join continuation lines
    for m in re.finditer(r"^(?:ARG|ENV)\s+(.+?)$", text, re.M):
        for k, v in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)=("[^"]*"|\S+)', m.group(1)):
            v = v.strip('"').rstrip("\\").strip()
            if re.fullmatch(r"\$\{\w+(:-[^}]*)?\}", v) and k in out:
                continue                     # ENV X=${X}: keeps the ARG's literal default
            out[k] = v
    return out


class Validate(unittest.TestCase):
    def test_every_pin_set_is_valid(self):
        for n in pins.names():
            err, _ = pins.problems(pins.load(n))
            self.assertEqual(err, [], n)

    def test_strict_refuses_unpinned_pip(self):
        err, pending = pins.problems(pins.load("vllm-kimi-k3"), strict=True)
        self.assertTrue(any("pyyaml" in e for e in err))

    def test_rules(self):
        base = pins.load("vllm-kimi-k3")
        for change, needle in [
            (lambda s: s["components"]["vllm"].update(commit="862bfd8"), "full 40-hex"),
            (lambda s: s["base_image"].update(gfx942="rocm/vllm-dev:latest"), "neither a digest"),
            (lambda s: s.update(pip=[{"name": "x", "version": ">=1"}]), "not exact"),
            (lambda s: s.update(pip=[{"name": "x", "version": None}]), "no pending reason"),
        ]:
            s = json.loads(json.dumps(base))
            change(s)
            err, _ = pins.problems(s)
            self.assertTrue(any(needle in e for e in err), (needle, err))

    def test_build_args_refuse_another_arch_or_base(self):
        with self.assertRaises(pins.PinError):
            pins.build_args(pins.load("vllm-mori121"), "gfx950")
        s = pins.load("vllm-kimi-k3")
        with self.assertRaises(pins.PinError):
            pins.build_env(s, "gfx942", "rocm/vllm-dev:other")
        env = pins.build_env(s, "gfx950", s["base_image"]["gfx950"])
        self.assertEqual(env["GPU_ARCH"], "gfx950")

    def test_mori_only_where_the_set_asks(self):
        s = pins.load("sglang-disagg")
        self.assertEqual(pins.build_env(s, "gfx942", s["base_image"]["gfx942"])["MORI_BUILD"], 1)
        self.assertEqual(pins.build_env(s, "gfx950", s["base_image"]["gfx950"])["MORI_BUILD"], 0)

    def test_cli(self):
        p = subprocess.run([sys.executable, str(PINS_DIR / "pins.py"), "build-args", "vllm-glm51", "--arch", "gfx942"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--build-arg PINS=vllm-glm51", p.stdout)
        self.assertIn("ci_base-dedbf6be8b1afa17a6220473b9c8c98242ac1c03@sha256:", p.stdout)


class LegacyEquivalence(unittest.TestCase):
    """Each pin set builds the sources its legacy Dockerfile builds."""

    def setUp(self):
        self.diffs = []

    def same(self, name, what, got, want):
        if got != want and not ((name, what) in DEVIATIONS or ("*", what) in DEVIATIONS):
            self.diffs.append(f"{name} {what}: pin set {got!r}, legacy {want!r}")

    def check_vllm(self, name):
        s = pins.load(name)
        text = legacy_text(name)
        a = legacy_args(text)
        c = s["components"]
        arch_from_mad = "MAD_SYSTEM_GPU_ARCHITECTURE must be gfx942 or gfx950" in text
        self.same(name, "gpu_archs", s["gpu_archs"], ["gfx942", "gfx950"] if arch_from_mad else [a["GFX_COMPILATION_ARCH"]])
        if not arch_from_mad:
            self.same(name, "pytorch arch", a["PYTORCH_ROCM_ARCH"], a["GFX_COMPILATION_ARCH"])
            self.same(name, "mori gpu arch", a["MORI_GPU_ARCHS"], a["GFX_COMPILATION_ARCH"])
        for arch in s["gpu_archs"]:
            self.same(name, f"base {arch}", s["base_image"][arch].split("@")[0], a["BASE_IMAGE"])
        self.same(name, "max_jobs", str(s["build"].get("max_jobs")), a["MAX_JOBS"])
        self.same(name, "nvcc_threads", str(s["build"].get("nvcc_threads", "")), a.get("NVCC_THREADS", ""))
        self.same(name, "nic_arch", s["build"]["nic_arch"], a["NIC_COMPILATION_ARCH"])
        self.same(name, "pip constraints", s["build"]["pip_constraints"],
                  [a["SETUPTOOLS_CONSTRAINT"]] if "SETUPTOOLS_CONSTRAINT" in a else [])
        self.same(name, "mori built", c["mori"].get("enabled", True), a.get("WITH_MORI_BUILD", "1") == "1")
        self.same(name, "mori", (c["mori"]["repo"], c["mori"]["commit"]),
                  (a["MORI_REPO"], resolve(a["MORI_REPO"], a["MORI_REF"])))
        aiter = c["aiter"]
        if "AITER_WHEEL_URL" in a:
            self.same(name, "aiter wheel", (aiter["wheel"]["url"], aiter["wheel"]["version"]),
                      (a["AITER_WHEEL_URL"], a["AITER_VERSION"]))
        else:
            self.same(name, "aiter", source(aiter), legacy_source(a["AITER_REPO"], a["AITER_REF"]))
            self.same(name, "aiter GPU_ARCHS", bool(aiter.get("gpu_archs_env")), 'GPU_ARCHS="${K3_GFX_ARCH}"' in text)
            self.same(name, "aiter conv check", bool(aiter.get("check_triton_conv")), "aiter/ops/triton/conv/" in text)
        fly = re.findall(r'pip install ([^"\n]*?)"?flydsl==([0-9.]+)', text)
        self.same(name, "flydsl", (aiter["flydsl"]["version"], " ".join(aiter["flydsl"]["pip_flags"])),
                  (fly[0][1], fly[0][0].strip()))
        self.same(name, "vllm", source(c["vllm"]), legacy_source(a["VLLM_REPO"], a["VLLM_REF"]))
        self.same(name, "router", (*source(c["router"]), c["router"]["rust_toolchain"]),
                  (*legacy_source(a["ROUTER_REPO"], a["ROUTER_REF"]), a["RUST_TOOLCHAIN"]))
        after = [p["name"] for p in s.get("pip") or [] if p.get("stage") == "after_engine"]
        self.same(name, "pip after engine", after, ["pyyaml"] if "RUN pip install pyyaml" in text else [])
        t, subs = c["transports"], c["transports"]["subcomponents"]
        self.same(name, "transports", t["enabled"], a["WITH_NIXL"] == "1")
        for sub, repo_k, ref_k in (("ucx", "_UCX_SOURCE", "_UCX_BRANCH"), ("rixl", "_RIXL_SOURCE", "_RIXL_BRANCH")):
            self.same(name, sub, (subs[sub]["repo"], subs[sub]["commit"]),
                      (a[repo_k], resolve(a[repo_k], a[ref_k])))
        rocshmem = a.get("ROCSHMEM_REF") or re.search(r"projects/rocshmem && git checkout (\S+?) ", text).group(1)
        self.same(name, "rocshmem", subs["rocshmem"]["commit"], rocshmem)
        deepep = re.search(r"git clone (https://github.com/ROCm/DeepEP.git)( && cd DeepEP && git checkout (\S+))?", text)
        self.same(name, "deepep", subs["deepep"]["commit"], deepep.group(3))
        tar = re.search(r"wget -q (\S+googletest\S+\.tar\.gz)", text).group(1)
        self.same(name, "googletest", subs["googletest"]["tarball"], tar)
        self.assertEqual(self.diffs, [])

    def test_mori121(self):
        self.check_vllm("vllm-mori121")

    def test_glm51(self):
        self.check_vllm("vllm-glm51")

    def test_kimi_k3(self):
        self.check_vllm("vllm-kimi-k3")

    def test_sglang(self):
        name = "sglang-disagg"
        s = pins.load(name)
        text = legacy_text(name)
        a = legacy_args(text)
        stacks = dict((arch, img) for img, arch in re.findall(r"^FROM (\S+) AS stack-(gfx\d+)", text, re.M))
        self.same(name, "gpu_archs", s["gpu_archs"], sorted(stacks))
        for arch, img in stacks.items():
            self.same(name, f"base {arch}", s["base_image"][arch].split("@")[0], img)
        mori = s["components"]["mori"]
        self.same(name, "mori", mori["commit"], a["MORI_COMMIT"])
        auto = re.search(r'if \[ "\$\{MAD_SYSTEM_GPU_ARCHITECTURE:-gfx942\}" = "(gfx\d+)" \]; then INSTALL_MORI=1', text)
        self.same(name, "mori archs", mori["archs"], [auto.group(1)])
        legacy_pip = re.findall(r"^RUN pip install (--upgrade )?(.+)$", text, re.M)
        want = [(n, bool(up)) for up, names_ in legacy_pip for n in names_.split()]
        got = [(p["name"], bool(p.get("upgrade"))) for p in s["pip"]]
        self.same(name, "pip", sorted(got), sorted(want))
        self.assertEqual(self.diffs, [])


class Patches(unittest.TestCase):
    """A patched component is its upstream commit plus patches that rebuild the fork's tree.

    Each patch was made as `git diff <upstream merge-base> <fork commit>`, and applying it
    to the upstream commit gives the fork commit's tree exactly (checked with
    docker/patches/verify.py, which needs GitHub). Here: the pin set and the patch header
    agree on both commits, no source is a personal fork, and every patch file is used.
    """

    def test_headers_match_the_pin_sets(self):
        used = set()
        for n in pins.names():
            for comp, c in pins.load(n)["components"].items():
                for p in c.get("patches") or []:
                    used.add(p["file"])
                    head = (DOCKER / "patches" / p["file"]).read_text().split("\ndiff --git", 1)[0]
                    self.assertIn(f"Upstream: {c['repo']} @ {c['commit']}", head, (n, comp))
                    self.assertIn(f"fork commit {c['fork_commit']}", head, (n, comp))
        on_disk = {str(f.relative_to(DOCKER / "patches")) for f in (DOCKER / "patches").rglob("*.patch")}
        self.assertEqual(on_disk, used)

    def test_no_personal_fork_is_a_source(self):
        for n in pins.names():
            text = (PINS_DIR / f"{n}.json").read_text()
            for owner in FORK_OWNERS:
                self.assertNotIn(owner, text, n)


class Dockerfiles(unittest.TestCase):
    """The two Dockerfiles read every value they use from the pin set's build env."""

    def test_every_variable_is_provided(self):
        s = pins.load("vllm-kimi-k3")
        env = set(pins.build_env(s, "gfx942", s["base_image"]["gfx942"]))
        builtin = {"BASE_IMAGE", "PINS", "MAD_SYSTEM_GPU_ARCHITECTURE", "MAX_JOBS", "PATH", "LD_LIBRARY_PATH",
                   "AITER_PATCHES", "VLLM_PATCHES", "ROUTER_PATCHES",
                   "PYTHONPATH", "_ROCM_DIR", "_UCX_INSTALL_DIR", "_RIXL_INSTALL_DIR", "_W", "_ARG_MAX_JOBS",
                   "GPU_ARCHS"}
        for f in ("di_vllm.ubuntu.amd.Dockerfile", "di_sglang.ubuntu.amd.Dockerfile"):
            used = set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)", (DOCKER / f).read_text()))
            self.assertEqual(sorted(used - env - builtin), [], f)

    def test_run_steps_parse(self):
        for f in ("di_vllm.ubuntu.amd.Dockerfile", "di_sglang.ubuntu.amd.Dockerfile"):
            text = re.sub(r"\\\n", " ", (DOCKER / f).read_text())
            runs = re.findall(r"^RUN (.+)$", text, re.M)
            self.assertTrue(runs, f)
            for i, body in enumerate(runs):
                p = subprocess.run(["bash", "-n"], input=body, capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, f"{f} RUN #{i + 1}: {p.stderr}\n{body[:300]}")

    def test_no_model_or_fork_in_the_dockerfiles(self):
        for f in ("di_vllm.ubuntu.amd.Dockerfile", "di_sglang.ubuntu.amd.Dockerfile"):
            text = (DOCKER / f).read_text()
            for word in ("kimi", "glm", "shikamd123", "raviguptaamd", "Kimi", "GLM"):
                body = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
                self.assertNotIn(word, body, f)


if __name__ == "__main__":
    unittest.main()
