"""The end-to-end examples (docs/context_as_code_developer_guide.md), each run through the real launcher and role
scripts on each GPU it targets -- the launcher applies the card's workload file itself
(common/config/workload.py), so no madengine is needed. What each example claims, its engine
commands show. When madengine is importable, its reader of the same files is checked to agree.

    python3 -m unittest scripts/common/tests/test_e2e_examples.py -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

import yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1]
sys.path.insert(0, str(HERE / "snapshot"))
import snapshot as S  # noqa: E402

sys.path.insert(0, str(SCRIPTS / "common" / "config"))
import workload as W  # noqa: E402

try:
    from madengine.deployment.layered_config import resolve_for_model
except ImportError:  # madengine is optional: MAD runs without it
    resolve_for_model = None


def example_cards():
    for d in ("vllm_dissag", "vllm_multinode", "sglang_disagg"):
        for c in json.loads((SCRIPTS / d / "models.json").read_text()):
            if "e2e_example" in c.get("tags", []):
                yield d, c


def mad_env(c, d, runtime=None):
    """The card's environment as MAD's launcher builds it: workload file < card env_vars < runtime."""
    card_env = {k: str(v) for k, v in c["env_vars"].items()}
    data = W.load(W.find(c["env_vars"]["MAD_CONFIG"], str(SCRIPTS / d)), {**card_env, **(runtime or {})})
    kind = (runtime or {}).get("BENCHMARK_SCRIPT") or card_env.get("BENCHMARK_SCRIPT")
    return {**W.env_of(data, kind), **card_env, **(runtime or {})}


def card(catalog, name):
    return next(c for c in json.loads((SCRIPTS / catalog / "models.json").read_text()) if c["name"] == name)


def on(c, arch, **env):
    c = json.loads(json.dumps(c))
    c["env_vars"].update({"MAD_GPU_ARCH": arch, **env})
    return c


def run(catalog, c):
    return S.snapshot_card(catalog, c, keep_output=True)


def argv(got, rank=None):
    ranks = got["ranks"] if rank is None else [got["ranks"][rank]]
    return [w for r in ranks for b in r["engine"]["blocks"] for w in b["argv"]]


def value(words, flag):
    return words[words.index(flag) + 1]


class Inheritance(unittest.TestCase):
    """The workloads are a few lines each on shared layers (OmegaConf defaults)."""

    def test_every_example_is_inherited_and_resolves(self):
        for d, c in example_cards():
            with self.subTest(card=c["name"]):
                self.assertIn("defaults:", (SCRIPTS / d / c["env_vars"]["MAD_CONFIG"]).read_text())
                env = mad_env(c, d)
                self.assertEqual((env["RECIPE_CHECK"], env["MAD_BEST_CONFIG"]), ("1", "1"))   # from base.yaml
                self.assertTrue(env.get("MODEL_NAME"))

    def test_interpolation_names_the_shape(self):
        env = mad_env(card("vllm_multinode", "pyt_e2e_qwen3-32b_replicas_lora"), "vllm_multinode")
        self.assertEqual(env["PERF_TAGS"], "vllm_multinode,replicas,tp1")

    def test_submit_time_config_wins_over_every_layer(self):
        env = mad_env(card("vllm_dissag", "pyt_e2e_kimi-k3_disagg_niah"), "vllm_dissag",
                      {"NIAH_WORDS": "2000", "MAD_BEST_CONFIG": "0"})
        self.assertEqual((env["NIAH_WORDS"], env["MAD_BEST_CONFIG"]), ("2000", "0"))


@unittest.skipIf(resolve_for_model is None, "madengine not importable (optional)")
class SameAsMadengine(unittest.TestCase):
    """madengine reads the same workload files at submit time; both readers must agree."""

    def test_every_example_resolves_the_same(self):
        for d, c in example_cards():
            for runtime in ({}, {"BENCHMARK_SCRIPT": "accuracy", "NIAH_WORDS": "7"}):
                with self.subTest(card=c["name"], runtime=runtime):
                    theirs, _ = resolve_for_model(c, SCRIPTS / d, runtime_env=runtime)
                    self.assertEqual(mad_env(c, d, runtime), theirs)


# Each reproduction and the existing card it reproduces: the same engine commands on every rank.
REPRODUCES = {
    ("vllm_dissag", "pyt_e2e_kimi-k3_disagg_niah"): ("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4"),
    ("vllm_dissag", "pyt_e2e_llama-3.1-405b_disagg"): ("vllm_dissag", "pyt_vllm_disagg_nixl_llama-3.1-405b-fp8"),
    ("sglang_disagg", "pyt_e2e_deepseek-r1_sglang"): ("sglang_disagg", "pyt_sglang_disagg_mori_dp_deepseek-r1"),
    ("sglang_disagg", "pyt_e2e_qwen3-32b_sglang"): ("sglang_disagg", "pyt_sglang_disagg_mori_io_qwen3-32b"),
}


def engine_commands(catalog, name):
    """What the card runs on each rank, live (the snapshot harness's run, as submitted)."""
    g = S.snapshot_card(catalog, card(catalog, name))
    return g["inputs"]["gpu_arch"], [[b["argv"] for b in (r.get("engine") or {}).get("blocks", [])] for r in g["ranks"]]


class Reproductions(unittest.TestCase):
    """An example marked `reproduction` runs exactly what an existing card runs; one marked `demo`
    shows a mechanism. Every example is one or the other."""

    def test_every_example_is_a_reproduction_or_a_demo(self):
        for d, c in example_cards():
            with self.subTest(card=c["name"]):
                kinds = {"reproduction", "demo"} & set(c["tags"])
                self.assertEqual(len(kinds), 1, c["tags"])
                self.assertEqual("reproduction" in kinds, (d, c["name"]) in REPRODUCES)

    def test_each_reproduction_runs_the_engine_commands_of_its_card(self):
        for (d, name), (sd, source) in REPRODUCES.items():
            with self.subTest(example=name, source=source):
                arch, got = engine_commands(d, name)
                src_arch, want = engine_commands(sd, source)
                self.assertEqual(arch, src_arch)
                self.assertTrue(want and any(want), "the source card runs nothing")
                self.assertEqual(got, want)


try:
    from madengine.deployment.slurm_multi_env import emit_config
    from madengine.execution import image_spec as madengine_image_spec
    from madengine.stack.wheels import Wheel
except ImportError:  # madengine is optional: the generalization is madengine's build step
    emit_config = None

MAD = SCRIPTS.parent
# A Kimi-K3 image someone built with a Dockerfile of their own, on the Kimi-K3 stack's base: the
# stack's MoRI (from source, at its commit) and FlyDSL, AITER from source at a newer commit, and a
# vLLM wheel from the frameworks index in place of the patched source build. This is what the build's
# probe reads -- the Python packages of the image and of its base (versions illustrative).
KIMI_BASE = "rocm/vllm-dev:ci_base-0fcd9b99cc9d63202da4c858d8ebc6582c9e2491"
BASE_INFO = {"python": "3.12.11", "rocm_version": "7.1.0",
             "packages": {"torch": "2.9.0+rocm7.1.0", "triton": "3.5.0", "numpy": "2.2.6"}}
NEW_VLLM = "0.12.1.dev0+rocm7.1.0.g5e3f9a01"
HAND_BUILT = {"python": "3.12.11", "rocm_version": "7.1.0", "packages": dict(
    BASE_INFO["packages"], **{"mori": "1.2.3.dev12+gfe12a11a", "flydsl": "0.2.4",
                              "amd-aiter": "0.1.17.dev40+g1a2b3c4d", "vllm": NEW_VLLM, "msgspec": "0.19.0"})}
NEW_VLLM_URL = "https://rocm.frameworks.amd.com/whl-multi-arch/vllm/vllm/vllm-0.12.1.dev0-cp312-cp312-linux_x86_64.whl"


def served(name, version):
    """The matrix's frameworks index, as madengine's build reads it: it serves the newer vLLM, whose
    metadata asks for a newer AITER than the working image has."""
    if (name, version) == ("vllm", NEW_VLLM):
        return "frameworks_vllm", Wheel(name=name, version=version, python="cp312", platform="linux_x86_64",
                                        url=NEW_VLLM_URL, requires=["torch==2.9.0+rocm7.1.0", "amd-aiter>=0.1.18",
                                                                    "mori", "transformers>=4.57"])
    return None


@unittest.skipIf(emit_config is None, "madengine not importable: the generalization runs in its build")
class FromABuildToItsGeneralizedForm(unittest.TestCase):
    """The build manifest is the specific instance; what the build writes beside it is the
    generalized form: the card on the shared layers, its image as a spec over compat, and what compat
    lacks merged into a copy of the matrix (docs/context_as_code_developer_guide.md, "From a build to the matrix")."""

    def setUp(self):
        (MAD / ".madengine").mkdir(exist_ok=True)              # where madengine writes (ignored by git)
        self.out = Path(tempfile.mkdtemp(dir=MAD / ".madengine"))
        self._cwd = os.getcwd()
        os.chdir(MAD)                          # madengine builds from the MAD root
        c = card("vllm_dissag", "pyt_e2e_kimi-k3_disagg_niah")
        self.model = dict(c, name="vllm_dissag/" + c["name"], scripts="scripts/vllm_dissag/" + c["scripts"])

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self.out)

    def written(self, got):
        return yaml.safe_load(Path(got["path"]).read_text())

    def test_a_card_built_from_its_spec_is_already_general(self):
        spec = madengine_image_spec.effective_image_spec(self.model, {}, str(MAD))
        for arch in ("gfx942", "gfx950"):
            with self.subTest(arch=arch):
                plan = madengine_image_spec.resolve(spec, "AMD", arch, root=str(MAD), name=self.model["name"])
                got = emit_config(self.model, self.out, {"build_plan": plan.to_dict()})
                d = self.written(got)
                self.assertEqual(d["defaults"], ["../../../scripts/common/workloads/base.yaml",
                                                 "../../../scripts/common/workloads/family/kimi-k3.yaml",
                                                 "../../../scripts/common/workloads/bench/niah.yaml"])
                self.assertEqual(d["model"]["env"], {"MODEL_NAME": "Kimi-K3-MXFP4", "xP": "2", "yD": "2",
                                                     "MAD_BEST_REGISTRY": "vllm_dissag/workloads/e2e/best_known.example.jsonl"})
                self.assertEqual(d["image"]["spec"], {"extends": "\\${images.vllm-kimi-k3}"})  # the stack itself
                self.assertNotIn("compat", got)                                       # nothing to add

    def test_a_hand_built_image_adds_what_it_proves_to_compat(self):
        build = {"base_docker": KIMI_BASE, "image_info": HAND_BUILT, "base_info": BASE_INFO}
        got = emit_config(self.model, self.out, build, served)
        d = self.written(got)
        # Placed under the stack it is closest to: the Kimi-K3 stack's MoRI (matched by the commit in
        # its version), FlyDSL (by its version), base and native libraries are inherited; only what
        # differs is written. torch, triton, numpy are the base's and msgspec is pip's job: none of them
        # is the image's to record.
        self.assertEqual(d["image"]["spec"], {
            "extends": "\\${images.vllm-kimi-k3}",
            "components": {"amd-aiter": "1a2b3c4d@di",                  # drafted below
                           "vllm": NEW_VLLM + "@frameworks_vllm"}})     # new: added to the matrix below
        unresolved = {u["component"]: u for u in d["image"]["unresolved"]}
        aiter = unresolved["amd-aiter"]                    # a source build at a commit compat lacks
        self.assertEqual(aiter["commit"], "1a2b3c4d")
        [(key, entry)] = aiter["draft"].items()
        self.assertEqual(key, "1a2b3c4d@di")
        self.assertEqual(entry["source"]["repo"], "https://github.com/ROCm/aiter.git")
        self.assertEqual(unresolved["vllm"]["requires"], "amd-aiter>=0.1.18")   # the working run says otherwise
        self.assertEqual(unresolved["vllm"]["ran_with"], "amd-aiter==0.1.17.dev40+g1a2b3c4d")
        matrix = yaml.safe_load(Path(got["compat"]).read_text())
        new = matrix["components"]["vllm"][NEW_VLLM + "@frameworks_vllm"]
        self.assertEqual((new["url"], new["requires"], new["python"]),
                         (NEW_VLLM_URL, ["mori"], ["3.12"]))         # torch is the base's
        self.assertNotIn("1a2b3c4d@di", matrix["components"]["amd-aiter"])    # a draft is confirmed by a person
        self.assertNotIn("msgspec", matrix["components"])


@unittest.skipIf(emit_config is None, "madengine not importable: the generalization runs in its build")
class CompatOverTime(unittest.TestCase):
    """docs/context_as_code_developer_guide.md, "How compat gets more accurate over time": run 1 writes what it proved,
    its PR merges it, and run 2 on the same stack adds nothing -- its image now a spec that composes
    into the whole stack on every GPU."""

    def setUp(self):
        (MAD / ".madengine").mkdir(exist_ok=True)
        self.work = Path(tempfile.mkdtemp(dir=MAD / ".madengine"))      # a checkout the PR merges into
        shutil.copytree(MAD / "compat", self.work / "compat")
        for d in ("common", "vllm_dissag"):
            shutil.copytree(SCRIPTS / d, self.work / "scripts" / d, symlinks=True)
        self._cwd = os.getcwd()
        os.chdir(self.work)
        c = card("vllm_dissag", "pyt_e2e_kimi-k3_disagg_niah")
        self.model = dict(c, name="vllm_dissag/" + c["name"], scripts="scripts/vllm_dissag/" + c["scripts"])
        self.build = {"base_docker": KIMI_BASE, "image_info": HAND_BUILT, "base_info": BASE_INFO}

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self.work)

    def test_a_merged_run_makes_the_next_one_general(self):
        from madengine.stack import compose, generalize
        run1 = emit_config(dict(self.model), self.work / "run1", self.build, served)
        image = yaml.safe_load(Path(run1["path"]).read_text())["image"]
        # The PR: the matrix the run wrote, and the drafted AITER build its author confirmed.
        draft = next(u["draft"] for u in image["unresolved"] if u.get("draft"))
        text = generalize.insert_builds(Path(run1["compat"]).read_text(), {"amd-aiter": draft})
        (self.work / "compat" / "rocm.yaml").write_text(text)
        run2 = emit_config(dict(self.model), self.work / "run2", self.build, served)
        again = yaml.safe_load(Path(run2["path"]).read_text())["image"]
        self.assertNotIn("compat", run2)                                  # nothing new to add
        self.assertNotIn("unresolved", again)                             # nothing left for a person
        spec = madengine_image_spec.effective_image_spec(
            {"name": "next", "_fs_rel_path": "vllm_dissag",
             "image_spec": {"extends": "${images.vllm-kimi-k3}", **{k: v for k, v in again["spec"].items()
                                                                     if k != "extends"}}}, {}, str(self.work))
        kimi = madengine_image_spec.named_specs(str(self.work))["vllm-kimi-k3"]
        matrix = madengine_image_spec.load_matrix(str(self.work))
        for arch in ("gfx942", "gfx950"):
            with self.subTest(arch=arch):
                got = {i["name"]: i["key"] for i in compose.compose(spec, matrix, arch)["items"]}
                was = {i["name"]: i["key"] for i in compose.compose(kimi, matrix, arch)["items"]}
                self.assertEqual({k: v for k, v in got.items() if was.get(k) != v},
                                 {"amd-aiter": "1a2b3c4d@di", "vllm": NEW_VLLM + "@frameworks_vllm"})
                self.assertEqual(set(got), set(was))                      # the whole stack: native libraries too


class FromACardToItsExample(unittest.TestCase):
    """Emitting an existing card gives the example built from it: the same layers and the card's own
    values, the rest left to the layers and to the recipe of the GPU it runs on."""

    def test_the_kimi_k3_card_emits_the_kimi_k3_example(self):
        sys.path.insert(0, str(SCRIPTS / "common" / "config"))
        import emit
        at = SCRIPTS / "vllm_dissag" / "workloads" / "e2e"
        got = emit.compose("vllm_dissag", card("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4"), str(at))
        example = yaml.safe_load((at / "kimi-k3-disagg-niah.yaml").read_text())
        # The example adds the root layer (the recipe check, best-known configurations) and its registry.
        self.assertEqual(["../../../common/workloads/base.yaml"] + got["defaults"], example["defaults"])
        self.assertEqual(dict(got["model"]["env"], MAD_BEST_REGISTRY=example["model"]["env"]["MAD_BEST_REGISTRY"]),
                         example["model"]["env"])
        self.assertEqual(got["image"], {"image_spec": "\\${images.vllm-kimi-k3}"})


class PerGpuStack(unittest.TestCase):
    """compat/images/vllm-per-gpu.yaml: composed from the stacks the matrix has, one per GPU."""

    def test_each_gpu_gets_the_stack_validated_there(self):
        sys.path.insert(0, str(SCRIPTS / "common"))
        sys.path.insert(0, str(SCRIPTS / "common" / "recipe"))
        import catalog
        import recipe
        c = catalog.load()
        demo = {"image_spec": "${images.vllm-per-gpu}"}
        for arch, source in (("gfx942", "vllm-mori121"), ("gfx950", "vllm-kimi-k3")):
            with self.subTest(arch=arch):
                self.assertEqual(catalog.components_of_card(demo, arch, c),
                                 catalog.components_of_card({"image_spec": "${images.%s}" % source}, arch, c))
        caps = recipe.engine_builds(c)["vllm@vllm-per-gpu"]
        self.assertEqual(sorted(caps), ["gfx942", "gfx950"])

    @unittest.skipIf(emit_config is None, "madengine not importable: it composes the specs")
    def test_composed_for_each_gpu_it_is_that_gpus_stack(self):
        from madengine.stack import compose
        specs = madengine_image_spec.named_specs(str(MAD))
        matrix = madengine_image_spec.load_matrix(str(MAD))
        for arch, source in (("gfx942", "vllm-mori121"), ("gfx950", "vllm-kimi-k3")):
            with self.subTest(arch=arch):
                mine, theirs = (compose._for_arch(specs[n], arch) for n in ("vllm-per-gpu", source))
                self.assertEqual({k: v for k, v in mine.items() if k != "base"},
                                 {k: v for k, v in theirs.items() if k != "base"})
                self.assertEqual(compose.compose(specs["vllm-per-gpu"], matrix, arch)["base"],
                                 compose.compose(specs[source], matrix, arch)["base"])


class Examples(unittest.TestCase):
    def test_kimi_k3_runs_each_gpus_variant_with_its_best_known_configuration(self):
        c = card("vllm_dissag", "pyt_e2e_kimi-k3_disagg_niah")
        g942 = run("vllm_dissag", on(c, "gfx942"))
        self.assertEqual(g942["exit_code"], 0)
        self.assertEqual(value(argv(g942, 0), "--quantization-config"), '{"moe":{"weight":"int4_per_group_32"}}')
        self.assertIn("best-known configuration for Kimi-K3-MXFP4 on mi300x-oam-8", g942["output"])
        g950 = run("vllm_dissag", on(c, "gfx950"))
        self.assertEqual(g950["exit_code"], 0)
        self.assertNotIn("--quantization-config", argv(g950))                       # native MXFP4
        self.assertNotIn("best-known configuration", g950["output"])                  # no gfx950 record
        off = run("vllm_dissag", on(c, "gfx942", MAD_BEST_CONFIG="0"))
        self.assertNotIn("best-known configuration", off["output"])

    def test_deepseek_mtp_speculates_on_decode_and_sizes_kv_from_the_gpu(self):
        c = card("vllm_dissag", "pyt_e2e_deepseek-v3_disagg_mtp")
        for arch, gpu_bytes in (("gfx942", 192e9), ("gfx950", 288e9)):
            with self.subTest(arch=arch):
                got = run("vllm_dissag", on(c, arch))
                self.assertEqual(got["exit_code"], 0)
                prefill, decode = argv(got, 0), argv(got, 1)
                self.assertNotIn("--speculative-config", prefill)
                self.assertEqual(json.loads(value(decode, "--speculative-config"))["method"], "deepseek_mtp")
                self.assertEqual(value(decode, "--kv-cache-memory-bytes"), str(int(0.1 * gpu_bytes)))

    def test_disaggregated_quantization_needs_fp4_compute(self):
        c = card("vllm_dissag", "pyt_e2e_deepseek-v3_dq")
        refused = run("vllm_dissag", on(c, "gfx942"))
        self.assertEqual(refused["exit_code"], 1)
        self.assertIn("activations 'mxfp4'", refused["output"])
        self.assertEqual(refused["ranks"], [])                                        # nothing started
        ckpts = Path(tempfile.mkdtemp())
        try:
            for name in ("DeepSeek-V3-MXFP4-W4A4", "DeepSeek-V3-MXFP4-W4A16"):
                (ckpts / name).mkdir()
                (ckpts / name / "config.json").write_text("{}")
            got = run("vllm_dissag", on(c, "gfx950", MODEL_DIR_CANDIDATES=str(ckpts)))
            self.assertEqual(got["exit_code"], 0, got.get("stderr_tail"))
            self.assertIn(str(ckpts / "DeepSeek-V3-MXFP4-W4A4"), argv(got, 0))       # prefill's own checkpoint
            self.assertIn(str(ckpts / "DeepSeek-V3-MXFP4-W4A16"), argv(got, 1))      # decode's own
        finally:
            shutil.rmtree(ckpts)

    def test_helios_is_expressed_and_refused_with_the_reason_until_bring_up(self):
        c = card("vllm_dissag", "pyt_e2e_kimi-k3_helios")
        on942 = run("vllm_dissag", on(c, "gfx942"))
        self.assertIn("MAD_PLATFORM=mi455x-helios-tray is cdna5, but these nodes are gfx942", on942["output"])
        on_cdna5 = run("vllm_dissag", on(c, "cdna5"))
        self.assertEqual(on_cdna5["exit_code"], 1)
        out = on_cdna5["output"]
        self.assertIn("GPUS_PER_NODE=4 xP=18 yD=18", out)                    # the topology spec, the platform's GPUs
        self.assertIn("+ platform mi455x-helios-tray: cdna5, 4 GPUs per node", out)
        self.assertIn("supports only: gfx942", out)        # no CDNA 5 variant yet: the default one's GPU_ARCHS

    def test_qwen3_replicas_serve_lora_adapters_on_every_gpu(self):
        c = card("vllm_multinode", "pyt_e2e_qwen3-32b_replicas_lora")
        for arch in ("gfx942", "gfx950"):
            with self.subTest(arch=arch):
                got = run("vllm_multinode", on(c, arch))
                self.assertEqual(got["exit_code"], 0)
                blocks = got["ranks"][0]["engine"]["blocks"]
                self.assertEqual(len(blocks), 8)                                       # 8 GPUs / TP 1
                self.assertTrue(all("--enable-lora" in b["argv"] for b in blocks))
                self.assertIn("sql=/adapters/qwen3-32b-sql", blocks[0]["argv"])

    def test_embedding_replicas_are_pooling_servers(self):
        got = run("vllm_multinode", on(card("vllm_multinode", "pyt_e2e_qwen3-embedding_replicas"), "gfx950"))
        self.assertEqual(got["exit_code"], 0)
        blocks = got["ranks"][0]["engine"]["blocks"]
        self.assertEqual(len(blocks), 8)
        self.assertTrue(all(value(b["argv"], "--runner") == "pooling" for b in blocks))

    def test_sglang_runs_from_the_same_layers(self):
        got = run("sglang_disagg", on(card("sglang_disagg", "pyt_e2e_deepseek-r1_sglang"), "gfx950"))
        self.assertEqual(got["exit_code"], 0)
        self.assertIn("--enable-dp-attention", " ".join(argv(got)))


if __name__ == "__main__":
    unittest.main()
