"""The end-to-end examples (docs/e2e-examples.md), each run through the real launcher and role
scripts on each GPU it targets, with its workload file resolved by madengine's layered config --
the code both Jenkins modes use. What each example claims, its engine commands show.

Needs madengine's src on PYTHONPATH (it resolves the workload files):
    PYTHONPATH=<madengine>/src python3 -m unittest scripts/common/tests/test_e2e_examples.py -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[1]
sys.path.insert(0, str(HERE / "snapshot"))
import snapshot as S  # noqa: E402

try:
    from madengine.deployment.layered_config import resolve_for_model
except ImportError:  # pragma: no cover
    resolve_for_model = None


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


@unittest.skipIf(resolve_for_model is None, "needs madengine's src on PYTHONPATH")
class Inheritance(unittest.TestCase):
    """The workloads are a few lines each on shared layers (OmegaConf defaults)."""

    def test_every_example_is_inherited_and_resolves(self):
        for d in ("vllm_dissag", "vllm_multinode", "sglang_disagg"):
            for c in json.loads((SCRIPTS / d / "models.json").read_text()):
                if "e2e_example" not in c.get("tags", []):
                    continue
                with self.subTest(card=c["name"]):
                    text = (SCRIPTS / d / c["env_vars"]["MAD_CONFIG"]).read_text()
                    self.assertIn("defaults:", text)
                    env, _ = resolve_for_model(c, SCRIPTS / d)
                    self.assertEqual((env["RECIPE_CHECK"], env["MAD_BEST_CONFIG"]), ("1", "1"))   # from base.yaml
                    self.assertTrue(env.get("MODEL_NAME"))

    def test_interpolation_names_the_shape(self):
        env, _ = resolve_for_model(card("vllm_multinode", "pyt_e2e_qwen3-32b_replicas_lora"), SCRIPTS / "vllm_multinode")
        self.assertEqual(env["PERF_TAGS"], "vllm_multinode,replicas,tp1")

    def test_submit_time_config_wins_over_every_layer(self):
        env, _ = resolve_for_model(card("vllm_dissag", "pyt_e2e_kimi-k3_disagg_niah"), SCRIPTS / "vllm_dissag",
                                   runtime_env={"NIAH_WORDS": "2000", "MAD_BEST_CONFIG": "0"})
        self.assertEqual((env["NIAH_WORDS"], env["MAD_BEST_CONFIG"]), ("2000", "0"))


@unittest.skipIf(resolve_for_model is None, "needs madengine's src on PYTHONPATH")
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

    def test_llama_405b_colocated_gets_the_kv_format_its_flags_lack(self):
        got = run("vllm_multinode", on(card("vllm_multinode", "pyt_e2e_llama-3.1-405b_colocated"), "gfx942"))
        self.assertEqual(got["exit_code"], 0)
        words = argv(got, 0)
        self.assertEqual((value(words, "--tensor-parallel-size"), value(words, "--pipeline-parallel-size")), ("8", "2"))
        self.assertEqual(value(words, "--kv-cache-dtype"), "fp8")

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
