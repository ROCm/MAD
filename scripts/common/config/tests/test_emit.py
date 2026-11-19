"""config/emit.py: a card's configuration as one workload file, and that the launchers run from it.

    python3 -m unittest discover -s scripts/common/config/tests
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(SCRIPTS / "common" / "tests" / "snapshot"))
import emit  # noqa: E402
import snapshot as S  # noqa: E402
import workload as W  # noqa: E402


def card(catalog, name):
    return emit.card_of(catalog, name)


def run_from(catalog, c, data, arch="gfx942"):
    """The card run with only MAD_CONFIG (the generated data) submitted."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(emit.dump(data))
    try:
        return S.snapshot_card(catalog, c, keep_output=True, submit_env={"MAD_CONFIG": f.name, "MAD_GPU_ARCH": arch})
    finally:
        os.unlink(f.name)


def argv(got, rank=0):
    return [w for b in got["ranks"][rank]["engine"]["blocks"] for w in b["argv"]]


def docker_env(got, rank=0):
    return dict(e.split("=", 1) for e in got["ranks"][rank]["docker"]["env"] if "=" in e)


class Compose(unittest.TestCase):
    def test_holds_what_the_card_uses(self):
        c = card("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4")
        d = emit.compose("vllm_dissag", c)
        self.assertEqual(d["model"]["env"]["MODEL_NAME"], "Kimi-K3-MXFP4")
        self.assertEqual(set(d["recipes"]["Kimi-K3-MXFP4"]["variants"]), {"gfx942", "gfx950"})   # per GPU, kept
        self.assertEqual(set(d["connectors"]), {"moriio", "rixl"})
        self.assertEqual(d["image"]["docker_build_arg"]["PINS"], "vllm-kimi-k3")

    def test_an_alias_brings_the_recipe_it_names(self):
        c = card("vllm_dissag", "pyt_e2e_deepseek-v3_disagg_mtp")
        d = emit.compose("vllm_dissag", c)
        self.assertEqual(list(d["recipes"]), ["DeepSeek-V3-MTP", "DeepSeek-V3"])
        self.assertEqual(d["model"]["env"]["RECIPE_CHECK"], "1")          # from the workload's base layer

    def test_the_colocated_launcher_reads_the_disagg_recipes(self):
        d = emit.compose("vllm_multinode", card("vllm_multinode", "pyt_vllm_kimi-k3_mi300x_pp2xtp8"))
        self.assertIn("Kimi-K3", d["recipes"])
        self.assertEqual(set(d["connectors"]), {"moriio", "rixl"})

    def test_every_card_generates_a_file_the_reader_takes(self):
        for catalog in emit.CATALOGS:
            for c in json.loads((SCRIPTS / catalog / "models.json").read_text()):
                with self.subTest(card=c["name"]):
                    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                        f.write(emit.dump(emit.compose(catalog, c)))
                    try:
                        W.load(f.name)
                    finally:
                        os.unlink(f.name)

    def test_shell_text_stays_literal(self):
        data = emit._escape({"a": "x ${HOME} y"})
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump({"version": 2, "model": {"env": data}}, f)
        try:
            self.assertEqual(W.env_of(W.load(f.name))["a"], "x ${HOME} y")
        finally:
            os.unlink(f.name)


class RunsFromTheFile(unittest.TestCase):
    """Editing the generated file changes the run: the launcher reads it, not the old files."""

    def test_a_recipe_edit_reaches_the_engine(self):
        c = card("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4")
        d = emit.compose("vllm_dissag", c)
        pre = d["recipes"]["Kimi-K3-MXFP4"]["variants"]["gfx942"]["prefill"]
        pre["dp"] = pre["dp"].replace("--max-num-seqs 8", "--max-num-seqs 77")
        got = run_from("vllm_dissag", c, d)
        self.assertEqual(got["exit_code"], 0, got.get("stderr_tail"))
        self.assertIn("MAD_RECIPE_FILE", got["ranks"][0]["docker"]["env"])     # the job's copy of it
        self.assertIn("[config] workload: recipes from the file", got["output"])
        self.assertEqual(argv(got)[argv(got).index("--max-num-seqs") + 1], "77")

    def test_a_connector_edit_reaches_the_container(self):
        c = card("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4")
        d = emit.compose("vllm_dissag", c)
        d["connectors"]["moriio"]["HSA_NO_SCRATCH_RECLAIM"] = "0"
        got = run_from("vllm_dissag", c, d)
        self.assertEqual(docker_env(got)["HSA_NO_SCRATCH_RECLAIM"], "0")

    def test_an_sglang_connector_edit_is_forwarded(self):
        c = card("sglang_disagg", "pyt_sglang_disagg_mori_io_qwen3-32b")
        d = emit.compose("sglang_disagg", c)
        d["connectors"]["mori"]["GLOO_TIMEOUT_MS"] = "12345"          # a container default, changed
        d["connectors"]["mori"]["MAD_TEST_ONLY_IN_FILE"] = "1"         # not in the container's file
        same = d["connectors"]["mori"]["SGLANG_USE_AITER"]             # unchanged: left to the container
        got = run_from("sglang_disagg", c, d)
        env = docker_env(got)
        self.assertEqual((env.get("GLOO_TIMEOUT_MS"), env.get("MAD_TEST_ONLY_IN_FILE")), ("12345", "1"))
        self.assertNotIn(f"SGLANG_USE_AITER={same}", got["ranks"][0]["docker"]["env"])

    def test_an_env_edit_wins_over_the_recipe(self):
        c = card("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4")
        d = emit.compose("vllm_dissag", c)
        d["model"]["env"]["NCCL_DEBUG"] = "INFO"
        got = run_from("vllm_dissag", c, d)
        self.assertEqual(docker_env(got)["NCCL_DEBUG"], "INFO")


if __name__ == "__main__":
    unittest.main()
