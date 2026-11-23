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


def run_from(catalog, c, edit=None, arch="gfx942"):
    """The card run with only MAD_CONFIG submitted: its generated file, after `edit(data)`."""
    d = tempfile.mkdtemp()
    data = emit.compose(catalog, c, d)
    if edit:
        edit(data)
    path = os.path.join(d, "card.yaml")
    with open(path, "w") as f:
        f.write(emit.dump(data))
    return S.snapshot_card(catalog, c, keep_output=True, submit_env={"MAD_CONFIG": path, "MAD_GPU_ARCH": arch})


def env_file(path):
    out = {}
    for line in open(path):
        line = line.rstrip("\n")
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def argv(got, rank=0):
    return [w for b in got["ranks"][rank]["engine"]["blocks"] for w in b["argv"]]


def docker_env(got, rank=0):
    return dict(e.split("=", 1) for e in got["ranks"][rank]["docker"]["env"] if "=" in e)


class Compose(unittest.TestCase):
    def test_a_card_fits_the_shared_layers_and_keeps_only_its_own_part(self):
        d = emit.compose("vllm_dissag", card("vllm_dissag", "pyt_vllm_disagg_nixl_llama-3.1-405b-fp8"), str(SCRIPTS))
        self.assertEqual(d["defaults"], ["common/workloads/serve/vllm-disagg-nixl.yaml"])
        own = d["model"]["env"]
        self.assertNotIn("RUN_MORI", own)                                   # the layer gives it
        self.assertEqual((own["MODEL_NAME"], own["xP"]), ("Llama-3.1-405B-Instruct-FP8-KV", "1"))
        self.assertNotIn("recipes", d)                                       # the recipe stays in models.yaml
        self.assertNotIn("connectors", d)                                    # the env stays in connectors/

    def test_a_layer_is_used_only_when_the_card_has_all_of_it(self):
        d = emit.compose("vllm_dissag", card("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4"), str(SCRIPTS))
        for p in d.get("defaults", []):
            values = W.env_of(W.load(os.path.join(str(SCRIPTS), p)), d["model"]["env"].get("BENCHMARK_SCRIPT"))
            env, _ = emit.card_env("vllm_dissag", card("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4"))
            self.assertTrue(all(env.get(k) == v for k, v in values.items()), p)

    def test_an_example_keeps_its_own_layers(self):
        d = emit.compose("vllm_dissag", card("vllm_dissag", "pyt_e2e_deepseek-v3_disagg_mtp"), str(SCRIPTS))
        self.assertEqual(d["defaults"], ["common/workloads/base.yaml", "common/workloads/serve/vllm-disagg-wide-ep.yaml",
                                         "common/workloads/bench/accuracy.yaml"])
        # family/deepseek-v3 names DeepSeek-V3; this example runs DeepSeek-V3-MTP, so it is its own.
        self.assertEqual(d["model"]["env"]["MODEL_NAME"], "DeepSeek-V3-MTP")

    def test_every_card_generates_a_file_the_reader_takes(self):
        for catalog in emit.CATALOGS:
            for c in json.loads((SCRIPTS / catalog / "models.json").read_text()):
                with self.subTest(card=c["name"]):
                    d = tempfile.mkdtemp()
                    path = os.path.join(d, "c.yaml")
                    with open(path, "w") as f:
                        f.write(emit.dump(emit.compose(catalog, c, d)))
                    W.load(path)

    def test_shell_text_stays_literal(self):
        data = emit._escape({"a": "x ${HOME} y"})
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump({"version": 2, "model": {"env": data}}, f)
        try:
            self.assertEqual(W.env_of(W.load(f.name))["a"], "x ${HOME} y")
        finally:
            os.unlink(f.name)

    def test_shared_values_no_layer_holds_are_suggested(self):
        composed = [("vllm_dissag", emit.compose("vllm_dissag", c, str(SCRIPTS)))
                    for c in json.loads((SCRIPTS / "vllm_dissag" / "models.json").read_text())]
        got = emit.suggest(composed)
        self.assertTrue(any(v == {"RUN_MORI": "0", "RUN_DEEPEP": "1"} for _, v, _ in got), got)


class RunsFromTheFile(unittest.TestCase):
    """The launcher runs the file: its own part, and the recipe or connector env a card chooses to
    override in it, reach the engines and containers."""

    def test_a_recipe_override_reaches_the_engine(self):
        c = card("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4")
        recipe = yaml.safe_load((SCRIPTS / "vllm_dissag" / "models.yaml").read_text())["Kimi-K3-MXFP4"]
        pre = recipe["variants"]["gfx942"]["prefill"]
        pre["dp"] = pre["dp"].replace("--max-num-seqs 8", "--max-num-seqs 77")
        got = run_from("vllm_dissag", c, lambda d: d.update(recipes={"Kimi-K3-MXFP4": recipe}))
        self.assertEqual(got["exit_code"], 0, got.get("stderr_tail"))
        self.assertIn("MAD_RECIPE_FILE", got["ranks"][0]["docker"]["env"])     # the job's copy of it
        self.assertIn("[config] workload: recipes from the file", got["output"])
        self.assertEqual(argv(got)[argv(got).index("--max-num-seqs") + 1], "77")

    def test_a_connector_override_reaches_the_container(self):
        c = card("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4")
        moriio = dict(env_file(SCRIPTS / "vllm_dissag" / "connectors" / "moriio.env"), HSA_NO_SCRATCH_RECLAIM="0")
        got = run_from("vllm_dissag", c, lambda d: d.update(connectors={"moriio": moriio}))
        self.assertEqual(docker_env(got)["HSA_NO_SCRATCH_RECLAIM"], "0")

    def test_an_sglang_connector_override_is_forwarded(self):
        c = card("sglang_disagg", "pyt_sglang_disagg_mori_io_qwen3-32b")
        mori = env_file(SCRIPTS / "sglang_disagg" / "connectors" / "mori.env")
        same = mori["SGLANG_USE_AITER"]
        mori.update(GLOO_TIMEOUT_MS="12345", MAD_TEST_ONLY_IN_FILE="1")
        got = run_from("sglang_disagg", c, lambda d: d.update(connectors={"mori": mori}))
        env = docker_env(got)
        self.assertEqual((env.get("GLOO_TIMEOUT_MS"), env.get("MAD_TEST_ONLY_IN_FILE")), ("12345", "1"))
        self.assertNotIn(f"SGLANG_USE_AITER={same}", got["ranks"][0]["docker"]["env"])

    def test_the_cards_own_part_wins_over_the_recipe(self):
        c = card("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4")
        got = run_from("vllm_dissag", c, lambda d: d["model"]["env"].update(NCCL_DEBUG="INFO"))
        self.assertEqual(docker_env(got)["NCCL_DEBUG"], "INFO")


class Advice(unittest.TestCase):
    def test_advice_never_fails_and_names_what_a_card_could_gain(self):
        import subprocess
        p = subprocess.run([sys.executable, str(HERE.parent / "advise.py")], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)
        self.assertIn("has a GPU-agnostic twin, pyt_vllm_disagg_mori_kimi-k3-mxfp4_auto", p.stdout)
        self.assertNotIn("_kimi_k3_recipe_env", p.stdout)          # anchors are not recipes
