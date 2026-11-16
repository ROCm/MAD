"""Recipe roles through a whole launcher run: what each node's engine is given.

A recipe's roles can add what its flags lack (a conversion an accelerator needs, the KV-cache
format) and give a role its own artifact. Each test changes one recipe in a copy of scripts/
and runs the real launcher and role scripts under the snapshot harness's stand-ins.
Run from the MAD repo root:
    python3 -m unittest discover -s scripts/common/recipe/tests -v
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

SCRIPTS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPTS / "common" / "tests" / "snapshot"))
import snapshot as S  # noqa: E402

INT4 = '{"moe":{"weight":"int4_per_group_32"}}'


class RolesEndToEnd(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.scripts = self.root / "scripts"
        shutil.copytree(SCRIPTS, self.scripts, ignore=shutil.ignore_patterns("__pycache__", "golden"))

    def tearDown(self):
        self._tmp.cleanup()

    def card(self, catalog, name):
        return next(c for c in json.loads((SCRIPTS / catalog / "models.json").read_text()) if c["name"] == name)

    def edit_recipe(self, catalog, model, fn):
        y = self.scripts / catalog / "models.yaml"
        m = yaml.safe_load(y.read_text())
        fn(m[model])
        y.write_text(yaml.safe_dump(m, sort_keys=False))

    def run_card(self, catalog, card):
        saved = S.SCRIPTS
        S.SCRIPTS = self.scripts
        try:
            got = S.snapshot_card(catalog, card)
        finally:
            S.SCRIPTS = saved
        self.assertEqual(got["exit_code"], 0, got.get("stderr_tail") or got.get("output_tail"))
        return got

    @staticmethod
    def argv(rank):
        return [line for b in rank["engine"]["blocks"] for line in b["argv"]]

    def test_conversion_added_where_the_flags_lack_it(self):
        """The MI300X Kimi recipe with its int4 flag removed: gfx942 still gets it, from the roles."""
        def drop_int4(r):
            for role in ("prefill", "decode"):
                flags = r["variants"]["gfx942"][role]
                flags["dp"] = flags["dp"].replace(f"--quantization-config '{INT4}'", "")
        self.edit_recipe("vllm_dissag", "Kimi-K3-MXFP4", drop_int4)
        got = self.run_card("vllm_dissag", self.card("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4"))
        for rank in got["ranks"]:
            self.assertIn("-e MAD_ROLE_ENGINE_ARGS_PREFILL", " ".join(rank["docker"]["flags"]) + " "
                          + " ".join("-e " + e.split("=", 1)[0] for e in rank["docker"]["env"]))
            words = self.argv(rank)
            self.assertIn("--quantization-config", words)
            self.assertEqual(words[words.index("--quantization-config") + 1], INT4)

    def test_a_role_serves_its_own_artifact(self):
        decode_ckpt = self.root / "decode-ckpt"
        decode_ckpt.mkdir()
        (decode_ckpt / "config.json").write_text("{}")
        self.edit_recipe("vllm_dissag", "DeepSeek-V3", lambda r: r["roles"].update(
            {"decode": {"artifact": str(decode_ckpt)}}))
        got = self.run_card("vllm_dissag", self.card("vllm_dissag", "pyt_vllm_disagg_mori_deepseek-v3"))
        prefill, decode = got["ranks"][0], got["ranks"][-1]
        self.assertNotIn(str(decode_ckpt), " ".join(self.argv(prefill)))
        words = self.argv(decode)
        self.assertIn(str(decode_ckpt), words)
        self.assertIn("--served-model-name", words)

    def checkpoint(self, name):
        d = self.root / name
        d.mkdir()
        (d / "config.json").write_text("{}")
        return str(d)

    def test_sglang_role_serves_its_own_artifact(self):
        ckpt = self.checkpoint("prefill-ckpt")
        self.edit_recipe("sglang_disagg", "DeepSeek-V3", lambda r: r["roles"].update({"prefill": {"artifact": ckpt}}))
        got = self.run_card("sglang_disagg", self.card("sglang_disagg", "pyt_sglang_disagg_mori_io_deepseek-v3"))
        prefill, decode = " ".join(self.argv(got["ranks"][0])), " ".join(self.argv(got["ranks"][-1]))
        self.assertIn(f"--model-path {ckpt}", prefill)
        self.assertIn("--served-model-name", prefill)
        self.assertNotIn(ckpt, decode)

    def test_colocated_server_gets_its_artifact_and_the_conversion(self):
        ckpt = self.checkpoint("server-ckpt")
        self.edit_recipe("vllm_dissag", "Kimi-K3", lambda r: r["roles"].update({"server": {"artifact": ckpt}}))
        card = self.card("vllm_multinode", "pyt_vllm_kimi-k3_mi300x_wideep_moriep")
        card["env_vars"]["COLOCATED_EXTRA_ARGS"] = card["env_vars"]["COLOCATED_EXTRA_ARGS"].replace(
            f"--quantization-config '{INT4}'", "")
        got = self.run_card("vllm_multinode", card)
        for rank in got["ranks"]:
            words = self.argv(rank)
            self.assertEqual(words[:3], ["vllm", "serve", ckpt])
            self.assertEqual(words[words.index("--quantization-config") + 1], INT4)

    def test_memory_from_the_gpu_reproduces_the_absolute_size(self):
        """DeepSeek-V3's 20 GB KV budget as a tenth of a 200 GB GPU: the same engine command."""
        card = self.card("vllm_dissag", "pyt_vllm_disagg_mori_deepseek-v3")
        before = self.run_card("vllm_dissag", card)

        def relative(r):
            del r["env"]["KV_CACHE_MEMORY_BYTES"]
            r["memory"] = {"KV_CACHE_MEMORY_BYTES": {"fraction": 0.1}}
        self.edit_recipe("vllm_dissag", "DeepSeek-V3", relative)
        card["env_vars"]["MAD_GPU_MEMORY_BYTES"] = "200000000000"
        after = self.run_card("vllm_dissag", card)
        for b, a in zip(before["ranks"], after["ranks"]):
            self.assertEqual(self.argv(a), self.argv(b))
        self.assertIn("--kv-cache-memory-bytes", self.argv(after["ranks"][0]))
        words = self.argv(after["ranks"][0])
        self.assertEqual(words[words.index("--kv-cache-memory-bytes") + 1], "20000000000")

    def test_one_recipe_runs_the_variant_of_the_gpu_it_lands_on(self):
        """Kimi-K3-MXFP4 (gfx942 and gfx950 variants) on gfx950 nodes: the gfx950 recipe, natively."""
        card = self.card("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4")
        card["env_vars"]["MAD_GPU_ARCH"] = "gfx950"
        words = self.argv(self.run_card("vllm_dissag", card)["ranks"][0])
        self.assertNotIn("--quantization-config", words)                 # no int4 conversion
        self.assertEqual(words[words.index("--max-num-batched-tokens") + 1], "4096")
        on942 = self.argv(self.run_card("vllm_dissag", self.card("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4"))["ranks"][0])
        self.assertIn("--quantization-config", on942)

    def best(self, **rec):
        r = {"model": "DeepSeek-V3", "launcher": "vllm_dissag", "platform": "mi300x-oam-8",
             "throughput": 1000, "date": "2026-11-16", "config": "run/config.resolved.json",
             "apply": {"env": {"KV_CACHE_MEMORY_BYTES": "30000000000"}}, **rec}
        (self.scripts / "common" / "perf" / "best_known.jsonl").write_text(json.dumps(r) + "\n")

    def kv_bytes(self, card):
        words = self.argv(self.run_card("vllm_dissag", card)["ranks"][0])
        return words[words.index("--kv-cache-memory-bytes") + 1]

    def test_the_best_known_configuration_is_the_default_run(self):
        card = self.card("vllm_dissag", "pyt_vllm_disagg_mori_deepseek-v3")
        self.assertEqual(self.kv_bytes(card), "20000000000")             # the recipe
        self.best()
        self.assertEqual(self.kv_bytes(card), "30000000000")             # the best-known record over it

    def test_config_still_wins_and_it_can_be_switched_off(self):
        self.best()
        card = self.card("vllm_dissag", "pyt_vllm_disagg_mori_deepseek-v3")
        given = json.loads(json.dumps(card)); given["env_vars"]["KV_CACHE_MEMORY_BYTES"] = "25000000000"
        self.assertEqual(self.kv_bytes(given), "25000000000")            # passed through config
        off = json.loads(json.dumps(card)); off["env_vars"]["MAD_BEST_CONFIG"] = "0"
        self.assertEqual(self.kv_bytes(off), "20000000000")

    def test_a_record_for_another_shape_or_platform_does_not_apply(self):
        card = self.card("vllm_dissag", "pyt_vllm_disagg_mori_deepseek-v3")
        self.best(shape={"xP": 99})
        self.assertEqual(self.kv_bytes(card), "20000000000")
        self.best(platform="mi355x-ubb-8")
        self.assertEqual(self.kv_bytes(card), "20000000000")

    def test_a_card_without_a_gpu_name_runs_like_each_gpus_own_card(self):
        """The generic Kimi-K3-MXFP4 card: on gfx950 nodes, the same engine commands as the
        MI355X card (the gfx950 variant: native MXFP4, EP_TP_SIZE 1); on gfx942, as the MI300X one."""
        generic = self.card("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4_auto")
        on950 = json.loads(json.dumps(generic)); on950["env_vars"]["MAD_GPU_ARCH"] = "gfx950"
        engines = lambda got: [[b["argv"] for b in r["engine"]["blocks"]] for r in got["ranks"]]
        mi355x = self.card("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4_mi355x")
        self.assertEqual(engines(self.run_card("vllm_dissag", on950)), engines(self.run_card("vllm_dissag", mi355x)))
        mi300x = self.card("vllm_dissag", "pyt_vllm_disagg_mori_kimi-k3-mxfp4")
        self.assertEqual(engines(self.run_card("vllm_dissag", generic)), engines(self.run_card("vllm_dissag", mi300x)))

    def test_a_colocated_card_without_a_gpu_name_converts_only_on_gfx942(self):
        generic = self.card("vllm_multinode", "pyt_vllm_kimi-k3_wideep_moriep")
        self.assertIn("--quantization-config", self.argv(self.run_card("vllm_multinode", generic)["ranks"][0]))
        on950 = json.loads(json.dumps(generic)); on950["env_vars"]["MAD_GPU_ARCH"] = "gfx950"
        self.assertNotIn("--quantization-config", self.argv(self.run_card("vllm_multinode", on950)["ranks"][0]))

    def test_speculative_decoding_reaches_the_server(self):
        self.edit_recipe("vllm_dissag", "Kimi-K3", lambda r: r["roles"].update(
            {"server": {"speculative": {"method": "eagle3", "model": "org/draft", "num_speculative_tokens": 3}}}))
        words = self.argv(self.run_card("vllm_multinode", self.card("vllm_multinode", "pyt_vllm_kimi-k3_wideep_moriep"))["ranks"][0])
        conf = json.loads(words[words.index("--speculative-config") + 1])
        self.assertEqual(conf["method"], "eagle3")


if __name__ == "__main__":
    unittest.main()
