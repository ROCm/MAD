"""scripts/common/perf: the performance model and the best-known-configuration registry."""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import model as M  # noqa: E402
import registry as R  # noqa: E402

CAT = M._catalog.load(M.CATALOG)
MODELS = M._load(os.path.join(os.path.dirname(HERE), "models.yaml"))
DSV3 = MODELS["DeepSeek-V3"]


class Model(unittest.TestCase):
    def test_wider_ep_reads_less_weight_per_gpu(self):
        a = M.estimate(CAT, DSV3, "mi355x-ubb-8", gpus=16, batch=64)
        b = M.estimate(CAT, DSV3, "mi355x-ubb-8", gpus=64, batch=64)
        self.assertGreater(b["decode_tokens_per_s_per_gpu"], a["decode_tokens_per_s_per_gpu"])

    def test_the_all_to_all_link_follows_the_scale_up_domain(self):
        self.assertEqual(M.estimate(CAT, DSV3, "mi455x-helios-tray", gpus=72)["all_to_all_link"], "scale-up")
        self.assertEqual(M.estimate(CAT, DSV3, "mi455x-helios-tray", gpus=144)["all_to_all_link"], "scale-out")
        self.assertEqual(M.estimate(CAT, DSV3, "mi300x-oam-8", gpus=16)["all_to_all_link"], "scale-out")
        self.assertIsNone(M.estimate(CAT, MODELS["Llama-3.1-405B"], "mi300x-oam-8", gpus=16)["all_to_all_link"])

    def test_prefill_scales_with_the_formats_peak(self):
        bf, f8 = (M.estimate(CAT, DSV3, "mi300x-oam-8", compute=c, weights=c)["prefill_tokens_per_s_per_gpu"]
                  for c in ("bf16", "fp8_e4m3"))
        self.assertAlmostEqual(f8 / bf, 2615 / 1307, places=3)

    def test_fits_and_unknown_peak(self):
        self.assertFalse(M.estimate(CAT, DSV3, "mi300x-oam-8", gpus=2, weights="bf16")["fits"])
        with self.assertRaisesRegex(ValueError, "no peak TFLOPS for mxfp4"):
            M.estimate(CAT, DSV3, "mi300x-oam-8", compute="mxfp4")

    def test_search_ranks_fitting_layouts_best_first(self):
        rows = M.search(CAT, DSV3, ["mi300x-oam-8", "mi355x-ubb-8"])
        self.assertTrue(rows and all(r["fits"] for r in rows))
        tps = [r["decode_tokens_per_s_per_gpu"] for r in rows]
        self.assertEqual(tps, sorted(tps, reverse=True))
        self.assertEqual(len({(r["platform"], r["gpus"], r["batch"], r["compute"]) for r in rows}), len(rows))


class ToSpec(unittest.TestCase):
    def test_the_best_layout_becomes_a_valid_spec(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "topology"))
        import topology as T
        best = M.search(CAT, DSV3, ["mi455x-helios-tray"])[0]
        spec = T.validate(M.to_spec(CAT, best))
        self.assertEqual(T.gpus_per_node(spec), 4)
        if best["gpus"] <= 72:
            self.assertIn({"same_domain": ["decode"]}, spec["placement"])
        T.legacy_env(spec, "vllm_dissag", {})      # a launcher runs it


class Registry(unittest.TestCase):
    def test_best_meets_the_slo_and_accuracy(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reg.jsonl")
            base = {"model": "DeepSeek-V3", "platform": "mi355x-ubb-8", "image": "vllm-mori121", "launcher": "vllm_dissag",
                    "serving_type": "disagg", "slo": {"tpot_ms": 50}, "config": "run1/config.resolved.json"}
            R.record(path, dict(base, throughput=900, tpot_ms=40, accuracy={"gsm8k": 0.95}))
            R.record(path, dict(base, throughput=1200, tpot_ms=70, accuracy={"gsm8k": 0.95}))   # misses SLO
            R.record(path, dict(base, throughput=1000, tpot_ms=45, accuracy={"gsm8k": 0.80}))
            self.assertEqual(R.best(path, model="DeepSeek-V3")["throughput"], 1000)
            self.assertEqual(R.best(path, accuracy_floor={"gsm8k": 0.9}, model="DeepSeek-V3")["throughput"], 900)
            self.assertIsNone(R.best(path, platform="mi300x-oam-8"))
            with self.assertRaisesRegex(ValueError, "needs throughput"):
                R.record(path, {k: v for k, v in base.items()})


class Promote(unittest.TestCase):
    def test_the_best_run_replaces_the_default(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "recipe"))
        import models
        with tempfile.TemporaryDirectory() as d:
            runs, bk = os.path.join(d, "runs.jsonl"), os.path.join(d, "best.jsonl")
            with open(bk, "w") as f:
                f.write("# header\n")
            base = {"model": "DeepSeek-V3", "platform": "mi300x-oam-8", "launcher": "vllm_dissag",
                    "shape": {"xP": 1, "yD": 1}, "config": "c", "apply": {"env": {"K": "1"}}}
            R.record(runs, dict(base, throughput=10))
            R.promote(runs, bk, model="DeepSeek-V3")
            R.record(runs, dict(base, throughput=20, apply={"env": {"K": "2"}}))
            R.promote(runs, bk, model="DeepSeek-V3")
            rows = [l for l in open(bk) if not l.startswith("#")]
            self.assertEqual(len(rows), 1)                       # replaced, not appended
            env = {"MAD_BEST_REGISTRY": bk, "MAD_LAUNCHER": "vllm_dissag", "MAD_PLATFORM": "mi300x-oam-8",
                   "xP": "1", "yD": "1"}
            self.assertEqual(models.best_record("DeepSeek-V3", env)["apply"]["env"]["K"], "2")
            self.assertIsNone(models.best_record("DeepSeek-V3", dict(env, MAD_BEST_CONFIG="0")))


if __name__ == "__main__":
    unittest.main()
