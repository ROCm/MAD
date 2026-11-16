#!/usr/bin/env python3
"""scripts/common/recipe/models.py: aliases and per-accelerator variants of a recipe."""
import os
import subprocess
import sys
import tempfile
import unittest

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import models as M  # noqa: E402

MODELS = {
    "Base": {
        "default_arch": "a1",
        "env": {"COMMON": "1"},
        "roles": {"default": {"kv_cache": "fp8"}},
        "variants": {
            "a1": {"env": {"ONLY_A1": "x", "COMMON": "1"}, "prefill": {"dp": "--a1"}},
            "a2": {"env": {"COMMON": None, "ONLY_A2": "y"}, "prefill": {"dp": "--a2"}},
        },
    },
    "Base-A2": {"alias": "Base", "arch": "a2"},
    "Plain": {"env": {"K": "v"}, "prefill": {"dp": "--p"}},
    "Loop1": {"alias": "Loop2"},
    "Loop2": {"alias": "Loop1"},
}


class Effective(unittest.TestCase):
    def test_a_recipe_without_variants_is_as_written(self):
        self.assertEqual(M.effective(MODELS, "Plain"), MODELS["Plain"])
        self.assertEqual(M.effective(MODELS, "Plain", "a2"), MODELS["Plain"])

    def test_default_arch_variant_applies_without_an_arch(self):
        r = M.effective(MODELS, "Base")
        self.assertEqual(r["env"], {"COMMON": "1", "ONLY_A1": "x"})
        self.assertEqual(r["prefill"], {"dp": "--a1"})
        self.assertNotIn("variants", r)
        self.assertNotIn("default_arch", r)

    def test_alias_pins_its_arch_and_null_removes(self):
        r = M.effective(MODELS, "Base-A2")
        self.assertEqual(r["env"], {"ONLY_A2": "y"})
        self.assertEqual(r["prefill"], {"dp": "--a2"})
        self.assertEqual(r["roles"], MODELS["Base"]["roles"])

    def test_an_explicit_arch(self):
        self.assertEqual(M.effective(MODELS, "Base", "a2")["prefill"], {"dp": "--a2"})

    def test_alias_loop_and_missing_target(self):
        with self.assertRaises(ValueError):
            M.effective(MODELS, "Loop1")
        with self.assertRaises(ValueError):
            M.effective({"X": {"alias": "Nope"}}, "X")

    def test_write_for_readers_that_know_no_aliases(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = os.path.join(d, "m.yaml"), os.path.join(d, "eff.yaml")
            with open(src, "w") as f:
                yaml.safe_dump(MODELS, f)
            p = subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), "models.py"), src, "Base-A2",
                                "--write", out], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(out) as f:
                self.assertEqual(yaml.safe_load(f), {"Base-A2": M.effective(MODELS, "Base-A2")})
            self.assertEqual(M.main([src, "Base-A2", "--uses-variants"]), 0)
            self.assertEqual(M.main([src, "Plain", "--uses-variants"]), 1)


class Memory(unittest.TestCase):
    R = {"env": {"EXPLICIT": "5"},
         "memory": {"KV": {"fraction": 0.1}, "HEAP": {"gib": 16}, "RAW": {"bytes": 7}, "EXPLICIT": {"gib": 1}}}

    def test_sizes_from_the_gpu_and_explicit_values_win(self):
        r = M.with_memory(self.R, 200_000_000_000)
        self.assertEqual(r["env"], {"EXPLICIT": "5", "KV": "20000000000", "HEAP": str(16 * 2**30), "RAW": "7"})
        self.assertNotIn("memory", r)

    def test_a_fraction_needs_the_gpu_size(self):
        with self.assertRaises(ValueError):
            M.with_memory(self.R, None)
        self.assertEqual(M.with_memory({"memory": {"H": {"gib": 1}}}, None)["env"], {"H": str(2**30)})

    def test_cli_needs_memory_and_write(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = os.path.join(d, "m.yaml"), os.path.join(d, "eff.yaml")
            with open(src, "w") as f:
                yaml.safe_dump({"X": self.R, "Y": {"memory": {"H": {"gib": 1}}}}, f)
            self.assertEqual(M.main([src, "X", "--needs-memory"]), 0)
            self.assertEqual(M.main([src, "Y", "--needs-memory"]), 1)          # no fraction: no probe
            self.assertEqual(M.main([src, "X", "--uses-variants"]), 0)         # needs resolving
            self.assertEqual(M.main([src, "X", "--write", out, "--hbm-bytes", "100"]), 0)
            with open(out) as f:
                self.assertEqual(yaml.safe_load(f)["X"]["env"]["KV"], "10")


class RealRecipes(unittest.TestCase):
    """Every recipe in the repository loads; the merged Kimi-K3-MXFP4 keeps both GPUs' recipes."""

    def test_every_recipe_loads(self):
        scripts = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
        for d in ("vllm_dissag", "sglang_disagg"):
            with open(os.path.join(scripts, d, "models.yaml")) as f:
                models = yaml.safe_load(f)
            for name in models:
                with self.subTest(recipe=name):
                    M.effective(models, name)


if __name__ == "__main__":
    unittest.main()
