#!/usr/bin/env python3
"""Offline tests of scripts/common/recipe/recipe.py against compat/rocm.yaml.

    python3 scripts/common/recipe/tests/test_recipe.py
"""
import copy
import glob
import io
import json
import shlex
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import recipe as R  # noqa: E402

SCRIPTS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MAD_ROOT = os.path.dirname(SCRIPTS)
VLLM_RECIPES = os.path.join(SCRIPTS, "vllm_dissag", "models.yaml")
KIMI_BUILD = "vllm@vllm-kimi-k3"
INT4_ARG = """--quantization-config '{"moe":{"weight":"int4_per_group_32"}}'"""

CAT = R.load_catalog()
with open(VLLM_RECIPES) as _f:
    _RAW = yaml.safe_load(_f)
RECIPES = {name: R.models.effective(_RAW, name) for name in _RAW}   # as the launchers read them

# A disaggregated-quantization recipe in the shape of arXiv 2609.26333: prefill is compute
# bound and runs W4A4 FP4 from its own checkpoint; decode is bandwidth bound and runs
# weight-only 4-bit with 16-bit activations from another. The KV cache both share is fp8.
DQ = {
    "roles": {
        "default": {"kv_cache": "fp8"},
        "prefill": {"artifact": "org/model-w4a4-mxfp4", "quantization": {"weights": "mxfp4", "activations": "mxfp4"}},
        "decode": {"artifact": "org/model-w4a16", "quantization": {"weights": "mxfp4", "activations": "bf16"}},
    }
}


def without_archs(recipe):
    """The recipe without its GPU_ARCHS: what its roles and flags alone resolve to."""
    r = copy.deepcopy(recipe)
    r.get("env", {}).pop("GPU_ARCHS", None)
    return r


def used(res, role, need):
    return next(r for r in res if r["role"] == role)[need]["used"]


class Catalog(unittest.TestCase):
    def test_catalog_is_consistent(self):
        self.assertEqual(R.validate(CAT), [])

    def test_validate_checks_fabrics_and_platforms(self):
        bad = copy.deepcopy(CAT)
        bad["fabric_detection"]["order"].append("nope")
        bad["platforms"]["x"] = {"accelerator": "gfx942", "product": "MI999", "gpus_per_node": 0,
                                 "scale_up": {"fabric": "warp"}}
        problems = R.validate(bad)
        for want in ("unknown fabric 'nope'", "'MI999' is not a product", "gpus_per_node", "scale-up fabric 'warp'"):
            self.assertTrue(any(want in p for p in problems), (want, problems))

    def test_validate_names_a_bad_reference(self):
        bad = copy.deepcopy(CAT)
        bad["kernels"][KIMI_BUILD]["gfx942"]["weights"].append("fp5_imaginary")
        bad["kernels"]["vllm@x"] = {"gfx000": {}}
        problems = R.validate(bad)
        self.assertTrue(any("fp5_imaginary" in p for p in problems), problems)
        self.assertTrue(any("gfx000" in p for p in problems), problems)


class KimiRecipes(unittest.TestCase):
    """The declared roles reach the verdict the hand-written GPU_ARCHS states."""

    def test_every_card_resolves_exactly_where_it_runs(self):
        """A card runs on its pin set's GPUs, less its skip_gpu_arch and outside its recipe's
        GPU_ARCHS. With the card's own build, the recipe's roles must resolve on exactly those."""
        checked = 0
        for path in glob.glob(os.path.join(SCRIPTS, "*", "models.json")):
            with open(path) as f:
                cards = json.load(f)
            for card in cards:
                model = (card.get("env_vars") or {}).get("MODEL_NAME")
                pins = (card.get("docker_build_arg") or {}).get("PINS")
                recipe_file = (card.get("env_vars") or {}).get("MODELS_YAML") or (
                    VLLM_RECIPES if "vllm" in os.path.basename(os.path.dirname(path))
                    else os.path.join(os.path.dirname(path), "models.yaml"))
                if not (model and pins) or not os.path.isfile(recipe_file):
                    continue
                with open(recipe_file) as f:
                    recipe = R.models.effective(yaml.safe_load(f) or {}, model)
                if not recipe or not recipe.get("roles"):
                    continue
                with open(os.path.join(MAD_ROOT, "docker", "pins", pins + ".json")) as f:
                    pin_archs = json.load(f)["gpu_archs"]
                skip = str(card.get("skip_gpu_arch") or "").replace(",", " ").split()
                only = str((recipe.get("env") or {}).get("GPU_ARCHS", "")).replace(",", " ").split()
                roles = ["server"] if "vllm_multinode" in path else None
                for arch in ("gfx942", "gfx950"):
                    runs = arch in pin_archs and arch not in skip and (not only or arch in only)
                    with self.subTest(card=card["name"], arch=arch):
                        flags = (card.get("env_vars") or {}).get("COLOCATED_EXTRA_ARGS", "") if roles else None
                        _, problems = R.check(CAT, recipe, arch, R.build_of_pin_set(CAT, pins), roles, flags=flags)
                        self.assertEqual(not problems, runs, problems)
                        checked += 1
        self.assertGreater(checked, 60)

    def test_gfx942_converts_mxfp4_to_int4_and_the_flags_say_so(self):
        res, problems = R.check(CAT, RECIPES["Kimi-K3-MXFP4"], "gfx942", KIMI_BUILD)
        self.assertEqual(problems, [])
        for role in ("prefill", "decode"):
            r = next(x for x in res if x["role"] == role)
            self.assertEqual(r["weights"], {"asked": "mxfp4", "used": "int4_g32", "converted": True})
            self.assertEqual(r["activations"]["used"], "fp8_e4m3fnuz")   # the family's gfx942 member
            self.assertEqual(r["engine_args"], [INT4_ARG])
            self.assertEqual(r["kv_cache_setting"]["value"], "fp8")

    def test_gfx950_is_native_mxfp4(self):
        res, problems = R.check(CAT, RECIPES["Kimi-K3-MXFP4-MI355X"], "gfx950", KIMI_BUILD)
        self.assertEqual(problems, [])
        self.assertEqual(used(res, "prefill", "weights"), "mxfp4")
        self.assertEqual(used(res, "decode", "kv_cache"), "fp8_e4m3")
        self.assertEqual(res[0]["engine_args"], [])

    def test_gpu_archs_is_the_declared_support_set(self):
        _, problems = R.check(CAT, RECIPES["Kimi-K3-MXFP4"], "gfx950", KIMI_BUILD)
        self.assertEqual(problems, ["the recipe is written for gfx942 (GPU_ARCHS), not gfx950"])

    def test_mi300x_flags_on_gfx950_are_refused_for_their_conversion(self):
        _, problems = R.check(CAT, without_archs(RECIPES["Kimi-K3-MXFP4"]), "gfx950", KIMI_BUILD)
        self.assertTrue(problems and all("does not need" in p for p in problems), problems)

    def test_mi355x_flags_on_gfx942_get_the_conversion_added(self):
        res, problems = R.check(CAT, without_archs(RECIPES["Kimi-K3-MXFP4-MI355X"]), "gfx942", KIMI_BUILD)
        self.assertEqual(problems, [])
        self.assertEqual([r["inject"] for r in res], [[INT4_ARG], [INT4_ARG]])
        exports = R.shell_exports(res)
        self.assertIn("export MAD_ROLE_ENGINE_ARGS_PREFILL=", exports)
        line = next(x for x in exports.splitlines() if "PREFILL" in x)
        words = shlex.split(shlex.split(line.split("=", 1)[1])[0])
        self.assertEqual(words, ["--quantization-config", '{"moe":{"weight":"int4_per_group_32"}}'])

    def test_a_kv_format_the_flags_do_not_set_is_added(self):
        recipe = copy.deepcopy(RECIPES["Qwen3-32B"])
        recipe["roles"]["default"]["kv_cache"] = "fp8"
        res, problems = R.check(CAT, recipe, "gfx942", "vllm@vllm-mori121")
        self.assertEqual(problems, [])
        self.assertEqual(res[0]["inject"], ["--kv-cache-dtype fp8"])

    def test_artifacts_are_exported_per_role(self):
        res, problems = R.check(CAT, DQ, "gfx950", KIMI_BUILD)
        self.assertEqual(problems, [])
        exports = R.shell_exports(res)
        self.assertIn("export MAD_ROLE_ARTIFACT_PREFILL=org/model-w4a4-mxfp4", exports)
        self.assertIn("export MAD_ROLE_ARTIFACT_DECODE=org/model-w4a16", exports)

    def test_colocated_flags_from_the_card_are_compared(self):
        int4 = "--max-num-seqs 8 " + INT4_ARG
        self.assertEqual(R.check(CAT, RECIPES["Kimi-K3"], "gfx942", KIMI_BUILD, ["server"], flags=int4)[1], [])
        _, problems = R.check(CAT, without_archs(RECIPES["Kimi-K3"]), "gfx950", KIMI_BUILD, ["server"], flags=int4)
        self.assertTrue(problems and "does not need" in problems[0], problems)

    def test_colocated_server_role_checks_needs_not_flags(self):
        res, problems = R.check(CAT, RECIPES["Kimi-K3"], "gfx942", KIMI_BUILD, roles=["server"])
        self.assertEqual(problems, [])
        self.assertEqual(used(res, "server", "weights"), "int4_g32")

    def test_without_a_build_the_first_catalog_build_that_resolves_is_named(self):
        res, problems = R.check(CAT, RECIPES["Kimi-K3"], "gfx942")
        self.assertEqual(problems, [])
        self.assertIn(KIMI_BUILD, res[0]["notes"][0])



class BringUp(unittest.TestCase):
    def test_cdna5_lists_every_gap(self):
        r = R.bringup(CAT, "cdna5")
        self.assertEqual((r["pin_sets"], r["builds"]), ({}, []))
        self.assertEqual(len(r["missing"]), 3)          # target, pin set, engine build

    def test_gfx950_is_ready(self):
        r = R.bringup(CAT, "gfx950")
        self.assertEqual(r["missing"], [])
        self.assertIn("vllm-kimi-k3", r["pin_sets"])
        self.assertIn("sglang@sglang-disagg", r["builds"])


class Refusals(unittest.TestCase):
    def test_unknown_accelerator(self):
        _, problems = R.check(CAT, DQ, "gfx999")
        self.assertTrue(any("gfx999" in p and "not in the catalog" in p for p in problems), problems)

    def test_build_that_does_not_list_the_accelerator(self):
        _, problems = R.check(CAT, DQ, "cdna5", KIMI_BUILD)
        self.assertTrue(any("lists no support for cdna5" in p for p in problems), problems)

    def test_placement_the_build_lacks(self):
        dq = copy.deepcopy(DQ)
        dq["roles"]["prefill"]["placement"] = "nvme_stream"
        _, problems = R.check(CAT, dq, "gfx950", KIMI_BUILD)
        self.assertTrue(any("placement 'nvme_stream'" in p for p in problems), problems)

    def test_kv_cache_handed_over_in_two_formats(self):
        dq = copy.deepcopy(DQ)
        dq["roles"]["decode"]["kv_cache"] = "bf16"
        _, problems = R.check(CAT, dq, "gfx950", KIMI_BUILD)
        self.assertTrue(any("KV cache handed from prefill" in p for p in problems), problems)

    def test_nvfp4_is_not_assumed_anywhere(self):
        for arch in CAT["accelerators"]:
            with self.subTest(arch=arch):
                _, problems = R.check(CAT, {"roles": {"server": {"quantization": {"activations": "nvfp4"}}}}, arch)
                self.assertTrue(problems)


class DisaggregatedQuantization(unittest.TestCase):
    """Per-role formats and artifacts: the case the roles model exists for."""

    def test_resolves_where_fp4_compute_exists(self):
        for arch, build in (("gfx950", KIMI_BUILD), ("cdna5", None)):
            with self.subTest(arch=arch):
                res, problems = R.check(CAT, DQ, arch, build)
                self.assertEqual(problems, [])
                self.assertEqual(used(res, "prefill", "activations"), "mxfp4")
                self.assertEqual(used(res, "decode", "activations"), "bf16")
                self.assertEqual(next(r for r in res if r["role"] == "prefill")["artifact"], "org/model-w4a4-mxfp4")
                self.assertEqual(next(r for r in res if r["role"] == "decode")["artifact"], "org/model-w4a16")

    def test_refused_on_gfx942_naming_fp4_activations(self):
        _, problems = R.check(CAT, DQ, "gfx942", KIMI_BUILD)
        self.assertTrue(any(p.startswith("prefill: activations 'mxfp4'") for p in problems), problems)
        self.assertFalse(any(p.startswith("decode:") for p in problems), "weight-only decode fits gfx942")

    def test_a_variant_overrides_a_role_on_one_accelerator(self):
        dq = copy.deepcopy(DQ)
        dq["roles"]["variants"] = {"gfx942": {"prefill": {"quantization": {"activations": "fp8"}}}}
        res, problems = R.check(CAT, dq, "gfx942", KIMI_BUILD)
        self.assertEqual(problems, [])
        self.assertEqual(used(res, "prefill", "activations"), "fp8_e4m3fnuz")
        self.assertEqual(used(res, "prefill", "weights"), "int4_g32")
        res, _ = R.check(CAT, dq, "gfx950", KIMI_BUILD)
        self.assertEqual(used(res, "prefill", "activations"), "mxfp4")   # other accelerators keep the role's


class Cli(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = R.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def test_check_refuses_with_the_reason(self):
        rc, _, err = self.run_cli("check", "--recipes", VLLM_RECIPES, "--model", "Kimi-K3-MXFP4-MI355X",
                                  "--arch", "gfx942", "--build", KIMI_BUILD)
        self.assertEqual(rc, 1)
        self.assertIn("Kimi-K3-MXFP4-MI355X on gfx942", err)

    def test_recipe_without_roles_resolves_nothing(self):
        rc, out, _ = self.run_cli("check", "--recipes", VLLM_RECIPES, "--model", "gpt-oss-120b", "--arch", "gfx942")
        self.assertEqual((rc, out.strip()), (0, "gpt-oss-120b: declares no roles; nothing to resolve"))

    def test_no_such_recipe_is_its_own_status(self):
        self.assertEqual(self.run_cli("check", "--recipes", VLLM_RECIPES, "--model", "Nope", "--arch", "gfx942")[0], 2)

    def test_explain_is_json(self):
        rc, out, _ = self.run_cli("explain", "--recipes", VLLM_RECIPES, "--model", "Kimi-K3-MXFP4",
                                  "--arch", "gfx942", "--build", KIMI_BUILD)
        self.assertEqual(rc, 0)
        self.assertEqual([r["role"] for r in json.loads(out)], ["prefill", "decode"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
