"""Topology v2: platforms, GPUs, partitions, scale-up domains, placement, fabric tiers, kinds.

Run from the MAD repo root:  python3 -m unittest discover -s scripts/common/topology/tests -v
"""
import copy
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import topology as T  # noqa: E402

V2 = HERE.parent / "specs" / "v2"


def spec(name):
    return json.loads((V2 / name).read_text())


def wide(name, **kw):
    return {"name": name, "parallel": {"mode": "wide_ep", "ep_tp": 1}, **kw}


class Examples(unittest.TestCase):
    def test_every_example_validates(self):
        for f in sorted(V2.glob("*.json")):
            with self.subTest(spec=f.name):
                T.validate(json.loads(f.read_text()))

    def test_helios_wide_ep_72_per_rack(self):
        s = T.validate(spec("helios_wide_ep72.json"))
        self.assertEqual((T.nnodes(s), T.gpus_per_node(s)), (36, 4))
        self.assertEqual({T.resolve(s, r)["domain"] for r in range(18)}, {0})
        self.assertEqual({T.resolve(s, r)["domain"] for r in range(18, 36)}, {1})
        self.assertEqual(T.resolve(s, 18)["dp_size"], 72)
        env = T.legacy_env(s, "vllm_dissag", {})
        self.assertEqual((env["GPUS_PER_NODE"], env["xP"], env["yD"], env["WIDE_EP"]), (4, 18, 18, 1))

    def test_cpx_counts_logical_gpus(self):
        s = T.validate(spec("mi355x_cpx_multitenant.json"))
        self.assertEqual(T.gpus_per_node(s), 64)
        self.assertEqual(T.check(s), ([], []))       # TP-8 replicas fit the 64-logical-GPU domain

    def test_heterogeneous_pools_resolve_but_no_launcher_runs_them_yet(self):
        s = T.validate(spec("hetero_prefill_decode.json"))
        self.assertEqual(T.resolve(s, 2)["gpus_per_node"], 4)    # decode on Helios trays
        with self.assertRaisesRegex(T.TopologyError, "one node type"):
            T.legacy_env(s, "vllm_dissag", {})

    def test_a_draft_role_is_expressible_not_runnable(self):
        s = T.validate(spec("speculative_draft.json"))
        self.assertEqual(T.resolve(s, 1)["kind"], "draft")
        with self.assertRaisesRegex(T.TopologyError, "serving type"):
            T.legacy_env(s, "vllm_multinode", {})


class Refusals(unittest.TestCase):
    def base(self, **kw):
        return {"engine": "vllm", "layout": "disagg", "platform": "mi455x-helios-tray",
                "roles": [wide("prefill", gpus=8), wide("decode", gpus=8)],
                "router": {"host": "prefill", "index": 0}, **kw}

    def test_gpus_must_fill_nodes(self):
        s = self.base(); s["roles"][0]["gpus"] = 6
        with self.assertRaisesRegex(T.TopologyError, "whole number of 4-GPU nodes"):
            T.validate(s)

    def test_same_domain_that_rank_order_cannot_meet(self):
        s = self.base(placement=[{"same_domain": ["decode"]}]); s["roles"][0]["gpus"] = 68
        with self.assertRaisesRegex(T.TopologyError, "span domains"):
            T.validate(s)

    def test_spread_domains(self):
        with self.assertRaisesRegex(T.TopologyError, "share domain 0"):
            T.validate(self.base(placement=[{"spread_domains": ["prefill", "decode"]}]))

    def test_strict_refuses_an_ep_group_split_across_domains(self):
        s = self.base(strict=True); s["roles"][0]["gpus"] = 68; s["roles"][1]["gpus"] = 8
        with self.assertRaisesRegex(T.TopologyError, "would fit one scale-up domain"):
            T.validate(s)
        s.pop("strict")
        self.assertTrue(any("would fit one" in n for n in T.check(T.validate(s))[1]))

    def test_strict_refuses_tp_beyond_the_domain(self):
        s = {"engine": "vllm", "layout": "colocated", "platform": "mi300x-oam-8", "strict": True,
             "roles": [{"name": "server", "nodes": 2, "parallel": {"mode": "tp_pp", "tp": 16, "pp": 1}}]}
        with self.assertRaisesRegex(T.TopologyError, "TP group of 16"):
            T.validate(s)

    def test_partitions(self):
        with self.assertRaisesRegex(T.TopologyError, "not known"):
            T.validate(self.base(partition="CPX"))         # CDNA 5 partition sizes are not stated
        s = self.base(); s.pop("platform"); s["partition"] = "CPX"; s["gpus_per_node"] = 8
        with self.assertRaisesRegex(T.TopologyError, "needs a platform"):
            T.validate(s)

    def test_kv_boundary(self):
        s = self.base(); s["roles"][0]["kv_format"] = "fp8"; s["roles"][1]["kv_format"] = "bf16"
        with self.assertRaisesRegex(T.TopologyError, "KV cache handed"):
            T.validate(s)

    def test_unknown_platform_and_kind(self):
        with self.assertRaisesRegex(T.TopologyError, "unknown platform"):
            T.validate(self.base(platform="nope"))
        s = self.base(); s["roles"][0]["kind"] = "oracle"
        with self.assertRaisesRegex(T.TopologyError, "kind 'oracle'"):
            T.validate(s)


class SchedulerHints(unittest.TestCase):
    def test_slurm_segment_per_scale_up_domain(self):
        h = T.scheduler_hints(spec("helios_wide_ep72.json"), "slurm")
        self.assertEqual(h["options"], ["--segment=18"])

    def test_kubernetes_affinity_needs_the_sites_label(self):
        self.assertEqual(T.scheduler_hints(spec("helios_wide_ep72.json"), "kubernetes")["options"], [])
        h = T.scheduler_hints(spec("helios_wide_ep72.json"), "kubernetes", "example.com/rack")
        self.assertEqual(len(h["options"]), 2)
        self.assertEqual(h["options"][0]["podAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][0]
                         ["topologyKey"], "example.com/rack")

    def test_single_node_domains_ask_for_nothing(self):
        self.assertEqual(T.scheduler_hints(spec("speculative_draft.json"), "slurm")["options"], [])


class V1Unchanged(unittest.TestCase):
    def test_a_v1_spec_resolves_without_v2_keys(self):
        for f in sorted((HERE.parent / "specs").glob("*.json")):
            s = T.validate(json.loads(f.read_text()))
            for r in range(T.nnodes(s)):
                with self.subTest(spec=f.name, rank=r):
                    self.assertFalse({"kind", "domain", "gpus_per_node"} & set(T.resolve(copy.deepcopy(s), r)))


if __name__ == "__main__":
    unittest.main()
