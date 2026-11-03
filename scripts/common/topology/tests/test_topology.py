"""The topology spec resolves to what every card's nodes actually do.

For every card in the multi-node catalogs, the spec built from the card's settings
(topology.from_env) must resolve each node rank to the role that rank's role script
takes in the snapshot goldens (scripts/common/tests/snapshot/golden): prefill or
decode master/child and the router host for the disagg launchers, head or worker for
the colocated one, and for vLLM wide-EP the DP size and start rank in its vllm serve
argv. Run from the MAD repo root:
    python3 -m unittest discover -s scripts/common/topology/tests -v
"""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import topology as T  # noqa: E402

SCRIPTS = HERE.parents[2]
GOLDEN = SCRIPTS / "common" / "tests" / "snapshot" / "golden"


def _arg(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


class GoldenRoles(unittest.TestCase):
    def test_every_card(self):
        checked = 0
        for launcher in T.LAUNCHERS:
            for card in json.loads((SCRIPTS / launcher / "models.json").read_text()):
                gold = json.loads((GOLDEN / launcher / f"{card['name']}.json").read_text())
                if gold["exit_code"] != 0:
                    continue        # stops before any node runs (madengine-only card)
                env = {k: str(v) for k, v in (card.get("env_vars") or {}).items()}
                env["SLURM_NNODES"] = str(gold["inputs"]["nodes"])
                spec = T.from_env(env, launcher)
                self.assertEqual(T.nnodes(spec), len(gold["ranks"]), card["name"])
                for r in gold["ranks"]:
                    with self.subTest(card=card["name"], rank=r["rank"]):
                        self._check(launcher, T.resolve(spec, int(r["rank"])), r["engine"]["blocks"])
                        checked += 1
        self.assertGreater(checked, 80)

    def _check(self, launcher, node, blocks):
        headers = [b["header"] for b in blocks]
        if launcher == "vllm_multinode":
            want = "head" if node["role_index"] == 0 else "worker"
            self.assertIn(f"role={want}", headers[0])
            return
        if launcher == "sglang_disagg":
            roles = [h.split("role=")[1].split()[0] for h in headers]
            self.assertEqual(roles[0], node["role"])
            self.assertEqual("router" in roles, node["router"])
            return
        # vllm_dissag: log_prefix is the pool, role is master (the pool's first node) or child
        h = headers[0]
        self.assertIn(f"log_prefix={node['role']}", h)
        self.assertIn("role=master" if node["role_index"] == 0 else "role=child", h)
        if "dp_size" in node:
            argv = blocks[0]["argv"]
            self.assertEqual(_arg(argv, "--data-parallel-size"), str(node["dp_size"]))
            start = _arg(argv, "--data-parallel-start-rank")
            if start is not None:
                self.assertEqual(start, str(node["dp_start_rank"]))


class Spec(unittest.TestCase):
    def test_router_on_prefill_master(self):
        spec = T.from_env({"xP": "2", "yD": "1", "WIDE_EP": "1"}, "vllm_dissag")
        self.assertEqual([T.resolve(spec, r)["router"] for r in range(3)], [True, False, False])
        self.assertEqual(T.resolve(spec, 2)["role"], "decode")
        self.assertEqual(T.resolve(spec, 1)["dp_start_rank"], 8)

    def test_legacy_shim(self):
        self.assertEqual(T.from_env({"RUN_MORI": "1"}, "vllm_dissag")["roles"][0]["parallel"]["mode"], "wide_ep")
        self.assertEqual(T.from_env({"RUN_MORI": "1", "CONNECTOR": "moriio"}, "vllm_dissag")
                         ["roles"][0]["parallel"]["mode"], "tp")

    def test_invalid_specs_are_refused(self):
        with self.assertRaises(T.TopologyError):
            T.validate({"layout": "disagg", "roles": [{"name": "p", "nodes": 0}]})
        with self.assertRaises(T.TopologyError):
            T.validate({"layout": "colocated", "gpus_per_node": 8,
                        "roles": [{"name": "s", "nodes": 2, "parallel": {"mode": "tp_pp", "tp": 8, "pp": 1}}]})
        with self.assertRaises(T.TopologyError):
            T.validate({"layout": "disagg", "roles": [{"name": "p", "nodes": 1}], "router": {"host": "x"}})
        with self.assertRaises(T.TopologyError):
            T.resolve(T.from_env({}, "sglang_disagg"), 2)


if __name__ == "__main__":
    unittest.main()
