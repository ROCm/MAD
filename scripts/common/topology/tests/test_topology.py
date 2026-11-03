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


SPECS = HERE.parent / "specs"
SPEC_LAUNCHER = {"vllm_disagg_2p2d_wide_ep_tp2.json": "vllm_dissag",
                 "sglang_disagg_1p2d_dp_ep.json": "sglang_disagg",
                 "vllm_colocated_2n_tp8_pp2.json": "vllm_multinode"}


class SpecFile(unittest.TestCase):
    """MAD_TOPOLOGY: a spec file in place of the launcher's xP / yD / ... settings.
    That every card runs the same from its spec file is the snapshot harness's
    `--spec-file` check; these are the rules around it."""

    def _cli(self, spec_path, launcher, **env):
        import os, subprocess
        e = {"PATH": os.environ["PATH"], "MAD_TOPOLOGY": str(spec_path), **env}
        return subprocess.run([sys.executable, str(HERE.parent / "topology.py"), "legacy-env", "--launcher", launcher],
                              env=e, capture_output=True, text=True)

    def test_examples(self):
        self.assertEqual(sorted(p.name for p in SPECS.glob("*.json")), sorted(SPEC_LAUNCHER))
        for name, launcher in SPEC_LAUNCHER.items():
            T.legacy_env(T.load(str(SPECS / name)), launcher)

    def test_launcher_names_kept(self):
        p = self._cli(SPECS / "vllm_disagg_2p2d_wide_ep_tp2.json", "vllm_dissag")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.split(), ["GPUS_PER_NODE=8", "xP=2", "yD=2", "WIDE_EP=1", "EP_TP_SIZE=2"])

    def test_relative_to_scripts(self):
        p = self._cli("common/topology/specs/sglang_disagg_1p2d_dp_ep.json", "sglang_disagg")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("yD=2", p.stdout.split())

    def test_conflicting_setting_refused(self):
        p = self._cli(SPECS / "vllm_disagg_2p2d_wide_ep_tp2.json", "vllm_dissag", xP="1")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("xP=1 disagrees with MAD_TOPOLOGY (xP=2)", p.stderr)
        self.assertEqual(self._cli(SPECS / "vllm_disagg_2p2d_wide_ep_tp2.json", "vllm_dissag", xP="2").returncode, 0)

    def test_shapes_a_launcher_cannot_run(self):
        disagg = T.load(str(SPECS / "vllm_disagg_2p2d_wide_ep_tp2.json"))
        bad = {
            "wrong engine": (disagg, "sglang_disagg"),
            "wrong layout": (T.load(str(SPECS / "vllm_colocated_2n_tp8_pp2.json")), "vllm_dissag"),
            "router elsewhere": (dict(disagg, router={"host": "decode", "index": 0}), "vllm_dissag"),
            "mixed parallelism": (dict(disagg, roles=[disagg["roles"][0], dict(disagg["roles"][1],
                                  parallel={"mode": "tp"})]), "vllm_dissag"),
            "decode first": (dict(disagg, roles=disagg["roles"][::-1]), "vllm_dissag"),
        }
        for why, (spec, launcher) in bad.items():
            with self.subTest(why), self.assertRaises(T.TopologyError):
                T.legacy_env(spec, launcher)

    def test_apply_stops_the_launcher(self):
        import os, subprocess
        sh = f'. {HERE.parent / "apply.sh"} vllm_dissag; echo "xP=$xP"'
        ok = subprocess.run(["bash", "-c", sh], capture_output=True, text=True,
                            env={"PATH": os.environ["PATH"], "MAD_TOPOLOGY": str(SPECS / "vllm_disagg_2p2d_wide_ep_tp2.json")})
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("xP=2", ok.stdout.splitlines())
        bad = subprocess.run(["bash", "-c", sh], capture_output=True, text=True,
                             env={"PATH": os.environ["PATH"], "MAD_TOPOLOGY": "/nonexistent.json"})
        self.assertEqual(bad.returncode, 1)
        self.assertNotIn("xP=", bad.stdout)
        unset = subprocess.run(["bash", "-c", sh], capture_output=True, text=True, env={"PATH": os.environ["PATH"]})
        self.assertEqual(unset.stdout.strip(), "xP=")


class RoleScriptsFollowTheSpec(unittest.TestCase):
    """The role scripts take their part from MAD_TOPO_* when given, over NODE_RANK:
    node rank 0 told it is a decode master (or a colocated worker) acts as one."""

    def _dry(self, script, env):
        import os, subprocess
        base = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), "DRY_RUN": "1",
                "NODE_RANK": "0", "xP": "1", "yD": "1", "MODEL_PATH": "/m/M", "MASTER_ADDR": "10.0.0.1",
                "IPADDRS": "10.0.0.1,10.0.0.2", "GPUS_PER_NODE": "8", "SLURM_JOB_ID": "T"}
        base.update(env)
        p = subprocess.run(["bash", str(script)], env=base, capture_output=True, text=True, timeout=120)
        return [l for l in p.stdout.splitlines() if l.startswith("===DRYRUN")]

    def test_vllm_disagg(self):
        d = SCRIPTS / "vllm_dissag"
        h = self._dry(d / "vllm_disagg.sh", {"NIXL_COOKBOOK_PATH": str(d), "CONNECTOR": "moriio", "WIDE_EP": "1",
                                             "EP_BACKEND": "mori", "MODEL_NAME": "DeepSeek-V3",
                                             "PROXY_TYPE": "vllm_router", "ROUTER_PORT": "30000",
                                             "MAD_TOPO_ROLE": "decode", "MAD_TOPO_ROLE_INDEX": "0"})
        self.assertTrue(h and "log_prefix=decode" in h[0] and "role=master" in h[0], h)

    def test_sglang_disagg(self):
        d = SCRIPTS / "sglang_disagg"
        h = self._dry(d / "sglang_disagg_mori_io_ep.sh", {"MOONCAKE_COOKBOOK_PATH": str(d), "_SGL_HELP": "",
                                                          "MODEL_NAME": "Qwen3-32B",
                                                          "MAD_TOPO_ROLE": "decode", "MAD_TOPO_ROLE_INDEX": "0"})
        self.assertEqual([x.split("role=")[1].split()[0] for x in h], ["decode"], h)

    def test_colocated(self):
        d = SCRIPTS / "vllm_multinode"
        h = self._dry(d / "serve_colocated.sh", {"SHARED_DIR": str(SCRIPTS / "vllm_dissag"), "MODEL_NAME": "M",
                                                 "NNODES": "2", "MAD_TOPO_ROLE": "server", "MAD_TOPO_ROLE_INDEX": "1"})
        self.assertTrue(h and "role=worker" in h[0], h)


if __name__ == "__main__":
    unittest.main()
