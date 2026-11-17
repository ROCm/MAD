"""Offline tests for render.py: catalog layering, topology, compat and check. No GPUs.

    python3 -m unittest discover -s scripts/sglang_disagg/tests
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("render", HERE / "render.py")
render = importlib.util.module_from_spec(spec)
sys.modules["render"] = render  # dataclasses look their module up while it loads
spec.loader.exec_module(render)
CAT = render.Catalog(HERE / "models.yaml")


def settings(**env):
    env = {"MODEL_NAME": "Llama-3.1-8B-Instruct", "_SGL_HELP": "", **{k: str(v) for k, v in env.items()}}
    if "IPADDRS" not in env:
        n = int(env.get("xP", 1)) + int(env.get("yD", 1)) + int(env.get("ROUTER_NODE", 0))
        env["IPADDRS"] = ",".join(f"10.0.0.{i + 1}" for i in range(n))
    return render.Settings.from_env(env)


def flags(argv):
    """argv as flag -> values, the last occurrence winning (as argparse reads it)."""
    out, cur = {}, None
    for tok in argv:
        if tok.startswith("--"):
            cur = tok
            out[cur] = []
        else:
            out[cur].append(tok)
    return out


def shell(**env):
    full = {"PATH": os.environ["PATH"], "_SGL_HELP": "", "MODEL_NAME": "Llama-3.1-8B-Instruct", **env}
    return subprocess.run([sys.executable, str(HERE / "render.py"), "shell", "--rank", env.get("RANK", "0")],
                          env=full, capture_output=True, text=True)


class Catalog(unittest.TestCase):
    def test_model_overrides_defaults(self):
        f = CAT.server_flags("amd-Llama-3.3-70B-Instruct-FP8-KV", "tp", "decode", "mori")
        self.assertEqual(f["--mem-fraction-static"], 0.65)
        self.assertEqual(f["--load-balance-method"], "round_robin")
        self.assertEqual(f["--disaggregation-transfer-backend"], "mori")

    def test_extends_with_null_drops_the_parent_value(self):
        v3 = CAT.server_flags("DeepSeek-V3", "dp_ep", "prefill", "mori")
        r1 = CAT.server_flags("DeepSeek-R1", "dp_ep", "prefill", "mori")
        self.assertEqual(v3["--mem-fraction-static"], 0.65)
        self.assertEqual(r1["--mem-fraction-static"], 0.73)
        self.assertEqual(r1["--moe-a2a-backend"], "mori")

    def test_agentic_overlays(self):
        f = CAT.server_flags("Llama-3.1-8B-Instruct", "tp", "decode", "mori", ["radix_cache", "server_metrics"])
        self.assertNotIn("--disable-radix-cache", f)
        self.assertIs(f["--enable-metrics"], True)

    def test_launcher_flags_are_refused_in_the_catalog(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as tmp:
            tmp.write("defaults:\n  base:\n    --tp-size: 4\nmodels:\n  M: {}\n")
            tmp.flush()
            with self.assertRaisesRegex(render.RenderError, "set by the launcher"):
                render.Catalog(Path(tmp.name)).server_flags("M", "tp", "prefill", "mori")

    def test_ranges_and_lists_expand(self):
        self.assertEqual(render.flags_to_argv({"--a": "1..3", "--b": [4, 5], "--c": True}),
                         ["--a", "1", "2", "3", "--b", "4", "5", "--c"])


class Topology(unittest.TestCase):
    def test_tp_2p2d_is_four_servers(self):
        nodes, leaders = render.build_topology(settings(xP=2, yD=2))
        self.assertEqual([nodes[r].role for r in range(4)], ["prefill", "prefill", "decode", "decode"])
        self.assertEqual(leaders, {"prefill": ["10.0.0.1", "10.0.0.2"], "decode": ["10.0.0.3", "10.0.0.4"]})
        self.assertTrue(nodes[0].runs_router)
        self.assertNotIn("--nnodes", render.server_argv(CAT, settings(xP=2, yD=2), nodes[1], None))

    def test_router_node_is_last_and_runs_no_server(self):
        s = settings(ROUTER_NODE=1)
        nodes, leaders = render.build_topology(s)
        self.assertEqual(nodes[2].role, "router")
        self.assertFalse(nodes[0].runs_router)
        argv = render.router_argv(s, leaders)
        self.assertNotIn("http://10.0.0.3:3000", argv)
        out = shell(IPADDRS="10.0.0.1,10.0.0.2,10.0.0.3", ROUTER_NODE="1", RANK="2")
        self.assertIn("ROUTER_ARGV=(", out.stdout)
        self.assertNotIn("SERVER_ARGV", out.stdout)

    def test_dp_ep_server_spans_its_role(self):
        s = settings(MODEL_NAME="DeepSeek-V3", DP_MODE=1, xP=2, yD=1)
        nodes, leaders = render.build_topology(s)
        f = flags(render.server_argv(CAT, s, nodes[1], None))
        self.assertEqual(f["--tp-size"], ["16"])
        self.assertEqual(f["--dp-size"], ["16"])
        self.assertEqual(f["--nnodes"], ["2"])
        self.assertEqual(f["--node-rank"], ["1"])
        self.assertEqual(f["--dist-init-addr"], ["10.0.0.1:5757"])
        self.assertEqual(leaders["prefill"], ["10.0.0.1"])

    def test_tp_server_across_two_nodes(self):
        with self.assertRaisesRegex(render.RenderError, "cannot be split"):
            render.build_topology(settings(xP=1, yD=1, GENERIC_TP_SIZE=16))
        s = settings(xP=2, yD=2, GENERIC_TP_SIZE=16)
        nodes, leaders = render.build_topology(s)
        self.assertEqual(leaders["prefill"], ["10.0.0.1"])
        self.assertEqual(flags(render.server_argv(CAT, s, nodes[3], None))["--node-rank"], ["1"])

    def test_wrong_address_count(self):
        with self.assertRaisesRegex(render.RenderError, "address"):
            render.build_topology(settings(xP=2, yD=1, IPADDRS="10.0.0.1,10.0.0.2"))

    def test_the_spec_decides_the_role(self):
        out = shell(IPADDRS="10.0.0.1,10.0.0.2", RANK="0", MAD_TOPO_ROLE="decode", MAD_TOPO_ROLE_INDEX="0")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("ROLE=decode", out.stdout)
        self.assertIn("RUNS_ROUTER=0", out.stdout)
        self.assertIn("--host 10.0.0.1", out.stdout)
        self.assertIn("following the topology spec", out.stderr)

    def test_an_agreeing_spec_says_nothing(self):
        out = shell(IPADDRS="10.0.0.1,10.0.0.2", RANK="1", MAD_TOPO_ROLE="decode",
                    MAD_TOPO_ROLE_INDEX="0", MAD_TOPO_ROUTER="0")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertNotIn("[topology]", out.stderr)

    def test_a_spec_role_the_run_lacks_fails(self):
        out = shell(IPADDRS="10.0.0.1,10.0.0.2", RANK="1", MAD_TOPO_ROLE="prefill", MAD_TOPO_ROLE_INDEX="1")
        self.assertEqual(out.returncode, 2)
        self.assertIn("has no place for", out.stderr)

    def test_router_node_seen_from_the_container(self):
        """node.sh passes no ROUTER_NODE; ROUTER_ADDR or MAD_TOPO_NNODES says it."""
        ips = "10.0.0.1,10.0.0.2,10.0.0.3"
        by_addr = shell(IPADDRS=ips, RANK="2", MASTER_ADDR="10.0.0.1", ROUTER_ADDR="10.0.0.3")
        by_topo = shell(IPADDRS=ips, RANK="2", MAD_TOPO_NNODES="3", MAD_TOPO_ROLE="router",
                        MAD_TOPO_ROLE_INDEX="0", MAD_TOPO_ROUTER="1")
        for out in (by_addr, by_topo):
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("ROLE=router", out.stdout)
            self.assertIn("ROUTER_ARGV=(", out.stdout)


class Validation(unittest.TestCase):
    def test_dp_mode_needs_a_dp_ep_block(self):
        with self.assertRaisesRegex(render.RenderError, "does not support DP_MODE=1"):
            render.validate(CAT, settings(DP_MODE=1))

    def test_backend_mode_combination(self):
        with self.assertRaisesRegex(render.RenderError, "not validated"):
            render.validate(CAT, settings(MODEL_NAME="DeepSeek-V3", DP_MODE=1, KV_TRANSFER_BACKEND="mooncake"))

    def test_unknown_model(self):
        with self.assertRaisesRegex(render.RenderError, "unknown MODEL_NAME"):
            render.validate(CAT, settings(MODEL_NAME="nope"))


class Compat(unittest.TestCase):
    OLD = {"--cuda-graph-bs", "--prefill-round-robin-balance"}
    NEW = {"--cuda-graph-bs-decode"}

    def test_renamed_and_removed_on_a_newer_sglang(self):
        f, notes = CAT.compat({"--cuda-graph-bs": [8], "--prefill-round-robin-balance": True}, self.NEW)
        self.assertEqual(f, {"--cuda-graph-bs-decode": [8]})
        self.assertEqual(len(notes), 2)

    def test_unchanged_where_accepted_or_unknown(self):
        written = {"--cuda-graph-bs": [8], "--prefill-round-robin-balance": True}
        self.assertEqual(CAT.compat(written, self.OLD)[0], written)
        self.assertEqual(CAT.compat(written, None)[0], written)

    def test_check_reports_a_flag_the_image_lacks(self):
        every = set()
        for name in CAT.names():
            for mode in CAT.modes_for(name):
                for role in render.ROLES:
                    every |= set(CAT.server_flags(name, mode, role, "mori", list(CAT.overlays)))
        every |= render.LAUNCHER_FLAGS | {"--disaggregation-transfer-backend"}
        self.assertEqual(render.cmd_check(CAT, every), 0)
        self.assertEqual(render.cmd_check(CAT, every - {"--attention-backend"}), 1)
        self.assertEqual(render.cmd_check(CAT, None), 1)


if __name__ == "__main__":
    unittest.main()
