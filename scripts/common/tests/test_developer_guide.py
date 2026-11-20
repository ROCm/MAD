#!/usr/bin/env python3
"""docs/context_as_code_developer_guide.md, "Change anything, see what the run does": each change
the guide makes to the DeepSeek-V3 baseline (scripts/vllm_dissag/workloads/guide/), run through the
real launchers (the snapshot harness's stand-ins for SLURM, Docker and the GPUs), with the effect
the guide shows. Standard library and PyYAML; madengine optional."""
import json
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_e2e_examples as T  # noqa: E402

GUIDE = T.SCRIPTS / "vllm_dissag" / "workloads" / "guide"
BASE_CARD = T.card("vllm_dissag", "pyt_e2e_deepseek-v3_disagg_mtp")    # a DeepSeek-V3 card on vllm-per-gpu


def go(name, arch="gfx942", nodes=None, **env):
    c = T.on(BASE_CARD, arch, MAD_CONFIG=str(GUIDE / name), **env)
    if nodes:
        c["slurm"]["nodes"] = c["distributed"]["nnodes"] = nodes
    return T.run("vllm_dissag", c)


def env(got, rank=0):
    return dict(e.split("=", 1) for e in got["ranks"][rank]["docker"]["env"] if "=" in e)


def roles(got):
    return [env(got, i).get("MAD_TOPO_ROLE") for i in range(len(got["ranks"]))]


def flag(got, rank, name):
    words = T.argv(got, rank)
    return words[words.index(name) + 1] if name in words else None


def recipe_line(got):
    return next(line for line in got["output"].splitlines() if line.startswith("[recipe]"))


class Baseline(unittest.TestCase):
    def test_one_prefill_and_one_decode_node_over_moriio(self):
        got = go("deepseek-v3.yaml")
        self.assertEqual((got["exit_code"], roles(got)), (0, ["prefill", "decode"]))
        self.assertIn("MoRIIOConnector", flag(got, 0, "--kv-transfer-config"))
        self.assertEqual(flag(got, 0, "--kv-cache-memory-bytes"), "20000000000")


class ChangeTheServingCluster(unittest.TestCase):
    def test_2p2d_needs_its_four_nodes_and_says_so_before_anything_runs(self):
        refused = go("scale-2p2d.yaml")
        self.assertEqual((refused["exit_code"], refused["ranks"]), (1, []))
        self.assertIn("topology needs 4 nodes but the allocation has 2", refused["output"])
        got = go("scale-2p2d.yaml", nodes=4)
        self.assertEqual(roles(got), ["prefill", "prefill", "decode", "decode"])
        self.assertEqual(flag(got, 0, "--data-parallel-size"), "16")             # 8 -> 16: two nodes per pool
        for rank in (1, 3):                                                       # each pool's second node
            self.assertEqual(flag(got, rank, "--data-parallel-start-rank"), "8")
            self.assertIn("--headless", T.argv(got, rank))

    def test_another_serving_layer_moves_the_kv_cache_over_nixl(self):
        got = go("serve-nixl.yaml")
        self.assertEqual(got["exit_code"], 0)
        self.assertIn("NixlConnector", flag(got, 0, "--kv-transfer-config"))
        self.assertEqual(env(got)["CONNECTOR"], "rixl")
        self.assertNotIn("MORI_IO_SL", env(got))                                  # MoRI-IO's env is gone


class TieValuesTogether(unittest.TestCase):
    def test_one_knob_sizes_both_pools_and_names_the_run(self):
        self.assertEqual(roles(go("tied.yaml", nodes=4)), ["prefill", "prefill", "decode", "decode"])
        self.assertEqual(roles(go("tied.yaml", POOL="1")), ["prefill", "decode"])      # set at submit
        data = T.W.load(str(GUIDE / "tied.yaml"), {"POOL": "1"})
        self.assertEqual(T.W.env_of(data)["PERF_TAGS"], "deepseek-v3,1p1d,sweep")

    @unittest.skipIf(T.resolve_for_model is None, "madengine not importable (optional)")
    def test_madengine_ties_them_the_same_way(self):
        c = {"name": "guide", "env_vars": {"MAD_CONFIG": "workloads/guide/tied.yaml"}}
        for runtime in ({}, {"POOL": "1"}, {"xP": "3"}):
            with self.subTest(runtime=runtime):
                theirs, _ = T.resolve_for_model(c, T.SCRIPTS / "vllm_dissag", runtime_env=runtime)
                self.assertEqual(T.mad_env(c, "vllm_dissag", runtime), theirs)


class AddAModelByChangingItsRecipe(unittest.TestCase):
    def test_the_new_recipe_changes_only_what_it_says_on_each_gpu(self):
        for arch, kv in (("gfx942", "48000000000"), ("gfx950", "72000000000")):  # a quarter of 192 / 288 GB
            with self.subTest(arch=arch):
                got = go("new-model.yaml", arch=arch)
                self.assertEqual(got["exit_code"], 0, got["output"][-800:])
                self.assertIn("DeepSeek-V3-LongCtx resolves", recipe_line(got))   # its alias's roles
                for rank in (0, 1):
                    self.assertEqual(flag(got, rank, "--max-model-len"), "131072")
                    self.assertEqual(flag(got, rank, "--kv-cache-memory-bytes"), kv)
                self.assertIsNone(flag(got, 0, "--max-num-seqs"))                 # decode only
                self.assertEqual(flag(got, 1, "--max-num-seqs"), "64")


class ChangeTheGpuAndTheSite(unittest.TestCase):
    def test_each_gpu_gets_its_formats_and_its_stack(self):
        self.assertIn("on gfx942 with vllm@vllm-mori121: prefill weights=fp8_e4m3fnuz",
                      recipe_line(go("deepseek-v3.yaml")))
        self.assertIn("on gfx950 with vllm@vllm-kimi-k3: prefill weights=fp8_e4m3,",
                      recipe_line(go("deepseek-v3.yaml", arch="gfx950")))

    def test_a_site_profile_changes_the_fabric_not_the_engine(self):
        cx7 = go("deepseek-v3.yaml", MAD_CLUSTER="example-slurm-mi300x-cx7")
        ainic = go("deepseek-v3.yaml", arch="gfx950", MAD_CLUSTER="example-slurm-mi355x-ainic")
        self.assertEqual(T.argv(ainic), T.argv(cx7))                              # the same engine commands
        self.assertEqual(env(ainic)["NCCL_IB_HCA"], ",".join(f"rdma{i}" for i in range(8)))
        self.assertEqual((env(cx7)["NCCL_IB_GID_INDEX"], env(ainic)["NCCL_IB_GID_INDEX"]), ("3", "1"))
        self.assertEqual((env(cx7)["NCCL_SOCKET_IFNAME"], env(ainic)["NCCL_SOCKET_IFNAME"]), ("eth0", "eno0"))


class OverrideOneRun(unittest.TestCase):
    def test_a_submit_time_value_wins_and_nothing_else_moves(self):
        base, got = go("deepseek-v3.yaml"), go("deepseek-v3.yaml", BENCHMARK_COMBINATIONS="8192/1024")
        self.assertEqual(env(got)["BENCHMARK_COMBINATIONS"], "8192/1024")
        self.assertEqual(T.argv(got), T.argv(base))


if __name__ == "__main__":
    unittest.main()
