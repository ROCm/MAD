#!/usr/bin/env python3
"""What the context-as-code guides (docs/context_as_code_user_guide.md,
docs/context_as_code_developer_guide.md) show, on the real cards: switching a run's cluster and
recipe, the DeepSeek-V3 MTP card built by inheritance, values that follow a submitted one -- each run
through the real launchers (the snapshot harness's stand-ins for SLURM, Docker and the GPUs) -- and
every `snapshot.py --show` command the guides give runs as written. Standard library and PyYAML."""
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_e2e_examples as T  # noqa: E402

MAD = T.SCRIPTS.parent
KIMI = T.card("vllm_dissag", "pyt_e2e_kimi-k3_disagg_niah")
MTP = T.card("vllm_dissag", "pyt_e2e_deepseek-v3_disagg_mtp")


def go(card, arch, catalog="vllm_dissag", **env):
    return T.run(catalog, T.on(card, arch, **env))


def env(got, rank=0):
    return dict(e.split("=", 1) for e in got["ranks"][rank]["docker"]["env"] if "=" in e)


def flag(got, rank, name):
    words = T.argv(got, rank)
    return words[words.index(name) + 1] if name in words else None


def recipe_line(got):
    return next(line for line in got["output"].splitlines() if line.startswith("[recipe]") and "resolves" in line)


class SwitchTheCluster(unittest.TestCase):
    def test_the_kimi_k3_card_on_the_two_example_sites(self):
        cx7 = go(KIMI, "gfx942", MAD_CLUSTER="example-slurm-mi300x-cx7")
        self.assertEqual((env(cx7)["EP_TP_SIZE"], env(cx7)["NCCL_IB_GID_INDEX"], env(cx7)["NCCL_SOCKET_IFNAME"]),
                         ("2", "3", "eth0"))
        self.assertEqual(flag(cx7, 0, "--data-parallel-size"), "8")
        self.assertIn('"int4_per_group_32"', flag(cx7, 0, "--quantization-config"))
        self.assertIn("best-known configuration for Kimi-K3-MXFP4 on mi300x-oam-8", cx7["output"])
        ainic = go(KIMI, "gfx950", MAD_CLUSTER="example-slurm-mi355x-ainic")
        self.assertEqual((env(ainic)["EP_TP_SIZE"], env(ainic)["NCCL_IB_HCA"], env(ainic)["NCCL_IB_GID_INDEX"],
                          env(ainic)["NCCL_SOCKET_IFNAME"]), ("1", ",".join(f"rdma{i}" for i in range(8)), "1", "eno0"))
        self.assertEqual((flag(ainic, 0, "--data-parallel-size"), flag(ainic, 0, "--max-num-batched-tokens")), ("16", "4096"))
        self.assertIsNone(flag(ainic, 0, "--quantization-config"))                      # native MXFP4
        self.assertNotIn("best-known configuration", ainic["output"])

    def test_your_own_site_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("scheduler: slurm\nplatform: mi325x-oam-8\nfabric: thor2\n"
                    "env: {SHARED_MOUNT: /shared, NVME_ROOT: /mnt/nvme}\n")
        try:
            got = go(KIMI, "gfx942", MAD_CLUSTER_FILE=f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual((env(got)["NCCL_IB_HCA"], env(got)["NCCL_SOCKET_IFNAME"]),
                         (",".join(f"bnxt_re{i}" for i in range(8)), "fenic0"))
        self.assertNotIn("best-known configuration", got["output"])                     # recorded for mi300x-oam-8


class SwitchTheRecipe(unittest.TestCase):
    def test_another_recipe_of_the_same_model(self):
        base, other = go(KIMI, "gfx942"), go(KIMI, "gfx942", MODEL_NAME="Kimi-K3")
        self.assertEqual((flag(base, 0, "--kv-cache-memory-bytes"), flag(other, 0, "--kv-cache-memory-bytes")),
                         ("8000000000", "40000000000"))
        self.assertEqual(flag(other, 0, "--max-model-len"), "1000000")
        self.assertIn("Kimi-K3 resolves on gfx942 with vllm@vllm-kimi-k3", recipe_line(other))

    def test_one_recipe_value_at_submit(self):
        got = go(KIMI, "gfx950", EP_TP_SIZE="2")
        self.assertIn("[config] EP_TP_SIZE=2 (submit) over the recipe's 1", got["output"])
        self.assertEqual(flag(got, 0, "--data-parallel-size"), "8")

    def test_the_base_recipe_instead_of_its_variant(self):
        got = go(MTP, "gfx942", MODEL_NAME="DeepSeek-V3")
        self.assertIsNone(flag(got, 1, "--speculative-config"))
        self.assertEqual(flag(got, 0, "--kv-cache-memory-bytes"), "20000000000")


class AModelCardBuiltByInheritance(unittest.TestCase):
    def test_deepseek_v3_mtp_on_each_gpu(self):
        for arch, build, kv in (("gfx942", "vllm@vllm-mori121", "19200000000"),
                                ("gfx950", "vllm@vllm-kimi-k3", "28800000000")):
            with self.subTest(arch=arch):
                got = go(MTP, arch)
                self.assertIn(f"DeepSeek-V3-MTP resolves on {arch} with {build}", recipe_line(got))
                self.assertEqual((flag(got, 0, "--kv-cache-memory-bytes"), flag(got, 1, "--kv-cache-memory-bytes")), (kv, kv))
                self.assertIsNone(flag(got, 0, "--speculative-config"))
                self.assertIn('"deepseek_mtp"', flag(got, 1, "--speculative-config"))


class ValuesFollowASubmittedOne(unittest.TestCase):
    def test_the_replicas_layer_names_the_run_from_tp_size(self):
        f = T.SCRIPTS / "vllm_multinode" / "workloads" / "e2e" / "qwen3-32b-replicas-lora.yaml"
        self.assertEqual(T.W.env_of(T.W.load(str(f)))["PERF_TAGS"], "vllm_multinode,replicas,tp1")
        self.assertEqual(T.W.env_of(T.W.load(str(f), {"TP_SIZE": "2"}))["PERF_TAGS"], "vllm_multinode,replicas,tp2")

    @unittest.skipIf(T.resolve_for_model is None, "madengine not importable (optional)")
    def test_madengine_resolves_it_the_same_way(self):
        c = T.card("vllm_multinode", "pyt_e2e_qwen3-32b_replicas_lora")
        for runtime in ({}, {"TP_SIZE": "2"}):
            with self.subTest(runtime=runtime):
                theirs, _ = T.resolve_for_model(c, T.SCRIPTS / "vllm_multinode", runtime_env=runtime)
                self.assertEqual(T.mad_env(c, "vllm_multinode", runtime), theirs)


class TheGuidesCommandsRun(unittest.TestCase):
    """Every `snapshot.py --show` command in the two guides runs as written, from the MAD root."""

    def test_every_show_command_runs(self):
        commands = []
        for doc in ("context_as_code_developer_guide.md", "context_as_code_user_guide.md"):
            text = (MAD / "docs" / doc).read_text().replace("\\\n", " ")
            commands += [c.strip() for c in re.findall(r"(python3 scripts/common/tests/snapshot/snapshot.py --show [^`\n]*)", text)
                         if "<" not in c and "[" not in c and "$" not in c]      # standalone ones (the block
                                                                                  # test runs the rest in order)
        self.assertGreaterEqual(len(commands), 10)
        for cmd in sorted(set(commands)):
            with self.subTest(cmd=cmd):
                p = subprocess.run(shlex.split(cmd), cwd=MAD, capture_output=True, text=True, timeout=600)
                self.assertEqual(p.returncode, 0, p.stderr[-800:])
                self.assertIn(": exit 0", p.stdout)


class TheGuidesBlocksRun(unittest.TestCase):
    """Every bash block in the two guides runs as pasted, in order, from the MAD root -- with madengine
    and sbatch stood in for, Kubernetes submissions as dry runs, <placeholders> filled, and HOME a
    scratch directory (so the files the blocks write are real and later blocks use them). Blocks that
    read a finished job's logs are skipped."""

    def test_every_bash_block_runs(self):
        home = Path(tempfile.mkdtemp())
        made = MAD / "run_configs"
        self.assertFalse(made.exists(), "a run_configs/ in the MAD root would be overwritten")
        try:
            for doc in ("context_as_code_user_guide.md", "context_as_code_developer_guide.md"):
                blocks = re.findall(r"```bash\n(.*?)```", (MAD / "docs" / doc).read_text(), re.S)
                self.assertTrue(blocks, doc)
                for block in blocks:
                    if "LOG_PATH" in block:
                        continue
                    script = re.sub(r"<([a-z][^<>\n]*)>", "x", block.replace("\\\n", " "))
                    script = re.sub(r"(k8s/submit\.py(?![^\n]*--dry-run)[^\n]*)", r"\1 --dry-run", script)
                    with self.subTest(doc=doc, block=block[:80]):
                        # Not -e: a refusal shown on purpose (recipe.py check) exits non-zero.
                        stubs = ('madengine() { case "$*" in *--emit-config*) mkdir -p run_configs/compat;; esac; }\n'
                                 'sbatch() { :; }\n')
                        p = subprocess.run(["bash", "-c", stubs + script],
                                           cwd=MAD, capture_output=True, text=True, timeout=1800,
                                           env=dict(os.environ, HOME=str(home)))
                        out = p.stdout + p.stderr
                        for broken in ("Traceback", "command not found", "No such file", "usage:", ": exit 1"):
                            self.assertNotIn(broken, out, out[-1200:])
        finally:
            import shutil
            shutil.rmtree(home)
            shutil.rmtree(made, ignore_errors=True)       # what the stubbed --emit-config build made


if __name__ == "__main__":
    unittest.main()
