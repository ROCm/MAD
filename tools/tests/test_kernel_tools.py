"""GPU-free tests for .claude/skills/mad-common/kernel-tools.py.

The fixtures are real rocprofv3 7.14 and Triton output from one MI350X run of a
1024x1024x1024 BF16 torch.compile matmul, trimmed to a few dispatches.

    python3 -m unittest discover -s tools/tests -p 'test_*.py'
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / ".claude/skills/mad-common/kernel-tools.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures/kernel"


def run(*args: str) -> dict:
    proc = subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


class KernelToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_context_counters_overrides_one_tool_and_keeps_the_mounts(self) -> None:
        base = {"docker_mounts": {"/myworkspace/fam_cache": "/w"}, "tools": [{"name": "rocm_trace_lite"}]}
        ctx = run("context", "--mode", "counters", "--group", "memory", "--kernel", "triton_mm",
                  "--run-dir", str(self.tmp), "--context", json.dumps(base))
        self.assertEqual(ctx["docker_mounts"]["/myworkspace/fam_cache"], "/w")
        self.assertEqual(ctx["docker_mounts"]["/mad_kernel"], str(self.tmp))
        self.assertEqual([t["name"] for t in ctx["tools"]], ["rocprofv3_compute"])
        self.assertEqual(ctx["tools"][0]["cmd"], "env PATH=/mad_kernel/shim:$PATH")
        shim = (self.tmp / "shim" / "python3").read_text()
        self.assertIn("--kernel-include-regex triton_mm", shim)
        self.assertIn("-i /mad_kernel/counters.txt", shim)
        self.assertIn('-- "$real" "$@"', shim)
        self.assertTrue(os.access(self.tmp / "shim" / "python", os.X_OK))
        self.assertEqual(ctx["docker_env_vars"]["MAD_CODE_OBJECT_DIR"], "/mad_kernel/co")
        passes = (self.tmp / "counters.txt").read_text().splitlines()
        self.assertTrue(all(line.startswith("pmc: ") for line in passes))
        self.assertIn("pmc: FETCH_SIZE", passes)

    def test_context_moves_the_previous_output_of_its_mode_aside(self) -> None:
        old = self.tmp / "rocprof" / "counters" / "host"
        old.mkdir(parents=True)
        (old / "1_counter_collection.csv").write_text("")
        (self.tmp / "rocprof" / "dispatch").mkdir()
        run("context", "--mode", "counters", "--group", "compute", "--run-dir", str(self.tmp))
        self.assertFalse((self.tmp / "rocprof" / "counters").exists())
        self.assertTrue((self.tmp / "rocprof" / "counters.prev-1" / "host").is_dir())
        self.assertTrue((self.tmp / "rocprof" / "dispatch").is_dir())
        self.assertIn("-d /mad_kernel/rocprof/counters --", (self.tmp / "shim" / "python3").read_text())

    def test_context_pc_sampling_uses_stochastic_cycles(self) -> None:
        ctx = run("context", "--mode", "pc_sampling", "--run-dir", str(self.tmp))
        cmd = (self.tmp / "shim" / "python3").read_text()
        self.assertEqual(ctx["tools"][0]["name"], "rocprofv3_pc_sampling")
        self.assertIn("--pc-sampling-method stochastic --pc-sampling-unit cycles", cmd)
        self.assertIn("--output-format csv", cmd)

    def test_card_computes_the_roofline_from_real_counters(self) -> None:
        shutil.copytree(FIXTURES / "counters", self.tmp / "rocprof" / "counters")
        card = run("card", "--run-dir", str(self.tmp))
        self.assertEqual(card["arch"], "gfx950")
        top = card["kernels"][0]
        self.assertEqual(top["kernel"], "triton_tem_fused_mm_0")
        # 2 * 1024^3 FLOPs: the MFMA count reproduces the matmul exactly.
        self.assertEqual(top["mfma_flops"], 2 * 1024 ** 3)
        self.assertIn(top["bound"], {"compute", "bandwidth", "latency"})
        self.assertGreater(top["pct_peak_bandwidth"], 0)
        self.assertEqual(top["per_wave"]["mfma"], 64.0)
        self.assertTrue((self.tmp / "kernel_card.json").exists())

    def test_asm_matches_the_dispatched_variant_and_finds_lds(self) -> None:
        shutil.copytree(FIXTURES / "counters", self.tmp / "rocprof" / "counters")
        shutil.copytree(FIXTURES / "co", self.tmp / "co")
        run("card", "--run-dir", str(self.tmp))
        summary = run("asm", "--run-dir", str(self.tmp))
        self.assertEqual(summary["status"], "matched")
        variant = summary["variants"][0]
        self.assertEqual(variant["mix"]["mfma"], 64)
        self.assertEqual(variant["vgpr"], 64)
        self.assertEqual(variant["lds_bytes"], 32768)
        self.assertEqual(variant["memory_widths"]["vmem:dwordx4"], 32)
        self.assertLessEqual(variant["occupancy_waves_per_simd"], 8)

    def test_asm_reports_a_missing_code_object(self) -> None:
        summary = run("asm", "--run-dir", str(self.tmp), "--kernel", "Cijk_Alik_Bljk_BBS")
        self.assertEqual(summary["status"], "no code object")

    def test_pcsample_groups_stalls_for_the_hottest_kernel(self) -> None:
        shutil.copytree(FIXTURES / "pc_sampling", self.tmp / "rocprof" / "pc_sampling")
        summary = run("pcsample", "--run-dir", str(self.tmp))
        self.assertEqual(summary["kernel"], "triton_tem_fused_mm_0")
        self.assertGreater(summary["samples"], 100)
        self.assertAlmostEqual(sum(s["share"] for s in summary["stall_reasons"]), 1.0, delta=0.02)

    def test_context_warm_and_dispatch_share_the_code_object_cache(self) -> None:
        base = '{"tools": [{"name": "rocprofv3_compute"}], "docker_env_vars": {"FAM_CACHE": "/c"}}'
        (self.tmp / "co" / "inductor").mkdir(parents=True)
        warm = run("context", "--mode", "warm", "--run-dir", str(self.tmp), "--context", base)
        self.assertFalse((self.tmp / "co").exists())
        self.assertTrue((self.tmp / "co.prev-1" / "inductor").is_dir())
        self.assertEqual(warm["tools"], [])
        self.assertEqual(warm["docker_env_vars"]["TORCHINDUCTOR_CACHE_DIR"], "/mad_kernel/co/inductor")
        self.assertEqual(warm["docker_env_vars"]["FAM_CACHE"], "/c")
        self.assertFalse((self.tmp / "shim").exists())
        dispatch = run("context", "--mode", "dispatch", "--run-dir", str(self.tmp), "--context", base)
        self.assertEqual(dispatch["tools"], [{"name": "rocprofv3", "cmd": "env PATH=/mad_kernel/shim:$PATH"}])
        self.assertEqual(dispatch["docker_mounts"]["/mad_kernel"], str(self.tmp.resolve()))
        self.assertTrue((self.tmp / "context_dispatch.json").exists())
        shim = (self.tmp / "shim" / "python3").read_text()
        self.assertIn("rocprofv3 --kernel-trace --output-format csv -d /mad_kernel/rocprof/dispatch --", shim)

    def test_rank_orders_kernels_by_gpu_time(self) -> None:
        trace = self.tmp / "rocprof" / "dispatch" / "host"
        trace.mkdir(parents=True)
        shutil.copy(FIXTURES / "pc_sampling" / "pcs_kernel_trace.csv", trace / "1_kernel_trace.csv")
        rank = run("rank", "--run-dir", str(self.tmp), "--top", "3")
        self.assertGreater(rank["dispatches"], 0)
        totals = [k["total_us"] for k in rank["kernels"]]
        self.assertEqual(totals, sorted(totals, reverse=True))
        self.assertLessEqual(sum(k["share_pct"] for k in rank["kernels"]), 100.1)


if __name__ == "__main__":
    unittest.main()
