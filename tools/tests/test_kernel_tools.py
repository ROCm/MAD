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
        self.assertEqual(ctx["tools"][0]["cmd"], "bash /mad_kernel/shim/enter.sh")
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
        self.assertEqual(dispatch["tools"], [{"name": "rocprofv3", "cmd": "bash /mad_kernel/shim/enter.sh"}])
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

    def test_enter_wraps_a_python_shebang_and_leaves_bash_alone(self) -> None:
        run("context", "--mode", "dispatch", "--run-dir", str(self.tmp))
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        entry = bin_dir / "clip_benchmark"
        entry.write_text("#!/usr/bin/python3\nprint('ran')\n")
        entry.chmod(0o755)
        profiler = bin_dir / "rocprofv3"
        profiler.write_text("#!/usr/bin/python3\n")
        profiler.chmod(0o755)
        bash_tool = bin_dir / "helper"
        bash_tool.write_text("#!/bin/bash\necho helper\n")
        bash_tool.chmod(0o755)
        script = self.tmp / "run.sh"
        seen = self.tmp / "seen_pythonpath"
        script.write_text(
            "#!/bin/bash\n# clip_benchmark helper rocprofv3\n"
            f"printf '%s\\n' \"$PYTHONPATH\" > {seen}\nexit 0\n")
        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:/bin"
        env.pop("PYTHONPATH", None)
        enter = str(self.tmp / "shim" / "enter.sh")
        subprocess.run(["bash", enter, "bash", str(script)], env=env, check=True)
        self.assertIn(str(self.tmp / "shim"), seen.read_text())
        wrapped = (self.tmp / "shim" / "clip_benchmark").read_text()
        self.assertIn(str(entry), wrapped)
        self.assertIn(str(profiler), wrapped)
        self.assertIn('exec "$real" "$@"', wrapped)
        self.assertFalse((self.tmp / "shim" / "helper").exists())
        self.assertFalse((self.tmp / "shim" / "rocprofv3").exists())
        self.assertIn('command -v', (self.tmp / "shim" / "python3").read_text())
        (self.tmp / "shim" / "profiler.cmd").write_text("rocprofv3 --kernel-trace NEW --\n")
        subprocess.run(["bash", enter, "bash", str(script)], env=env, check=True)
        refreshed = (self.tmp / "shim" / "clip_benchmark").read_text()
        self.assertIn("rocprofv3 --kernel-trace NEW --", refreshed)
        self.assertNotIn("rocprof/dispatch", refreshed)
        site = (self.tmp / "shim" / "sitecustomize.py").read_text()
        self.assertIn('set_start_method("spawn")', site)
        py = os.environ.copy()
        py["PYTHONPATH"] = str(self.tmp / "shim")
        py.pop("PYTHONSAFEPATH", None)
        out = subprocess.run(
            [sys.executable, "-c",
             "import multiprocessing as m; print(m.get_start_method())"],
            env=py, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "spawn")

    def test_context_serialize_is_opt_in(self) -> None:
        plain = run("context", "--mode", "counters", "--kernel", "k", "--run-dir", str(self.tmp))
        self.assertNotIn("AMD_SERIALIZE_KERNEL", plain["docker_env_vars"])
        serial = run("context", "--mode", "counters", "--kernel", "k", "--serialize",
                     "--run-dir", str(self.tmp))
        self.assertEqual(serial["docker_env_vars"]["AMD_SERIALIZE_KERNEL"], "3")

    def test_context_thread_trace_needs_one_kernel_and_a_short_range(self) -> None:
        missing = subprocess.run([sys.executable, str(TOOL), "context", "--mode", "thread_trace",
                                  "--run-dir", str(self.tmp)], capture_output=True, text=True)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("needs --kernel", missing.stderr)
        ctx = run("context", "--mode", "thread_trace", "--kernel", "^_fw_skmm_kernel$",
                  "--iteration-range", "[300-305]", "--run-dir", str(self.tmp))
        self.assertEqual(ctx["tools"][0]["name"], "rocprofv3")
        shim = (self.tmp / "shim" / "python3").read_text()
        self.assertIn("--att --att-target-cu 1", shim)
        self.assertIn("--kernel-iteration-range '[300-305]'", shim)
        self.assertIn("-d /mad_kernel/rocprof/thread_trace --", shim)
        self.assertTrue((self.tmp / "context.json").exists())

    def test_threadtrace_splits_variants_and_matches_the_saved_code_object(self) -> None:
        out = self.tmp / "rocprof" / "thread_trace"
        out.mkdir(parents=True)
        header = "CodeObj,Vaddr,Instruction,Hitcount,Latency,Stall,Idle,Source\n"

        def stats(name: str, code: str, kernel: str, wait: int, mfma: int) -> None:
            (out / name).write_text(
                header
                + f'{code},0,"; {kernel}",0,0,0,0,""\n'
                + f'{code},8,"s_waitcnt vmcnt(0)",2,{wait},1,0,""\n'
                + f'{code},16,"v_mfma_f32_16x16x16_bf16 v[0:3], v[4:5], v[6:7], v[0:3]",1,{mfma},0,0,""\n')

        stats("stats_ui_output_agent_1_dispatch_10.csv", "7", "_fw_skmm_kernel", 80, 20)
        stats("stats_ui_output_agent_1_dispatch_11.csv", "9", "_fw_skmm_kernel", 40, 60)
        blob = b"code-object-bytes"
        (out / "1_gfx950_code_object_id_7.out").write_bytes(blob)
        cached = self.tmp / "co" / "triton" / "ABC"
        cached.mkdir(parents=True)
        (cached / "_fw_skmm_kernel.hsaco").write_bytes(blob)
        summary = run("threadtrace", "--run-dir", str(self.tmp), "--kernel", "_fw_skmm_kernel")
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["dispatches_traced"], 2)
        by_code = {tuple(v["code_objects"]): v for v in summary["variants"]}
        first = by_code[("7",)]
        self.assertEqual(first["static_mfma"], 1)
        self.assertAlmostEqual(first["by_class"]["waitcnt"]["latency_share"], 0.8)
        self.assertEqual(first["code_object_files"][0]["matches"],
                         ["co/triton/ABC/_fw_skmm_kernel.hsaco"])
        self.assertAlmostEqual(by_code[("9",)]["by_class"]["mfma"]["latency_share"], 0.6)


if __name__ == "__main__":
    unittest.main()
