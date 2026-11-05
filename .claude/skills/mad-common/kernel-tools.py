#!/usr/bin/env python3
"""Kernel and assembly depth for /mad-profile, /mad-tune and the workflows.

Standard library only. Every subcommand works on one host run directory and
writes its JSON result there, so a later step (or another agent) can read it.

    context   print the --additional-context for a warm-up run (no profiler),
              a dispatch trace, a kernel-counter run, a PC-sampling run or a
              thread trace, all sharing one code-object cache; write the
              counter file
    rank      rank kernels by GPU time from the dispatch trace
    card      turn the rocprofv3 counter CSVs into a kernel card: duration,
              bytes, MFMA FLOPs, roofline position, bound, instruction mix
    asm       summarize the kernel's code object: registers, LDS, spills,
              occupancy, instruction classes, memory widths, hot loops
    pcsample  summarize rocprofv3 PC samples: hottest instructions and
              stall reasons for the kernel
    threadtrace
              summarize a rocprofv3 --att trace: per compiled variant, where
              its cycles go, and which saved code object that variant is

Usage:

    python3 .claude/skills/mad-common/kernel-tools.py context --mode counters \\
        --group compute --kernel triton_mm --run-dir /var/tmp/mad-kernel/k1 \\
        --context '{"docker_mounts": {...}}'
    python3 .claude/skills/mad-common/kernel-tools.py card --run-dir /var/tmp/mad-kernel/k1
    python3 .claude/skills/mad-common/kernel-tools.py asm --run-dir /var/tmp/mad-kernel/k1
    python3 .claude/skills/mad-common/kernel-tools.py pcsample --run-dir /var/tmp/mad-kernel/k1
    python3 .claude/skills/mad-common/kernel-tools.py threadtrace --run-dir /var/tmp/mad-kernel/k1

Run --mode warm once before the profiled runs: a cold torch.compile under a
profiler spends its time compiling and autotuning, and the autotune candidates
(triton_mm and friends) then dominate the trace.

The method behind the numbers is in kernel-analysis.md next to this file.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

CONTAINER_DIR = "/mad_kernel"

# One rocprofv3 pass per line. Every counter here was collected on gfx950
# (MI350X, ROCm 7.14) in one run; FETCH_SIZE and WRITE_SIZE each need their own pass.
GROUPS = {
    "compute": [
        "SQ_WAVES SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_ACTIVE_INST_VALU GRBM_GUI_ACTIVE",
        "SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_SALU SQ_INSTS_SMEM SQ_INSTS_LDS SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR",
        "SQ_VALU_MFMA_BUSY_CYCLES SQ_INSTS_VALU_MFMA_MOPS_BF16 SQ_INSTS_VALU_MFMA_MOPS_F16 SQ_INSTS_VALU_MFMA_MOPS_F32 SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE",
        "FETCH_SIZE",
        "WRITE_SIZE",
    ],
    "memory": [
        "SQ_WAVES SQ_WAVE_CYCLES SQ_WAIT_ANY SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR SQ_INSTS_LDS GRBM_GUI_ACTIVE",
        "FETCH_SIZE",
        "WRITE_SIZE",
        "TCC_HIT_sum TCC_MISS_sum TCP_TCC_READ_REQ_sum",
        "SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE SQ_INSTS_VALU_MFMA_MOPS_BF16 SQ_INSTS_VALU_MFMA_MOPS_F16",
    ],
}
MOPS = ("SQ_INSTS_VALU_MFMA_MOPS_BF16", "SQ_INSTS_VALU_MFMA_MOPS_F16",
        "SQ_INSTS_VALU_MFMA_MOPS_F32", "SQ_INSTS_VALU_MFMA_MOPS_F8")

# Public peak HBM bandwidth (TB/s) and dense BF16 matrix throughput (TFLOP/s).
PEAKS_BY_PRODUCT = {
    "MI355X": (8.0, 2516.6), "MI350X": (8.0, 2306.9), "MI325X": (6.0, 1307.4),
    "MI300X": (5.3, 1307.4), "MI300A": (5.3, 980.6), "MI250X": (3.2, 383.0),
    "MI250": (3.2, 362.1), "MI210": (1.6, 181.0),
}
PEAKS_BY_ARCH = {"gfx950": "MI350X", "gfx942": "MI300X", "gfx90a": "MI250X"}
# LDS per CU (KiB), SIMDs per CU, max waves per SIMD, VGPR file per SIMD lane.
ARCH_LIMITS = {
    "gfx950": {"lds_kib": 160, "simds": 4, "max_waves": 8, "vgprs": 512},
    "gfx942": {"lds_kib": 64, "simds": 4, "max_waves": 8, "vgprs": 512},
    "gfx90a": {"lds_kib": 64, "simds": 4, "max_waves": 8, "vgprs": 512},
}
LATENCY_FRACTION = 0.10


def write_json(run_dir: Path, name: str, data: dict) -> None:
    path = run_dir / name
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))
    print(f"wrote {path}", file=sys.stderr)


def parse_context(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


# ---------------------------------------------------------------- context

SHIM = """#!/bin/bash
# The first python3/python the model script starts runs under rocprofv3. Its
# children inherit the tool through the environment. The bash launcher is not
# profiled: with counters on, a forked bash subshell deadlocks.
self={inside}/shim
real=$(PATH="${{PATH//$self:/}}" command -v "${{0##*/}}")
[ -n "${{MAD_ROCPROF_ACTIVE:-}}" ] && exec "$real" "$@"
export MAD_ROCPROF_ACTIVE=1
exec {profiler} "$real" "$@"
"""


def write_shim(run_dir: Path, profiler: str) -> None:
    shim = run_dir / "shim"
    shim.mkdir(exist_ok=True)
    for name in ("python3", "python"):
        path = shim / name
        path.write_text(SHIM.format(inside=CONTAINER_DIR, profiler=profiler))
        path.chmod(0o755)


def move_aside(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        n = 1
        while path.with_name(f"{path.name}.prev-{n}").exists():
            n += 1
        path.rename(path.with_name(f"{path.name}.prev-{n}"))


def cmd_context(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = parse_context(args.context)
    inside = CONTAINER_DIR
    profiler = ["rocprofv3", "--kernel-trace"]
    tool, cmd = "", ""
    out = f"{inside}/rocprof/{args.mode}"
    if args.mode == "warm":
        # Start from an empty cache so the autotune results come from an unprofiled run.
        move_aside(run_dir / "co")
    else:
        # card, rank and pcsample read every CSV under this directory. The
        # container writes it as root; rocprof/ stays ours so it can be renamed.
        (run_dir / "rocprof").mkdir(exist_ok=True)
        move_aside(run_dir / "rocprof" / args.mode)
    if args.mode == "dispatch":
        # A rocprofv3 kernel trace also records kernels replayed from HIP graphs.
        tool = "rocprofv3"
    elif args.mode == "counters":
        (run_dir / "counters.txt").write_text(
            "".join(f"pmc: {line}\n" for line in GROUPS[args.group]))
        profiler += ["-i", f"{inside}/counters.txt"]
        if args.kernel:
            profiler += ["--kernel-include-regex", args.kernel]
        tool = "rocprofv3_compute"
    elif args.mode == "pc_sampling":
        profiler += ["--pc-sampling-beta-enabled", "--pc-sampling-method", "stochastic",
                     "--pc-sampling-unit", "cycles",
                     "--pc-sampling-interval", str(args.interval)]
        tool = "rocprofv3_pc_sampling"
    elif args.mode == "thread_trace":
        if not args.kernel:
            sys.exit("thread_trace needs --kernel, a regex for one kernel. "
                     "An unfiltered --att trace records every launch.")
        # One CU and a short iteration range: a full trace of every launch is
        # gigabytes, and --att records the CU named by --att-target-cu only.
        profiler += ["--att", "--att-target-cu", str(args.att_cu),
                     "--kernel-include-regex", args.kernel,
                     "--kernel-iteration-range", args.iteration_range]
        tool = "rocprofv3"
    if tool:
        # No -o: the default %hostname%/%pid% names keep child processes apart.
        profiler += ["--output-format", "csv", "-d", out, "--"]
        write_shim(run_dir, " ".join(shlex.quote(part) for part in profiler))
        cmd = f"env PATH={inside}/shim:$PATH"

    ctx.setdefault("docker_mounts", {})[inside] = str(run_dir)
    env = ctx.setdefault("docker_env_vars", {})
    env["MAD_CODE_OBJECT_DIR"] = f"{inside}/co"
    if args.serialize:
        # Counter collection can stall a host wait (a synchronize or a copy to
        # the CPU) when the host has queued tens of thousands of tiny launches.
        # Waiting after every launch keeps the queue short.
        env["AMD_SERIALIZE_KERNEL"] = "3"
    env.setdefault("TRITON_CACHE_DIR", f"{inside}/co/triton")
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{inside}/co/inductor")
    tools = [t for t in ctx.get("tools", []) if not t.get("name", "").startswith(("rocprof", "rocm_trace_lite"))]
    if tool:
        tools.append({"name": tool, "cmd": cmd} if cmd else {"name": tool})
    # Always set tools: a run from a manifest keeps the previous run's tools
    # when the new context has no tools key.
    ctx["tools"] = tools
    name = ("context.json" if args.mode in ("counters", "pc_sampling", "thread_trace")
            else f"context_{args.mode}.json")
    (run_dir / name).write_text(json.dumps(ctx, indent=2) + "\n")
    print(json.dumps(ctx))


# ---------------------------------------------------------------- card

def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def gpu_agent(run_dir: Path) -> dict:
    for path in sorted(run_dir.rglob("*agent_info.csv")):
        for row in read_csv(path):
            if row.get("Agent_Type") == "GPU":
                return row
    return {}


def peaks(agent: dict, arch: str, bw: float | None, tflops: float | None) -> dict:
    product = agent.get("Product_Name", "")
    key = next((k for k in PEAKS_BY_PRODUCT if k in product), PEAKS_BY_ARCH.get(arch))
    base_bw, base_tf = PEAKS_BY_PRODUCT.get(key, (None, None))
    return {"product": product or key, "hbm_tbps": bw or base_bw,
            "bf16_tflops": tflops or base_tf,
            "source": "override" if (bw or tflops) else "public spec sheet"}


def number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_counters(run_dir: Path) -> dict:
    """Per kernel variant: counter values per dispatch, durations, launch shape.

    Autotuning launches candidates under the same name, so a variant is the name
    plus its registers, LDS and workgroup size.
    """
    kernels: dict = {}
    files = sorted(run_dir.rglob("*counter_collection.csv"))
    for path in files:
        for row in read_csv(path):
            name = row.get("Kernel_Name", "")
            key = (name,) + tuple(row.get(c) for c in ("VGPR_Count", "Accum_VGPR_Count",
                                                        "LDS_Block_Size", "Workgroup_Size"))
            k = kernels.setdefault(key, {"values": collections.defaultdict(dict),
                                         "durations": {}, "row": row})
            dispatch = (str(path), row.get("Dispatch_Id"))
            value = number(row.get("Counter_Value"))
            if value is not None:
                counter = row.get("Counter_Name")
                k["values"][counter][dispatch] = k["values"][counter].get(dispatch, 0.0) + value
            start, end = number(row.get("Start_Timestamp")), number(row.get("End_Timestamp"))
            if start is not None and end is not None and end > start:
                k["durations"][dispatch] = end - start
    return kernels


def classify(card: dict) -> tuple[str, str]:
    bw, fl = card.get("pct_peak_bandwidth"), card.get("pct_peak_mfma")
    if bw is None or card.get("duration_us") is None:
        return "unknown", "FETCH_SIZE/WRITE_SIZE or the duration is missing"
    fl = fl or 0.0
    if max(bw, fl) < LATENCY_FRACTION * 100:
        return "latency", (f"{bw:.1f}% of peak bandwidth and {fl:.1f}% of peak MFMA: "
                           "neither unit is busy, so launch, occupancy or dependency latency dominates")
    if fl >= bw:
        return "compute", f"{fl:.1f}% of peak MFMA against {bw:.1f}% of peak bandwidth"
    return "bandwidth", f"{bw:.1f}% of peak bandwidth against {fl:.1f}% of peak MFMA"


def build_card(name: str, k: dict, peak: dict) -> dict:
    mean = {c: statistics.fmean(v.values()) for c, v in k["values"].items() if v}
    row = k["row"]
    dispatches = max((len(v) for v in k["values"].values()), default=0)
    durations = list(k["durations"].values())
    card: dict = {
        "kernel": name,
        "dispatches_profiled": dispatches,
        "duration_us": round(statistics.median(durations) / 1000, 3) if durations else None,
        "grid": number(row.get("Grid_Size")), "workgroup": number(row.get("Workgroup_Size")),
        "vgpr": number(row.get("VGPR_Count")), "agpr": number(row.get("Accum_VGPR_Count")),
        "sgpr": number(row.get("SGPR_Count")), "lds_bytes": number(row.get("LDS_Block_Size")),
        "scratch_bytes": number(row.get("Scratch_Size")),
        "counters": {c: round(v, 1) for c, v in sorted(mean.items())},
    }
    ns = statistics.median(durations) if durations else None
    if "FETCH_SIZE" in mean and "WRITE_SIZE" in mean:
        card["bytes"] = (mean["FETCH_SIZE"] + mean["WRITE_SIZE"]) * 1024
    mops = [mean[m] for m in MOPS if m in mean]
    if mops:
        card["mfma_flops"] = 512 * sum(mops)
    if ns:
        if "bytes" in card:
            tbps = card["bytes"] / ns / 1000
            card["achieved_tbps"] = round(tbps, 3)
            if peak["hbm_tbps"]:
                card["pct_peak_bandwidth"] = round(100 * tbps / peak["hbm_tbps"], 2)
        if "mfma_flops" in card:
            tflops = card["mfma_flops"] / ns / 1000
            card["achieved_mfma_tflops"] = round(tflops, 2)
            if peak["bf16_tflops"]:
                card["pct_peak_mfma"] = round(100 * tflops / peak["bf16_tflops"], 2)
    if card.get("bytes") and "mfma_flops" in card:
        card["flops_per_byte"] = round(card["mfma_flops"] / card["bytes"], 2)
        if peak["hbm_tbps"] and peak["bf16_tflops"]:
            card["ridge_flops_per_byte"] = round(peak["bf16_tflops"] / peak["hbm_tbps"], 1)
    waves = mean.get("SQ_WAVES")
    if waves:
        card["per_wave"] = {c.replace("SQ_INSTS_", "").lower(): round(mean[c] / waves, 1)
                            for c in sorted(mean) if c.startswith("SQ_INSTS_") and "MOPS" not in c}
    if mean.get("SQ_WAVE_CYCLES") and "SQ_WAIT_ANY" in mean:
        card["wait_share"] = round(mean["SQ_WAIT_ANY"] / mean["SQ_WAVE_CYCLES"], 3)
    hit, miss = mean.get("TCC_HIT_sum"), mean.get("TCC_MISS_sum")
    if hit is not None and miss is not None and hit + miss > 0:
        card["l2_hit_rate"] = round(hit / (hit + miss), 3)
    if mean.get("SQ_LDS_IDX_ACTIVE") and "SQ_LDS_BANK_CONFLICT" in mean:
        card["lds_bank_conflict_ratio"] = round(mean["SQ_LDS_BANK_CONFLICT"] / mean["SQ_LDS_IDX_ACTIVE"], 3)
    card["bound"], card["bound_reason"] = classify(card)
    return card


def cmd_card(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    kernels = collect_counters(run_dir / "rocprof" / "counters")
    if not kernels:
        sys.exit(f"no *counter_collection.csv under {run_dir}/rocprof/counters")
    agent = gpu_agent(run_dir)
    arch = args.arch or agent.get("Name", "")
    peak = peaks(agent, arch, args.peak_bw, args.peak_tflops)
    cards = [build_card(key[0], k, peak) for key, k in kernels.items()
             if not args.kernel or re.search(args.kernel, key[0])]
    for card in cards:
        card["total_us"] = round((card["duration_us"] or 0) * card["dispatches_profiled"], 1)
    cards.sort(key=lambda c: c["total_us"], reverse=True)
    names = collections.Counter(c["kernel"] for c in cards)
    for card in cards:
        card["variants_with_this_name"] = names[card["kernel"]]
    write_json(run_dir, "kernel_card.json", {"arch": arch, "peaks": peak, "kernels": cards})


# ---------------------------------------------------------------- asm

INSN_CLASSES = (
    ("mfma", ("v_mfma", "v_smfmac")),
    ("agpr_move", ("v_accvgpr",)),
    ("lds", ("ds_",)),
    ("scratch", ("scratch_",)),
    ("vmem", ("global_", "buffer_", "flat_")),
    ("smem", ("s_load", "s_buffer_load", "s_store", "s_buffer_store")),
    ("waitcnt", ("s_waitcnt",)),
    ("barrier", ("s_barrier",)),
    ("branch", ("s_cbranch", "s_branch", "s_setpc", "s_swappc")),
    ("salu", ("s_",)),
    ("valu", ("v_",)),
)
INSN_RE = re.compile(r"^\s+([a-z][a-z0-9_]*)(\s|$)")
LABEL_RE = re.compile(r"^(\.LBB[0-9_]+):")
BRANCH_RE = re.compile(r"^\s+s_c?branch\w*\s+(\.LBB[0-9_]+)")
META_RE = re.compile(r"^\s*-?\s*\.(vgpr_count|agpr_count|sgpr_count|vgpr_spill_count|sgpr_spill_count|"
                     r"group_segment_fixed_size|private_segment_fixed_size|wavefront_size):\s+(\d+)")
OCC_RE = re.compile(r"^; Occupancy: (\d+)")
WIDTH_RE = re.compile(r"(dwordx\d|dword|b128|b96|b64|b32|b16|ubyte|sbyte|ushort|sshort|short|byte|d16\w*)")


def insn_class(op: str) -> str:
    for name, prefixes in INSN_CLASSES:
        if op.startswith(prefixes):
            return name
    return "other"


def mix(ops: list[str]) -> dict:
    counts = collections.Counter(insn_class(op) for op in ops)
    return dict(counts.most_common())


def kernel_body(lines: list[str], name: str) -> list[str]:
    start = next((i for i, l in enumerate(lines) if l.startswith(f"{name}:")), None)
    if start is None:
        return []
    end = next((i for i in range(start, len(lines)) if lines[i].startswith(".Lfunc_end")), len(lines))
    return lines[start:end]


def summarize_amdgcn(path: Path, name: str, limits: dict) -> dict:
    lines = path.read_text(errors="replace").splitlines()
    body = kernel_body(lines, name) or lines
    ops, labels, branches = [], {}, []
    for line in body:
        label = LABEL_RE.match(line)
        if label:
            labels[label.group(1)] = len(ops)
            continue
        m = INSN_RE.match(line)
        if not m or m.group(1).startswith("."):
            continue
        b = BRANCH_RE.match(line)
        if b:
            branches.append((len(ops), b.group(1)))
        ops.append(m.group(1))
    meta: dict = {}
    occupancy = None
    for line in lines:
        m = META_RE.match(line)
        if m:
            meta.setdefault(m.group(1), int(m.group(2)))
        o = OCC_RE.match(line)
        if o:
            occupancy = int(o.group(1))
    sidecar = path.with_suffix(".json")
    triton = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    lds = meta.get("group_segment_fixed_size", 0) + int(triton.get("shared", 0) or 0)
    loops = []
    for at, target in branches:
        if target in labels and labels[target] <= at:
            seg = ops[labels[target]:at + 1]
            loops.append({"label": target, "instructions": len(seg), "mix": mix(seg)})
    loops.sort(key=lambda l: l["instructions"], reverse=True)
    mem_ops = collections.Counter(op for op in ops if insn_class(op) in ("vmem", "lds", "scratch"))
    widths = collections.Counter()
    for op, n in mem_ops.items():
        w = WIDTH_RE.search(op)
        widths[f"{insn_class(op)}:{w.group(1) if w else '?'}"] += n
    waves_per_wg = int(triton.get("num_warps") or 0) or None
    summary = {
        "kernel": name, "source": path.name,
        "instructions": len(ops), "mix": mix(ops),
        "memory_ops": dict(mem_ops.most_common(12)), "memory_widths": dict(widths.most_common()),
        "loops": loops[:3],
        "vgpr": meta.get("vgpr_count"), "agpr": meta.get("agpr_count"), "sgpr": meta.get("sgpr_count"),
        "vgpr_spill": meta.get("vgpr_spill_count"), "sgpr_spill": meta.get("sgpr_spill_count"),
        "scratch_bytes": meta.get("private_segment_fixed_size"), "lds_bytes": lds,
        "waves_per_workgroup": waves_per_wg,
        "compiler_occupancy": occupancy,
    }
    if triton:
        summary["triton"] = {k: triton.get(k) for k in ("num_warps", "num_stages", "waves_per_eu",
                                                          "matrix_instr_nonkdim", "kpack", "shared")}
    summary["occupancy_waves_per_simd"], summary["occupancy_limiter"] = occupancy_estimate(summary, limits)
    return summary


def occupancy_estimate(s: dict, limits: dict) -> tuple[int | None, str]:
    """Waves per SIMD: the lowest of the register, LDS and hardware limits."""
    candidates = {"hardware": limits["max_waves"]}
    regs = (s.get("vgpr") or 0) + (s.get("agpr") or 0)
    if s.get("compiler_occupancy"):
        candidates["registers"] = s["compiler_occupancy"]
    elif regs:
        granule = -(-regs // 8) * 8
        candidates["registers"] = limits["vgprs"] // granule
    if s.get("lds_bytes") and s.get("waves_per_workgroup"):
        wgs = (limits["lds_kib"] * 1024) // s["lds_bytes"]
        candidates["lds"] = (wgs * s["waves_per_workgroup"]) // limits["simds"]
    limiter = min(candidates, key=candidates.get)
    return min(candidates[limiter], limits["max_waves"]), limiter


def disassemble(path: Path, arch: str) -> Path | None:
    tool = os.environ.get("LLVM_OBJDUMP") or shutil.which("llvm-objdump") or "/opt/rocm/llvm/bin/llvm-objdump"
    if not Path(tool).exists() and not shutil.which(tool):
        return None
    out = path.with_suffix(".objdump.s")
    proc = subprocess.run([tool, "-d", f"--mcpu={arch}", str(path)], capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    out.write_text(proc.stdout)
    return out


def cmd_asm(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    roots = [Path(p) for p in args.code_objects] or [run_dir / "co"]
    card_path = run_dir / "kernel_card.json"
    card = json.loads(card_path.read_text()) if card_path.exists() else {}
    arch = args.arch or card.get("arch") or "gfx950"
    limits = ARCH_LIMITS.get(arch, ARCH_LIMITS["gfx950"])
    agent = gpu_agent(run_dir)
    if agent.get("Lds_Size_In_Kb"):
        limits = dict(limits, lds_kib=int(agent["Lds_Size_In_Kb"]),
                      simds=int(agent.get("Simd_Per_Cu") or limits["simds"]),
                      max_waves=int(agent.get("Max_Waves_Per_Simd") or limits["max_waves"]))
    target = args.kernel or next((k["kernel"] for k in card.get("kernels", [])), None)
    if not target:
        sys.exit("name the kernel with --kernel, or run `card` first")
    want = next((k for k in card.get("kernels", []) if k["kernel"] == target), {})
    files = [p for r in roots for p in r.rglob(f"{target}.amdgcn")]
    if not files:
        for hsaco in (p for r in roots for p in r.rglob(f"{target}.hsaco")):
            listing = disassemble(hsaco, arch)
            if listing:
                files.append(listing)
    if not files:
        write_json(run_dir, "kernel_asm_summary.json", {
            "kernel": target, "status": "no code object",
            "note": "No .amdgcn or .hsaco with this name under " + ", ".join(map(str, roots))
                    + ". Library kernels (hipBLASLt, rocBLAS, CK) ship inside shared libraries; "
                      "use PC sampling for them."})
        return
    variants = [summarize_amdgcn(p, target, limits) for p in sorted(set(files))]
    seen, unique = set(), []
    for v in variants:
        key = json.dumps({k: v[k] for k in ("mix", "vgpr", "agpr", "lds_bytes")}, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(v)
    matched = [v for v in unique if want.get("vgpr") is None or v["vgpr"] == want.get("vgpr")]
    status = "matched" if len(matched) == 1 else ("ambiguous" if matched else "unmatched")
    write_json(run_dir, "kernel_asm_summary.json", {
        "kernel": target, "arch": arch, "status": status,
        "match_rule": "VGPR count from the counter CSV" if want else "name only",
        "variants_found": len(unique), "variants": matched or unique})


# ---------------------------------------------------------------- pcsample

def cmd_pcsample(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    # Dispatch ids restart in every profiled process, so key them by the file prefix.
    out = run_dir / "rocprof" / "pc_sampling"
    names = {}
    for path in out.rglob("*kernel_trace.csv"):
        prefix = (path.parent, path.name[: -len("kernel_trace.csv")])
        for row in read_csv(path):
            names[prefix + (row.get("Dispatch_Id"),)] = row.get("Kernel_Name", "")
    samples = []
    for path in out.rglob("*pc_sampling*.csv"):
        prefix = (path.parent, path.name.split("pc_sampling")[0])
        for row in read_csv(path):
            row["kernel"] = names.get(prefix + (row.get("Dispatch_Id"),), "?")
            samples.append(row)
    if not samples:
        write_json(run_dir, "kernel_pc_samples.json", {
            "status": "no samples", "note": "No *pc_sampling*.csv under rocprof/pc_sampling"})
        return
    per_kernel = collections.Counter(r["kernel"] for r in samples)
    if args.kernel:
        target = args.kernel
        rows = [r for r in samples if re.search(args.kernel, r["kernel"])]
    else:
        target = next(k for k, _ in per_kernel.most_common() if k != "?") if len(per_kernel) > 1 else "?"
        rows = [r for r in samples if r["kernel"] == target]
    total = len(rows) or 1

    def share(counter: collections.Counter, top: int) -> list:
        return [{"value": v, "samples": n, "share": round(n / total, 3)} for v, n in counter.most_common(top)]

    short = lambda s: s.replace("ROCPROFILER_PC_SAMPLING_INSTRUCTION_", "")
    write_json(run_dir, "kernel_pc_samples.json", {
        "kernel": target, "samples": len(rows),
        "kernels_sampled": {k[:120]: n for k, n in per_kernel.most_common(5)},
        "issued_share": round(sum(r.get("Wave_Issued_Instruction") == "1" for r in rows) / total, 3),
        "stall_reasons": share(collections.Counter(short(r.get("Stall_Reason", "")) for r in rows), 8),
        "instruction_types": share(collections.Counter(short(r.get("Instruction_Type", "")) for r in rows), 8),
        "hot_instructions": share(collections.Counter(r.get("Instruction", "") for r in rows), 12),
    })


# ---------------------------------------------------------------- thread trace

def att_class(instruction: str) -> str | None:
    """Instruction class for one row of a rocprofv3 --att stats CSV."""
    text = instruction.strip()
    if not text or text.startswith(";"):
        return None
    op = text.split()[0]
    if op.startswith("v_mfma"):
        return "mfma"
    if op.startswith("s_waitcnt"):
        return "waitcnt"
    if op.startswith("s_barrier"):
        return "barrier"
    if op.startswith("ds_"):
        return "lds"
    if op.startswith(("buffer_", "global_", "flat_")):
        return "vmem"
    if op.startswith(("s_load", "s_buffer_load")):
        return "smem"
    if op.startswith("v_"):
        return "valu"
    if op.startswith("s_"):
        return "salu"
    return "other"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_threadtrace(args: argparse.Namespace) -> None:
    """Reduce a rocprofv3 --att run to one record per compiled variant.

    Each stats_*.csv is one traced launch. Launches that saved the same code
    object are one variant. The .out files --att writes are matched by content
    to the code objects under co/, which is how two kernels that share a name
    and a register count are told apart.
    """
    run_dir = Path(args.run_dir)
    out = run_dir / "rocprof" / "thread_trace"
    names: dict[str, dict] = {}
    for path in out.rglob("*kernel_trace.csv"):
        for row in read_csv(path):
            names[row.get("Dispatch_Id", "")] = row
    dispatches = []
    for path in sorted(out.rglob("stats_*.csv")):
        rows = read_csv(path)
        kernel = ""
        instructions = []
        code_objs = []
        for row in rows:
            text = row.get("Instruction", "").strip()
            if text.startswith(";") and not kernel:
                kernel = text.lstrip("; ").split()[0]
            if att_class(text):
                instructions.append(row)
                code_objs.append(row.get("CodeObj", ""))
        found = re.search(r"dispatch_(\d+)", path.name)
        dispatch_id = found.group(1) if found else ""
        traced = names.get(dispatch_id, {})
        dispatches.append({
            "dispatch_id": dispatch_id,
            "kernel": traced.get("Kernel_Name") or kernel,
            "kernel_id": traced.get("Kernel_Id", ""),
            "code_objects": tuple(sorted(set(code_objs))),
            "instructions": instructions,
        })
    if args.kernel:
        dispatches = [d for d in dispatches if re.search(args.kernel, d["kernel"])]
    if not dispatches:
        write_json(run_dir, "kernel_thread_trace.json", {
            "status": "no trace",
            "note": "No stats_*.csv under rocprof/thread_trace. "
                    "Run context --mode thread_trace first."})
        return

    cache: dict[str, list[str]] = {}
    for path in (run_dir / "co").rglob("*"):
        if path.is_file() and path.suffix in (".hsaco", ".out", ".co"):
            cache.setdefault(sha256(path), []).append(str(path.relative_to(run_dir)))
    saved = {}
    for path in out.rglob("*.out"):
        found = re.search(r"code_object_id_(\d+)", path.name)
        if found:
            saved[found.group(1)] = path

    groups: dict[tuple, list] = {}
    for dispatch in dispatches:
        groups.setdefault(dispatch["code_objects"], []).append(dispatch)
    variants = []
    for code_objects, items in groups.items():
        per_class: dict[str, list] = collections.defaultdict(lambda: [0, 0, 0])
        per_inst: dict[str, list] = collections.defaultdict(lambda: [0, 0, 0])
        static: collections.Counter = collections.Counter()
        for row in items[0]["instructions"]:
            kind = att_class(row.get("Instruction", ""))
            if kind:
                static[kind] += 1
        for item in items:
            for row in item["instructions"]:
                kind = att_class(row.get("Instruction", ""))
                if not kind:
                    continue
                hit = int(row.get("Hitcount") or 0)
                latency = int(row.get("Latency") or 0)
                stall = int(row.get("Stall") or 0)
                per_class[kind][0] += hit
                per_class[kind][1] += latency
                per_class[kind][2] += stall
                inst = per_inst[row.get("Instruction", "")]
                inst[0] += latency
                inst[1] += stall
                inst[2] += hit
        latency_total = sum(v[1] for v in per_class.values()) or 1
        stall_total = sum(v[2] for v in per_class.values()) or 1
        objects = []
        for code_id in code_objects:
            path = saved.get(code_id)
            if path is None:
                objects.append({"id": code_id})
                continue
            digest = sha256(path)
            objects.append({"id": code_id, "file": path.name, "sha256": digest,
                            "matches": cache.get(digest, [])})
        variants.append({
            "code_objects": list(code_objects),
            "dispatches_traced": len(items),
            "dispatch_ids": [d["dispatch_id"] for d in items],
            "kernel_ids": sorted({d["kernel_id"] for d in items if d["kernel_id"]}),
            "static_instructions": sum(static.values()),
            "static_mfma": static["mfma"],
            "issues": sum(v[0] for v in per_class.values()),
            "latency_cycles": sum(v[1] for v in per_class.values()),
            "stall_cycles": sum(v[2] for v in per_class.values()),
            "by_class": {kind: {"issues": v[0],
                                "latency_share": round(v[1] / latency_total, 3),
                                "stall_share": round(v[2] / stall_total, 3)}
                         for kind, v in sorted(per_class.items(), key=lambda kv: -kv[1][1])},
            "top_instructions": [{"instruction": inst, "latency_share": round(v[0] / latency_total, 3),
                                  "stall_share": round(v[1] / stall_total, 3), "hits": v[2]}
                                 for inst, v in sorted(per_inst.items(), key=lambda kv: -kv[1][0])[:6]],
            "code_object_files": objects,
        })
    kernels = collections.Counter(d["kernel"] for d in dispatches)
    write_json(run_dir, "kernel_thread_trace.json", {
        "status": "ok",
        "kernel": args.kernel or kernels.most_common(1)[0][0],
        "dispatches_traced": len(dispatches),
        "variants": variants,
        "scope": "one compute unit (--att-target-cu); a short kernel-iteration range",
        "device_support": "rocprofv3 documents thread trace for MI200 and MI300. "
                          "The same command completed on gfx950 with rocprofv3 1.3.2.",
    })


# ---------------------------------------------------------------- rank

def cmd_rank(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    traces = sorted((run_dir / "rocprof" / "dispatch").rglob("*kernel_trace.csv"))
    if not traces:
        sys.exit(f"no *kernel_trace.csv under {run_dir}/rocprof/dispatch; run context --mode dispatch first")
    total, count = collections.Counter(), collections.Counter()
    for path in traces:
        for row in read_csv(path):
            try:
                total[row["Kernel_Name"]] += (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1000
            except (KeyError, ValueError):
                continue
            count[row["Kernel_Name"]] += 1
    busy = sum(total.values()) or 1.0
    write_json(run_dir, "kernel_rank.json", {
        "dispatches": sum(count.values()),
        "busy_us": round(busy, 1),
        "kernels": [{"kernel": name, "dispatches": count[name], "total_us": round(us, 1),
                     "avg_us": round(us / count[name], 2), "share_pct": round(100 * us / busy, 1)}
                    for name, us in total.most_common(args.top)],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("context", help="additional-context for a kernel-counter or PC-sampling run")
    p.add_argument("--mode", choices=("warm", "dispatch", "counters", "pc_sampling", "thread_trace"),
                   default="counters",
                   help="warm: no profiler, fills the code-object cache; dispatch: rocprofv3 kernel trace on it")
    p.add_argument("--group", choices=sorted(GROUPS), default="compute")
    p.add_argument("--kernel", default="", help="regex passed to --kernel-include-regex")
    p.add_argument("--interval", type=int, default=1048576, help="PC-sampling interval in cycles")
    p.add_argument("--att-cu", type=int, default=1, help="compute unit for --att-target-cu")
    p.add_argument("--iteration-range", default="[1-8]",
                   help="kernel launches --att records, as a rocprofv3 --kernel-iteration-range")
    p.add_argument("--serialize", action="store_true",
                   help="set AMD_SERIALIZE_KERNEL=3 so the host waits after every launch; "
                        "for an eager model whose counter run stalls at a host wait")
    p.add_argument("--run-dir", required=True, help="host directory mounted at " + CONTAINER_DIR)
    p.add_argument("--context", default="", help="base additional-context (weights mount, env vars)")
    p.set_defaults(func=cmd_context)
    p = sub.add_parser("rank", help="kernels by GPU time from the dispatch-mode kernel trace")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--top", type=int, default=20)
    p.set_defaults(func=cmd_rank)
    for name, func, help_text in (("card", cmd_card, "kernel card from counter CSVs"),
                                  ("asm", cmd_asm, "code-object summary"),
                                  ("pcsample", cmd_pcsample, "PC-sampling summary"),
                                  ("threadtrace", cmd_threadtrace, "rocprofv3 --att summary")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--kernel", default="")
        p.add_argument("--arch", default="")
        if name == "card":
            p.add_argument("--peak-bw", type=float, help="peak HBM TB/s override")
            p.add_argument("--peak-tflops", type=float, help="peak dense BF16 TFLOP/s override")
        if name == "asm":
            p.add_argument("--code-objects", action="append", default=[],
                           help="directory holding .amdgcn/.hsaco (default <run-dir>/co)")
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
